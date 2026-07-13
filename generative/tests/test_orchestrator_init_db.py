"""#231: `orchestrator.main()` ruft `db.init_db()` nie auf — zwei Datenverlust-Facetten.

Facette 1 — frische `ATOMIC_DB_PATH`: `get_db()` legt die Datei ohne Schema an,
jeder DB-Write (pipeline_runs-Insert) degradiert still zu
`[warn] DB-Write fehlgeschlagen: no such table: ...` (die Exception wird im
Write-Block Z.2755ff. abgefangen, der Lauf selbst "gelingt").

Facette 2 — bestehende DB mit ALTEM Schema (z.B. vor #197/#220): die
`_add_column`-Migrationen in `init_db()` laufen nie, ein Insert nach einer
Schema-Erweiterung (hier: `n_extracted`, #220) schlägt still fehl
(`table pipeline_runs has no column named n_extracted`).

Fährt `main()` mit `--no-llm --dry-run` auf dem Beispiel-PDF, LLM/Netz
deterministisch gestubbt — Harness wiederverwendet aus test_ci_smoke_e2e /
test_pipeline_run_persist_no_eval.

RED auf master-Stand: `orchestrator.main()` ruft `db.init_db()` an keiner
Stelle auf → in Facette 1 bleiben `pipeline_runs`/`note_evals` inexistent, in
Facette 2 bleibt die Spalte `n_extracted` unbekannt und der Insert scheitert
still (kein sichtbarer Testfehler, nur eine leere/unvollständige Tabelle —
daher prüfen die Assertions unten direkt auf DB-Zustand statt auf Exceptions).
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from generative import db
from generative import embeddings as embeddings_mod
from generative import orchestrator
from generative.agents import base as agents_base
from generative.agents import context_builder
from generative.agents import quality as quality_agent
from generative.config import AGENT_VERSION
from generative.tests.test_ci_smoke_e2e import (
    EXAMPLE_PDF,
    REPO_ROOT,
    _FakeEmbeddingModel,
    _fake_check_quality,
    _install_network_guard,
    _make_backends,
)
from shared.db_schema import SCHEMA_SQL

pytestmark = pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext (poppler) nicht installiert — CI hat es, lokal ggf. nachinstallieren.",
)

# Schema vor #197/#220: identisch zu SCHEMA_SQL, aber ohne die Spalte
# `n_extracted` in pipeline_runs — simuliert eine DB, die vor der
# Schema-Erweiterung angelegt und seither nie migriert wurde.
_OLD_SCHEMA_NO_N_EXTRACTED = SCHEMA_SQL.replace("    n_extracted       INT  DEFAULT 0,\n", "")
assert "n_extracted" not in _OLD_SCHEMA_NO_N_EXTRACTED, "Test-Fixture-Bug: altes Schema enthaelt noch n_extracted"


def _run_main_dry(tmp_path, monkeypatch, pdf_name: str):
    """Fährt orchestrator.main() mit --no-llm --dry-run auf einer PDF-Kopie.

    Gestubbtes LLM-Backend, geblockte Netz-Zugriffe, Fake-Embeddings — Harness
    identisch zu test_pipeline_run_persist_no_eval.py."""
    network_calls: list[str] = []
    _install_network_guard(monkeypatch, network_calls)

    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    monkeypatch.setenv("ATOMIC_AGENT_PROFILE", "fast")
    monkeypatch.delenv("ENABLE_NLI_VALIDATION", raising=False)
    monkeypatch.delenv("ATOMIC_AGENT_TRACING", raising=False)
    monkeypatch.delenv("ENABLE_ACRONYM_LLM_FALLBACK", raising=False)

    from generative import config as _config

    monkeypatch.setattr(_config, "ENABLE_LLM", _config.ENABLE_LLM)

    monkeypatch.setattr(embeddings_mod, "_model", lambda: _FakeEmbeddingModel())
    fake_vault = tmp_path / "fake-vault"
    fake_vault.mkdir()
    monkeypatch.setattr(context_builder, "VAULT", fake_vault)
    monkeypatch.setattr(quality_agent, "check_quality", _fake_check_quality)

    import generative.tools.pdf_enrich as pdf_enrich_mod

    monkeypatch.setattr(pdf_enrich_mod, "enrich", lambda *a, **k: None)

    sync_backend, async_backend = _make_backends(network_calls)
    monkeypatch.setattr(agents_base, "_backend_call_full", sync_backend)
    monkeypatch.setattr(agents_base, "_backend_call_full_async", async_backend)

    fixture_pdf = tmp_path / pdf_name
    shutil.copyfile(EXAMPLE_PDF, fixture_pdf)
    eval_dir = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / fixture_pdf.stem

    inbox_dir = tmp_path / "inbox"
    argv = ["--source", str(fixture_pdf), "--dry-run", "--no-llm", "--inbox-dir", str(inbox_dir)]

    try:
        orchestrator.main(argv)
    finally:
        agents_base.clear_llm_runtime_config()
        shutil.rmtree(eval_dir, ignore_errors=True)

    assert network_calls == [], f"Echter Netz-Zugriff aufgezeichnet: {network_calls}"


def test_orchestrator_creates_schema_on_fresh_db_path(tmp_path, monkeypatch, isolate_pipeline_side_effects):
    """Facette 1: frische ATOMIC_DB_PATH ohne Schema — main() muss init_db() ziehen.

    Die autouse-Fixture `isolate_pipeline_side_effects` ruft selbst init_db() auf
    ihrem eigenen tmp-Pfad — hier biegen wir db.DB_PATH danach auf einen NEUEN,
    noch nicht angelegten Pfad um, damit der Fresh-Path-Fall wirklich exerziert
    wird (kein File, kein Schema, bevor main() läuft)."""
    fresh_db_path = tmp_path / "fresh-db" / "atomic_analytics.db"
    assert not fresh_db_path.exists()
    monkeypatch.setattr(db, "DB_PATH", fresh_db_path)

    _run_main_dry(tmp_path, monkeypatch, "init-db-fresh-fixture.pdf")

    assert fresh_db_path.exists(), "main() sollte die DB-Datei am neuen Pfad anlegen"
    conn = sqlite3.connect(str(fresh_db_path))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "pipeline_runs" in tables, f"pipeline_runs fehlt — Tabellen: {tables}"
        assert "note_evals" in tables, f"note_evals fehlt — Tabellen: {tables}"
        rows = conn.execute("SELECT run_id, pipeline_version, pdf_source FROM pipeline_runs").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f"Erwartet: genau 1 pipeline_runs-Zeile, gefunden: {rows}"
    assert rows[0][1] == AGENT_VERSION
    assert rows[0][2] == "init-db-fresh-fixture.pdf"


def test_orchestrator_migrates_existing_db_missing_n_extracted_column(
    tmp_path, monkeypatch, isolate_pipeline_side_effects
):
    """Facette 2: bestehende DB mit ALTEM Schema (ohne n_extracted) — main() muss
    die `_add_column`-Migration nachziehen, sonst scheitert der Insert nach
    #220 still (`table pipeline_runs has no column named n_extracted`)."""
    old_db_path = tmp_path / "old-schema" / "atomic_analytics.db"
    old_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(old_db_path))
    try:
        conn.executescript(_OLD_SCHEMA_NO_N_EXTRACTED)
        conn.commit()
        cols_before = [r[1] for r in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
    finally:
        conn.close()
    assert "n_extracted" not in cols_before, "Test-Fixture-Bug: Alt-Schema hat n_extracted schon"

    monkeypatch.setattr(db, "DB_PATH", old_db_path)

    _run_main_dry(tmp_path, monkeypatch, "init-db-migrate-fixture.pdf")

    conn2 = sqlite3.connect(str(old_db_path))
    try:
        cols_after = [r[1] for r in conn2.execute("PRAGMA table_info(pipeline_runs)").fetchall()]
        assert "n_extracted" in cols_after, f"Migration griff nicht — Spalten: {cols_after}"
        rows = conn2.execute("SELECT run_id, pipeline_version, pdf_source FROM pipeline_runs").fetchall()
    finally:
        conn2.close()

    assert len(rows) == 1, f"Erwartet: genau 1 pipeline_runs-Zeile nach Migration, gefunden: {rows}"
    assert rows[0][1] == AGENT_VERSION
    assert rows[0][2] == "init-db-migrate-fixture.pdf"
