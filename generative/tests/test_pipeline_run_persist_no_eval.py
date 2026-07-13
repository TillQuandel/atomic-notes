"""#198 P1: pipeline_runs-Insert läuft auch bei deaktiviertem Inline-Eval.

Regressionstest für die "Insert-Lücke": Bei Profil fast/balanced (oder
ATOMIC_AGENT_INLINE_EVAL=0) kehrte `orchestrator.main()` VOR dem pipeline_runs-Insert
zurück. Der Lauf schrieb einen vollständigen Trace, bekam aber keine DB-Zeile — war
also keiner Pipeline-Version zuordenbar und verschwand aus allen versions-gefilterten
Dashboard-Ansichten.

Fährt `main()` mit `fast`-Profil (Inline-Eval AUS) auf dem echten Beispiel-PDF, LLM/Netz
deterministisch gestubbt (Harness wiederverwendet aus test_ci_smoke_e2e). Danach muss
GENAU EINE pipeline_runs-Zeile mit der aktuellen pipeline_version existieren. Die DB ist
ausschließlich die tmp-DB der autouse-Fixture `isolate_pipeline_side_effects` — nie die
produktive .cache/atomic_analytics.db.

RED auf master-Stand: Insert steht dort erst NACH dem inline_eval-skip-Return → 0 Zeilen.
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

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

pytestmark = pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext (poppler) nicht installiert — CI hat es, lokal ggf. nachinstallieren.",
)


def test_pipeline_run_persisted_when_inline_eval_disabled(tmp_path, monkeypatch, isolate_pipeline_side_effects):
    network_calls: list[str] = []
    _install_network_guard(monkeypatch, network_calls)

    # Inline-Eval AUS (fast-Profil) — genau der Pfad, der früher vor dem Insert zurückkehrte.
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    monkeypatch.setenv("ATOMIC_AGENT_PROFILE", "fast")
    monkeypatch.delenv("ENABLE_NLI_VALIDATION", raising=False)
    monkeypatch.delenv("ATOMIC_AGENT_TRACING", raising=False)
    monkeypatch.delenv("ENABLE_ACRONYM_LLM_FALLBACK", raising=False)

    # main(--no-llm) mutiert config.ENABLE_LLM = False als Modul-Attribut und stellt
    # es NIE zurück (bestehender Prozess-State-Leak, siehe orchestrator.main). Snapshot
    # per monkeypatch → Teardown restauriert den Ausgangswert, sonst laufen NACHfolgende
    # Tests (z. B. test_verifier_prepass) fälschlich im no-LLM-Modus.
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

    # Eigener PDF-Namespace + Cleanup der Dry-Run-eval-Kopien (vault_writer schreibt
    # sie hart nach .cache/eval/baseline/<stem>/<run_id>/ — nicht parametrisierbar,
    # #241: run_id-Unterordner seit dem Namespace-Fix). rmtree auf Stem-Ebene räumt
    # den run_id-Unterordner mit ab.
    fixture_pdf = tmp_path / "persist-fixture.pdf"
    shutil.copyfile(EXAMPLE_PDF, fixture_pdf)
    eval_dir = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / "persist-fixture"

    inbox_dir = tmp_path / "inbox"
    argv = ["--source", str(fixture_pdf), "--dry-run", "--no-llm", "--inbox-dir", str(inbox_dir)]

    try:
        orchestrator.main(argv)
    finally:
        agents_base.clear_llm_runtime_config()
        shutil.rmtree(eval_dir, ignore_errors=True)

    # Netz-Blockade als Sanity-Check: kein echter HTTP-Versuch.
    assert network_calls == [], f"Echter Netz-Zugriff aufgezeichnet: {network_calls}"

    # Der Lauf war ohne Inline-Eval — trotzdem muss die DB-Zeile existieren (#198 P1).
    conn = sqlite3.connect(str(isolate_pipeline_side_effects.db_path))
    try:
        rows = conn.execute("SELECT run_id, pipeline_version, pdf_source, n_generated FROM pipeline_runs").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f"Erwartet: genau 1 pipeline_runs-Zeile (Insert-Lücke #198 P1), gefunden: {rows}"
    assert rows[0][1] == AGENT_VERSION, f"pipeline_version falsch: {rows[0][1]!r} != {AGENT_VERSION!r}"
    assert rows[0][2] == "persist-fixture.pdf"
    assert rows[0][3] >= 1, "n_generated sollte die im Dry-Run verarbeiteten Notes widerspiegeln"
