"""#239: echte Wall-Clock (inkl. Stage-8) wird nach main() korrekt in
pipeline_runs persistiert — nicht nur die Zeit bis VOR Stage-8.

insert_run() (orchestrator.py, vor Stage-8) schreibt wall_clock_s zunaechst
identisch zu duration_s (siehe test_db.py::test_insert_run_stores_wall_clock_s).
Erst NACH Stage-8 korrigiert db.update_wall_clock_s() die Zeile auf die echte
Gesamtzeit. Dieser Test faehrt main() end-to-end (mit gestubbtem LLM-Backend,
Muster aus test_ci_smoke_e2e/test_pipeline_run_persist_no_eval) und erzwingt
Stage-8 trotz fast-Profil ueber den ATOMIC_AGENT_INLINE_EVAL-Env-Override
(dieselbe Override-Semantik wie test_inline_eval_toggle.py).

`orchestrator.run_stage8_eval` selbst wird auf eine Sleep-Fake umgebogen (statt
den LLM-Judge zu stubben): entkoppelt den Test von Hard-Gate-/Vault-Routing-
Details der Fixture-Backends (ob eine Note ueberhaupt vault-approved wird, ist
fuer die Wall-Clock-Frage irrelevant) und erzeugt eine deterministisch messbare
Zeitspanne zwischen insert_run() und Stage-8-Ende — ohne Sleep waere die
Rundung auf eine Nachkommastelle bei 0 evaluierten Notes zu knapp fuer eine
stabile Unterscheidung von duration_s.

RED auf Vor-Fix-Stand: wall_clock_s bleibt nach main() identisch zu duration_s
(die Korrektur nach Stage-8 fehlt komplett) — GREEN: wall_clock_s > duration_s,
weil zwischen dem insert_run-Zeitpunkt und dem Stage-8-Ende messbar Zeit
vergangen ist.
"""

from __future__ import annotations

import shutil
import sqlite3
import time as _time

import pytest

from generative import embeddings as embeddings_mod
from generative import orchestrator
from generative.agents import base as agents_base
from generative.agents import context_builder
from generative.agents import quality as quality_agent
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


def _sleepy_stage8_eval(note_files, source_path, run_meta, *, fresh_run=False):
    """Ersetzt den echten Stage-8-Judge-Call — 150ms Sleep statt echter LLM-Arbeit,
    aber genug, um duration_s (vor Stage-8) und wall_clock_s (nach Stage-8) auf
    der 1-Nachkommastellen-Aufloesung der Zeit-Messung sicher zu trennen."""
    _time.sleep(0.15)
    return [], 0, 0


def test_wall_clock_s_corrected_after_stage8(tmp_path, monkeypatch, isolate_pipeline_side_effects):
    network_calls: list[str] = []
    _install_network_guard(monkeypatch, network_calls)

    # fast-Profil (kein Refine, niedriger Concept-Cap — bewährte, stabile
    # Harness aus test_pipeline_run_persist_no_eval.py) + expliziter Env-
    # Override erzwingt Stage-8 trotzdem an (dieselbe Override-Semantik wie
    # test_inline_eval_toggle.py::test_inline_eval_env_override_still_wins).
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    monkeypatch.setenv("ATOMIC_AGENT_PROFILE", "fast")
    monkeypatch.setenv("ATOMIC_AGENT_INLINE_EVAL", "1")
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

    # Stage-8 selbst gestubbt: kein echter LLM-Judge-Call, dafuer ein Sleep als
    # deterministisch messbare Zeitspanne (siehe Docstring _sleepy_stage8_eval).
    monkeypatch.setattr(orchestrator, "run_stage8_eval", _sleepy_stage8_eval)

    fixture_pdf = tmp_path / "wall-clock-fixture.pdf"
    shutil.copyfile(EXAMPLE_PDF, fixture_pdf)
    eval_dir = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / "wall-clock-fixture"

    inbox_dir = tmp_path / "inbox"
    argv = ["--source", str(fixture_pdf), "--dry-run", "--no-llm", "--inbox-dir", str(inbox_dir)]

    try:
        orchestrator.main(argv)
    finally:
        agents_base.clear_llm_runtime_config()
        shutil.rmtree(eval_dir, ignore_errors=True)

    assert network_calls == [], f"Echter Netz-Zugriff aufgezeichnet: {network_calls}"

    conn = sqlite3.connect(str(isolate_pipeline_side_effects.db_path))
    try:
        row = conn.execute("SELECT duration_s, wall_clock_s FROM pipeline_runs").fetchone()
    finally:
        conn.close()

    assert row is not None, "Erwartet: genau 1 pipeline_runs-Zeile"
    duration_s, wall_clock_s = row
    assert wall_clock_s > 0
    # Kernbefund #239: wall_clock_s (nach Stage-8) darf NICHT identisch zu
    # duration_s (vor Stage-8) bleiben — Stage-8 hat real Zeit verbraucht.
    assert wall_clock_s > duration_s, (
        f"wall_clock_s ({wall_clock_s}) == duration_s ({duration_s}) — "
        "Stage-8-Korrektur (update_wall_clock_s) griff nicht."
    )
