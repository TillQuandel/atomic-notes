"""Tests fuer die Quell-Pfad-Aufloesung im CLI-Hauptpfad von `orchestrator.main`
(#186-Nachbesserung, Cross-Model-Review).

Der `--load-drafts`-Zweig und die extractive-/eval_chunk_recall-Einstiegspunkte
nutzen bereits `shared.path_safety.resolve_source_path` (Apostroph-/
Anfuehrungszeichen-Glob-Fallback). Der normale `--source`-Pfad in
`orchestrator.main` wurde im Review uebersehen und brach mit einem nackten
`sys.exit` bei reinen Apostroph-Varianten ab. Kein voller Pipeline-Lauf hier --
`_run_extraction_stages` wird als Seam gestubbt, analog zu
test_maintainer_optin.py/test_orchestrator_export.py.
"""

from __future__ import annotations

import pytest

from generative import orchestrator


@pytest.fixture(autouse=True)
def _kein_globaler_llm_state(monkeypatch):
    """orchestrator.main ruft set_llm_runtime_config VOR der Quell-Aufloesung --
    das setzt sonst das globale _LLM_RUNTIME_SETTINGS und leakt Backend-Kwargs
    (call_timeout_sec) in spaeter laufende Tests (test_phoenix_span). Hier als
    No-op patchen; Phoenix-Tracing analog zu test_orchestrator_export.py stummschalten."""
    from generative.agents import base as agents_base

    monkeypatch.setattr(agents_base, "set_llm_runtime_config", lambda _cfg: None)
    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", lambda: None)


def test_missing_source_exits_with_datei_nicht_gefunden(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    missing = tmp_path / "nicht-vorhanden.pdf"
    with pytest.raises(SystemExit) as exc:
        orchestrator.main(["--source", str(missing)])
    msg = str(exc.value)
    assert "Datei nicht gefunden" in msg
    assert "nicht-vorhanden.pdf" in msg


def test_source_apostrophe_variant_resolved_via_glob_fallback(monkeypatch, tmp_path):
    # Datei liegt mit typografischem Apostroph (U+2019) vor, --source wird mit
    # dem geraden ' aufgerufen -- muss trotzdem gefunden werden (wie extractive/
    # eval_chunk_recall bereits per resolve_source_path).
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    real = tmp_path / "Porst’s-Buch.pdf"
    real.write_bytes(b"%PDF-1.4")
    queried = tmp_path / "Porst's-Buch.pdf"

    captured = {}

    def stop_here(_args, source_path, _runtime_config):
        captured["source_path"] = source_path
        raise RuntimeError("stop-here")

    monkeypatch.setattr(orchestrator, "_run_extraction_stages", stop_here)

    with pytest.raises(RuntimeError, match="stop-here"):
        orchestrator.main(["--source", str(queried)])

    assert captured["source_path"] == real
