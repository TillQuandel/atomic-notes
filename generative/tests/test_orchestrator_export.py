"""Tests für das `--export-format`/`--export-dir`-Wiring in orchestrator.main
(Output-Projekt F4). Kein voller Pipeline-Lauf hier — die Format-Validierung
läuft fail-fast direkt nach `parse_args`, bevor irgendeine teure/mutierende
Stufe beginnt (siehe `_setup_phoenix_tracing`-Monkeypatch unten als Beweis).
Die eigentliche Export-Logik ist in test_export_runner.py getestet.
"""

from __future__ import annotations

import pytest

from generative import orchestrator


def test_invalid_export_format_exits_before_heavy_setup(monkeypatch):
    def _boom():
        raise AssertionError("_setup_phoenix_tracing lief VOR der Export-Format-Validierung — Fail-Fast verletzt")

    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", _boom)
    with pytest.raises(SystemExit) as exc:
        orchestrator.main(["--source", "irrelevant.pdf", "--export-format", "bogus-format"])
    assert "bogus-format" in str(exc.value)


def test_future_format_exits_with_geplant_hint(monkeypatch):
    def _boom():
        raise AssertionError("sollte nicht laufen")

    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", _boom)
    with pytest.raises(SystemExit) as exc:
        orchestrator.main(["--source", "irrelevant.pdf", "--export-format", "rtf"])
    assert "geplant" in str(exc.value).lower()


def test_valid_export_format_passes_fail_fast_check(monkeypatch):
    # Gueltiges Format darf die Validierung NICHT stoppen -- Beweis: der
    # naechste Schritt (_setup_phoenix_tracing) wird tatsaechlich erreicht.
    # ATOMIC_AGENT_GUI=1 unterdrueckt _auto_version_bump()/_auto_start_dashboard()
    # (echte, mutierende/prozess-startende Auto-Aktionen) -- ein Test darf
    # niemals config.py mutieren oder einen Dashboard-Prozess spawnen.
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    calls = []
    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", lambda: calls.append("reached"))
    # Ab hier weiterlaufen lassen wuerde einen echten Pipeline-Lauf brauchen --
    # wir brechen kontrolliert ab, sobald der Fail-Fast-Punkt erwiesen ist.
    monkeypatch.setattr(orchestrator, "load_runtime_config", lambda: (_ for _ in ()).throw(RuntimeError("stop-here")))
    with pytest.raises(RuntimeError, match="stop-here"):
        orchestrator.main(["--source", "irrelevant.pdf", "--export-format", "pdf,docx"])
    assert calls == ["reached"]


def test_no_export_format_flag_skips_validation_entirely(monkeypatch):
    # Ohne --export-format darf keine Format-Validierung/kein Import passieren --
    # bestehendes Verhalten (kein Flag gesetzt) bleibt unangetastet.
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")  # s. Begruendung oben
    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", lambda: None)
    monkeypatch.setattr(orchestrator, "load_runtime_config", lambda: (_ for _ in ()).throw(RuntimeError("stop-here")))
    with pytest.raises(RuntimeError, match="stop-here"):
        orchestrator.main(["--source", "irrelevant.pdf"])
