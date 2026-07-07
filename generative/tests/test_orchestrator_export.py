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
    # Braucht echtes pandoc+typst: die Deps sind hier NICHT gemockt, also laeuft
    # der reale export_available()-Check gegen ein Binaerformat (pdf,docx). Ohne
    # die [export]-Deps quittiert der Fail-Fast-Pfad mit SystemExit -- dann
    # sauber skippen statt fehlschlagen (Muster wie test_export_convert.py).
    pytest.importorskip("pypandoc")
    pytest.importorskip("typst")
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


def test_binary_format_without_deps_exits_before_heavy_setup(monkeypatch):
    # Review-Fund 1: --export-format pdf ohne pandoc/typst muss SOFORT
    # abbrechen (mit pip-Hint), nicht erst nach dem kompletten Lauf.
    from generative.pipeline import export_convert

    monkeypatch.setattr(export_convert, "export_available", lambda: (False, "pandoc kaputt"))

    def _boom():
        raise AssertionError("_setup_phoenix_tracing lief VOR dem Deps-Check — Fail-Fast verletzt")

    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", _boom)
    with pytest.raises(SystemExit) as exc:
        orchestrator.main(["--source", "irrelevant.pdf", "--export-format", "pdf"])
    msg = str(exc.value)
    assert "pandoc kaputt" in msg
    assert "pip install" in msg
    assert "atomic-notes[export]" in msg


def test_pure_formats_do_not_check_export_deps(monkeypatch):
    # json/portable-md/obsidian-md brauchen keine Export-Deps -- der Check darf
    # fuer sie gar nicht laufen (sonst braeche json-Export ohne Deps).
    from generative.pipeline import export_convert

    def _deps_boom():
        raise AssertionError("export_available darf fuer reine Formate nicht aufgerufen werden")

    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    monkeypatch.setattr(export_convert, "export_available", _deps_boom)
    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", lambda: None)
    monkeypatch.setattr(orchestrator, "load_runtime_config", lambda: (_ for _ in ()).throw(RuntimeError("stop-here")))
    with pytest.raises(RuntimeError, match="stop-here"):
        orchestrator.main(["--source", "irrelevant.pdf", "--export-format", "json,portable-md,obsidian-md"])
