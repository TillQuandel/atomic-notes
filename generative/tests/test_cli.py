"""Tests für den atomic-notes-Konsolen-Entry-Point (generative/cli.py)."""
from __future__ import annotations

import pytest

from generative import cli


def test_run_delegates_an_orchestrator(monkeypatch):
    """`atomic-notes run <args>` reicht die Rest-Argumente an orchestrator.main durch."""
    import generative.orchestrator as orch

    captured = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return None

    monkeypatch.setattr(orch, "main", fake_main)
    rc = cli.main(["run", "--source", "x.pdf", "--dry-run"])
    assert rc == 0
    assert captured["argv"] == ["--source", "x.pdf", "--dry-run"]


def test_doctor_delegiert_an_doctor_main(monkeypatch):
    """`atomic-notes doctor` ruft generative.doctor.main und reicht den Exit-Code durch."""
    from generative import doctor

    monkeypatch.setattr(doctor, "main", lambda: 7)
    assert cli.main(["doctor"]) == 7


def test_ohne_argumente_usage_und_exitcode_2(capsys):
    rc = cli.main([])
    assert rc == 2
    assert "run" in capsys.readouterr().out


def test_unbekanntes_kommando_exitcode_2(capsys):
    rc = cli.main(["frobnicate"])
    assert rc == 2


def test_help_exitcode_0(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    assert "atomic-notes" in capsys.readouterr().out


def test_gui_invalid_port_returns_2_no_crash(capsys):
    # #5: `gui --port abc` darf nicht mit ValueError crashen, sondern sauber abbrechen.
    rc = cli.main(["gui", "--port", "abc"])
    assert rc == 2
    assert "port" in capsys.readouterr().err.lower()


def test_gui_missing_port_value_returns_2_no_crash(capsys):
    # #5: `gui --port` ohne Wert darf nicht mit IndexError crashen.
    rc = cli.main(["gui", "--port"])
    assert rc == 2
    assert "port" in capsys.readouterr().err.lower()


def test_gui_parses_valid_args(monkeypatch):
    # Gültige Argumente werden korrekt an serve() durchgereicht.
    import generative.gui.app as gui_app
    captured = {}
    monkeypatch.setattr(gui_app, "serve",
                        lambda **kw: captured.update(kw))
    rc = cli.main(["gui", "--port", "9001", "--no-browser"])
    assert rc == 0
    assert captured == {"port": 9001, "open_browser": False}


def test_help_mentions_backend_env_and_doctor(capsys):
    # #49/M9: ATOMIC_AGENT_BACKEND im --help auffindbar (bisher nur README/doctor)
    cli.main(["--help"])
    out = capsys.readouterr().out
    assert "ATOMIC_AGENT_BACKEND" in out
    assert "doctor" in out
