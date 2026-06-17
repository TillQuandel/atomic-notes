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


def test_help_mentions_backend_env_and_doctor(capsys):
    # #49/M9: ATOMIC_AGENT_BACKEND im --help auffindbar (bisher nur README/doctor)
    cli.main(["--help"])
    out = capsys.readouterr().out
    assert "ATOMIC_AGENT_BACKEND" in out
    assert "doctor" in out
