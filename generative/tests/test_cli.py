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


def test_doctor_ist_platzhalter(capsys):
    """`atomic-notes doctor` existiert, meldet aber 'nicht implementiert' (M1-S2)."""
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "doctor" in out
    assert "M1-S2" in out


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
