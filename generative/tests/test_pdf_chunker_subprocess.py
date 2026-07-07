"""S5/S6 (#150): Timeout + Argument-Injection-Haertung der poppler-Subprozesse.

pdftotext/pdfinfo bekommen ein `timeout=120` (defektes/boesartiges PDF darf die
Pipeline nicht unbegrenzt haengen lassen) und der PDF-Pfad wird vor der Uebergabe
absolutiert (ein relativer Name mit fuehrendem "-" kann sonst als poppler-Option
fehlinterpretiert werden).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from generative.pipeline import pdf_chunker


class _FakeResult:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


# --- S5: Timeout -----------------------------------------------------------


def test_pdf_to_pages_passes_timeout(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeResult(stdout="Seiteninhalt\f")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pdf_chunker.pdf_to_pages(tmp_path / "x.pdf")
    assert captured["kwargs"].get("timeout") == pdf_chunker._PDF_SUBPROCESS_TIMEOUT_S


def test_pdf_to_pages_timeout_exits_with_hint(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        pdf_chunker.pdf_to_pages(tmp_path / "haengt.pdf")
    # Handlungsanleitende Meldung (kein nackter Stacktrace), nennt den Timeout.
    assert "nicht geantwortet" in str(exc.value)


def test_pdf_metadata_timeout_returns_empty(monkeypatch, tmp_path):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    # pdfinfo ist optional (fail-soft): Timeout -> {} statt Crash.
    assert pdf_chunker.pdf_metadata(tmp_path / "haengt.pdf") == {}


def test_pdf_metadata_passes_timeout(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeResult(stdout="Pages: 1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pdf_chunker.pdf_metadata(tmp_path / "x.pdf")
    assert captured["kwargs"].get("timeout") == pdf_chunker._PDF_SUBPROCESS_TIMEOUT_S


# --- S6: Argument-Injection (Pfad absolutiert) -----------------------------


def test_pdf_to_pages_absolutizes_relative_path(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeResult(stdout="Text\f")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Relativer Name mit fuehrendem "-": darf NICHT als poppler-Option ankommen.
    pdf_chunker.pdf_to_pages(Path("-oconfig.pdf"))
    path_arg = captured["argv"][1]
    assert os.path.isabs(path_arg)
    assert not path_arg.startswith("-")


def test_pdf_metadata_absolutizes_relative_path(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _FakeResult(stdout="Pages: 1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    pdf_chunker.pdf_metadata(Path("-oconfig.pdf"))
    path_arg = captured["argv"][1]
    assert os.path.isabs(path_arg)
    assert not path_arg.startswith("-")
