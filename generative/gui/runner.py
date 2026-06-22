"""Subprocess-Runner fuer die Live-GUI.

Faehrt `python -m generative.orchestrator …` als Subprocess und streamt dessen
stdout zeilenweise durch den RunParser zu strukturierten Events. Ein Subprocess
(statt In-Process-Aufruf) isoliert den Lauf sauber: der Orchestrator liest
VAULT/BACKEND beim Import aus ENV (config.py:7-12) und ruft `sys.exit()` —
beides unkritisch in einem eigenen Prozess.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator

from generative.gui.run_parser import RunParser


def build_argv(pdf_path: str, *, dry_run: bool, extra: list[str] | None = None) -> list[str]:
    argv = [sys.executable, "-m", "generative.orchestrator", "--source", pdf_path]
    if dry_run:
        argv.append("--dry-run")
    if extra:
        argv.extend(extra)
    return argv


def iter_run_events(
    argv: list[str],
    *,
    env: dict | None = None,
    cwd: str | None = None,
) -> Iterator[dict]:
    """Startet den Subprocess und yieldet geparste Events (inkl. started/error)."""
    yield {"type": "started", "argv": argv}
    run_env = {**os.environ, **(env or {})}
    # Unbuffered Python-Subprocess, damit stdout live ankommt; UTF-8 erzwingen
    # (Umlaute/⚠️ in den Notes-Titeln und Dry-Run-Flags).
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=run_env,
        cwd=cwd,
    )
    parser = RunParser()
    assert proc.stdout is not None
    for line in proc.stdout:
        for ev in parser.feed(line):
            yield ev
    for ev in parser.flush():
        yield ev
    rc = proc.wait()
    if rc != 0:
        yield {"type": "error", "returncode": rc}
