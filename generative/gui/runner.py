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
    on_proc=None,
) -> Iterator[dict]:
    """Startet den Subprocess und yieldet geparste Events (inkl. started/error).

    on_proc: optionaler Callback, der mit dem Popen-Handle aufgerufen wird, sobald
    der Subprocess läuft — erlaubt dem Aufrufer (RunSession), den Lauf zu canceln.
    """
    yield {"type": "started", "argv": argv}
    run_env = {**os.environ, **(env or {})}
    # Unbuffered Python-Subprocess, damit stdout live ankommt; UTF-8 erzwingen
    # (Umlaute/⚠️ in den Notes-Titeln und Dry-Run-Flags).
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    # Markiert den Lauf als GUI-getrieben → der Orchestrator unterdrückt seine
    # schreibenden Auto-Aktionen (Version-Bump in config.py, Eval-Dashboard-Spawn
    # auf :8051). Ein Vorschau-Lauf darf weder Quellcode mutieren noch Prozesse leaken.
    run_env["ATOMIC_AGENT_GUI"] = "1"
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
    if on_proc is not None:
        on_proc(proc)
    parser = RunParser()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            for ev in parser.feed(line):
                yield ev
        for ev in parser.flush():
            yield ev
        rc = proc.wait()
        # Terminal-Event: IMMER `exited` (auch bei rc==0). `done` aus
        # `=== Fertig ===` ist NICHT das Ende — der Orchestrator druckt danach
        # noch Routing-Report + Stage-8-Eval. Erst `exited` schliesst den Stream.
        yield {"type": "exited", "returncode": rc}
    finally:
        # Generator vorzeitig geschlossen (SSE-Client trennt, Lauf abgebrochen):
        # Child nicht verwaisen lassen.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
