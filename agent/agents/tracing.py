"""Austauschbares Tracing-Backend für die atomic-agent-Pipeline.

Swap-Beispiel (nach SDK-Migration):
    from agents.tracing import set_tracing_backend
    set_tracing_backend(OtelBackend(endpoint="..."))

# Note: LLM-call tracing still uses _trace() in agents/base.py (legacy path).
# Both write to the same run JSONL file. Full migration to TracingBackend
# (so LLM calls also flow through the backend, enabling Langfuse spans)
# is planned as part of the SDK migration (Plan 2).
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
from typing import Protocol

from config import CACHE_DIR


class TracingBackend(Protocol):
    def write(self, entry: dict) -> None: ...


class JsonlBackend:
    """Default-Backend: schreibt in .cache/runs/<run-id>.jsonl."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._file: Path | None = None
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        with self._lock:
            if self._file is None:
                self._run_dir.mkdir(parents=True, exist_ok=True)
                self._file = self._run_dir / f"{self._run_id}.jsonl"
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Aktives Backend — austauschbar via set_tracing_backend()
_RUN_ID = time.strftime("%Y%m%d-%H%M%S")
_backend: TracingBackend = JsonlBackend(
    run_dir=CACHE_DIR / "runs",
    run_id=_RUN_ID,
)


def set_tracing_backend(backend: TracingBackend) -> None:
    """Ersetzt das aktive Backend. Aufruf vor dem ersten trace_event()."""
    global _backend
    _backend = backend


def flush_tracing() -> None:
    """Flushes das aktive Backend — aufrufen am Pipeline-Ende."""
    if hasattr(_backend, "flush"):
        _backend.flush()


def trace_event(agent: str, event_type: str, data: dict) -> None:
    """Schreibt ein strukturiertes Event. Backend-agnostisch."""
    _backend.write({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type": event_type,
        "agent": agent,
        **data,
    })


def trace_run_start(model_config: dict) -> None:
    """Schreibt Run-Start-Entry mit Model-Config."""
    _backend.write({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type": "run_start",
        "run_id": _RUN_ID,
        "model_config": model_config,
    })


# Auto-Aktivierung via ATOMIC_AGENT_TRACING=langfuse
if os.getenv("ATOMIC_AGENT_TRACING") == "langfuse":
    try:
        import atexit
        from agents.langfuse_backend import LangfuseBackend as _LF  # type: ignore[import]
        _backend = _LF()
        atexit.register(_backend.flush)
    except Exception:
        pass  # graceful fallback auf JsonlBackend
