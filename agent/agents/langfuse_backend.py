"""Optionaler Langfuse-Backend für atomic-agent Tracing.

Nur für die Entwicklung. Endnutzer des Obsidian-Plugins
nutzen JsonlBackend. Aktivierung via ATOMIC_AGENT_TRACING=langfuse.

Nutzt die Langfuse REST API direkt (kein SDK) — kompatibel mit Python 3.14+.
Setup: .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""
from __future__ import annotations
import base64
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any


class LangfuseBackend:
    """TracingBackend via Langfuse REST API (kein SDK, Python-3.14-kompatibel).

    Einen Trace pro Run (run_start-Event). LLM-Calls → Spans. Events → Events.
    Graceful no-op bei Verbindungsfehler.
    """

    def __init__(self) -> None:
        self._public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self._host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
        self._available = bool(self._public_key and self._secret_key)
        self._trace_id: str | None = None
        self._batch: list[dict] = []

        if not self._available:
            import sys
            print("[LangfuseBackend] Kein API-Key — deaktiviert.", file=sys.stderr)

    def _auth(self) -> str:
        creds = f"{self._public_key}:{self._secret_key}"
        return "Basic " + base64.b64encode(creds.encode()).decode()

    def _send(self, batch: list[dict]) -> None:
        try:
            import urllib.request
            body = json.dumps({"batch": batch}).encode("utf-8")
            req = urllib.request.Request(
                f"{self._host}/api/public/ingestion",
                data=body,
                headers={
                    "Authorization": self._auth(),
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 201, 207):
                    raise RuntimeError(f"HTTP {resp.status}")
        except Exception as e:
            import sys
            print(f"[LangfuseBackend] Sendefehler: {e} — deaktiviert.", file=sys.stderr)
            self._available = False

    def write(self, entry: dict) -> None:
        if not self._available:
            return

        etype = entry.get("type")
        agent = entry.get("agent", "unknown")
        ts = entry.get("ts", datetime.utcnow().isoformat())

        try:
            if etype == "run_start":
                self._trace_id = str(uuid.uuid4())  # Langfuse braucht UUID, run_id als Name
                self._batch.append({
                    "id": str(uuid.uuid4()),
                    "type": "trace-create",
                    "timestamp": ts,
                    "body": {
                        "id": self._trace_id,
                        "name": self._trace_id,
                        "metadata": {"model_config": entry.get("model_config", {})},
                        "tags": ["atomic-agent"],
                    },
                })

            elif etype is None and self._trace_id:  # LLM-Call
                end_dt = datetime.fromisoformat(ts)
                start_dt = end_dt - timedelta(milliseconds=entry.get("duration_ms", 0))
                self._batch.append({
                    "id": str(uuid.uuid4()),
                    "type": "span-create",
                    "timestamp": ts,
                    "body": {
                        "id": str(uuid.uuid4()),
                        "traceId": self._trace_id,
                        "name": f"{agent}/{entry.get('model', '?')}",
                        "startTime": start_dt.isoformat(),
                        "endTime": end_dt.isoformat(),
                        "metadata": {
                            "cached": entry.get("cached", False),
                            "error": entry.get("error"),
                        },
                        "usage": {
                            "input": entry.get("input_tokens", 0),
                            "output": entry.get("output_tokens", 0),
                            "unit": "TOKENS",
                        },
                        "level": "ERROR" if entry.get("error") else "DEFAULT",
                    },
                })

            elif etype and self._trace_id:  # Strukturiertes Event
                meta = {k: v for k, v in entry.items() if k not in ("ts", "type", "agent")}
                self._batch.append({
                    "id": str(uuid.uuid4()),
                    "type": "event-create",
                    "timestamp": ts,
                    "body": {
                        "id": str(uuid.uuid4()),
                        "traceId": self._trace_id,
                        "name": f"{agent}/{etype}",
                        "metadata": meta,
                        "level": "DEFAULT",
                    },
                })

            # Batch senden wenn groß genug
            if len(self._batch) >= 20:
                self._send(self._batch)
                self._batch = []

        except Exception as e:
            import sys
            print(f"[LangfuseBackend] Fehler: {e} — deaktiviert.", file=sys.stderr)
            self._available = False

    def flush(self) -> None:
        """Verbleibende Events senden."""
        if self._available and self._batch:
            self._send(self._batch)
            self._batch = []
