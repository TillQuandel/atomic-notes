"""Subscription-Backend: ruft claude CLI via subprocess auf.

Modell-Mapping: vollständige Modell-IDs (anthropic/claude-opus-4-7) →
CLI-Shorthand (opus/haiku) via _to_cli_model(). Erlaubt einheitliche
Modell-IDs in config.py für beide Backends.
"""
from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import time

from generative.config import CLAUDE_BIN, CALL_TIMEOUT_SEC

_TRANSIENT_RC = {3221226505}
_MAX_RETRIES = 2
_TIMEOUT_RETRIES = int(os.getenv("ATOMIC_AGENT_TIMEOUT_RETRIES", "0"))

_CLI_ALIASES: dict[str, str] = {
    "anthropic/claude-opus-4-7":           "opus",
    "anthropic/claude-haiku-4-5-20251001": "haiku",
    "anthropic/claude-sonnet-4-6":         "sonnet",
}


def _to_cli_model(model: str) -> str:
    """Mappt vollständige Modell-ID auf claude-CLI-Shorthand.
    Unbekannte Strings werden unverändert durchgereicht.
    """
    return _CLI_ALIASES.get(model, model)


_CLI_INSTALL_HINT = (
    "Installation: `npm install -g @anthropic-ai/claude-code`, danach einmal "
    "`claude` starten und einloggen (Pro/Max-Abo). Alternative ohne CLI: "
    "ATOMIC_AGENT_BACKEND=litellm + API-Key. Setup prüfen: `atomic-notes doctor`."
)

# Fehlerklassen, bei denen Retries sinnlos sind: sofort mit Handlungsanleitung scheitern.
_AUTH_PATTERNS = ("/login", "not logged in", "invalid api key", "authentication",
                  "unauthorized", "oauth token")
_RATE_PATTERNS = ("429", "rate_limit", "rate limit")


def _fail_fast_hint(text: str) -> str | None:
    """Auth-/Rate-Limit-Fehler erkennen — Meldung mit nächstem Schritt statt Retry."""
    low = (text or "").lower()
    if any(p in low for p in _AUTH_PATTERNS):
        return (
            "claude-CLI nicht eingeloggt oder Session abgelaufen — einmal `claude` "
            "starten und einloggen. Alternative: ATOMIC_AGENT_BACKEND=litellm + "
            "API-Key. Setup prüfen: `atomic-notes doctor`."
        )
    if any(p in low for p in _RATE_PATTERNS):
        return (
            "Subscription-Rate-Limit erreicht (5-h-Fenster, ≈8 volle Pipeline-Läufe) — "
            "bis zum Fenster-Reset warten oder ATOMIC_AGENT_BACKEND=litellm nutzen."
        )
    return None


def _build_argv(model: str) -> list[str]:
    return [
        CLAUDE_BIN, "-p",
        "--output-format", "json",
        "--model", _to_cli_model(model),
        "--exclude-dynamic-system-prompt-sections",
    ]


def _parse_cli_json(raw: str):
    from generative.agents.base import CallResult
    d = json.loads(raw)
    if d.get("is_error"):
        raise RuntimeError(f"claude CLI Fehler: {d.get('result', '')[:300]}")
    usage = d.get("usage", {}) or {}
    return CallResult(
        text=(d.get("result") or "").strip(),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
        cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0)),
        duration_ms=int(d.get("duration_ms", 0)),
    )


def call_full(
    prompt: str,
    *,
    model: str,
    agent: str = "unknown",
    call_timeout_sec: int | None = None,
    timeout_retries: int | None = None,
):
    """Synchroner Subprocess-Aufruf. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    timeout_retries = _TIMEOUT_RETRIES if timeout_retries is None else timeout_retries
    for attempt in range(_MAX_RETRIES + 1):
        try:
            _env = os.environ.copy()
            _env["CLAUDE_INTERNAL_CALL"] = "1"
            proc = subprocess.run(
                _build_argv(model),
                input=prompt,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=call_timeout_sec,
                env=_env,
            )
        except subprocess.TimeoutExpired:
            # Runtime config controls timeout retries, but transient process retries
            # remain bounded by the existing backend retry ceiling.
            effective_timeout_retries = min(_MAX_RETRIES, timeout_retries)
            if attempt < effective_timeout_retries:
                print(f"      [cli-retry] {agent}/{model} timeout (attempt {attempt+1}/{effective_timeout_retries+1}) â€” 10s Pause", file=sys.stderr)
                time.sleep(10.0)
                continue
            raise RuntimeError(f"claude CLI Timeout nach {call_timeout_sec}s ({agent}/{model})")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"claude-CLI '{CLAUDE_BIN}' nicht gefunden. {_CLI_INSTALL_HINT}"
            ) from e
        except OSError as e:
            raise RuntimeError(f"claude CLI nicht aufrufbar: {e}") from e

        if proc.returncode in _TRANSIENT_RC and attempt < _MAX_RETRIES:
            print(f"      [cli-retry] {agent}/{model} rc={proc.returncode} (attempt {attempt+1}/{_MAX_RETRIES+1})", file=sys.stderr)
            time.sleep(1.0)
            continue
        if proc.returncode == 1:
            try:
                d = json.loads(proc.stdout or "")
            except json.JSONDecodeError:
                d = {}
            if d.get("is_error"):
                hint = _fail_fast_hint(str(d.get("result", "")))
                if hint:
                    raise RuntimeError(
                        f"claude CLI: {str(d.get('result', ''))[:200]} | {hint}"
                    )
                if attempt < _MAX_RETRIES:
                    print(f"      [cli-retry] {agent}/{model} is_error (attempt {attempt+1}/{_MAX_RETRIES+1}) — 10s Pause", file=sys.stderr)
                    time.sleep(10.0)
                    continue
        break

    assert proc is not None
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:500]
        hint = _fail_fast_hint(err)
        raise RuntimeError(
            f"claude CLI fehlgeschlagen (rc={proc.returncode}): {err}"
            + (f" | {hint}" if hint else "")
        )

    try:
        return _parse_cli_json(proc.stdout)
    except (json.JSONDecodeError, RuntimeError) as e:
        raise RuntimeError(f"JSON-Parse: {e} | stdout[:200]={proc.stdout[:200]}") from e


async def call_full_async(
    prompt: str,
    *,
    model: str,
    agent: str = "unknown",
    call_timeout_sec: int | None = None,
    timeout_retries: int | None = None,
):
    """Asynchroner Subprocess-Aufruf. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    timeout_retries = _TIMEOUT_RETRIES if timeout_retries is None else timeout_retries
    stdout = ""
    rc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            _env = os.environ.copy()
            _env["CLAUDE_INTERNAL_CALL"] = "1"
            proc = await asyncio.create_subprocess_exec(
                *_build_argv(model),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"claude-CLI '{CLAUDE_BIN}' nicht gefunden. {_CLI_INSTALL_HINT}"
            ) from e
        except OSError as e:
            raise RuntimeError(f"claude CLI nicht aufrufbar: {e}") from e
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=call_timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # Runtime config controls timeout retries, but transient process retries
            # remain bounded by the existing backend retry ceiling.
            effective_timeout_retries = min(_MAX_RETRIES, timeout_retries)
            if attempt < effective_timeout_retries:
                print(f"      [cli-retry] {agent}/{model} timeout (attempt {attempt+1}/{effective_timeout_retries+1}) â€” 10s Pause", file=sys.stderr)
                await asyncio.sleep(10.0)
                continue
            raise RuntimeError(f"claude CLI Timeout nach {call_timeout_sec}s ({agent}/{model})")

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        rc = proc.returncode

        if rc in _TRANSIENT_RC and attempt < _MAX_RETRIES:
            print(f"      [cli-retry] {agent}/{model} rc={rc} (attempt {attempt+1}/{_MAX_RETRIES+1})", file=sys.stderr)
            await asyncio.sleep(1.0)
            continue
        if rc == 1:
            try:
                d = json.loads(stdout or "")
            except json.JSONDecodeError:
                d = {}
            if d.get("is_error"):
                hint = _fail_fast_hint(str(d.get("result", "")))
                if hint:
                    raise RuntimeError(
                        f"claude CLI: {str(d.get('result', ''))[:200]} | {hint}"
                    )
                if attempt < _MAX_RETRIES:
                    print(f"      [cli-retry] {agent}/{model} is_error (attempt {attempt+1}/{_MAX_RETRIES+1}) — 10s Pause", file=sys.stderr)
                    await asyncio.sleep(10.0)
                    continue
        break

    if rc != 0:
        err = (stderr or stdout or "")[:500]
        hint = _fail_fast_hint(err)
        raise RuntimeError(
            f"claude CLI fehlgeschlagen (rc={rc}): {err}"
            + (f" | {hint}" if hint else "")
        )

    try:
        return _parse_cli_json(stdout)
    except (json.JSONDecodeError, RuntimeError) as e:
        raise RuntimeError(f"JSON-Parse: {e} | stdout[:200]={stdout[:200]}") from e
