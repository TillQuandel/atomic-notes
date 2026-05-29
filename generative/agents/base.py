"""Gemeinsame LLM-Hilfe für alle Agenten.

Dispatch: BACKEND=subscription (claude CLI) | litellm (API-basiert).
Cache + Trace liegen hier; der eigentliche LLM-Call liegt im Backend.
Response-Cache: ``.cache/llm/<sha256(prompt+model)[:16]>.json``
Trace-Hook: jeder Call schreibt eine Zeile nach ``.cache/runs/<run-id>.jsonl``
"""
from __future__ import annotations
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import MODEL_OPUS, CACHE_DIR, BACKEND

# Re-Export für Backwards-Compat — Agenten importieren aus agents.base
from agents.tracing import trace_event, trace_run_start, set_tracing_backend, flush_tracing  # noqa: F401
from agents.tracing import _RUN_ID  # single source of truth for run ID

# OTel-Tracer für Phoenix. Bleibt None bis orchestrator._setup_phoenix_tracing()
# ihn via set_llm_tracer() explizit setzt (nur bei ATOMIC_AGENT_TRACING=phoenix).
# Kein impliziter ProxyTracer: ein anderswo gesetzter globaler Provider soll NICHT
# ungewollt LLM-Spans erzeugen. None → _llm_span ist garantiert No-Op.
try:
    from opentelemetry.trace import Status as _OtelStatus, StatusCode as _OtelStatusCode
except Exception:  # opentelemetry nicht installiert
    _OtelStatus = _OtelStatusCode = None
_OTEL_TRACER = None


def set_llm_tracer(tracer) -> None:
    """Aktiviert die LLM-Call-Instrumentierung mit einem konkreten OTel-Tracer.
    Aufruf aus orchestrator._setup_phoenix_tracing(). None = Tracing aus."""
    global _OTEL_TRACER
    _OTEL_TRACER = tracer

# Backend-Dispatch
if BACKEND == "litellm":
    from agents._litellm_backend import call_full as _backend_call_full
    from agents._litellm_backend import call_full_async as _backend_call_full_async
else:
    from agents._subscription_backend import call_full as _backend_call_full
    from agents._subscription_backend import call_full_async as _backend_call_full_async


@dataclass
class CallResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    duration_ms: int = 0
    cached: bool = False


# Thread-lokaler Record des letzten LLM-Calls (Issue #17): erlaubt einem
# Stage-6-Crash-Handler, Prompt + rohen Output des gecrashten Calls eindeutig
# zuzuordnen — auch bei N parallelen Notes via asyncio.to_thread (je eigener Thread).
_LAST_CALL = threading.local()


def _record_call(agent: str, prompt: str, raw_output: str, error: Optional[str] = None) -> None:
    _LAST_CALL.record = {
        "agent": agent,
        "prompt": prompt,
        "raw_output": raw_output,
        "error": error,
    }


def get_last_call_record() -> Optional[dict]:
    """Letzter LLM-Call des aktuellen Threads (oder None). Nur sync-Pfad."""
    return getattr(_LAST_CALL, "record", None)


def clear_last_call_record() -> None:
    if hasattr(_LAST_CALL, "record"):
        del _LAST_CALL.record


_LLM_CACHE_DIR = CACHE_DIR / "llm"
_RUN_DIR = CACHE_DIR / "runs"

# Run-Namespace für --fresh-run: leerer String = normaler Cache (shared across runs).
# Gesetzt via set_cache_namespace(salt) aus orchestrator.py bei --fresh-run.
_CACHE_NAMESPACE: str = ""


@dataclass(frozen=True)
class LLMRuntimeSettings:
    call_timeout_sec: int
    timeout_retries: int


_LLM_RUNTIME_SETTINGS: LLMRuntimeSettings | None = None


def set_llm_runtime_config(runtime_config) -> None:
    """Set per-run backend settings resolved by runtime_config.load_runtime_config()."""
    global _LLM_RUNTIME_SETTINGS
    _LLM_RUNTIME_SETTINGS = LLMRuntimeSettings(
        call_timeout_sec=int(runtime_config.call_timeout_sec),
        timeout_retries=int(runtime_config.timeout_retries),
    )


def clear_llm_runtime_config() -> None:
    """Clear per-run backend settings so direct backend defaults apply again."""
    global _LLM_RUNTIME_SETTINGS
    _LLM_RUNTIME_SETTINGS = None


def _backend_runtime_kwargs() -> dict[str, int]:
    if _LLM_RUNTIME_SETTINGS is None:
        return {}
    return {
        "call_timeout_sec": _LLM_RUNTIME_SETTINGS.call_timeout_sec,
        "timeout_retries": _LLM_RUNTIME_SETTINGS.timeout_retries,
    }


def set_cache_namespace(salt: str) -> None:
    """Setzt einen Run-spezifischen Salt für den Cache-Key.
    Leerer String = normales Caching (Wiederverwendung über Runs).
    Nicht-leerer String = frischer Cache-Namespace (kein Hit aus alten Runs).
    Typischer Aufruf: set_cache_namespace(_RUN_ID) bei --fresh-run.
    """
    global _CACHE_NAMESPACE
    _CACHE_NAMESPACE = salt


def _ensure_dirs() -> None:
    _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prompt: str, model: str, agent: str = "") -> str:
    h = hashlib.sha256()
    h.update((_CACHE_NAMESPACE or "").encode("utf-8"))
    h.update(b"\0")
    h.update((model or "").encode("utf-8"))
    h.update(b"\0")
    h.update((agent or "").encode("utf-8"))
    h.update(b"\0")
    h.update((prompt or "").encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_get(key: str) -> Optional[CallResult]:
    p = _LLM_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return CallResult(
            text=d["text"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cache_read_tokens=d.get("cache_read_tokens", 0),
            cache_creation_tokens=d.get("cache_creation_tokens", 0),
            duration_ms=d.get("duration_ms", 0),
            cached=True,
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _cache_put(key: str, result: CallResult) -> None:
    _ensure_dirs()
    p = _LLM_CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps({
        "text": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "duration_ms": result.duration_ms,
    }, ensure_ascii=False), encoding="utf-8")


def _trace(agent: str, prompt: str, model: str, result: CallResult,
           error: Optional[str] = None) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "model": model,
        "prompt_hash": _cache_key(prompt, model),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "duration_ms": result.duration_ms,
        "cached": result.cached,
        "error": error,
    }
    from agents.tracing import _backend as _tracing_backend
    _tracing_backend.write(entry)


from contextlib import contextmanager as _contextmanager


@_contextmanager
def _llm_span(agent: str, prompt: str, model: str):
    """OTel-LLM-Span für einen einzelnen Call. No-op wenn Tracing aus.

    Wird VOR dem Cache-Check geöffnet, damit auch Cache-Hits in Phoenix
    erscheinen (sonst fehlt bei Re-Runs die halbe Pipeline).
    """
    if _OTEL_TRACER is None:
        yield None
        return
    with _OTEL_TRACER.start_as_current_span(agent) as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", prompt)
        span.set_attribute("llm.model_name", model)
        yield span


def _annotate_llm_span(span, result: "CallResult", *, cache_hit: bool = False,
                       error: Optional[str] = None) -> None:
    """Schreibt Output + Usage in den LLM-Span. No-op wenn span is None."""
    if span is None:
        return
    span.set_attribute("output.value", str(result.text or ""))
    span.set_attribute("llm.token_count.prompt", result.input_tokens)
    span.set_attribute("llm.token_count.completion", result.output_tokens)
    span.set_attribute("cache.hit", cache_hit)
    if error:
        span.set_status(_OtelStatus(_OtelStatusCode.ERROR, error))


def call_claude(prompt: str, *, model: str = MODEL_OPUS, agent: str = "unknown",
                use_cache: bool = True) -> str:
    """Synchroner Aufruf. Gibt nur den Text zurück (Backwards-Compat)."""
    return call_claude_full(prompt, model=model, agent=agent, use_cache=use_cache).text


def call_claude_full(prompt: str, *, model: str = MODEL_OPUS, agent: str = "unknown",
                     use_cache: bool = True) -> CallResult:
    """Synchroner Aufruf mit vollem CallResult (text + usage)."""
    with _llm_span(agent, prompt, model) as span:
        key = _cache_key(prompt, model, agent)
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                _annotate_llm_span(span, cached, cache_hit=True)
                _trace(agent, prompt, model, cached)
                _record_call(agent, prompt, cached.text)
                return cached

        try:
            result = _backend_call_full(prompt, model=model, agent=agent, **_backend_runtime_kwargs())
        except RuntimeError as e:
            result = CallResult(text="")
            _annotate_llm_span(span, result, error=str(e))
            _trace(agent, prompt, model, result, error=str(e))
            _record_call(agent, prompt, "", error=str(e))
            raise

        _annotate_llm_span(span, result)
        if use_cache:
            _cache_put(key, result)
        _trace(agent, prompt, model, result)
        _record_call(agent, prompt, result.text)
        return result


async def call_claude_async(prompt: str, *, model: str = MODEL_OPUS, agent: str = "unknown",
                            use_cache: bool = True) -> str:
    return (await call_claude_full_async(prompt, model=model, agent=agent, use_cache=use_cache)).text


async def call_claude_full_async(prompt: str, *, model: str = MODEL_OPUS, agent: str = "unknown",
                                 use_cache: bool = True) -> CallResult:
    with _llm_span(agent, prompt, model) as span:
        key = _cache_key(prompt, model, agent)
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                _annotate_llm_span(span, cached, cache_hit=True)
                _trace(agent, prompt, model, cached)
                return cached

        try:
            result = await _backend_call_full_async(prompt, model=model, agent=agent, **_backend_runtime_kwargs())
        except RuntimeError as e:
            result = CallResult(text="")
            _annotate_llm_span(span, result, error=str(e))
            _trace(agent, prompt, model, result, error=str(e))
            raise

        _annotate_llm_span(span, result)
        if use_cache:
            _cache_put(key, result)
        _trace(agent, prompt, model, result)
        return result


# Provider-agnostische Aliase für Backend-Abstraktion
call_llm_full = call_claude_full
call_llm_full_async = call_claude_full_async
