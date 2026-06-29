"""litellm-Backend: provider-agnostischer LLM-Aufruf.

Unterstützte Provider (Auswahl):
  anthropic/claude-opus-4-7           → ANTHROPIC_API_KEY
  openai/gpt-4o                       → OPENAI_API_KEY
  gemini/gemini-2.0-flash             → GEMINI_API_KEY
  ollama/llama3                       → OLLAMA_API_BASE (default: localhost:11434)

Vollständige Provider-Liste: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations
import time

import litellm

# litellm registriert intern async Callbacks die beim Event-Loop-Close nicht sauber
# awaited werden → RuntimeWarning + hängender Prozess bei asyncio.run()-Kontext.
litellm.success_callback = []
litellm.failure_callback = []
litellm._async_success_callback = []  # undokumentierte interne Liste (Gemini-Finding)

from generative.config import CALL_TIMEOUT_SEC

_MAX_RETRIES = 2


def _parse_response(resp, duration_ms: float):
    from generative.agents.base import CallResult

    usage = resp.usage
    # Anthropic: cache_read_input_tokens / cache_creation_input_tokens
    # OpenAI: usage.prompt_tokens_details.cached_tokens (kein creation-Äquivalent)
    anthropic_cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    openai_cached = int(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details else 0
    return CallResult(
        text=(resp.choices[0].message.content or "").strip(),
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_read_tokens=anthropic_cache_read or openai_cached,
        cache_creation_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        duration_ms=int(duration_ms),
    )


def call_full(
    prompt: str,
    *,
    model: str,
    agent: str = "unknown",
    call_timeout_sec: int | None = None,
    timeout_retries: int | None = None,
):
    """Synchroner LLM-Aufruf via litellm. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    t0 = time.time()
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            request_timeout=call_timeout_sec,
            num_retries=_MAX_RETRIES,
        )
    except Exception as e:
        from generative.pipeline.error_hints import litellm_error_hint

        raise RuntimeError(litellm_error_hint(agent, model, e)) from e
    return _parse_response(resp, (time.time() - t0) * 1000)


async def call_full_async(
    prompt: str,
    *,
    model: str,
    agent: str = "unknown",
    call_timeout_sec: int | None = None,
    timeout_retries: int | None = None,
):
    """Asynchroner LLM-Aufruf via litellm. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    t0 = time.time()
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            request_timeout=call_timeout_sec,
            num_retries=_MAX_RETRIES,
        )
    except Exception as e:
        from generative.pipeline.error_hints import litellm_error_hint

        raise RuntimeError(litellm_error_hint(agent, model, e)) from e
    return _parse_response(resp, (time.time() - t0) * 1000)
