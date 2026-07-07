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

from generative.config import CALL_TIMEOUT_SEC

# litellm registriert intern async Callbacks die beim Event-Loop-Close nicht sauber
# awaited werden → RuntimeWarning + hängender Prozess bei asyncio.run()-Kontext.
litellm.success_callback = []
litellm.failure_callback = []
litellm._async_success_callback = []  # undokumentierte interne Liste (Gemini-Finding)

_MAX_RETRIES = 2


def _resolve_num_retries(timeout_retries: int | None) -> int:
    """Mappt den Runtime-Retry-Knopf auf litellm ``num_retries`` (Parität #148).

    Semantik-Unterschied der beiden Backends:
      - Subscription: ``timeout_retries`` begrenzt NUR Timeout-Retries; transiente
        Prozessfehler behalten fix ``_MAX_RETRIES`` Retries.
      - litellm: am bare-``completion()``-Level gibt es genau EINEN Retry-Knopf,
        ``num_retries``. Er deckt Timeout + APIError + APIConnectionError einheitlich
        ab (verifiziert in litellm 1.90.0, ``utils.py``-Wrapper: retried nur bei
        ``openai.APIError | Timeout | APIConnectionError``).

    ``RetryPolicy(TimeoutErrorRetries=…)`` käme der Timeout-only-Semantik näher, würde
    aber APIConnectionError-Retries still auf 0 senken (nicht im Policy-Mapping
    abgedeckt → ``None`` → kein Retry) und damit die Robustheit gegen Netzfehler
    heimlich verschlechtern. Daher: den einen Runtime-Wert direkt auf ``num_retries``
    mappen, damit der konfigurierte Wert auf dem API-Pfad tatsächlich greift.
    Rest-Differenz: litellm wendet ihn auf alle transienten API-Fehler an, nicht nur
    Timeouts. ``None`` (kein Runtime-Config gesetzt) → heutiger Default ``_MAX_RETRIES``.
    """
    return _MAX_RETRIES if timeout_retries is None else timeout_retries


def _build_messages(prompt: str, cache_prefix: str | None, model: str) -> list[dict]:
    """Baut die litellm-``messages``. Opt-in Prompt-Caching (#148).

    - ``cache_prefix is None`` → exakt heutiges Verhalten: ein einziger String-Content.
    - ``cache_prefix`` gesetzt + ``anthropic/``-Modell → Content als Block-Liste; der
      statische Prefix-Block trägt ``cache_control: {"type": "ephemeral"}``, damit die
      Anthropic-Prompt-Cache greift. Format verifiziert in litellm 1.90.0
      (``prompt_templates/factory.py``: text-Block mit optionalem ``cache_control``;
      ``ChatCompletionCachedContent = {"type": "ephemeral"}``).
    - ``cache_prefix`` gesetzt + Nicht-Anthropic-Provider → kein ``cache_control``;
      Prefix + Rest werden konkateniert wie bisher (ein String).
    """
    if cache_prefix is None:
        return [{"role": "user", "content": prompt}]
    if model.startswith("anthropic/"):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    return [{"role": "user", "content": cache_prefix + prompt}]


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
    cache_prefix: str | None = None,
):
    """Synchroner LLM-Aufruf via litellm. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    t0 = time.time()
    try:
        resp = litellm.completion(
            model=model,
            messages=_build_messages(prompt, cache_prefix, model),
            request_timeout=call_timeout_sec,
            num_retries=_resolve_num_retries(timeout_retries),
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
    cache_prefix: str | None = None,
):
    """Asynchroner LLM-Aufruf via litellm. Cache/Trace übernimmt base.py."""
    call_timeout_sec = CALL_TIMEOUT_SEC if call_timeout_sec is None else call_timeout_sec
    t0 = time.time()
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=_build_messages(prompt, cache_prefix, model),
            request_timeout=call_timeout_sec,
            num_retries=_resolve_num_retries(timeout_retries),
        )
    except Exception as e:
        from generative.pipeline.error_hints import litellm_error_hint

        raise RuntimeError(litellm_error_hint(agent, model, e)) from e
    return _parse_response(resp, (time.time() - t0) * 1000)
