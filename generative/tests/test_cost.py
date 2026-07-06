from unittest.mock import patch

import pytest

from generative.config import MODEL_CONFIG, compute_cost_per_call


def test_opus_cost():
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(
            model="claude-opus-4-7", input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=0
        )
    assert abs(cost - 90.0) < 0.01  # $15 input + $75 output per M


def test_opus_cost_with_prefix():
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(
            model="anthropic/claude-opus-4-7", input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=0
        )
    assert abs(cost - 90.0) < 0.01  # provider-Prefix wird gestrippt


def test_haiku_cache_read():
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(
            model="claude-haiku-4-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000
        )
    assert abs(cost - 0.03) < 0.001  # $0.03/M cache-read


def test_haiku_cost_with_date_suffix():
    """#100: MODEL_HAIKU trägt einen Datums-Suffix (-20251001), der Pricing-Key
    nicht — die Normalisierung muss ihn zusätzlich zum Provider-Prefix strippen,
    sonst verfehlt der eigene Default-String die eigene Tabelle (fail-silent 0.0)."""
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(
            model="anthropic/claude-haiku-4-5-20251001",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=0,
        )
    assert abs(cost - 4.8) < 0.01  # $0.80 input + $4.0 output per M


def test_unknown_model_returns_zero():
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call("unknown-model", 1000, 1000)
    assert cost == 0.0


@pytest.mark.parametrize("model", sorted(set(MODEL_CONFIG.values())))
def test_all_active_agent_models_have_nonzero_cost(model):
    """Regressionsschutz gegen künftige Model-Bumps: jeder in MODEL_CONFIG aktiv
    zugewiesene Model-String muss im API-Backend einen Preis > 0 liefern."""
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(model, 1000, 1000)
    assert cost > 0.0, f"{model!r} verfehlt MODEL_PRICING — fail-silent 0.0"


def test_subscription_returns_zero():
    with patch("generative.config.BACKEND", "subscription"):
        cost = compute_cost_per_call("claude-opus-4-7", 1000, 1000)
    assert cost == 0.0


def test_gemini_cost():
    with patch("generative.config.BACKEND", "api"):
        cost = compute_cost_per_call(model="gemini-2.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 0.375) < 0.001  # $0.075 + $0.30 per M


def test_cost_from_jsonl_trace():
    import json
    import tempfile
    import os

    calls = [
        {
            "agent": "planner",
            "model": "claude-opus-4-7",
            "input_tokens": 5000,
            "output_tokens": 1000,
            "cache_read_tokens": 20000,
        },
        {
            "agent": "extractor",
            "model": "claude-opus-4-7",
            "input_tokens": 3000,
            "output_tokens": 1200,
            "cache_read_tokens": 15000,
        },
        {
            "agent": "critic",
            "model": "claude-haiku-4-5",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_tokens": 5000,
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
        path = f.name

    try:
        from unittest.mock import patch

        with patch("generative.config.BACKEND", "api"):
            total = sum(
                compute_cost_per_call(c["model"], c["input_tokens"], c["output_tokens"], c.get("cache_read_tokens", 0))
                for c in calls
            )
        assert total > 0
        assert total < 1.0  # kleiner Run, unter $1
    finally:
        os.unlink(path)
