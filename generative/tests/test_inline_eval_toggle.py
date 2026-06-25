from __future__ import annotations



from generative.orchestrator import inline_eval_enabled
from generative.runtime_config import load_runtime_config


def test_inline_eval_enabled_by_default():
    cfg = load_runtime_config(env={})
    assert inline_eval_enabled(cfg) is True


def test_inline_eval_disabled_by_fast_profile():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "fast"})
    assert inline_eval_enabled(cfg) is False


def test_inline_eval_env_override_still_wins():
    cfg = load_runtime_config(env={
        "ATOMIC_AGENT_PROFILE": "fast",
        "ATOMIC_AGENT_INLINE_EVAL": "1",
    })
    assert inline_eval_enabled(cfg) is True
