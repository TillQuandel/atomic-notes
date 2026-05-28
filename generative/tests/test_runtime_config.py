from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from runtime_config import load_runtime_config


def test_default_profile_preserves_legacy_behavior():
    cfg = load_runtime_config(env={})

    assert cfg.profile == "legacy"
    assert cfg.inline_eval is True
    assert cfg.call_timeout_sec == 300
    assert cfg.timeout_retries == 0
    assert cfg.max_concepts is None
    assert cfg.refine.enabled is True
    assert cfg.refine.min_trigger_a_score == 2
    assert cfg.refine.max_refines_per_run is None


def test_fast_profile_prioritizes_short_feedback_loop():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "fast"})

    assert cfg.profile == "fast"
    assert cfg.inline_eval is False
    assert cfg.max_concepts == 3
    assert cfg.timeout_retries == 0
    assert cfg.refine.enabled is False
    assert cfg.refine.max_refines_per_run == 0


def test_balanced_profile_refines_only_high_roi_cases():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    assert cfg.profile == "balanced"
    assert cfg.inline_eval is False
    assert cfg.refine.enabled is True
    assert cfg.refine.min_trigger_a_score == 3
    assert cfg.refine.score2_enabled is False
    assert cfg.refine.score3_requires_hint is True
    assert cfg.refine.trigger_b_enabled is True
    assert cfg.refine.max_refines_per_run == 2


def test_env_override_beats_profile():
    cfg = load_runtime_config(env={
        "ATOMIC_AGENT_PROFILE": "fast",
        "ATOMIC_AGENT_INLINE_EVAL": "1",
        "ATOMIC_AGENT_MAX_CONCEPTS": "7",
        "ATOMIC_AGENT_TIMEOUT_RETRIES": "1",
    })

    assert cfg.profile == "fast"
    assert cfg.inline_eval is True
    assert cfg.max_concepts == 7
    assert cfg.timeout_retries == 1


def test_runtime_config_is_immutable():
    cfg = load_runtime_config(env={})

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.inline_eval = False
