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


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "turbo"})


def test_invalid_bool_override_raises():
    with pytest.raises(ValueError):
        load_runtime_config(env={"ATOMIC_AGENT_INLINE_EVAL": "maybe"})


def test_negative_int_override_raises():
    with pytest.raises(ValueError):
        load_runtime_config(env={"ATOMIC_AGENT_CALL_TIMEOUT": "-5"})
    with pytest.raises(ValueError):
        load_runtime_config(env={"ATOMIC_AGENT_MAX_CONCEPTS": "-1"})


# ---------------------------------------------------------------------------
# Task 2: Pure refine decision helpers
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from runtime_config import RefineTrigger, refine_accepted, should_attempt_refine


def _draft(score, hard_gates=True, hint="", flags=None):
    return SimpleNamespace(
        critic_score=score,
        hard_gates_pass=hard_gates,
        revision_hint=hint,
        quality_flags=flags or [],
    )


def test_fast_profile_never_attempts_refine():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "fast"})

    decision = should_attempt_refine(
        _draft(4, hard_gates=False, hint="fix anchors"),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )

    assert decision.attempt is False
    assert decision.trigger is RefineTrigger.NONE


def test_balanced_rejects_score2_refine():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    decision = should_attempt_refine(
        _draft(2, hard_gates=False, hint="fix"),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )

    assert decision.attempt is False


def test_balanced_allows_score3_only_with_hint():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    without_hint = should_attempt_refine(
        _draft(3, hard_gates=False, hint=""),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )
    with_hint = should_attempt_refine(
        _draft(3, hard_gates=False, hint="fix standalone"),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )

    assert without_hint.attempt is False
    assert with_hint.attempt is True
    assert with_hint.trigger is RefineTrigger.TRIGGER_A


def test_balanced_allows_score4_hard_gate_failure_with_hint():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    decision = should_attempt_refine(
        _draft(4, hard_gates=False, hint="fix source anchors"),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )

    assert decision.attempt is True
    assert decision.trigger is RefineTrigger.TRIGGER_B


def test_refine_requires_concept_context():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "quality"})

    decision = should_attempt_refine(
        _draft(4, hard_gates=False, hint="fix"),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=False,
    )

    assert decision.attempt is False


def test_refine_acceptance_preserves_current_gate():
    assert refine_accepted(_draft(4, hard_gates=True), auto_threshold=4) is True
    assert refine_accepted(_draft(5, hard_gates=False), auto_threshold=4) is False
    assert refine_accepted(_draft(3, hard_gates=True), auto_threshold=4) is False
