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


def test_trigger_b_fallthrough_without_hint_or_disabled():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    # trigger_b fires (score >= auto_threshold, hard_gates False) but has_hint is False → no_trigger
    decision = should_attempt_refine(
        _draft(4, hard_gates=False, hint=""),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
    )
    assert decision.attempt is False
    assert decision.trigger is RefineTrigger.NONE
    assert decision.reason == "no_trigger"

    # trigger_b_enabled=False, so even with a hint it falls through to no_trigger
    policy = dataclasses.replace(cfg.refine, trigger_b_enabled=False)
    decision = should_attempt_refine(
        _draft(4, hard_gates=False, hint="fix"),
        policy,
        auto_threshold=4,
        has_concept_context=True,
    )
    assert decision.attempt is False
    assert decision.reason == "no_trigger"


def test_score2_disabled_reason_when_score2_in_band():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    # min_trigger_a_score=2 puts score 2 inside the trigger_a band; score2_enabled=False → score2_disabled
    policy = dataclasses.replace(cfg.refine, enabled=True, min_trigger_a_score=2, score2_enabled=False)
    decision = should_attempt_refine(
        _draft(2, hard_gates=False, hint="fix"),
        policy,
        auto_threshold=4,
        has_concept_context=True,
    )
    assert decision.attempt is False
    assert decision.reason == "score2_disabled"


def test_synthesized_hint_alone_enables_trigger_a():
    cfg = load_runtime_config(env={"ATOMIC_AGENT_PROFILE": "balanced"})

    # No draft hint; synthesized_hint satisfies has_hint → trigger_a fires
    decision = should_attempt_refine(
        _draft(3, hard_gates=False, hint=""),
        cfg.refine,
        auto_threshold=4,
        has_concept_context=True,
        synthesized_hint="fix standalone",
    )
    assert decision.attempt is True
    assert decision.trigger is RefineTrigger.TRIGGER_A


# ---------------------------------------------------------------------------
# Task 3: Thread-safe refine budget
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor

from runtime_config import RunBudget


def test_run_budget_allows_unlimited_when_limit_is_none():
    budget = RunBudget(max_refines_per_run=None)

    assert [budget.try_consume() for _ in range(5)] == [True, True, True, True, True]
    assert budget.consumed == 5


def test_run_budget_stops_at_limit():
    budget = RunBudget(max_refines_per_run=2)

    assert [budget.try_consume() for _ in range(4)] == [True, True, False, False]
    assert budget.consumed == 2


def test_run_budget_is_thread_safe():
    budget = RunBudget(max_refines_per_run=2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: budget.try_consume(), range(20)))

    assert sum(1 for result in results if result) == 2
    assert budget.consumed == 2


# ---------------------------------------------------------------------------
# Task 5: Concept cap helper
# ---------------------------------------------------------------------------

from runtime_config import cap_actionable_concepts


def _concept(title, action="create", origin="primary"):
    return SimpleNamespace(title=title, action=action, origin=origin)


def test_concept_cap_limits_only_actionable_concepts():
    concepts = [
        _concept("A"),
        _concept("B", origin="secondary_mention"),
        _concept("C"),
        _concept("D", action="skip"),
        _concept("E"),
    ]

    capped, dropped = cap_actionable_concepts(concepts, 2)

    assert [c.title for c in capped] == ["A", "B", "C", "D"]
    assert [c.title for c in dropped] == ["E"]


def test_cap_is_run_wide_across_multiple_chapter_plans():
    # Three "chapters", each with 2 actionable concepts; run cap = 3 total.
    chapters = [
        [_concept("A1"), _concept("A2")],
        [_concept("B1"), _concept("B2")],
        [_concept("C1"), _concept("C2")],
    ]
    remaining = 3
    kept_actionable_titles = []
    for concepts in chapters:
        capped, _dropped = cap_actionable_concepts(concepts, remaining)
        kept = [c for c in capped if c.action != "skip" and c.origin != "secondary_mention"]
        kept_actionable_titles += [c.title for c in kept]
        remaining = max(0, remaining - len(kept))
    assert kept_actionable_titles == ["A1", "A2", "B1"]  # exactly 3 run-wide, not 6


from runtime_config import count_actionable, is_actionable_concept


def test_is_actionable_and_count():
    cs = [_concept("A"), _concept("B", action="skip"), _concept("C", origin="secondary_mention"), _concept("D")]
    assert is_actionable_concept(cs[0]) is True
    assert is_actionable_concept(cs[1]) is False
    assert is_actionable_concept(cs[2]) is False
    assert count_actionable(cs) == 2
