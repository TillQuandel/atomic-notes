from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping


_FALSE = {"0", "false", "False", "no", "off"}
_TRUE = {"1", "true", "True", "yes", "on"}


@dataclass(frozen=True)
class RefinePolicy:
    enabled: bool
    min_trigger_a_score: int
    score2_enabled: bool
    score3_requires_hint: bool
    trigger_b_enabled: bool
    max_refines_per_note: int | None
    max_refines_per_run: int | None


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str
    inline_eval: bool
    call_timeout_sec: int
    timeout_retries: int
    max_concepts: int | None
    max_concurrent_calls: int
    refine: RefinePolicy


LEGACY = RuntimeConfig(
    profile="legacy",
    inline_eval=True,
    call_timeout_sec=300,
    timeout_retries=0,
    max_concepts=None,
    max_concurrent_calls=4,
    refine=RefinePolicy(
        enabled=True,
        min_trigger_a_score=2,
        score2_enabled=True,
        score3_requires_hint=False,
        trigger_b_enabled=True,
        max_refines_per_note=1,
        max_refines_per_run=None,
    ),
)

FAST = replace(
    LEGACY,
    profile="fast",
    inline_eval=False,
    max_concepts=3,
    refine=replace(LEGACY.refine, enabled=False, max_refines_per_run=0),
)

BALANCED = replace(
    LEGACY,
    profile="balanced",
    inline_eval=False,
    refine=replace(
        LEGACY.refine,
        min_trigger_a_score=3,
        score2_enabled=False,
        score3_requires_hint=True,
        trigger_b_enabled=True,
        max_refines_per_note=1,
        max_refines_per_run=2,
    ),
)

QUALITY = replace(
    LEGACY,
    profile="quality",
    inline_eval=True,
    timeout_retries=1,
    refine=replace(LEGACY.refine, max_refines_per_run=None),
)

PRESETS = {
    "legacy": LEGACY,
    "fast": FAST,
    "balanced": BALANCED,
    "quality": QUALITY,
}


def _parse_bool(value: str) -> bool:
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"Invalid boolean runtime config value: {value!r}")


def _parse_optional_int(value: str) -> int | None:
    if value == "" or value.lower() == "none":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Runtime config integer values must be >= 0")
    return parsed


def _parse_nonneg_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Runtime config integer values must be >= 0")
    return parsed


def load_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    import os

    source = os.environ if env is None else env
    profile = source.get("ATOMIC_AGENT_PROFILE", source.get("ATOMIC_AGENT_RUNTIME_MODE", "legacy"))
    if profile not in PRESETS:
        raise ValueError(f"Unknown ATOMIC_AGENT_PROFILE: {profile!r}")

    cfg = PRESETS[profile]
    cfg = replace(cfg, profile=profile)

    if "ATOMIC_AGENT_INLINE_EVAL" in source:
        cfg = replace(cfg, inline_eval=_parse_bool(source["ATOMIC_AGENT_INLINE_EVAL"]))
    if "ATOMIC_AGENT_CALL_TIMEOUT" in source:
        cfg = replace(cfg, call_timeout_sec=_parse_nonneg_int(source["ATOMIC_AGENT_CALL_TIMEOUT"]))
    if "ATOMIC_AGENT_TIMEOUT_RETRIES" in source:
        cfg = replace(cfg, timeout_retries=_parse_nonneg_int(source["ATOMIC_AGENT_TIMEOUT_RETRIES"]))
    if "ATOMIC_AGENT_MAX_CONCEPTS" in source:
        cfg = replace(cfg, max_concepts=_parse_optional_int(source["ATOMIC_AGENT_MAX_CONCEPTS"]))
    if "ATOMIC_AGENT_MAX_CONCURRENT_CALLS" in source:
        cfg = replace(cfg, max_concurrent_calls=_parse_nonneg_int(source["ATOMIC_AGENT_MAX_CONCURRENT_CALLS"]))
    if "ATOMIC_AGENT_MAX_REFINES_PER_RUN" in source:
        cfg = replace(
            cfg,
            refine=replace(
                cfg.refine,
                max_refines_per_run=_parse_optional_int(source["ATOMIC_AGENT_MAX_REFINES_PER_RUN"]),
            ),
        )

    return cfg


# ---------------------------------------------------------------------------
# Pure refine decision helpers
# ---------------------------------------------------------------------------


class RefineTrigger(Enum):
    NONE = "none"
    TRIGGER_A = "score_band"
    TRIGGER_B = "hard_gate_failure"


@dataclass(frozen=True)
class RefineDecision:
    attempt: bool
    trigger: RefineTrigger
    reason: str


def should_attempt_refine(
    draft,
    policy: RefinePolicy,
    *,
    auto_threshold: int,
    has_concept_context: bool,
    synthesized_hint: str | None = None,
) -> RefineDecision:
    if not policy.enabled:
        return RefineDecision(False, RefineTrigger.NONE, "refine_disabled")
    if not has_concept_context:
        return RefineDecision(False, RefineTrigger.NONE, "missing_concept_context")

    has_hint = bool(getattr(draft, "revision_hint", "") or synthesized_hint)
    score = int(getattr(draft, "critic_score", 0))
    hard_gates_pass = bool(getattr(draft, "hard_gates_pass", False))

    trigger_a = policy.min_trigger_a_score <= score < auto_threshold
    if trigger_a:
        if score == 2 and not policy.score2_enabled:
            return RefineDecision(False, RefineTrigger.NONE, "score2_disabled")
        if score == 3 and policy.score3_requires_hint and not has_hint:
            return RefineDecision(False, RefineTrigger.NONE, "score3_without_hint")
        if not has_hint:
            return RefineDecision(False, RefineTrigger.NONE, "missing_revision_hint")
        return RefineDecision(True, RefineTrigger.TRIGGER_A, "score_band")

    trigger_b = score >= auto_threshold and not hard_gates_pass
    if trigger_b and policy.trigger_b_enabled and has_hint:
        return RefineDecision(True, RefineTrigger.TRIGGER_B, "hard_gate_failure")

    return RefineDecision(False, RefineTrigger.NONE, "no_trigger")


def refine_accepted(refined, *, auto_threshold: int) -> bool:
    return (
        bool(getattr(refined, "hard_gates_pass", False))
        and int(getattr(refined, "critic_score", 0)) >= auto_threshold
    )


# ---------------------------------------------------------------------------
# Mutable run state — kept outside frozen RuntimeConfig intentionally
# ---------------------------------------------------------------------------


class RunBudget:
    def __init__(self, *, max_refines_per_run: int | None):
        self._max_refines_per_run = max_refines_per_run
        self._consumed = 0
        self._lock = threading.Lock()

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def try_consume(self) -> bool:
        with self._lock:
            if self._max_refines_per_run is not None and self._consumed >= self._max_refines_per_run:
                return False
            self._consumed += 1
            return True


# ---------------------------------------------------------------------------
# Concept cap helper
# ---------------------------------------------------------------------------


def is_actionable_concept(concept) -> bool:
    """Prüft ob ein Konzept actionable ist (weder skip noch secondary_mention)."""
    return getattr(concept, "action", None) != "skip" and getattr(concept, "origin", None) != "secondary_mention"


def count_actionable(concepts) -> int:
    """Zählt actionable Konzepte in einer Liste."""
    return sum(1 for c in concepts if is_actionable_concept(c))


def cap_actionable_concepts(concepts: list, max_concepts: int | None) -> tuple[list, list]:
    actionable = [c for c in concepts if is_actionable_concept(c)]
    if max_concepts is None or len(actionable) <= max_concepts:
        return concepts, []

    kept_actionable_ids = {id(c) for c in actionable[:max_concepts]}
    capped = [
        c for c in concepts
        if id(c) in kept_actionable_ids or not is_actionable_concept(c)
    ]
    dropped = actionable[max_concepts:]
    return capped, dropped
