from __future__ import annotations

from dataclasses import dataclass, replace
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
