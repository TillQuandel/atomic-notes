# Runtime Config Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime configuration layer so pipeline speed/quality tradeoffs can be selected per run without editing code.

**Architecture:** Create an immutable `RuntimeConfig` with `legacy`, `fast`, `balanced`, and `quality` presets. Load one resolved config once at run startup, apply environment overrides, and pass it into the orchestrator. Keep mutable run state, such as refine budget usage, outside the frozen config.

**Tech Stack:** Python dataclasses, pytest, existing `orchestrator.py`, existing `config.py`, existing Claude CLI subscription backend.

---

## Cross-Model Review Summary

| Reviewer | Severity | Finding | Convergence |
|---|---|---|---|
| Gemini | high | The config access pattern must be explicit; avoid scattered globals. | with Opus |
| Gemini | high | Refine budget is mutable run state and must not live inside frozen config. | with Opus |
| Opus 4.8 | high | Refine attempt and refine acceptance are two separate decisions; test both. | solo |
| Opus 4.8 | high | `max_refines_per_run` is thread-sensitive because Stage 6 runs notes in parallel. | with Gemini |
| Opus 4.8 | medium | Default profile must preserve current behavior; `balanced` must not silently become default. | with Gemini |
| Gemini | medium | Log the resolved active config at run start. | compatible |

**Recommendation:** Proceed in phases. Phase 1 creates the config layer and behavior-preserving refactor. Phase 2 enables `fast` and `balanced` runtime behavior. YAML files and dashboard are deferred until the CLI/env layer is stable.

## Decisions Fixed For This Plan

- Default profile is `legacy`, not `balanced`.
- `legacy` preserves current behavior: inline eval on, no concept cap, refine score floor equivalent to current `_REFINE_MIN_SCORE=2`, timeout 300, timeout retries 0.
- `balanced` defines "score3 selective" as: score is exactly 3, `revision_hint` is present, refine policy is enabled, concept map key exists, and run/note budgets are available. Score 2 is not refined in `balanced`.
- `fast` disables refine and inline eval and caps actionable concepts at 3.
- `quality` keeps current refine generosity, keeps inline eval on, and may opt into one timeout retry.
- No dashboard and no YAML local config in this plan. A future UI or config file must read/write the same runtime config contract.

## File Structure

- Create: `generative/runtime_config.py`
  - Owns frozen config dataclasses, presets, env override parsing, and pure refine decision helpers.
- Modify: `generative/orchestrator.py`
  - Loads one `RuntimeConfig`, logs it, applies concept cap, routes inline eval through config, and delegates refine decisions to pure helpers.
- Test: `generative/tests/test_runtime_config.py`
  - Covers presets, env override precedence, immutable config, and refine decisions.
- Test: `generative/tests/test_inline_eval_toggle.py`
  - Updates inline eval tests to cover config-backed behavior.

---

### Task 1: Add Runtime Config Dataclasses And Presets

**Files:**
- Create: `generative/runtime_config.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Write failing preset tests**

Create `generative/tests/test_runtime_config.py` with:

```python
from __future__ import annotations

import dataclasses
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py -q
```

Expected: import failure for `runtime_config`.

- [ ] **Step 3: Implement minimal dataclasses and loader**

Create `generative/runtime_config.py`:

```python
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
        cfg = replace(cfg, call_timeout_sec=int(source["ATOMIC_AGENT_CALL_TIMEOUT"]))
    if "ATOMIC_AGENT_TIMEOUT_RETRIES" in source:
        cfg = replace(cfg, timeout_retries=int(source["ATOMIC_AGENT_TIMEOUT_RETRIES"]))
    if "ATOMIC_AGENT_MAX_CONCEPTS" in source:
        cfg = replace(cfg, max_concepts=_parse_optional_int(source["ATOMIC_AGENT_MAX_CONCEPTS"]))
    if "ATOMIC_AGENT_MAX_CONCURRENT_CALLS" in source:
        cfg = replace(cfg, max_concurrent_calls=int(source["ATOMIC_AGENT_MAX_CONCURRENT_CALLS"]))
    if "ATOMIC_AGENT_MAX_REFINES_PER_RUN" in source:
        cfg = replace(
            cfg,
            refine=replace(
                cfg.refine,
                max_refines_per_run=_parse_optional_int(source["ATOMIC_AGENT_MAX_REFINES_PER_RUN"]),
            ),
        )

    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py -q
```

Expected: all tests pass.

---

### Task 2: Extract Pure Refine Decisions

**Files:**
- Modify: `generative/runtime_config.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Write failing refine decision tests**

Append to `generative/tests/test_runtime_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py -q
```

Expected: import failure for `RefineTrigger`, `should_attempt_refine`, or `refine_accepted`.

- [ ] **Step 3: Implement pure refine helpers**

Append to `generative/runtime_config.py`:

```python
from dataclasses import dataclass
from enum import Enum


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
    return bool(getattr(refined, "hard_gates_pass", False)) and int(getattr(refined, "critic_score", 0)) >= auto_threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py -q
```

Expected: all tests pass.

---

### Task 3: Add Thread-Safe Refine Budget

**Files:**
- Modify: `generative/runtime_config.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Write failing budget tests**

Append to `generative/tests/test_runtime_config.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py::test_run_budget_stops_at_limit -q
```

Expected: import failure for `RunBudget`.

- [ ] **Step 3: Implement `RunBudget`**

Append to `generative/runtime_config.py`:

```python
import threading


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py -q
```

Expected: all tests pass.

---

### Task 4: Integrate Runtime Config Without Changing Legacy Behavior

**Files:**
- Modify: `generative/orchestrator.py`
- Test: `generative/tests/test_inline_eval_toggle.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Update inline eval tests for config-backed behavior**

Replace `generative/tests/test_inline_eval_toggle.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import inline_eval_enabled
from runtime_config import load_runtime_config


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest generative/tests/test_inline_eval_toggle.py -q
```

Expected: failure because `inline_eval_enabled` still expects env-like mapping or `None`.

- [ ] **Step 3: Change `inline_eval_enabled` to accept `RuntimeConfig`**

In `generative/orchestrator.py`, replace:

```python
def inline_eval_enabled(env=None) -> bool:
    """Whether Stage-8 inline quality evaluation should run for this process."""
    source = os.environ if env is None else env
    return source.get("ATOMIC_AGENT_INLINE_EVAL", "1") != "0"
```

with:

```python
def inline_eval_enabled(runtime_config) -> bool:
    """Whether Stage-8 inline quality evaluation should run for this process."""
    return bool(runtime_config.inline_eval)
```

- [ ] **Step 4: Load runtime config at run start and pass it to Stage 8**

In `generative/orchestrator.py`, import:

```python
from runtime_config import load_runtime_config
```

At the start of `main()`, after argument parsing and before pipeline stages, add:

```python
    runtime_config = load_runtime_config()
    print(
        "[runtime-config] "
        f"profile={runtime_config.profile} "
        f"inline_eval={runtime_config.inline_eval} "
        f"max_concepts={runtime_config.max_concepts} "
        f"max_refines_per_run={runtime_config.refine.max_refines_per_run} "
        f"timeout_retries={runtime_config.timeout_retries}"
    )
```

Replace:

```python
    if not inline_eval_enabled():
```

with:

```python
    if not inline_eval_enabled(runtime_config):
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```powershell
python -m pytest generative/tests/test_inline_eval_toggle.py generative/tests/test_runtime_config.py -q
```

Expected: all tests pass.

---

### Task 5: Apply Concept Cap In Fast Mode

**Files:**
- Modify: `generative/orchestrator.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Add pure concept cap helper and tests**

Append to `generative/runtime_config.py`:

```python
def cap_actionable_concepts(concepts: list, max_concepts: int | None) -> tuple[list, list]:
    actionable = [c for c in concepts if getattr(c, "action", None) != "skip" and getattr(c, "origin", None) != "secondary_mention"]
    if max_concepts is None or len(actionable) <= max_concepts:
        return concepts, []

    kept_actionable_ids = {id(c) for c in actionable[:max_concepts]}
    capped = [
        c for c in concepts
        if id(c) in kept_actionable_ids
        or getattr(c, "action", None) == "skip"
        or getattr(c, "origin", None) == "secondary_mention"
    ]
    dropped = actionable[max_concepts:]
    return capped, dropped
```

Append tests:

```python
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
```

- [ ] **Step 2: Run tests**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py::test_concept_cap_limits_only_actionable_concepts -q
```

Expected: pass.

- [ ] **Step 3: Integrate cap after hallucination filtering**

In `generative/orchestrator.py`, import:

```python
from runtime_config import cap_actionable_concepts
```

After `planner.filter_hallucinated(...)`, add:

```python
            concept_plan.concepts, capped = cap_actionable_concepts(
                concept_plan.concepts,
                runtime_config.max_concepts,
            )
            if capped:
                print(
                    f"      [runtime-config] max_concepts={runtime_config.max_concepts} "
                    f"-> {len(capped)} Konzept(e) übersprungen: "
                    f"{', '.join(c.title for c in capped[:3])}"
                    f"{'…' if len(capped) > 3 else ''}"
                )
```

Ensure `runtime_config` is available in `_run_extraction_stages` by adding it as a parameter and passing it from `main()`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py generative/tests/test_inline_eval_toggle.py -q
```

Expected: all tests pass.

---

### Task 6: Wire Refine Policy And Budget Into Orchestrator

**Files:**
- Modify: `generative/orchestrator.py`
- Test: `generative/tests/test_runtime_config.py`

- [ ] **Step 1: Import refine helpers**

In `generative/orchestrator.py`, import:

```python
from runtime_config import RunBudget, refine_accepted, should_attempt_refine
```

- [ ] **Step 2: Create one run budget per run**

In `main()`, after loading `runtime_config`, create:

```python
    refine_budget = RunBudget(max_refines_per_run=runtime_config.refine.max_refines_per_run)
```

Pass `runtime_config` and `refine_budget` through `process_all_notes_async(...)` into `_run_note_pipeline(...)`.

- [ ] **Step 3: Replace trigger condition with policy call**

In `_run_note_pipeline(...)`, replace the existing refine trigger block:

```python
    refine_trigger_a = (_REFINE_MIN_SCORE <= draft.critic_score < CRITIC_AUTO_THRESHOLD)
    refine_trigger_b = (draft.critic_score >= CRITIC_AUTO_THRESHOLD and not draft.hard_gates_pass)
```

and the `if ((refine_trigger_a or refine_trigger_b) ...):` entry condition with:

```python
    refine_decision = should_attempt_refine(
        draft,
        runtime_config.refine,
        auto_threshold=CRITIC_AUTO_THRESHOLD,
        has_concept_context=_refine_map_key in concept_map,
        synthesized_hint=synthesized_hint,
    )

    if refine_decision.attempt and refine_budget.try_consume():
```

Keep the existing hint construction and retry body. If `refine_decision.attempt` is true but `try_consume()` is false, print:

```python
        print("      [refine] übersprungen: Run-Budget ausgeschöpft")
```

- [ ] **Step 4: Replace acceptance logic**

Replace the current `better` block with:

```python
            better = refine_accepted(refined, auto_threshold=CRITIC_AUTO_THRESHOLD)
            if better:
                print(f"      [refine] Score {draft.critic_score}/{draft.hard_gates_pass} → "
                      f"{refined.critic_score}/{refined.hard_gates_pass} ✓")
                draft = refined
            else:
                print(f"      [refine] Score {refined.critic_score}/{refined.hard_gates_pass} ≤ "
                      f"{draft.critic_score}/{draft.hard_gates_pass}, Original behalten")
```

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py generative/tests/test_inline_eval_toggle.py -q
```

Expected: all tests pass.

---

### Task 7: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run backend/config tests**

Run:

```powershell
python -m pytest generative/tests/test_runtime_config.py generative/tests/test_inline_eval_toggle.py generative/tests/test_backends.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run generative non-slow suite**

Run:

```powershell
python -m pytest generative/tests -q -m "not slow"
```

Expected: all tests pass. Existing unrelated warnings may remain.

- [ ] **Step 3: Run a fast dry-run smoke only after tests pass**

Run with network access available:

```powershell
$env:ATOMIC_AGENT_PROFILE="fast"
python generative/orchestrator.py --source "C:/tmp/Porst-2014-Auszug-S1-40.pdf" --dry-run
```

Expected:
- output includes `[runtime-config] profile=fast`
- planner runs
- concept cap logs skipped concepts when more than 3 actionable concepts exist
- inline eval is skipped
- no self-refine calls are attempted

---

## Deferred Work

- Dashboard for editing runtime config.
- YAML/local config file.
- CLI `--profile` flag, unless env-based profile switching proves too clumsy.
- Token/time-based dynamic budgeting.
- Moving the subscription backend fully from env constants to `RuntimeConfig`; keep current env-backed timeout retry opt-in until backend lifecycle is explicit.

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-05-28-runtime-config-layer.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
