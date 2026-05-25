# Per-Agent Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jeden Agenten der atomic-agent-Pipeline nach Geschwindigkeit, Tokenverbrauch, Effizienz und Qualität tracken — um Flaschenhälse zu identifizieren (welcher Agent ist am teuersten/langsamsten?), Qualitätsprobleme zu lokalisieren (wo entstehen Fehler?) und gezielte Verbesserungen ableiten zu können. Modell-Vergleiche (z.B. Opus vs. Sonnet) sind ein Anwendungsfall davon, nicht das Primärziel.

**Architecture:** Ein austauschbares `TracingBackend`-Interface entkoppelt die Event-Erzeugung vom Storage. Default: `JsonlBackend` (schreibt in `.cache/runs/<run-id>.jsonl`). Später austauschbar gegen `OtelBackend` oder `LangfuseBackend` ohne Änderung an Agenten-Code. `MODEL_CONFIG` kommt in `config.py` und in den Eval-Output. Ein neues Script `eval_agent_stats.py` aggregiert die JSONL zu per-Agent-Tabellen.

**Tech Stack:** Python 3.10+, bestehendes `agents/base.py` Trace-System, `json`, `pathlib`. Optional für Multi-Run-Analyse: `duckdb` (kein Pflicht-Dependency, nur für `eval_agent_stats.py` Multi-File-Mode).

**OTel-Kompatibilität (Recherche 2026-05-18):** `write(entry: dict)` Interface ist korrekt — OTel ist reine Backend-Angelegenheit. Event-Felder werden so benannt dass ein späterer `OtelBackend` sie 1:1 mappen kann: `agent` → `gen_ai.agent.name`, `model` → `gen_ai.request.model`, `input_tokens` → `gen_ai.usage.input_tokens`, `output_tokens` → `gen_ai.usage.output_tokens`. Kein Umbau des Caller-Codes nötig.

**Multi-Run (Recherche 2026-05-18):** `eval_agent_stats.py` akzeptiert Glob-Pattern (`python eval_agent_stats.py runs/*.jsonl`). DuckDB-Query für Runs-Vergleich: `read_ndjson_auto('runs/*.jsonl', union_by_name=true)` — handelt nested `model_config` automatisch als STRUCT.

> **Gemini-Review 2026-05-18 (2.5 Pro):** 2 Befunde eingearbeitet:
> 1. `verifier.py` + `critic.py`: `try/finally` statt N× `_log_*` vor `return` — verhindert Logging-Vergessen bei zukünftigen Exit-Pfaden.
> 2. `eval_agent_stats.py`: `error_rate` pro Agent ergänzt (Reliability-Metrik für Opus-vs-Sonnet-Vergleich). Race-Condition (thread-safe via `_TRACE_LOCK`) und LLM-Call-Detection (kein `type`-Feld = älterer Entry-Format) als korrekt bestätigt.

---

## File Map

| File | Aktion | Verantwortlichkeit |
|---|---|---|
| `agents/tracing.py` | Create | `TracingBackend`-Protocol + `JsonlBackend` + `trace_event()` + `trace_run_start()` |
| `agents/base.py` | Modify | `trace_event` + `trace_run_start` aus `agents.tracing` re-exportieren (Backwards-Compat) |
| `config.py` | Modify | `MODEL_CONFIG`-Dict ergänzen |
| `agents/verifier.py` | Modify | `anchor_stats`-Event via `try/finally` in `run()` |
| `agents/critic.py` | Modify | `score_result`-Event via `try/finally` in `run()` |
| `orchestrator.py` | Modify | `trace_run_start` am Anfang + `note_outcome` + `plan_stats` |
| `eval_quality_v4.py` | Modify | `model_config` in `_aggregate`-Output |
| `eval_agent_stats.py` | Create | Aggregations-Script: JSONL → per-Agent-Tabelle |
| `agents/langfuse_backend.py` | Create | Optionaler `LangfuseBackend` (Dev-Tool, kein Pflicht-Dependency) |
| `tests/test_per_agent_tracking.py` | Create | Unit-Tests für alle neuen Funktionen |

---

### Task 1: `TracingBackend`-Interface + `JsonlBackend` in `agents/tracing.py`

> **Design-Prinzip:** Backend ist austauschbar. `trace_event()` und `trace_run_start()` kennen kein JSONL — sie delegieren an den aktiven Backend. Swap auf OTel/Langfuse = eine Zeile: `set_tracing_backend(OtelBackend())`.

**Files:**
- Create: `agents/tracing.py`
- Modify: `agents/base.py` (Re-Export für Backwards-Compat)
- Test: `tests/test_per_agent_tracking.py`

- [ ] **Step 1: Failing Tests schreiben**

```python
# tests/test_per_agent_tracking.py
import json
from pathlib import Path
import pytest


def test_trace_event_writes_jsonl(tmp_path, monkeypatch):
    import agents.base as base
    monkeypatch.setattr(base, "_TRACE_FILE", None)
    monkeypatch.setattr(base, "_RUN_DIR", tmp_path)
    monkeypatch.setattr(base, "_RUN_ID", "test-run")

    base.trace_event("verifier", "anchor_stats", {"total_in": 5, "confirmed": 4})

    lines = (tmp_path / "test-run.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"] == "anchor_stats"
    assert entry["agent"] == "verifier"
    assert entry["total_in"] == 5
    assert entry["confirmed"] == 4
    assert "ts" in entry


def test_trace_run_start_writes_model_config(tmp_path, monkeypatch):
    import agents.base as base
    monkeypatch.setattr(base, "_TRACE_FILE", None)
    monkeypatch.setattr(base, "_RUN_DIR", tmp_path)
    monkeypatch.setattr(base, "_RUN_ID", "test-run")

    base.trace_run_start({"planner": "opus", "extractor": "sonnet"})

    lines = (tmp_path / "test-run.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["type"] == "run_start"
    assert entry["model_config"]["extractor"] == "sonnet"
    assert entry["run_id"] == "test-run"
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
cd c:\Users\tillq\Obsidian_Vault\98-system\scripts\atomic-agent
pytest tests/test_per_agent_tracking.py -v
```
Erwartet: `AttributeError: module 'agents.base' has no attribute 'trace_event'`

- [ ] **Step 3: `agents/tracing.py` erstellen**

> OTel-Mapping-Hinweis: Feldnamen in Events so wählen dass `OtelBackend` sie ohne Transformation übernehmen kann. `input_tokens`/`output_tokens` statt `input`/`output` — matches `gen_ai.usage.input_tokens`.

```python
"""Austauschbares Tracing-Backend für die atomic-agent-Pipeline.

Swap-Beispiel (nach SDK-Migration):
    from agents.tracing import set_tracing_backend
    set_tracing_backend(OtelBackend(endpoint="..."))
"""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from typing import Protocol


class TracingBackend(Protocol):
    def write(self, entry: dict) -> None: ...


class JsonlBackend:
    """Default-Backend: schreibt in .cache/runs/<run-id>.jsonl."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self._run_dir = run_dir
        self._run_id = run_id
        self._file: Path | None = None
        self._lock = threading.Lock()

    def write(self, entry: dict) -> None:
        with self._lock:
            if self._file is None:
                self._run_dir.mkdir(parents=True, exist_ok=True)
                self._file = self._run_dir / f"{self._run_id}.jsonl"
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Aktives Backend — austauschbar via set_tracing_backend()
from config import CACHE_DIR
import time as _time
_RUN_ID = _time.strftime("%Y%m%d-%H%M%S")
_backend: TracingBackend = JsonlBackend(
    run_dir=CACHE_DIR / "runs",
    run_id=_RUN_ID,
)


def set_tracing_backend(backend: TracingBackend) -> None:
    """Ersetzt das aktive Backend. Aufruf vor dem ersten trace_event()."""
    global _backend
    _backend = backend


def trace_event(agent: str, event_type: str, data: dict) -> None:
    """Schreibt ein strukturiertes Event. Backend-agnostisch."""
    _backend.write({
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type": event_type,
        "agent": agent,
        **data,
    })


def trace_run_start(model_config: dict) -> None:
    """Schreibt Run-Start-Entry mit Model-Config."""
    _backend.write({
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type": "run_start",
        "run_id": _RUN_ID,
        "model_config": model_config,
    })
```

In `agents/base.py` Re-Export ergänzen (nach den bestehenden Imports):

```python
# Re-Export für Backwards-Compat — Agenten importieren aus agents.base
from agents.tracing import trace_event, trace_run_start, set_tracing_backend  # noqa: F401
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
pytest tests/test_per_agent_tracking.py -v
```
Erwartet: 2 passed

> **Swap-Check:** `OtelBackend` braucht nur `write(entry: dict)` zu implementieren. Feldnamen mappen direkt: `entry["input_tokens"]` → `gen_ai.usage.input_tokens`, `entry["agent"]` → `gen_ai.agent.name`. Kein Umbau im Agenten-Code.

- [ ] **Step 5: Commit**

```
git add agents/base.py tests/test_per_agent_tracking.py
git commit -m "feat: add trace_event() and trace_run_start() to base.py"
```

---

### Task 2: `MODEL_CONFIG` in `config.py`

**Files:**
- Modify: `config.py`
- Test: `tests/test_per_agent_tracking.py`

- [ ] **Step 1: Failing Test ergänzen**

```python
def test_model_config_has_required_keys():
    from config import MODEL_CONFIG
    required = {"planner", "extractor", "verifier", "cross_ref", "critic", "canonicalizer"}
    assert required <= set(MODEL_CONFIG.keys())
    for k, v in MODEL_CONFIG.items():
        assert v, f"MODEL_CONFIG['{k}'] ist leer"
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_model_config_has_required_keys -v
```
Erwartet: `ImportError: cannot import name 'MODEL_CONFIG'`

- [ ] **Step 3: `MODEL_CONFIG` in `config.py` nach `MODEL_LLM_DEDUP` (Zeile ~131) einfügen**

```python
# Snapshot aller Agent-Modell-Zuweisungen — für Run-Trace und Eval-Vergleiche.
MODEL_CONFIG = {
    "planner":       MODEL_PLANNER,
    "extractor":     MODEL_EXTRACTOR,
    "verifier":      MODEL_VERIFIER,
    "cross_ref":     MODEL_CROSS_REF,
    "critic":        MODEL_CRITIC,
    "canonicalizer": MODEL_CANONICALIZER,
}
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
pytest tests/test_per_agent_tracking.py -v
```
Erwartet: 3 passed

- [ ] **Step 5: Commit**

```
git add config.py tests/test_per_agent_tracking.py
git commit -m "feat: add MODEL_CONFIG snapshot to config.py"
```

---

### Task 3: Anchor-Stats-Tracking in `agents/verifier.py`

**Files:**
- Modify: `agents/verifier.py`
- Test: `tests/test_per_agent_tracking.py`

> `try/finally` statt 6× Aufruf vor `return` — verhindert Logging-Vergessen bei neuen Exit-Pfaden (Gemini-Befund).

- [ ] **Step 1: Failing Test ergänzen**

```python
def test_verifier_run_emits_anchor_stats(tmp_path, monkeypatch):
    import agents.base as base
    import agents.verifier as verifier
    from schemas.atomic_note import AtomicNoteDraft

    monkeypatch.setattr(base, "_TRACE_FILE", None)
    monkeypatch.setattr(base, "_RUN_DIR", tmp_path)
    monkeypatch.setattr(base, "_RUN_ID", "test-run")

    draft = AtomicNoteDraft(title="Test Note", body="Kurztext.", source_anchors=[])
    verifier.run(draft, chunk_text="[S. 1] Kurztext. Weiterer Text.")

    lines = (tmp_path / "test-run.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines]
    anchor_events = [e for e in events if e.get("type") == "anchor_stats"]
    assert len(anchor_events) == 1
    ev = anchor_events[0]
    assert ev["agent"] == "verifier"
    assert "total_in" in ev
    assert "confirmed" in ev
    assert "confirmation_rate" in ev
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_verifier_run_emits_anchor_stats -v
```

- [ ] **Step 3: Import + `try/finally` in `agents/verifier.py` einbauen**

Import-Zeile (bestehend) erweitern:
```python
from agents.base import call_claude, trace_event
```

Den gesamten Body von `run()` in `try/finally` einwickeln — `_total_in` als erste Zeile vor dem `try`:

```python
def run(draft: AtomicNoteDraft, chunk_text: str) -> AtomicNoteDraft:
    _total_in = len(draft.source_anchors)
    try:
        # --- bisheriger run()-Body unverändert ---
        if not draft.source_anchors:
            sync_anchors_from_body(draft)
            if not draft.source_anchors:
                return draft
        # ... (restlicher Code identisch) ...
        return draft
    finally:
        confirmed = sum(1 for a in draft.source_anchors if a.page or a.fuzzy_page)
        trace_event("verifier", "anchor_stats", {
            "title": draft.title,
            "total_in": _total_in,
            "confirmed": confirmed,
            "confirmation_rate": round(confirmed / _total_in, 3) if _total_in > 0 else 0.0,
        })
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
pytest tests/test_per_agent_tracking.py -v
pytest tests/test_verifier_prepass.py -v
```

- [ ] **Step 5: Commit**

```
git add agents/verifier.py tests/test_per_agent_tracking.py
git commit -m "feat: emit anchor_stats trace event from verifier.run() via try/finally"
```

---

### Task 4: Score-Tracking in `agents/critic.py`

**Files:**
- Modify: `agents/critic.py`
- Test: `tests/test_per_agent_tracking.py`

- [ ] **Step 1: Failing Test ergänzen**

```python
def test_critic_run_emits_score_result(tmp_path, monkeypatch):
    import agents.base as base
    import agents.critic as critic
    import config as _config
    from schemas.atomic_note import AtomicNoteDraft, TextAnchor

    monkeypatch.setattr(base, "_TRACE_FILE", None)
    monkeypatch.setattr(base, "_RUN_DIR", tmp_path)
    monkeypatch.setattr(base, "_RUN_ID", "test-run")
    monkeypatch.setattr(_config, "ENABLE_LLM", False)

    draft = AtomicNoteDraft(
        title="Test Note",
        body="Die Kategorialen Beschreibungen sind zentral. (S. 5) " * 5,
        source_anchors=[TextAnchor(quote="zentral", page="S. 5")],
    )
    critic.run(draft)

    lines = (tmp_path / "test-run.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(l) for l in lines]
    score_events = [e for e in events if e.get("type") == "score_result"]
    assert len(score_events) == 1
    ev = score_events[0]
    assert ev["agent"] == "critic"
    assert isinstance(ev["score"], int)
    assert isinstance(ev["hard_gates_pass"], bool)
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_critic_run_emits_score_result -v
```

- [ ] **Step 3: Import + `try/finally` in `agents/critic.py` einbauen**

Import-Zeile erweitern:
```python
from agents.base import call_claude, trace_event
```

Body von `run()` in `try/finally`:

```python
def run(draft, existing_concepts=None, concept_links=None):
    try:
        # --- bisheriger run()-Body unverändert ---
        ...
        return draft
    finally:
        trace_event("critic", "score_result", {
            "title": draft.title,
            "score": draft.critic_score,
            "hard_gates_pass": draft.hard_gates_pass,
        })
```

- [ ] **Step 4: Tests + bestehende Tests ausführen — alle PASS**

```
pytest tests/test_per_agent_tracking.py -v
pytest tests/ -x -q --ignore=tests/test_e2e_baseline.py
```

- [ ] **Step 5: Commit**

```
git add agents/critic.py tests/test_per_agent_tracking.py
git commit -m "feat: emit score_result trace event from critic.run() via try/finally"
```

---

### Task 5: `orchestrator.py` — Run-Start + Note-Outcome + Plan-Stats

**Files:**
- Modify: `orchestrator.py`

- [ ] **Step 1: `trace_run_start` am Anfang von `main()` einbauen**

In `main()`, direkt nach dem ersten `print`-Statement (~Zeile 730):

```python
from agents.base import trace_run_start as _trace_run_start, trace_event as _trace_event
from config import MODEL_CONFIG as _MODEL_CONFIG
_trace_run_start(_MODEL_CONFIG)
```

- [ ] **Step 2: `note_outcome`-Event im Vault-Writer-Loop einbauen** (~Zeile 976)

```python
for draft in drafts:
    vault_writer.write_note(draft, source_file=source_path.name,
                            dry_run=args.dry_run, source_meta=enriched_meta,
                            existing_concepts=existing_concepts)
    will_vault, _ = vault_writer.auto_write_decision(draft)
    _trace_event("orchestrator", "note_outcome", {
        "title": draft.title,
        "destination": "vault" if will_vault else "inbox",
        "critic_score": draft.critic_score,
        "hard_gates_pass": draft.hard_gates_pass,
    })
    written += 1
```

- [ ] **Step 3: `plan_stats`-Event direkt nach `print(f"   -> Inbox: ...")` (~Zeile 986)**

```python
_trace_event("orchestrator", "plan_stats", {
    "written": written,
    "vault": vault_count,
    "inbox": inbox_count,
    "vault_rate": round(vault_count / written, 3) if written > 0 else 0.0,
})
```

- [ ] **Step 4: Smoke-Test (dry-run) + JSONL prüfen**

```
python orchestrator.py --source "C:\Users\tillq\OneDrive\Dokumente\Literatur\Bates - 2017 - Information Behavior.pdf" --dry-run
```

```python
import json, pathlib
runs = sorted(pathlib.Path(".cache/runs").glob("*.jsonl"))
lines = runs[-1].read_text().splitlines()
entries = [json.loads(l) for l in lines]
types = {e.get("type", "llm_call") for e in entries}
print("Event-Typen:", types)
# Erwartet: {'run_start', 'anchor_stats', 'score_result', 'note_outcome', 'plan_stats', 'llm_call'}
run_start = next(e for e in entries if e.get("type") == "run_start")
print("model_config:", run_start["model_config"])
```

- [ ] **Step 5: Commit**

```
git add orchestrator.py
git commit -m "feat: trace run_start, note_outcome and plan_stats in orchestrator"
```

---

### Task 6: `model_config` in `eval_quality_v4.py`

**Files:**
- Modify: `eval_quality_v4.py`
- Test: `tests/test_per_agent_tracking.py`

- [ ] **Step 1: Failing Test ergänzen**

```python
def test_aggregate_includes_model_config(tmp_path):
    import eval_quality_v4 as eq

    result = eq._aggregate(
        note_path=tmp_path / "test.md",
        pdf_path=tmp_path / "test.pdf",
        pipeline_version="v0.0.0",
        timestamp="2026-01-01T00:00:00",
        language_pair="de-en",
        chunks=[],
        claim_scores=[],
        llm_meta={},
    )
    assert "model_config" in result
    assert isinstance(result["model_config"], dict)
    assert "extractor" in result["model_config"]
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_aggregate_includes_model_config -v
```

- [ ] **Step 3: `MODEL_CONFIG` importieren + in `_aggregate` ergänzen**

Import-Zeile (~Zeile 37) ergänzen:
```python
from config import AGENT_VERSION, CACHE_DIR, EVAL_ADAPTIVE_K_HIGH, EVAL_ADAPTIVE_K_MID, MODEL_OPUS, MODEL_CONFIG
```

Im Return-Dict von `_aggregate()`:
```python
"model_config": MODEL_CONFIG,
```

- [ ] **Step 4: Tests ausführen — muss PASS**

```
pytest tests/test_per_agent_tracking.py -v
```

- [ ] **Step 5: Commit**

```
git add eval_quality_v4.py tests/test_per_agent_tracking.py
git commit -m "feat: include model_config in eval_quality_v4 aggregate output"
```

---

### Task 7: `eval_agent_stats.py` — Aggregations-Script

**Files:**
- Create: `eval_agent_stats.py`
- Test: `tests/test_per_agent_tracking.py`

> `error_rate` pro Agent ergänzt (Gemini-Befund: Reliability-Metrik für Modell-Vergleich).

- [ ] **Step 1: Failing Test ergänzen**

```python
def test_eval_agent_stats_aggregates_llm_calls(tmp_path):
    import json, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    trace = tmp_path / "run.jsonl"
    entries = [
        {"type": "run_start", "run_id": "r1", "model_config": {"extractor": "opus"}, "ts": "2026-01-01T00:00:00"},
        {"agent": "extractor", "model": "opus", "input_tokens": 1000, "output_tokens": 200, "duration_ms": 3000, "cached": False, "error": None, "ts": "2026-01-01T00:00:01"},
        {"agent": "extractor", "model": "opus", "input_tokens": 1200, "output_tokens": 250, "duration_ms": 4000, "cached": False, "error": "Timeout", "ts": "2026-01-01T00:00:05"},
        {"type": "anchor_stats", "agent": "verifier", "total_in": 5, "confirmed": 4, "confirmation_rate": 0.8, "ts": "2026-01-01T00:00:06"},
        {"type": "score_result", "agent": "critic", "score": 4, "hard_gates_pass": True, "ts": "2026-01-01T00:00:07"},
        {"type": "note_outcome", "agent": "orchestrator", "destination": "vault", "critic_score": 4, "ts": "2026-01-01T00:00:08"},
        {"type": "plan_stats", "agent": "orchestrator", "written": 1, "vault": 1, "inbox": 0, "vault_rate": 1.0, "ts": "2026-01-01T00:00:09"},
    ]
    trace.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")

    import eval_agent_stats as eas
    stats = eas.aggregate(trace)

    assert stats["extractor"]["calls"] == 2
    assert stats["extractor"]["total_input_tokens"] == 2200
    assert stats["extractor"]["total_output_tokens"] == 450
    assert stats["extractor"]["avg_duration_ms"] == 3500
    assert stats["extractor"]["error_rate"] == pytest.approx(0.5)
    assert stats["verifier"]["avg_confirmation_rate"] == pytest.approx(0.8)
    assert stats["critic"]["avg_score"] == pytest.approx(4.0)
    assert stats["orchestrator"]["vault_rate"] == pytest.approx(1.0)
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_eval_agent_stats_aggregates_llm_calls -v
```
Erwartet: `ModuleNotFoundError: No module named 'eval_agent_stats'`

- [ ] **Step 3: `eval_agent_stats.py` erstellen**

> Multi-Run: Script akzeptiert mehrere JSONL-Files via Glob. DuckDB optional für komplexe Queries (`pip install duckdb`), Pandas-Fallback für einfache Fälle.

```python
"""Per-Agent Statistik-Aggregation aus Run-Trace-JSONL.

Usage:
  python eval_agent_stats.py                          # neuester Run
  python eval_agent_stats.py .cache/runs/20260518-123456.jsonl
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path


def aggregate(trace_path: Path) -> dict[str, dict]:
    """Liest JSONL und gibt per-Agent-Stats-Dict zurück.

    LLM-Call-Entries haben kein 'type'-Feld (base._trace()-Format).
    Strukturierte Events haben type: run_start | anchor_stats | score_result | ...
    """
    entries = [
        json.loads(l)
        for l in trace_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    llm_calls: dict[str, list] = defaultdict(list)
    anchor_rates: dict[str, list] = defaultdict(list)
    scores: dict[str, list] = defaultdict(list)
    outcomes: list[dict] = []
    plan: dict = {}

    for e in entries:
        agent = e.get("agent", "")
        etype = e.get("type")

        if etype is None and agent:      # LLM-Call (kein type-Feld)
            llm_calls[agent].append(e)
        elif etype == "anchor_stats":
            anchor_rates[agent].append(e.get("confirmation_rate", 0.0))
        elif etype == "score_result":
            scores[agent].append(e.get("score", 0))
        elif etype == "note_outcome":
            outcomes.append(e)
        elif etype == "plan_stats":
            plan = e

    stats: dict[str, dict] = {}

    for agent, calls in llm_calls.items():
        errors = [c for c in calls if c.get("error")]
        not_cached = [c for c in calls if not c.get("cached", False)]
        stats.setdefault(agent, {}).update({
            "calls": len(calls),
            "cached_calls": len(calls) - len(not_cached),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(calls), 3) if calls else 0.0,
            "total_input_tokens": sum(c.get("input_tokens", 0) for c in not_cached),
            "total_output_tokens": sum(c.get("output_tokens", 0) for c in not_cached),
            "avg_duration_ms": (
                int(sum(c.get("duration_ms", 0) for c in not_cached) / len(not_cached))
                if not_cached else 0
            ),
        })

    for agent, rates in anchor_rates.items():
        stats.setdefault(agent, {})["avg_confirmation_rate"] = (
            round(sum(rates) / len(rates), 3) if rates else 0.0
        )

    for agent, sc in scores.items():
        stats.setdefault(agent, {})["avg_score"] = (
            round(sum(sc) / len(sc), 2) if sc else 0.0
        )

    if outcomes:
        vault_n = sum(1 for o in outcomes if o.get("destination") == "vault")
        stats.setdefault("orchestrator", {}).update({
            "notes_total": len(outcomes),
            "notes_vault": vault_n,
            "vault_rate": round(vault_n / len(outcomes), 3),
        })

    if plan:
        stats.setdefault("orchestrator", {}).update({
            "vault_rate": plan.get("vault_rate", 0.0),
            "written": plan.get("written", 0),
        })

    return stats


def _find_latest_trace() -> Path | None:
    runs_dir = Path(__file__).parent / ".cache" / "runs"
    if not runs_dir.exists():
        return None
    files = sorted(runs_dir.glob("*.jsonl"))
    return files[-1] if files else None


def _print_table(stats: dict, model_config: dict | None, run_id: str) -> None:
    print(f"\n=== Agent-Stats: Run {run_id} ===")
    if model_config:
        print("Model-Config:", " | ".join(f"{k}={v}" for k, v in model_config.items()))
    print()
    print(f"{'Agent':<18} {'Calls':>6} {'Err%':>6} {'In-Tok':>9} {'Out-Tok':>8} {'Ø ms':>7} {'Ø Score':>8} {'Anker%':>8} {'Vault%':>7}")
    print("-" * 82)
    for agent, s in sorted(stats.items()):
        err_pct = f"{s['error_rate']:.0%}" if "error_rate" in s else ""
        score   = f"{s['avg_score']:.1f}"  if "avg_score" in s else ""
        anker   = f"{s['avg_confirmation_rate']:.0%}" if "avg_confirmation_rate" in s else ""
        vault   = f"{s['vault_rate']:.0%}" if "vault_rate" in s else ""
        print(
            f"{agent:<18}"
            f" {s.get('calls', ''):>6}"
            f" {err_pct:>6}"
            f" {s.get('total_input_tokens', ''):>9}"
            f" {s.get('total_output_tokens', ''):>8}"
            f" {s.get('avg_duration_ms', ''):>7}"
            f" {score:>8}"
            f" {anker:>8}"
            f" {vault:>7}"
        )


def main() -> None:
    # Mehrere Files per Glob oder einzelner Pfad
    import glob as _glob
    if len(sys.argv) > 1:
        paths = [Path(p) for pattern in sys.argv[1:] for p in _glob.glob(pattern)]
    else:
        latest = _find_latest_trace()
        paths = [latest] if latest else []
    if not paths:
        print("Kein Run-Trace gefunden unter .cache/runs/")
        sys.exit(1)

    if len(paths) == 1:
        trace_path = paths[0]
        entries = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        run_start = next((e for e in entries if e.get("type") == "run_start"), {})
        stats = aggregate(trace_path)
        _print_table(stats, run_start.get("model_config"), run_start.get("run_id", trace_path.stem))
    else:
        # Multi-Run: je Run eine Zeile
        print(f"\n=== Multi-Run-Vergleich ({len(paths)} Runs) ===")
        print(f"{'Run':<24} {'Agent':<14} {'Calls':>5} {'In-Tok':>8} {'Out-Tok':>8} {'Ø ms':>7} {'Vault%':>7}")
        print("-" * 75)
        for p in sorted(paths):
            entries = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            run_start = next((e for e in entries if e.get("type") == "run_start"), {})
            run_id = run_start.get("run_id", p.stem)
            model = run_start.get("model_config", {}).get("extractor", "?")
            stats = aggregate(p)
            for agent, s in sorted(stats.items()):
                vault = f"{s['vault_rate']:.0%}" if "vault_rate" in s else ""
                print(
                    f"{run_id[:24]:<24}"
                    f" {agent[:14]:<14}"
                    f" {s.get('calls', ''):>5}"
                    f" {s.get('total_input_tokens', ''):>8}"
                    f" {s.get('total_output_tokens', ''):>8}"
                    f" {s.get('avg_duration_ms', ''):>7}"
                    f" {vault:>7}"
                )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Alle Tests ausführen — muss PASS**

```
pytest tests/test_per_agent_tracking.py -v
pytest tests/ -x -q --ignore=tests/test_e2e_baseline.py
```
Erwartet: alle grün

- [ ] **Step 5: Commit**

```
git add eval_agent_stats.py tests/test_per_agent_tracking.py
git commit -m "feat: add eval_agent_stats.py for per-agent run aggregation"
```

---

---

### Task 8: `LangfuseBackend` in `agents/langfuse_backend.py`

> **Scope:** Reines Entwicklerwerkzeug für Till. Endnutzer des Obsidian-Plugins sehen davon nichts. Aktivierung via Env-Var `ATOMIC_AGENT_TRACING=langfuse`. Wenn `langfuse`-Package nicht installiert: graceful skip, kein Crash.
>
> **Gemini-Review 2026-05-18:** 4 Befunde eingearbeitet: (1) `span()` braucht `startTime`/`endTime` aus `ts` + `duration_ms`, sonst 0ms in UI. (2) `flush()` in `tracing.py` als public `flush_tracing()` kapseln. (3) `atexit.register()` für Flush bei Ctrl+C/Exception. (4) Error-Handling in `write()` — Langfuse-Ausfall darf Pipeline nicht crashen.

**Voraussetzungen (einmalig):**
```powershell
# Docker Desktop muss laufen
git clone https://github.com/langfuse/langfuse
cd langfuse
docker compose up -d  # → http://localhost:3000
# Account + Project anlegen, API-Keys kopieren
```

**Env-Vars setzen** (in `.env` im Repo-Root, bereits in `.gitignore`):
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
ATOMIC_AGENT_TRACING=langfuse
```

**Files:**
- Create: `agents/langfuse_backend.py`
- Modify: `agents/tracing.py` (Auto-Aktivierung via Env-Var)
- Test: `tests/test_per_agent_tracking.py`

- [ ] **Step 1: Failing Test schreiben**

```python
def test_langfuse_backend_write_does_not_crash_without_package(monkeypatch):
    """LangfuseBackend fällt graceful zurück wenn langfuse nicht installiert."""
    import sys
    # langfuse aus sys.modules entfernen falls installiert
    monkeypatch.setitem(sys.modules, "langfuse", None)

    from agents.langfuse_backend import LangfuseBackend
    backend = LangfuseBackend()
    # Kein Crash erwartet, nur no-op
    backend.write({"type": "run_start", "run_id": "test", "model_config": {}, "ts": "2026-01-01"})
    backend.write({"agent": "extractor", "input_tokens": 100, "output_tokens": 50, "duration_ms": 1000, "cached": False, "ts": "2026-01-01"})


def test_langfuse_backend_auto_activated_via_env(monkeypatch, tmp_path):
    """ATOMIC_AGENT_TRACING=langfuse aktiviert LangfuseBackend automatisch."""
    monkeypatch.setenv("ATOMIC_AGENT_TRACING", "langfuse")
    import importlib
    import agents.tracing as tracing
    importlib.reload(tracing)
    from agents.langfuse_backend import LangfuseBackend
    assert isinstance(tracing._backend, LangfuseBackend)
```

- [ ] **Step 2: Test ausführen — muss FAIL**

```
pytest tests/test_per_agent_tracking.py::test_langfuse_backend_write_does_not_crash_without_package -v
```

- [ ] **Step 3: `agents/langfuse_backend.py` erstellen**

```python
"""Optionaler Langfuse-Backend für atomic-agent Tracing.

Nur für die Entwicklung (Till). Endnutzer des Obsidian-Plugins
nutzen JsonlBackend. Aktivierung via ATOMIC_AGENT_TRACING=langfuse.

Setup:
  docker compose up -d  (langfuse/langfuse repo)
  pip install langfuse
  .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""
from __future__ import annotations
from typing import Any


class LangfuseBackend:
    """TracingBackend der Traces an eine lokale Langfuse-Instanz schickt.

    Graceful no-op wenn `langfuse`-Package nicht installiert.
    Einen Trace pro Run (gestartet durch run_start-Event).
    LLM-Calls → Spans. Strukturierte Events → Langfuse-Events.
    """

    def __init__(self) -> None:
        try:
            from langfuse import Langfuse
            self._client: Any = Langfuse()
            self._available = True
        except (ImportError, Exception):
            self._client = None
            self._available = False
        self._trace: Any = None

    def write(self, entry: dict) -> None:
        if not self._available:
            return

        etype = entry.get("type")
        agent = entry.get("agent", "unknown")

        if etype == "run_start":
            self._trace = self._client.trace(
                name=entry.get("run_id", "atomic-agent-run"),
                metadata={"model_config": entry.get("model_config", {})},
                tags=["atomic-agent"],
            )

        try:
            if etype == "run_start":
                self._trace = self._client.trace(
                    name=entry.get("run_id", "atomic-agent-run"),
                    metadata={"model_config": entry.get("model_config", {})},
                    tags=["atomic-agent"],
                )

            elif etype is None and self._trace:  # LLM-Call (kein type-Feld)
                from datetime import datetime, timedelta
                end_time = datetime.fromisoformat(entry["ts"])
                start_time = end_time - timedelta(milliseconds=entry.get("duration_ms", 0))
                self._trace.span(
                    name=f"{agent}/{entry.get('model', '?')}",
                    startTime=start_time,
                    endTime=end_time,
                    metadata={
                        "cached": entry.get("cached", False),
                        "error": entry.get("error"),
                    },
                    usage={
                        "input": entry.get("input_tokens", 0),
                        "output": entry.get("output_tokens", 0),
                        "unit": "TOKENS",
                    },
                    level="DEFAULT" if not entry.get("error") else "ERROR",
                )

            elif etype and self._trace:  # Strukturiertes Event
                meta = {k: v for k, v in entry.items() if k not in ("ts", "type", "agent")}
                self._trace.event(
                    name=f"{agent}/{etype}",
                    metadata=meta,
                    level="DEFAULT",
                )
        except Exception as e:
            import sys
            print(f"[LangfuseBackend] Fehler: {e} — Tracing für diesen Run deaktiviert.", file=sys.stderr)
            self._available = False

    def flush(self) -> None:
        """Alle gepufferten Events senden."""
        if self._available and self._client:
            self._client.flush()
```

- [ ] **Step 4: Auto-Aktivierung + `flush_tracing()` + `atexit` in `agents/tracing.py` einbauen**

Nach der `set_tracing_backend()`-Funktion eine öffentliche Flush-Funktion ergänzen:

```python
def flush_tracing() -> None:
    """Flushes das aktive Backend — aufrufen am Pipeline-Ende."""
    if hasattr(_backend, "flush"):
        _backend.flush()
```

Am Ende von `agents/tracing.py` Auto-Aktivierung mit `atexit`:

```python
import os as _os
if _os.getenv("ATOMIC_AGENT_TRACING") == "langfuse":
    try:
        import atexit
        from agents.langfuse_backend import LangfuseBackend as _LF
        _backend = _LF()
        atexit.register(_backend.flush)  # garantierter Flush bei Ctrl+C + normaler Terminierung
    except Exception:
        pass  # graceful fallback auf JsonlBackend
```

- [ ] **Step 5: `flush_tracing()` in `orchestrator.py` aufrufen** (redundant zu `atexit`, aber explizit am richtigen Zeitpunkt)

Nach dem `plan_stats`-Event:

```python
from agents.tracing import flush_tracing as _flush_tracing
_flush_tracing()
```

- [ ] **Step 6: Tests ausführen — alle PASS**

```
pytest tests/test_per_agent_tracking.py -v
pytest tests/ -x -q --ignore=tests/test_e2e_baseline.py
```

- [ ] **Step 7: Commit**

```
git add agents/langfuse_backend.py agents/tracing.py orchestrator.py tests/test_per_agent_tracking.py
git commit -m "feat: add optional LangfuseBackend for dev observability"
```

---

## Self-Review

**Spec-Coverage:**
- [x] Geschwindigkeit: `avg_duration_ms` per Agent
- [x] Tokenverbrauch: `total_input_tokens` + `total_output_tokens` per Agent
- [x] Effizienz: `vault_rate`, `avg_confirmation_rate`, `cached_calls`
- [x] Qualität: `avg_score` (Critic), `avg_confirmation_rate` (Verifier), `vault_rate`
- [x] Reliability: `error_rate` pro Agent (Gemini-Befund)
- [x] Modell-Vergleich: `model_config` in `run_start` + in `eval_quality_v4`-Output

**Bewusst offen — Planner-Quality-Metrik:** `concept_plan.concepts`-Anzahl nicht geloggt. `plan_stats.vault_rate` ist ein Proxy, aber kein vollständiges Plan-to-Vault-Tracking. Erfordert Änderung in `agents/planner.py` — Scope für v2.

**Placeholder-Check:** Keine TBDs. Alle Code-Blöcke vollständig.

**Type-Konsistenz:** `trace_event(agent, event_type, data)` in Task 1 definiert, in Tasks 3+4+5 identisch aufgerufen. `MODEL_CONFIG` in Task 2 definiert, in Tasks 5+6 importiert. `aggregate(trace_path)` in Task 7 definiert, im Test direkt aufgerufen.
