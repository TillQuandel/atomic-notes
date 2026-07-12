"""#198 P2: Kein Test schreibt in das produktive Trace-Verzeichnis.

RED-Nachweis (Vorher-Zustand): Tests wie test_call_record / test_eval_cache_namespace
rufen `agents.base.call_claude_full(...)` mit gestubbtem Backend auf. Der `_trace()`-
Hook schrieb dann eine Zeile nach `generative/.cache/runs/<run-id>.jsonl` — im frischen
Worktree entstand dadurch aus dem Nichts ein produktives `runs/`-Verzeichnis (empirisch
verifiziert: agent="verifier"/"critic"-Trace nach zwei Polluter-Tests).

Die autouse-Fixture `isolate_pipeline_side_effects` (generative/conftest.py) biegt den
Trace-Backend auf tmp um. Dieser Test beweist, dass die Umlenkung greift: ein getracter
Call landet in tmp, das produktive `<repo>/generative/.cache/runs/` bleibt unberührt.
"""

from __future__ import annotations

from generative.agents import base
from generative.agents import tracing
from generative.agents.base import CallResult
from generative.config import CACHE_DIR


def test_traced_call_goes_to_tmp_not_production(isolate_pipeline_side_effects, monkeypatch):
    real_runs = CACHE_DIR / "runs"
    trace_name = f"{tracing._RUN_ID}.jsonl"
    real_before = set(real_runs.glob("*.jsonl")) if real_runs.exists() else set()

    # Gestubbtes Backend → deterministischer, netz-freier Call, der aber den echten
    # _trace()-Schreibpfad exerziert.
    monkeypatch.setattr(base, "_backend_call_full", lambda prompt, **k: CallResult(text="x"))
    base.call_claude_full("prompt", agent="test", use_cache=False)

    # (a) Trace landete im tmp-Verzeichnis der Fixture.
    tmp_trace = isolate_pipeline_side_effects.runs_dir / trace_name
    assert tmp_trace.exists(), f"Trace nicht in tmp geschrieben: {isolate_pipeline_side_effects.runs_dir}"

    # (b) Das produktive runs/-Verzeichnis bekam keine neue Datei.
    real_after = set(real_runs.glob("*.jsonl")) if real_runs.exists() else set()
    assert real_after == real_before, f"Produktives runs/ verschmutzt: {real_after - real_before}"
