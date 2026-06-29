"""Tests für den Dashboard-Call-Count-Fix (Mahmood-Session 2026-06-26).

Per-Agent-„calls" zählte Nicht-LLM-Bookkeeping-Events (verifier anchor_stats,
critic score_result) als Call → Verifier zeigte „10 calls / 0 Tokens", obwohl
die meisten Records gar keine LLM-Calls sind. Fix: nur Records mit `model`-Feld
zählen. (Der KPI-Min-N-Guard ist bereits im Frontend über `n_notes < 20`
abgedeckt — kein Backend-Eingriff nötig.)
"""

from __future__ import annotations

from generative.eval_dashboard_server import _is_llm_call_record


def test_llm_call_record_recognised_by_model():
    rec = {"agent": "verifier", "model": "anthropic/claude-haiku-4-5", "input_tokens": 9, "output_tokens": 713}
    assert _is_llm_call_record(rec) is True


def test_anchor_stats_event_is_not_a_call():
    rec = {"type": "anchor_stats", "agent": "verifier", "total_in": 1, "confirmed": 3}
    assert _is_llm_call_record(rec) is False


def test_critic_score_result_event_is_not_a_call():
    rec = {"type": "score_result", "agent": "critic", "score": 4}
    assert _is_llm_call_record(rec) is False


def test_orchestrator_bookkeeping_event_is_not_a_call():
    rec = {"type": "note_outcome", "agent": "orchestrator", "outcome": "vault"}
    assert _is_llm_call_record(rec) is False
