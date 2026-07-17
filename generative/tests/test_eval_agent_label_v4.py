"""#319: eval_quality_v4.py nutzte intern weiterhin `agent="eval_quality_v3_..."`
fuer Cache-/Kostenzuordnung -- obwohl der Judge laengst v4 ist. Verwirrend bei
Auswertungen nach Agent-Label (Kosten-/Cache-Aufschluesselung zeigt "v3", obwohl
v4 laeuft).

Cache-Invalidierungs-Nebenwirkung (dokumentiert im PR, Fix-Richtung des Issues):
`agent` ist Teil des LLM-Call-Cache-Keys (`agents/base.py::_cache_key`) -- die
Umbenennung invalidiert den bestehenden Disk-Cache fuer Judge-/Repair-Calls
innerhalb der aktuellen eval_version (die naechste inhaltlich unveraenderte
Note trifft dort einmalig einen Cache-Miss statt -Hit). Der davon unabhaengige
Re-Eval-Hash-Guard (content_hash+eval_version+pipeline_version in
quality_history.jsonl) ist NICHT betroffen.

RED vor dem Fix: agent-Label traegt noch "eval_quality_v3_...".
"""

from __future__ import annotations

from generative.agents import base
import generative.eval_quality_v4 as eq


def test_call_judge_agent_label_uses_v4(monkeypatch):
    captured: dict = {}

    def fake_call(prompt, *, model, agent, use_cache, cache_namespace=None):
        captured["agent"] = agent
        return base.CallResult(text="[]")

    monkeypatch.setattr(eq.base, "call_llm_full", fake_call)

    item = eq.RetrievedContext(
        claim_idx=1,
        claim="Eine Testbehauptung.",
        contexts=[{"chunk_idx": 0, "pages": [1], "text": "Kontext."}],
        top_cosine=0.5,
        best_chunk_idx=0,
        best_page=1,
    )

    eq._call_judge("Titel", [item], variant="primary", use_cache=True)

    assert captured["agent"] == "eval_quality_v4_primary"


def test_repair_json_with_claude_agent_label_uses_v4(monkeypatch):
    captured: dict = {}

    def fake_call(prompt, *, model, agent, use_cache, cache_namespace=None):
        captured["agent"] = agent
        return base.CallResult(text="[]")

    monkeypatch.setattr(eq.base, "call_llm_full", fake_call)

    eq._repair_json_with_claude("kein valides JSON", [], use_cache=True)

    assert captured["agent"] == "eval_quality_v4_json_repair"
