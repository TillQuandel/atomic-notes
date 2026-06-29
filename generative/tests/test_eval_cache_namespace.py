"""Tests fuer die Entkopplung des Eval-Judge-Caches vom --fresh-run-Salt (H1).

Der teure LLM-Judge-Call soll content-adressiert gecacht werden und UNABHAENGIG
vom Run-Namespace sein, den ``--fresh-run`` via ``set_cache_namespace`` setzt.
So wird eine inhaltlich unveraenderte Note nicht erneut evaluiert, auch wenn die
uebrige Pipeline mit ``--fresh-run`` frisch generiert. AC1-neutral: der Cache
liefert die bit-identische frueher gezogene Judge-Antwort, kein neues Sampling.
"""

from __future__ import annotations

import pytest

from generative.agents import base
import generative.eval_quality_v4 as eq


@pytest.fixture(autouse=True)
def _restore_cache_namespace():
    saved = base._CACHE_NAMESPACE
    yield
    base._CACHE_NAMESPACE = saved


def test_cache_key_namespace_override_is_run_independent():
    """_cache_key(..., namespace=...) ueberschreibt den globalen Run-Salt."""
    base.set_cache_namespace("RUN_A")
    salted = base._cache_key("prompt", "model", "agent")
    override_a = base._cache_key("prompt", "model", "agent", namespace="")
    base.set_cache_namespace("RUN_B")
    override_b = base._cache_key("prompt", "model", "agent", namespace="")

    assert override_a == override_b  # stabil ueber Runs hinweg
    assert override_a != salted  # entkoppelt vom Run-Salt


def test_call_llm_full_namespace_override_hits_cache_across_runs(tmp_path, monkeypatch):
    """Gleicher Prompt + Override-Namespace = Cache-Hit, auch wenn der globale
    Run-Salt zwischen den Aufrufen wechselt (zwei --fresh-run-Laeufe simuliert)."""
    monkeypatch.setattr(base, "_LLM_CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(base, "_trace", lambda *a, **k: None)

    backend_calls: list[str] = []

    def fake_backend(prompt, *, model, agent, **kwargs):
        backend_calls.append(prompt)
        return base.CallResult(text="judge-rows", input_tokens=42, output_tokens=7)

    monkeypatch.setattr(base, "_backend_call_full", fake_backend)

    base.set_cache_namespace("RUN_A")
    base.call_llm_full("p", model="m", agent="eval_quality_v3_primary", cache_namespace="")
    base.set_cache_namespace("RUN_B")
    second = base.call_llm_full("p", model="m", agent="eval_quality_v3_primary", cache_namespace="")

    assert len(backend_calls) == 1  # zweiter Lauf war ein Cache-Hit
    assert second.cached is True
    assert second.text == "judge-rows"


def test_call_judge_passes_run_independent_eval_namespace(monkeypatch):
    """_call_judge gibt den Eval-Calls den run-unabhaengigen Namespace mit."""
    captured: dict = {}

    def fake_call(prompt, *, model, agent, use_cache, cache_namespace=None):
        captured["cache_namespace"] = cache_namespace
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

    assert captured["cache_namespace"] == eq.EVAL_CACHE_NAMESPACE
    # run-unabhaengig (kein _RUN_ID), aber versions-gescoped: ein EVAL_VERSION-Bump
    # invalidiert den Eval-Cache automatisch (Schutz gegen stille Staleness).
    assert eq.EVAL_VERSION in eq.EVAL_CACHE_NAMESPACE


def test_call_llm_full_traces_real_cache_key(tmp_path, monkeypatch):
    """Der Trace-Eintrag fuehrt den ECHTEN Cache-Key (inkl. agent + Override-Namespace),
    nicht den global-gesalzenen agent-losen Hash — sonst luegt die Cache-Hit-Analyse."""
    monkeypatch.setattr(base, "_LLM_CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(base, "_backend_call_full", lambda prompt, *, model, agent, **k: base.CallResult(text="x"))
    traced: dict = {}

    def fake_trace(agent, prompt, model, result, error=None, cache_key=None):
        traced["cache_key"] = cache_key

    monkeypatch.setattr(base, "_trace", fake_trace)

    base.set_cache_namespace("RUN_X")
    base.call_llm_full("p", model="m", agent="eval_quality_v3_primary", cache_namespace="eval-vX")

    expected = base._cache_key("p", "m", "eval_quality_v3_primary", namespace="eval-vX")
    assert traced["cache_key"] == expected
