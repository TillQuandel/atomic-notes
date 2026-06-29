"""Phoenix/OTel-Instrumentierung: LLM-Calls erzeugen Spans mit Prompt + Output.

Verifiziert die Review-kritischen Punkte (Gemini + Qwen 2026-05-27):
- LLM-Span pro Call mit input.value / output.value / openinference.span.kind=LLM
- Cache-Hit-Pfad erzeugt ebenfalls einen Span (sonst fehlen Re-Run-Calls in Phoenix)
- Verschachtelung: Call-Span ist Kind eines umgebenden Stage-Spans
"""

from __future__ import annotations

import asyncio

import pytest


from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import generative.agents.base as base


@pytest.fixture
def exporter(monkeypatch):
    """Patcht den Modul-Tracer auf einen In-Memory-Provider — isoliert vom globalen OTel-State."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(base, "_OTEL_TRACER", provider.get_tracer("test"))
    return exp


def test_live_call_emits_llm_span_with_io(exporter, monkeypatch):
    monkeypatch.setattr(
        base,
        "_backend_call_full",
        lambda prompt, *, model, agent: base.CallResult(text="ANTWORT", input_tokens=5, output_tokens=7),
    )
    base.call_claude_full("PROMPT", model="m", agent="extractor", use_cache=False)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "extractor"
    assert s.attributes["openinference.span.kind"] == "LLM"
    assert s.attributes["input.value"] == "PROMPT"
    assert s.attributes["output.value"] == "ANTWORT"
    assert s.attributes["llm.token_count.prompt"] == 5
    assert s.attributes["llm.token_count.completion"] == 7
    assert s.attributes["cache.hit"] is False


def test_cache_hit_still_emits_span(exporter, monkeypatch):
    monkeypatch.setattr(base, "_cache_get", lambda key: base.CallResult(text="CACHED", input_tokens=1, output_tokens=2))
    base.call_claude_full("PROMPT", model="m", agent="critic", use_cache=True)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["output.value"] == "CACHED"
    assert spans[0].attributes["cache.hit"] is True


def test_call_span_nested_under_stage_span(exporter, monkeypatch):
    monkeypatch.setattr(
        base,
        "_backend_call_full",
        lambda prompt, *, model, agent: base.CallResult(text="X"),
    )
    tracer = base._OTEL_TRACER
    with tracer.start_as_current_span("Stage") as stage:
        base.call_claude_full("P", model="m", agent="extractor", use_cache=False)
        stage_ctx = stage.get_span_context()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "extractor" in spans and "Stage" in spans
    # Call-Span muss denselben Trace haben und den Stage-Span als Parent
    assert spans["extractor"].context.trace_id == stage_ctx.trace_id
    assert spans["extractor"].parent.span_id == stage_ctx.span_id


def test_no_tracer_is_noop(monkeypatch):
    """Tracing aus (kein Provider) → kein Span, Hot-Path unverändert."""
    monkeypatch.setattr(base, "_OTEL_TRACER", None)
    monkeypatch.setattr(
        base,
        "_backend_call_full",
        lambda prompt, *, model, agent: base.CallResult(text="X"),
    )
    # Darf nicht crashen und liefert normal das Ergebnis
    assert base.call_claude_full("P", model="m", agent="a", use_cache=False).text == "X"


def test_async_call_span_nested_under_stage(exporter, monkeypatch):
    """call_claude_full_async (asyncio.gather/Semaphore-Pfad) bleibt unter dem Stage-Span."""

    async def _backend(prompt, *, model, agent):
        return base.CallResult(text="ASYNC")

    monkeypatch.setattr(base, "_backend_call_full_async", _backend)

    tracer = base._OTEL_TRACER
    with tracer.start_as_current_span("Stage") as stage:
        stage_ctx = stage.get_span_context()
        asyncio.run(base.call_claude_full_async("P", model="m", agent="extractor", use_cache=False))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["extractor"].attributes["output.value"] == "ASYNC"
    assert spans["extractor"].context.trace_id == stage_ctx.trace_id
    assert spans["extractor"].parent.span_id == stage_ctx.span_id


def test_to_thread_call_span_nested_under_stage(exporter, monkeypatch):
    """Stage-6-Pfad: sync call_claude_full via asyncio.to_thread bleibt unter dem Stage-Span."""
    monkeypatch.setattr(
        base,
        "_backend_call_full",
        lambda prompt, *, model, agent: base.CallResult(text="THREADED"),
    )

    async def _runner():
        return await asyncio.to_thread(base.call_claude_full, "P", model="m", agent="critic", use_cache=False)

    tracer = base._OTEL_TRACER
    with tracer.start_as_current_span("Stage6") as stage:
        stage_ctx = stage.get_span_context()
        asyncio.run(_runner())

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["critic"].attributes["output.value"] == "THREADED"
    assert spans["critic"].context.trace_id == stage_ctx.trace_id
    assert spans["critic"].parent.span_id == stage_ctx.span_id
