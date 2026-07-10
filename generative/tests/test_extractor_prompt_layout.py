"""Layout-Wächter für den Extractor-Prompt (#147, Prefix-Cache-Ordnung).

Anthropic-Prompt-Caching matcht token-präfix-basiert: Der cachebare Präfix
endet am ersten Token, das sich zwischen Fan-out-Calls unterscheidet. Diese
Tests erzwingen, dass alle call-variablen Blöcke (Whitelist-Sortierung,
Konzepte, existing, Task-Hints, Chunk) HINTER dem statischen Instruktions-Kern
liegen. Regression (variabler Block rutscht nach vorn) = die statischen Regeln
werden bei jedem Fan-out-Call wieder zur teuren Cache-Neuanlage (real gemessen
41 % der Input-Seite, Issue #147).
"""

import asyncio
import os

from generative.agents import extractor
from generative.schemas.atomic_note import ConceptItem
from generative.schemas.citation import CitationMeta

# Letzter Satz des statischen Instruktions-Kerns — alles danach ist call-variabel.
_STATIC_TAIL_SENTINEL = "stummes Weglassen."

_VARIABLE_PLACEHOLDERS = (
    "{source_meta}",
    "{tag_whitelist}",
    "{concepts}",
    "{existing}",
    "{background_block}",
    "{related_mentions_block}",
    "{task_hints}",
    "{chunk_title}",
    "{chunk_text}",
)


def test_template_variable_blocks_after_static_core():
    tpl = extractor._PROMPT
    static_end = tpl.index(_STATIC_TAIL_SENTINEL)
    for placeholder in _VARIABLE_PLACEHOLDERS:
        assert tpl.index(placeholder) > static_end, (
            f"{placeholder} liegt vor dem Ende des statischen Kerns — "
            "bricht den Prompt-Cache-Präfix aller Fan-out-Calls (#147)"
        )
    # Der Chunk bleibt das letzte Element des Prompts.
    assert tpl.rstrip().endswith("{chunk_text}")


def _concept(title: str) -> ConceptItem:
    return ConceptItem(title=title, priority="high", chapter="1", action="create")


def _capture_prompts(monkeypatch, outputs: list[str], calls: list[dict]) -> None:
    """Ersetzt den LLM-Call: captured Prompts, liefert vorgegebene Outputs."""

    async def fake_call(prompt, **_kw):
        calls.append({"prompt": prompt})
        return outputs[min(len(calls) - 1, len(outputs) - 1)]

    monkeypatch.setattr(extractor, "call_claude_async", fake_call)


_CITATION = CitationMeta(author="Kuhlthau, Carol", year="1991", title="ISP", doi=None, source_file="isp.pdf")
_WHITELIST = ["uni/ibi", "methoden"]


def test_fanout_prompts_share_static_prefix(monkeypatch):
    """Zwei Calls verschiedener Konzepte müssen den kompletten statischen Kern
    als gemeinsamen String-Präfix teilen — die Voraussetzung für Cache-Reads."""
    calls: list[dict] = []
    _capture_prompts(monkeypatch, [""], calls)

    for title, text in (("Konzept Alpha", "Text über Alpha."), ("Konzept Beta", "Ganz anderer Text über Beta.")):
        result = asyncio.run(
            extractor.run_per_concept(
                _concept(title), text, {"Alte Note": "04-wissen/alt.md"}, citation=_CITATION, tag_whitelist=_WHITELIST
            )
        )
        assert result is None  # leerer Fake-Output → None; Prompt ist trotzdem captured

    assert len(calls) == 2
    common = os.path.commonprefix([calls[0]["prompt"], calls[1]["prompt"]])
    assert _STATIC_TAIL_SENTINEL in common, (
        "Der statische Instruktions-Kern ist NICHT im gemeinsamen Präfix der "
        "Fan-out-Prompts — ein call-variabler Block steht zu weit vorn (#147)"
    )


def test_refine_hint_goes_to_variable_tail_not_prompt_start(monkeypatch):
    """Self-Refine-Hinweis darf den Prompt-Präfix nicht mehr brechen: Er steht
    im variablen Schwanz (nach 'existing', vor dem Textabschnitt), nicht vorn."""
    calls: list[dict] = []
    _capture_prompts(monkeypatch, [""], calls)

    asyncio.run(
        extractor.run_per_concept(
            _concept("Konzept Alpha"),
            "Text über Alpha.",
            {},
            citation=_CITATION,
            tag_whitelist=_WHITELIST,
            revision_hint="Empirie-Teil kürzen.",
        )
    )
    prompt = calls[0]["prompt"]
    assert prompt.startswith("Du extrahierst"), "Prompt beginnt nicht mehr mit dem statischen Kern"
    hint_pos = prompt.index("## Revision-Hinweis")
    assert hint_pos > prompt.index("## Bereits existierende Notes")
    assert hint_pos < prompt.index("## Textabschnitt:")


_TRUNCATED_OUTPUT = """<!--NOTE-->
title: Konzept Alpha
aliases: Alpha
tags: uni/ibi
proposed_tags:
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
# Konzept Alpha: Ein Kernsatz

Dieser Body bricht mitten im Satz ab und endet auf ein
<!--END-->
"""


def test_truncation_retry_shares_static_prefix(monkeypatch):
    """Der Trunkierungs-Retry stellt den Hinweis nicht mehr voran: Erst- und
    Retry-Prompt teilen den statischen Präfix (Cache-Read statt Neuanlage)."""
    calls: list[dict] = []
    _capture_prompts(monkeypatch, [_TRUNCATED_OUTPUT, ""], calls)

    asyncio.run(
        extractor.run_per_concept(
            _concept("Konzept Alpha"), "Text über Alpha.", {}, citation=_CITATION, tag_whitelist=_WHITELIST
        )
    )

    assert len(calls) == 2, "Trunkierungs-Retry hat nicht gefeuert — Fixture-Body endet nicht unvollständig?"
    retry_prompt = calls[1]["prompt"]
    assert retry_prompt.startswith("Du extrahierst")
    assert "## Trunkierungs-Hinweis" in retry_prompt
    assert retry_prompt.index("## Trunkierungs-Hinweis") < retry_prompt.index("## Textabschnitt:")
    common = os.path.commonprefix([calls[0]["prompt"], retry_prompt])
    assert _STATIC_TAIL_SENTINEL in common
