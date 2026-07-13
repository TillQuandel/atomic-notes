"""Regressionstest für den `--load-drafts`-Checkpoint-Roundtrip (#152).

`_save_draft_state` → `_load_draft_state` ist die intrikateste neue Logik des
RunContext-Refactors: Stage 1–5 werden beim Resume übersprungen, daher müssen
`citation` (via `_build_citation`, inkl. `physical_pages`-Zweit-Check #95/#96),
`q_title`, `word_count`, `dropped_total` und `extractor_failures` aus dem
persistierten State REKONSTRUIERT statt durchgereicht werden. Bis zu diesem
Test war das nur manuell verifiziert (siehe PR #227 Kontroll-Review) — dieser
Test fixiert das Verhalten Feld für Feld gegen die volle `RunContext`-
Dataclass (19 Felder, siehe `generative/schemas/run_context.py`).

Keine LLM-Calls, kein Netz-/Vault-Zugriff — reiner JSON-Roundtrip über
`tmp_path`. `pdf_chunker.pdf_uses_physical_pages` wird deterministisch
gestubbt (echtes PDF-Parsing ist nicht Gegenstand dieses Tests, siehe
`test_physical_pages.py`).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from generative import orchestrator
from generative.pipeline.pdf_chunker import Chunk
from generative.schemas.atomic_note import AtomicNoteDraft, ConceptItem, QualityReport, TextAnchor
from generative.schemas.citation import CitationMeta
from generative.schemas.run_context import RunContext

SOURCE_NAME = "Wilson - 2020 - Informationsverhalten und Kontext.pdf"
TEXT = "Menschen suchen und nutzen Information in einem sozialen Kontext ständig."


def _drafts() -> list[AtomicNoteDraft]:
    return [
        AtomicNoteDraft(
            title="Informationsverhalten: Kerncharakteristik",
            body="# Informationsverhalten: Kerncharakteristik\n\nErste Aussage (S. 3).",
            source_anchors=[TextAnchor(quote="Originalzitat A", page="S. 3", fuzzy_page=None)],
            related=["[[Anderes Konzept]]"],
            tags=["uni/ibi/konzept"],
            synthesis_confidence="high",
            aliases=["Info-Verhalten"],
            quality_flags=["⚠️ Testflag"],
            action="create",
            critic_score=4,
            hard_gates_pass=True,
        ),
        AtomicNoteDraft(
            title="Kontextsensitivität",
            body="# Kontextsensitivität\n\nZweite Aussage (S. 5).",
            source_anchors=[TextAnchor(quote="Zweites Zitat", page="S. 5")],
            related=[],
            tags=["uni/ibi/konzept"],
            synthesis_confidence="medium",
        ),
    ]


def _concept_map() -> dict:
    return {
        "Informationsverhalten: Kerncharakteristik": (
            ConceptItem(
                title="Informationsverhalten: Kerncharakteristik",
                priority="high",
                chapter="Kap. 2",
                action="create",
            ),
            "Kontext-Textauszug für die Extraktion.",
        ),
    }


def _quality_report() -> QualityReport:
    # Bewusst OHNE CrossRef-Override-Daten (crossref_title=None): die Override-
    # Blocklogik selbst ist bereits in test_citation_meta.py abgedeckt. Hier
    # geht es um die Rekonstruktion der Factory-Kette nach dem Reload, nicht um
    # die Override-Fallunterscheidung — pdf_meta bleibt daher unverfälscht die
    # Erwartung für author/year/title.
    return QualityReport(
        peer_reviewed=True,
        citation_count=12,
        retracted=False,
        flags=["⚠️ Testflag"],
    )


def _save_state(path: str) -> None:
    orchestrator._save_draft_state(
        path,
        drafts=_drafts(),
        concept_map=_concept_map(),
        existing_concepts={"Bestehendes Konzept": 1},
        concept_links={"Informationsverhalten: Kerncharakteristik": ["Kontextsensitivität", "Anderes Konzept"]},
        text=TEXT,
        chunks=[Chunk(title="Kapitel 2", text="[S. 3]\n\nText ...", index=0, page_start=3, page_end=5)],
        acronym_dict={"HIB": "Human Information Behavior"},
        quality_report=_quality_report(),
        pdf_meta={"Title": "Informationsverhalten und Kontext", "Author": "Wilson", "Year": "2020"},
        source_name=SOURCE_NAME,
        tag_whitelist=["uni/ibi/konzept", "uni/ibi/methode"],
        background_map={"Informationsverhalten: Kerncharakteristik": "Hintergrundtext"},
        filename_year="2020",
        related_mentions=["Kontextsensitivität", "Erwähntes Konzept"],
    )


def test_save_then_load_reconstructs_full_run_context(monkeypatch, tmp_path):
    path = str(tmp_path / "draft_state.json")

    # physical_pages (#95/#96 Zweit-Check) deterministisch stubben + Aufruf
    # protokollieren — kein echtes PDF nötig, aber der Aufruf-Pfad (source_path
    # aus dem geladenen State) muss stimmen.
    calls: list[Path] = []

    def _spy_physical_pages(source_path):
        calls.append(source_path)
        return True  # bewusst != CitationMeta-Default (False) — reine Passthrough-Prüfung

    monkeypatch.setattr(orchestrator.pdf_chunker, "pdf_uses_physical_pages", _spy_physical_pages)

    _save_state(path)
    ctx = orchestrator._load_draft_state(path)

    # -- Struktur: alle 19 RunContext-Felder vorhanden -----------------------
    assert isinstance(ctx, RunContext)
    field_names = {f.name for f in dataclasses.fields(RunContext)}
    assert len(field_names) == 19
    for name in field_names:
        assert hasattr(ctx, name), f"RunContext.{name} fehlt nach Reload"

    # -- Kern-Daten überleben den Roundtrip -----------------------------------
    assert [d.title for d in ctx.drafts] == [
        "Informationsverhalten: Kerncharakteristik",
        "Kontextsensitivität",
    ]
    assert all(isinstance(d, AtomicNoteDraft) for d in ctx.drafts)
    assert isinstance(ctx.drafts[0].source_anchors[0], TextAnchor)
    assert ctx.drafts[0].source_anchors[0].quote == "Originalzitat A"
    assert ctx.drafts[0].source_anchors[0].page == "S. 3"

    assert set(ctx.concept_map.keys()) == {"Informationsverhalten: Kerncharakteristik"}
    concept_item, ctext = ctx.concept_map["Informationsverhalten: Kerncharakteristik"]
    assert isinstance(concept_item, ConceptItem)
    assert concept_item.title == "Informationsverhalten: Kerncharakteristik"
    assert concept_item.priority == "high"
    assert ctext == "Kontext-Textauszug für die Extraktion."

    assert ctx.text == TEXT
    assert ctx.tag_whitelist == ["uni/ibi/konzept", "uni/ibi/methode"]

    # -- Typen/übrige direkt persistierte Felder ------------------------------
    assert ctx.existing_concepts == {"Bestehendes Konzept": 1}
    # concept_links: Liste -> set beim Reload (siehe _load_draft_state)
    assert ctx.concept_links == {
        "Informationsverhalten: Kerncharakteristik": {"Kontextsensitivität", "Anderes Konzept"}
    }
    assert isinstance(ctx.concept_links["Informationsverhalten: Kerncharakteristik"], set)
    assert len(ctx.chunks) == 1 and isinstance(ctx.chunks[0], Chunk)
    assert ctx.chunks[0].page_start == 3
    assert ctx.acronym_dict == {"HIB": "Human Information Behavior"}
    assert isinstance(ctx.quality_report, QualityReport)
    assert ctx.quality_report.flags == ["⚠️ Testflag"]
    assert ctx.pdf_meta == {"Title": "Informationsverhalten und Kontext", "Author": "Wilson", "Year": "2020"}
    assert ctx.source_path == Path(SOURCE_NAME)
    assert isinstance(ctx.source_path, Path)
    assert ctx.background_map == {"Informationsverhalten: Kerncharakteristik": "Hintergrundtext"}
    assert ctx.related_mentions == ["Kontextsensitivität", "Erwähntes Konzept"]

    # -- Rekonstruierte (NICHT persistierte) Felder — das eigentliche Ziel ---
    # fb_year: direkt aus dem gespeicherten filename_year (State-Feld), NICHT
    # zu verwechseln mit dem parse_filename_fallback-fb_year innerhalb der
    # citation-Factory (unten) — zwei unabhängige Ableitungen desselben Namens.
    assert ctx.fb_year == "2020"
    assert ctx.dropped_total == 0
    assert ctx.word_count == len(TEXT.split()) == 10
    # q_title laeuft im Reload-Pfad seit #234 durch den /Title-Trust-Cross-Check
    # (_quarantine_poisoned_embedded_title). Bei sauberem Input — kein
    # widersprechender InfoDictAuthor — bleibt der Titel unveraendert zitierfaehig.
    assert ctx.q_title == "Informationsverhalten und Kontext"  # aus pdf_meta["Title"]
    assert ctx.extractor_failures == []

    # citation: über dieselbe Factory wie der Normalpfad (_build_citation) neu
    # gebaut. Ohne CrossRef-Override-Daten entspricht sie 1:1 pdf_meta.
    assert isinstance(ctx.citation, CitationMeta)
    assert ctx.citation.author == "Wilson"
    assert ctx.citation.year == "2020"
    assert ctx.citation.title == "Informationsverhalten und Kontext"
    assert ctx.citation.source_file == SOURCE_NAME
    # physical_pages (#95/#96): via pdf_chunker.pdf_uses_physical_pages(source_path)
    # neu ermittelt (Zweit-Check, nicht persistiert) und durchgereicht.
    assert ctx.citation.physical_pages is True
    assert calls == [Path(SOURCE_NAME)]

    # -- Save-Datei selbst trägt die Kern-Daten (unabhängig von der Reload-Seite) --
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert [d["title"] for d in raw["drafts"]] == [
        "Informationsverhalten: Kerncharakteristik",
        "Kontextsensitivität",
    ]
    assert raw["tag_whitelist"] == ["uni/ibi/konzept", "uni/ibi/methode"]
    assert raw["text"] == TEXT


# --- #234: Reload-Pfad ist ein zweiter, unabhaengiger Trust-Punkt --------------
# Ein vor dem Fix gespeicherter (oder anderweitig vergifteter) Draft-State kann
# einen fremden Embedded-Titel + widersprechenden InfoDictAuthor tragen. Der
# --load-drafts-Pfad muss dieselbe /Title-Quarantaene anwenden wie der Normalpfad,
# damit der Gift-Titel nicht ueber den Resume in die Zitation zurueckkehrt.
POISON_SOURCE = "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf"
POISON_TITLE = "Conceptualisation and Measurement of Information Needs: A Literature Review"


def test_load_drafts_quarantines_poisoned_embedded_title(monkeypatch, tmp_path):
    path = str(tmp_path / "poisoned_state.json")
    monkeypatch.setattr(orchestrator.pdf_chunker, "pdf_uses_physical_pages", lambda _p: False)

    orchestrator._save_draft_state(
        path,
        drafts=_drafts(),
        concept_map=_concept_map(),
        existing_concepts={},
        concept_links={},
        text=TEXT,
        chunks=[Chunk(title="Kap. 2", text="[S. 3]\n\nText ...", index=0, page_start=3, page_end=5)],
        acronym_dict={},
        quality_report=_quality_report(),
        # Vergifteter pdf_meta: Fremd-Titel (Afzal) + widersprechender InfoDictAuthor.
        pdf_meta={"Title": POISON_TITLE, "Subject": "2017", "InfoDictAuthor": "Afzal"},
        source_name=POISON_SOURCE,
        tag_whitelist=[],
        background_map={},
        filename_year="2022",
        related_mentions=[],
    )
    ctx = orchestrator._load_draft_state(path)

    # Der Gift-Titel ist verworfen; q_title/Zitation nutzen den Dateiname-Titel.
    assert ctx.q_title != POISON_TITLE
    assert ctx.citation.title != POISON_TITLE
    assert ctx.pdf_meta.get("Title") in (None, "")
