"""RunContext: gebündelter Rückgabe-Zustand der Extraction-Stages (0–5).

Vorher gab `_run_extraction_stages` ein 19-stelliges Positions-Tupel zurück, das
`main()` per Reihenfolge auspackte. Ein Positions-Versehen an dieser Grenze war
die Quelle mehrerer Wiring-Bugs (q_title-NameError, quality-Modul-Shadowing) —
ein falsch platzierter Wert fiel erst zur Laufzeit im nächsten Stage auf. Diese
frozen-Dataclass ersetzt Positionen durch benannte Felder: einmal gefüllt, alle
lesen per Attribut (analog zu `CitationMeta`/`RuntimeConfig` — „einmal bauen,
alle lesen"). Sowohl der Normalpfad (`_run_extraction_stages`) als auch der
`--load-drafts`-Pfad (`_load_draft_state`) liefern dieselbe Struktur. Siehe #152.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from generative.schemas.atomic_note import AtomicNoteDraft, QualityReport
    from generative.schemas.citation import CitationMeta


@dataclass(frozen=True)
class RunContext:
    """Kanonischer Zustand nach den Extraction-Stages — nach Konstruktion unveränderlich.

    Feldreihenfolge = die Reihenfolge des früheren 19er-Tupels (Nachvollziehbarkeit
    beim Umbau); für den Zugriff irrelevant, weil ausschließlich per Attribut gelesen.
    `drafts` wird im weiteren `main()`-Verlauf durch Dedup-/Stage-6-Stufen ersetzt —
    dort in eine lokale Variable gebunden, nicht in-place auf dem RunContext mutiert.
    """

    drafts: list[AtomicNoteDraft]
    concept_map: dict
    existing_concepts: dict
    concept_links: dict
    text: str
    chunks: list
    acronym_dict: dict
    quality_report: QualityReport
    pdf_meta: dict
    source_path: Path
    tag_whitelist: list
    background_map: dict
    fb_year: Optional[str]
    dropped_total: int
    word_count: int
    related_mentions: list[str]
    q_title: Optional[str]
    citation: CitationMeta
    extractor_failures: list[tuple[str, str]]
