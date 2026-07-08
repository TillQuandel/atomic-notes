"""CitationMeta: Single Source of Truth für Zitations-Metadaten (Autor/Jahr/Titel/DOI).

Vorher gab es ZWEI parallele dict-Welten: `pdf_meta` (Extractor/Planner, OHNE
CrossRef-Korrekturen) vs. `enriched_meta` (nur Vault-Writer, MIT CrossRef-
Korrekturen aus `QualityReport`). `build_citation_meta` konstruiert EINMAL pro
Lauf ein `CitationMeta`-Objekt — danach frozen, alle Konsumenten (Extractor,
Planner, Vault-Writer) lesen aus demselben Objekt. Siehe Issue #96, Etappe E3a.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CitationMeta:
    """Kanonische Zitations-Metadaten für eine Quelle — nach Konstruktion unveränderlich."""

    author: Optional[str]
    year: Optional[str]
    title: Optional[str]
    doi: Optional[str]
    source_file: str
    # Additiv (#95, kein SCHEMA_VERSION-Bump): True wenn die Quelle KEINE
    # eingebetteten `/PageLabels` führt und `pdf_chunker.pdf_to_pages` deshalb auf
    # den `i+1`-Fallback (physische PDF-Position statt gedruckter Seite)
    # zurückfällt. Der Renderer (vault_writer/portable_md) kennzeichnet
    # Seitenangaben dann sichtbar als „PDF-S." statt unmarkiert als „S." — das
    # pipeline-interne `(S. N)`-Inline-Format im Draft-Body bleibt davon
    # unberührt (nur eine Render-Kennzeichnung, kein Format-Refactor).
    physical_pages: bool = False

    @property
    def display_year(self) -> str:
        """Jahr für Anzeige. `"[o. J.]"` (deutsche Zitierkonvention „ohne Jahr")
        statt stillem Verschwinden, wenn keine Quelle ein Jahr belegt — nie ein
        Jahr erfinden."""
        return self.year if self.year else "[o. J.]"

    @property
    def short_label(self) -> str:
        """`<Surname> <Jahr>` bzw. `<Surname> [o. J.]` für Footnote-Defs/Quellen-
        Block. Multi-Autor auf erste Surname + „et al." verkürzt. Fällt auf den
        Dateiname-Stem zurück, wenn gar kein Autor aufgelöst werden konnte."""
        author = (self.author or "").strip()
        if not author:
            return Path(self.source_file).stem
        parts = [p.strip() for p in re.split(r"\s*;\s*|\s+(?:und|and)\s+", author) if p.strip()]
        surname = parts[0].split(",", 1)[0].strip() if "," in parts[0] else parts[0].split()[-1]
        if len(parts) > 1:
            surname = f"{surname} et al."
        return f"{surname} {self.display_year}"

    def as_meta_dict(self) -> dict[str, str]:
        """Legacy-Brücke für Übergangsstellen, die noch Capitalized-Key-Dicts
        erwarten (z.B. `routing_report.is_source_unresolved`)."""
        d: dict[str, str] = {}
        if self.author:
            d["Author"] = self.author
        if self.year:
            d["Year"] = self.year
        if self.title:
            d["Title"] = self.title
        return d


def crossref_override_blocked(quality_report, q_title: str | None) -> bool:
    """True wenn ein per Title-RATEN gefundener CrossRef-Treffer (kein harter
    ID-Match) verworfen werden muss, weil sein Titel nicht zum erwarteten Titel
    passt (schwacher Match) — sonst verfälscht ein Fehltreffer Quelle, Autor,
    Jahr und alle Footnotes der Note (gleiche Klasse wie das OpenAlex-Title-Gate).

    Exakt die bisherige `_block_crossref_override`-Bedingung aus orchestrator.py
    (F2) — separat aufrufbar, weil main() dieselbe Bedingung für das Fail-
    Closed-Routing (`is_source_unresolved`) braucht. Analog zum bestehenden
    Muster, `parse_filename_fallback` deterministisch mehrfach aufzurufen statt
    das Ergebnis durchzureichen.
    """
    from generative.tools.pdf_enrich import _title_match_confident

    return bool(
        quality_report.doi_from_title_match
        and quality_report.crossref_title
        and not _title_match_confident(q_title or "", quality_report.crossref_title)
    )


def build_citation_meta(
    pdf_meta: dict,
    quality_report,
    q_title: str | None,
    source_file: str,
    physical_pages: bool = False,
) -> CitationMeta:
    """Konstruiert die kanonische CitationMeta EINMAL pro Lauf.

    Übernimmt exakt die CrossRef-Override-Blocklogik, die vorher in
    orchestrator.py (`enriched_meta`, F2) nur für den Vault-Writer galt:
    Title/Author/Year aus `pdf_meta`, überschrieben von CrossRef-Werten aus
    `quality_report` — außer der Override ist geblockt (schwacher Titel-Match)
    oder (nur beim Jahr) ein Filename-Jahr hat bereits Vorrang (v28-Regel).

    Keine Logik-Änderung ggü. dem bisherigen Verhalten — nur EIN Konstruktions-
    punkt statt der beiden getrennten Welten `pdf_meta`/`enriched_meta`. Dadurch
    sehen Extractor und Planner ab jetzt dieselben CrossRef-korrigierten Werte
    wie vorher nur der Vault-Writer (dokumentierte Verhaltensänderung, #96 E3a).

    `physical_pages` (#95): vom Aufrufer durchgereichtes Signal aus
    `pdf_chunker.pdf_uses_physical_pages(source_path)` — hier nicht selbst
    ermittelt, weil `build_citation_meta` `source_path` (die Datei) nicht kennt,
    nur `source_file` (den Namen für Anzeigezwecke).
    """
    from generative.pipeline.vault_writer import parse_filename_fallback

    pdf_meta = pdf_meta or {}
    fb = parse_filename_fallback(source_file)
    fb_year = fb.get("Year")

    author = pdf_meta.get("Author")
    # fb_year-Fallback: im Pipeline-Fluss redundant (apply_filename_citation_metadata
    # setzt das Filename-Jahr vorher autoritativ in pdf_meta), aber bei direkter
    # Factory-Nutzung repliziert er die alte _short_label-Doppel-Absicherung.
    year = pdf_meta.get("Year") or fb_year
    title = pdf_meta.get("Title")

    if not crossref_override_blocked(quality_report, q_title):
        if quality_report.crossref_title:
            title = quality_report.crossref_title
        if quality_report.crossref_author:
            author = quality_report.crossref_author
        if quality_report.crossref_year and not fb_year:
            # Filename-Year hat Vorrang (v28): CrossRef darf nur überschreiben wenn Filename kein Jahr hat
            year = quality_report.crossref_year

    doi = pdf_meta.get("DOI") or pdf_meta.get("doi")

    return CitationMeta(
        author=author, year=year, title=title, doi=doi, source_file=source_file, physical_pages=physical_pages
    )
