"""Deterministische Alt-Text-Einbettung aus PDF-UA-getaggten PDFs (Pfad C).

Aus getaggten PDFs (StructTreeRoot + /Figure-Strukturelemente) den
menschengeschriebenen /Alt-Text ziehen und an genau eine create-Note binden
(exakter source_anchor-Seitenmatch, nie fuzzy). Untagged-PDFs liefern nichts.

Empirisch motiviert: nur tagged-PDFs speichern eine semantische Figur-Einheit;
untagged-Vektor-Extraktion ist deterministisch widerlegt (Spike A/C, siehe
docs/superpowers/specs/2026-06-04-figure-embedding-feasibility.md).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from generative.schemas.atomic_note import AtomicNoteDraft

_XREF_REF_RE = re.compile(r"^(\d+)\s+\d+\s+R$")
_NO_ANCHOR = -1  # anchor_page-Sentinel fuer geskippte (nicht-bindbare) Figuren


@dataclass(frozen=True)
class TaggedFigure:
    """Eine aus einem getaggten PDF extrahierte Figur (Alt-Text-Pfad)."""

    anchor_page: int  # source_anchor-Seitennummer (pdftotext-Space)
    alt_text: str
    label: Optional[str] = None  # Caption-Label, z.B. "Abbildung 2", falls vorhanden


@dataclass(frozen=True)
class SkippedFigure:
    figure: TaggedFigure
    reason: str  # "no_match" | "ambiguous" | "textless_page"


@dataclass
class BindReport:
    bound: list[tuple[TaggedFigure, str]] = field(default_factory=list)  # (figure, draft_title)
    skipped: list[SkippedFigure] = field(default_factory=list)
    untagged: bool = False  # #50/M11: PDF nicht PDF-UA-getaggt → Abbildungen-Skip melden


def _figure_bullet(fig: TaggedFigure) -> str:
    label = fig.label or "Abbildung"
    return f"- **{label}** (S. {fig.anchor_page}): {fig.alt_text}"


def _page_text_flags(pdf_path: Path, page_count: int) -> list[bool]:
    """Pro PyMuPDF-Seite: liefert pdftotext Text? (gleicher \\f-Split wie pdf_chunker).

    pdftotext emittiert ein \\f-Segment pro Seite in Dokumentreihenfolge -> i-tes
    Segment == PyMuPDF-Seite i. So bleibt die Ausrichtung exakt, auch wenn einzelne
    Seiten textlos sind (die verwirft pdf_to_pages, hier markieren wir sie als False).
    """
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        return [False] * page_count  # fail-closed: ohne Mapping nichts binden
    if result.returncode != 0:
        return [False] * page_count  # fail-closed (precision-first)
    segments = result.stdout.split("\f")
    flags = [bool(seg.strip()) for seg in segments[:page_count]]
    flags += [False] * (page_count - len(flags))
    return flags


def embed_alt_figures(pdf_path: Path, drafts: list[AtomicNoteDraft]) -> BindReport:
    """Entry-Point: extrahiert Alt-Text getaggter Figuren und bindet sie an Drafts.

    Untagged-PDFs -> leerer Report, keine Mutation (Gate). Komponiert die vier
    reinen/IO-Bausteine: extract -> pdftotext-Seiten-Flags -> anchor-page-Mapping
    -> exakte Bindung. Figuren auf textlosen Seiten (anchor-page None) werden als
    unbindbar geskippt (precision-first).
    """
    import fitz  # PyMuPDF

    raw = extract_tagged_figures(pdf_path)
    if not raw:
        # leeres raw = untagged ODER getaggt-ohne-Figuren → unterscheiden, damit
        # M11 nur bei wirklich untagged PDFs den Skip meldet (kein Falsch-Alarm).
        return BindReport(untagged=not is_tagged_pdf(pdf_path))

    with fitz.open(str(pdf_path)) as doc:
        page_count = doc.page_count
    has_text = _page_text_flags(pdf_path, page_count)
    from generative.pipeline.pdf_chunker import _pdf_page_labels

    page_labels = _pdf_page_labels(pdf_path)

    figures: list[TaggedFigure] = []
    report = BindReport()
    for page_index, alt in raw:
        anchor_page = pdf_index_to_anchor_page(has_text, page_index, page_labels)
        if anchor_page is None:
            # Figur auf textloser Seite -> kein source_anchor kann sie tragen.
            # anchor_page-Sentinel, da es keine gueltige pdftotext-Seite gibt.
            report.skipped.append(SkippedFigure(TaggedFigure(_NO_ANCHOR, alt), "textless_page"))
            continue
        figures.append(TaggedFigure(anchor_page=anchor_page, alt_text=alt))

    bind_report = bind_figures_to_drafts(figures, drafts)
    report.bound.extend(bind_report.bound)
    report.skipped.extend(bind_report.skipped)
    return report


def bind_figures_to_drafts(figures: list[TaggedFigure], drafts: list[AtomicNoteDraft]) -> BindReport:
    """Bindet Figuren an genau eine create-Note mit exaktem source_anchor-Seitenmatch.

    Precision-first: nur ``action == "create"``-Drafts, nur exakte ``page``-Treffer
    (nie ``fuzzy_page``). 0 Treffer -> "no_match", >1 Treffer -> "ambiguous"; beide
    werden geskippt und im Report vermerkt, nie geraten. Mutiert ``draft.body``:
    gebundene Figuren landen gruppiert unter genau einer ``## Abbildungen``-Sektion.
    """
    report = BindReport()
    per_draft: dict[int, list[TaggedFigure]] = {}

    for fig in figures:
        target_page = f"S. {fig.anchor_page}"
        matches = [d for d in drafts if d.action == "create" and any(a.page == target_page for a in d.source_anchors)]
        if len(matches) == 1:
            per_draft.setdefault(id(matches[0]), []).append(fig)
            report.bound.append((fig, matches[0].title))
        elif len(matches) == 0:
            report.skipped.append(SkippedFigure(fig, "no_match"))
        else:
            report.skipped.append(SkippedFigure(fig, "ambiguous"))

    for draft in drafts:
        figs = per_draft.get(id(draft))
        if not figs:
            continue
        bullets = "\n".join(_figure_bullet(f) for f in figs)
        draft.body = draft.body.rstrip() + "\n\n## Abbildungen\n\n" + bullets

    return report


def _sanitize_alt(text: str) -> str:
    """Normalisiert untrusted /Alt-Text fuer die Markdown-Einbettung.

    Collabiert Whitespace/Zeilenumbrueche und neutralisiert die zwei realen
    Injection-Vektoren in einem Body-Bullet: Obsidian-Wikilinks (``[[...]]``)
    und Tabellen-Pipes (``|``).
    """
    text = " ".join(text.split())
    text = text.replace("[[", r"\[\[").replace("]]", r"\]\]")
    text = text.replace("|", r"\|")
    return text


def is_tagged_pdf(pdf_path: Path) -> bool:
    """True wenn das PDF PDF-UA-getaggt ist (StructTreeRoot im Catalog)."""
    import fitz  # PyMuPDF

    try:
        with fitz.open(str(pdf_path)) as doc:
            return doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[0] == "xref"
    except Exception:
        return False


def extract_tagged_figures(pdf_path: Path) -> list[tuple[int, str]]:
    """Liest /Figure-Strukturelemente mit /Alt-Text aus einem getaggten PDF.

    Gibt ``[(pymupdf_page_index_0based, alt_text), ...]`` zurueck. Gate: ohne
    ``StructTreeRoot`` im Catalog -> ``[]`` (untagged PDF, keine semantischen
    Figuren). Liest Werte ueber ``xref_get_key`` (loest indirekte Refs auf,
    dekodiert Strings) statt ueber rohes Regex-Parsing. Figuren ohne aufloesbares
    /Alt oder /Pg werden uebersprungen (precision-first: skip statt raten).
    """
    import fitz  # PyMuPDF

    figures: list[tuple[int, str]] = []
    with fitz.open(str(pdf_path)) as doc:
        if doc.xref_get_key(doc.pdf_catalog(), "StructTreeRoot")[0] != "xref":
            return []
        page_index = {doc[i].xref: i for i in range(doc.page_count)}
        for xref in range(1, doc.xref_length()):
            if doc.xref_get_key(xref, "S") != ("name", "/Figure"):
                continue
            alt_type, alt_val = doc.xref_get_key(xref, "Alt")
            pg_type, pg_val = doc.xref_get_key(xref, "Pg")
            if alt_type != "string" or pg_type != "xref":
                continue
            m = _XREF_REF_RE.match(pg_val)
            if not m:
                continue
            pg_xref = int(m.group(1))
            if pg_xref not in page_index:
                continue
            alt = _sanitize_alt(alt_val)
            if alt:
                figures.append((page_index[pg_xref], alt))
    return figures


def pdf_index_to_anchor_page(has_text: list[bool], pymupdf_index: int, page_labels: list | None = None) -> int | None:
    """Bildet einen 0-basierten PyMuPDF-Seitenindex auf die source_anchor-Seitennummer ab.

    ``source_anchors`` tragen dieselbe Seitenzahl wie ``pdf_chunker.pdf_to_pages``:
    das numerische Druckseiten-Label aus ``/PageLabels`` (Buch: PyMuPDF-Seite 178 →
    „159"), falls vorhanden — sonst die 1-basierte Zählung ueber NUR die
    textfuehrenden Seiten (= altes pdftotext-Verhalten, PDFs ohne Labels unveraendert).
    ``has_text[i]`` sagt, ob PyMuPDF-Seite i pdftotext-Text liefert.

    Liegt die Figur selbst auf einer textlosen Seite, kann kein source_anchor diese
    Seite tragen -> ``None`` (nicht exakt bindbar, precision-first: skip).
    """
    if not has_text[pymupdf_index]:
        return None
    # Mit numerischem Druckseiten-Label: konsistent zu pdf_to_pages/source_anchors.
    if page_labels and pymupdf_index < len(page_labels):
        label = str(page_labels[pymupdf_index]).strip()
        if label.isdigit():
            return int(label)
    return sum(1 for flag in has_text[: pymupdf_index + 1] if flag)
