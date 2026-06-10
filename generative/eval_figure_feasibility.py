"""Deterministische Feasibility-Probe fuer Figur-Einbettung Variante A.

Kein LLM-Call, kein Pipeline-Kern-Eingriff. Das Modul prueft nur, ob ein PDF
mechanische Figur-Signale liefert:
  - nummerierte Figure-/Abbildungs-Captions im pdftotext-Seitentext
  - Raster-Bildsignale via PyMuPDF
  - Vektor-/Drawing-Signale via PyMuPDF
  - Seitennaehe zwischen Caption und vorhandenen Chunk.page_start/page_end

Tabellen bleiben absichtlich ausserhalb dieser Lane.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_FIGURE_CAPTION_RE = re.compile(
    r"^\s*((?:Abbildung|Abb\.|Figure|Fig\.)\s*\d+[A-Za-z]?)\s*[:.\-]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:Tabelle|Table)\s+\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class FigureCaption:
    page: int
    label: str
    text: str
    line_number: int


@dataclass(frozen=True)
class CaptionChunkMatch:
    chunk_title: str
    chunk_index: int
    reason: str = "page_range"


@dataclass(frozen=True)
class PageVisualSignals:
    page: int
    raster_images: int
    vector_drawings: int
    captions: list[FigureCaption] = field(default_factory=list)


def find_figure_captions(page_text: str, page: int) -> list[FigureCaption]:
    """Findet nummerierte Figure-/Abbildungs-Captions in einem Seitentext.

    Bewusst konservativ: Nur zeilenanfangs stehende, nummerierte Captions zaehlen.
    Tabellen-Captions und Tabellenzeilen, die eine Abbildung nur referenzieren,
    werden ignoriert.
    """
    captions: list[FigureCaption] = []
    for line_number, line in enumerate(page_text.splitlines(), start=1):
        stripped = " ".join(line.split())
        if not stripped or _TABLE_CAPTION_RE.match(stripped):
            continue
        match = _FIGURE_CAPTION_RE.match(stripped)
        if not match:
            continue
        captions.append(FigureCaption(
            page=page,
            label=" ".join(match.group(1).split()),
            text=stripped,
            line_number=line_number,
        ))
    return captions


def match_caption_to_chunk(caption: FigureCaption, chunks: list[Any]) -> CaptionChunkMatch | None:
    """Ordnet eine Caption per Seitennaehe dem ersten passenden Chunk zu."""
    for chunk in chunks:
        page_start = getattr(chunk, "page_start", None)
        page_end = getattr(chunk, "page_end", None)
        if page_start is None or page_end is None:
            continue
        if page_start <= caption.page <= page_end:
            return CaptionChunkMatch(
                chunk_title=getattr(chunk, "title", ""),
                chunk_index=getattr(chunk, "index", -1),
            )
    return None


def classify_page_signals(signals: PageVisualSignals) -> str:
    """Klassifiziert eine Seite nach Variante-A-Tauglichkeit.

    Nur die zwei verifiziert relevanten Signale entscheiden: nummerierte
    Caption (chunk-ankerbar via Seiten-Range) und Raster-Bild (einbettbares
    Asset). ``vector_drawings`` fliesst bewusst NICHT ein -- ``get_drawings()``
    zaehlt Layout-Linien/Rahmen/Bullets auf praktisch jeder Seite und ist als
    Figur-Diskriminator wertlos (empirisch: >0 auf 100 % der Seiten). Der
    Rohwert bleibt reine Diagnostik im Report.
    """
    has_caption = bool(signals.captions)
    has_raster = signals.raster_images > 0
    if has_caption and has_raster:
        return "captioned_raster"
    if has_caption:
        return "captioned_no_raster"
    if has_raster:
        return "raster_uncaptioned"
    return "no_signal"


def warning_for_classification(classification: str) -> str | None:
    """Report-Warnung fuer Seiten ohne handlungsrelevantes Figur-Signal.

    Bewusst NICHT "no visual signal": ``vector_drawings`` (Layout-Rohsignal)
    kann >0 sein und wird im selben Report weiter ausgegeben. ``no_signal``
    heisst praezise "keine Caption und kein Raster".
    """
    if classification == "no_signal":
        return "no caption or raster signal"
    return None


def chunk_summary_rows(chunks: list[Any]) -> list[dict[str, Any]]:
    """Report-Metadaten fuer Chunks, ohne Chunk-Volltext."""
    return [
        {
            "title": getattr(chunk, "title", ""),
            "index": getattr(chunk, "index", -1),
            "page_start": getattr(chunk, "page_start", None),
            "page_end": getattr(chunk, "page_end", None),
        }
        for chunk in chunks
    ]


def _count_raster_image_placements(page: Any) -> int:
    """Zaehlt Bild-Platzierungen, nicht nur PDF-XObject-Ressourcen."""
    count = 0
    for image in page.get_images(full=True):
        xref = image[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        count += len(rects) if rects else 1
    return count


def _page_visual_signals(pdf_path: Path, pages: list[tuple[int, str]]) -> list[PageVisualSignals]:
    """Duenne PyMuPDF-Abstraktion fuer echte PDF-Signale."""
    import fitz  # PyMuPDF

    signals: list[PageVisualSignals] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_number, page_text in pages:
            if page_number < 1 or page_number > len(doc):
                continue
            page = doc[page_number - 1]
            raster_images = _count_raster_image_placements(page)
            vector_drawings = len(page.get_drawings())
            signals.append(PageVisualSignals(
                page=page_number,
                raster_images=raster_images,
                vector_drawings=vector_drawings,
                captions=find_figure_captions(page_text, page_number),
            ))
    return signals


def analyze_pdf(pdf_path: Path) -> dict[str, Any]:
    """Analysiert ein PDF fuer Figur-Feasibility und gibt ein JSON-faehiges Dict."""
    from generative.pipeline import pdf_chunker

    pages = pdf_chunker.pdf_to_pages(pdf_path)
    text = pdf_chunker.pdf_to_text(pdf_path)
    chunks = pdf_chunker.split_by_chapters(text)
    page_signals = _page_visual_signals(pdf_path, pages)

    captions = [caption for signal in page_signals for caption in signal.captions]
    caption_rows = []
    unmatched_captions = []
    for caption in captions:
        match = match_caption_to_chunk(caption, chunks)
        row = {
            "page": caption.page,
            "label": caption.label,
            "text": caption.text,
            "chunk": dataclasses.asdict(match) if match else None,
        }
        caption_rows.append(row)
        if match is None:
            unmatched_captions.append(dataclasses.asdict(caption))

    pages_report = []
    for signal in page_signals:
        classification = classify_page_signals(signal)
        pages_report.append({
            "page": signal.page,
            "raster_images": signal.raster_images,
            "vector_drawings": signal.vector_drawings,
            "captions": [dataclasses.asdict(c) for c in signal.captions],
            "classification": classification,
            "warning": warning_for_classification(classification),
        })

    return {
        "pdf": str(pdf_path),
        "pages": len(pages),
        "chunks": chunk_summary_rows(chunks),
        "captions": caption_rows,
        "unmatched_captions": unmatched_captions,
        "raster_images": sum(s.raster_images for s in page_signals),
        "vector_signals": sum(s.vector_drawings for s in page_signals),
        "page_signals": pages_report,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Figure Feasibility: {Path(report['pdf']).name}",
        "",
        f"- Pages: {report['pages']}",
        f"- Captions: {len(report['captions'])}",
        f"- Unmatched captions: {len(report['unmatched_captions'])}",
        f"- Raster image placements: {report['raster_images']}",
        f"- Vector drawing signals: {report['vector_signals']}",
        "",
        "| Page | Class | Raster | Vector | Captions |",
        "|---:|---|---:|---:|---|",
    ]
    for row in report["page_signals"]:
        captions = "; ".join(c["label"] for c in row["captions"])
        lines.append(
            f"| {row['page']} | {row['classification']} | "
            f"{row['raster_images']} | {row['vector_drawings']} | {captions} |"
        )
    return "\n".join(lines)


def report_has_no_hardcoded_literature_paths() -> bool:
    """Regression-Guard: keine lokalen Literaturpfade als Defaults im Tool."""
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "One" + "Drive",
        "Dokumente" + "/" + "Literatur",
        "Dokumente" + "\\" + "Literatur",
    )
    return not any(item in source for item in forbidden)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-freie Figur-Feasibility-Probe")
    parser.add_argument("pdf", type=Path, nargs="+", help="PDF-Datei(en) fuer die Probe")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args(argv)

    reports = []
    for pdf in args.pdf:
        if not pdf.exists():
            parser.error(f"PDF nicht gefunden: {pdf}")
        reports.append(analyze_pdf(pdf))

    if args.format == "markdown":
        print("\n\n".join(render_markdown(report) for report in reports))
    else:
        print(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
