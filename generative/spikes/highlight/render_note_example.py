# -*- coding: utf-8 -*-
"""Highlightet ALLE Zitate EINER atomic Note im Quell-PDF (Spike-Beleg).

Per-Zeile gemergte Rects (durchgehende Hülle statt boxiger Einzelwörter),
Provenance-Info pro Annotation, Original nie mutiert -> neue Datei + Seiten-Renders.

Aufruf: python render_note_example.py "<Note-Dateiname.md>" [out-basename]
"""

import os
import sys
from collections import defaultdict

import fitz

from spike_align import LIT, SOURCE_MAP, find_page, load_quotes, localize

OUT = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = "C:/tmp/render"


def merge_line_rects(rects):
    """Gruppiert Wort-Rects nach Zeile (y0-Bucket) -> eine Hülle pro Zeile."""
    lines = defaultdict(list)
    for r in rects:
        lines[round(fitz.Rect(r).y0, 0)].append(fitz.Rect(r))
    merged = []
    for _, rs in sorted(lines.items()):
        merged.append(
            fitz.Rect(min(r.x0 for r in rs), min(r.y0 for r in rs), max(r.x1 for r in rs), max(r.y1 for r in rs))
        )
    return merged


def main():
    note = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "note_example"
    quotes = [q for q in load_quotes(os.path.join(OUT, "quotes.json")) if q["note"] == note]
    if not quotes:
        sys.exit(f"Keine Zitate fuer Note {note!r} im Testset.")
    src = quotes[0]["source"]
    doc = fitz.open(f"{LIT}/{SOURCE_MAP[src]}")
    print(f"Note: {note}\nQuelle: {src}\nZitate: {len(quotes)}")

    touched = {}
    for q in quotes:
        page_no, char_score = find_page(doc, q["quote"])
        hit = localize(doc, page_no, q["quote"], 96.0, 0.9) if char_score >= 97 else None
        if not hit:
            print(f"  [Sidecar] cite=S.{q['page']} char={char_score:.0f} -- {q['quote'][:55]}")
            continue
        page = doc[page_no - 1]
        annot = page.add_highlight_annot(merge_line_rects(hit["rects"]))
        annot.set_info(title="atomic-notes", content=f"{note} :: {q['quote'][:60]}")
        annot.update()
        touched[page_no] = touched.get(page_no, 0) + 1
        print(
            f"  [Highlight] cite=S.{q['page']} -> gefunden PDF-S.{page_no} "
            f"score={hit['score']:.0f} -- {q['quote'][:55]}"
        )

    out_pdf = os.path.join(RENDER_DIR, f"{base}.pdf")
    doc.save(out_pdf, garbage=3, deflate=True)
    print(f"\nPDF: {out_pdf}  (Highlights auf Seiten {sorted(touched)})")
    for pno in sorted(touched):
        doc[pno - 1].get_pixmap(dpi=130).save(f"{RENDER_DIR}/{base}_p{pno}.png")
        print(f"  render: {base}_p{pno}.png")
    doc.close()


if __name__ == "__main__":
    main()
