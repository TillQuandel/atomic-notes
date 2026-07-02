# -*- coding: utf-8 -*-
"""Erzeugt ein highlightetes Beleg-PDF (Original bleibt unberuehrt, neue Datei)
und rendert die betroffenen Seiten als PNG zur visuellen Kontrolle.

Nur Spike-Beleg -- KEIN Pipeline-Code. Aufruf: python render_highlight.py <source-key>
"""
import os
import sys

import fitz

from spike_align import (LIT, SOURCE_MAP, find_page, load_quotes, localize)

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "Knowles"
    src = next(s for s in SOURCE_MAP if key in s)
    doc = fitz.open(f"{LIT}/{SOURCE_MAP[src]}")

    quotes = [q for q in load_quotes(os.path.join(OUT, "quotes.json"))
              if q["source"] == src]
    touched = set()
    n_hl = 0
    for q in quotes:
        page_no, char_score = find_page(doc, q["quote"])
        if char_score < 97:
            continue
        hit = localize(doc, page_no, q["quote"], 96.0, 0.9)
        if not hit:
            continue
        page = doc[page_no - 1]
        quads = [fitz.Rect(r).quad for r in hit["rects"]]
        annot = page.add_highlight_annot(quads)
        annot.set_info(title="atomic-notes", content="atomic-notes highlight (spike)")
        annot.update()
        touched.add(page_no - 1)
        n_hl += 1

    out_pdf = os.path.join(OUT, f"_highlighted_{key}.pdf")
    doc.save(out_pdf, garbage=3, deflate=True)
    print(f"{n_hl} Highlights auf {len(touched)} Seiten -> {out_pdf}")

    # betroffene Seiten als PNG rendern
    for pno in sorted(touched)[:3]:
        pix = doc[pno].get_pixmap(dpi=130)
        png = os.path.join(OUT, f"_hl_{key}_p{pno+1}.png")
        pix.save(png)
        print(f"  render: {png}")
    doc.close()


if __name__ == "__main__":
    main()
