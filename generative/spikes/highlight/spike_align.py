# -*- coding: utf-8 -*-
"""Integrations-Harness fuer den Alignment-Spike (Ansatz A).

Trennt die zwei Belange sauber, wie im Feature-Design beschlossen:

  1. PAGE-ORAKEL (simuliert den bestehenden Verifier): ein Char-Level-
     partial_ratio-Scan ueber alle Seiten liefert die wahre Fundseite +
     einen Praesenz-Score. Das ist NICHT das, was der Spike validiert --
     es steht fuer den Schritt, den `verify_citations.py` in der Pipeline
     schon leistet (Zitier-Seite != sauber highlightbare Seite; die
     `page`-Felder in quotes.json sind Notes-Zitierseiten, oft mit Offset).

  2. WORD-ALIGNMENT (der eigentliche Spike): aligner.locate() gegen den
     get_text("words")-Token-Strom GENAU der Orakel-Seite (harter Page-
     Constraint, window=0). Score- + Laengen-Ratio-Guardrail entscheiden
     highlight-or-not.

Stufe 0 column_boxes wird pro Orakel-Seite angewandt (no-op bei einspaltig);
auf der sauberen Prosa-Fundseite ist die native words-Reihenfolge korrekt,
die Figur-Fragmentierung trat nur auf der (falschen) Zitier-Seite auf.

Aufruf:  python spike_align.py [--min-score S] [--min-ratio R]
                               [--present T] [--verbose]
"""

import argparse
import json
import os
import re
import sys

import fitz
from rapidfuzz import fuzz

import aligner
from multi_column import column_boxes

HERE = os.path.dirname(os.path.abspath(__file__))
LIT = r"C:/Users/tillq/OneDrive/Dokumente/Literatur"

SOURCE_MAP = {
    "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf": "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf",
    "Knowles - From Pedagogy to Andragogy.pdf": "Knowles - From Pedagogy to Andragogy.pdf",
    "Mahmood und University of the Punjab - 2016 - Do People Overestimate Their Information Literacy Skills A Systematic Review of Empirical Evidence.pdf": "Mahmood und University of the Punjab - 2016 - Do People Overestimate Their Information Literacy Skills A Systematic Review of Empirical Evidence.pdf",
    "Merrill - First principles of instruction.pdf": "Merrill - First principles of instruction.pdf",
    "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf": "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf",
    "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf": "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
}


def load_quotes(path):
    with open(path, encoding="utf-8") as f:
        content = f.read().split("\n\nTOTAL:")[0]
    return json.loads(content)


# --- Page-Orakel (Verifier-Simulation) -------------------------------------


def _page_charstream(page):
    raw = page.get_text("rawdict")
    chars = []
    for b in raw["blocks"]:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    chars.append(ch["c"])
    return re.sub(r"\s+", "", aligner.normalize_text("".join(chars))).replace("-", "")


def find_page(doc, quote):
    """Char-Level-Scan ueber alle Seiten -> (best_page_1based, char_score)."""
    q = re.sub(r"\s+", "", aligner.normalize_text(aligner.strip_editorial_brackets(quote)))
    q = q.replace("-", "").replace("–", "").replace("—", "").replace("…", "")
    best_score, best_page = 0.0, None
    for p in range(doc.page_count):
        cs = _page_charstream(doc[p])
        if len(cs) < 10:
            continue
        sc = fuzz.partial_ratio(q, cs)
        if sc > best_score:
            best_score, best_page = sc, p + 1
    return best_page, best_score


# --- Word-Alignment (der Spike selbst) -------------------------------------


def page_tokens(page):
    """Stufe 0 column_boxes + native words-Reihenfolge je Spalte."""
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,wno)
    cols = column_boxes(page, footer_margin=0, header_margin=0, no_image_text=True)
    if len(cols) <= 1:
        return [(w[4], (w[0], w[1], w[2], w[3])) for w in words]
    buckets = [[] for _ in cols]
    leftover = []
    for w in words:
        pt = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        for ci, rect in enumerate(cols):
            if pt in rect:
                buckets[ci].append(w)
                break
        else:
            leftover.append(w)
    ordered = []
    for bucket in buckets:
        ordered.extend(bucket)
    ordered.extend(leftover)
    return [(w[4], (w[0], w[1], w[2], w[3])) for w in ordered]


def _norm_nospace(s):
    return re.sub(r"\s+", "", aligner.normalize_text(aligner.strip_editorial_brackets(s))).replace("-", "")


def geometry_matches(page, rects, quote, min_sim=95):
    """ORAKEL-UNABHAENGIGE Prüfung: reproduziert der Text UNTER den
    zurueckgegebenen Bboxes das Zitat? get_textbox extrahiert direkt aus der
    Seiten-Geometrie, nicht aus dem partial_ratio-Alignment -> entkoppelt die
    Erfolgsmetrik vom find_page/locate-Score (Cross-Review-Fund Zirkularitaet).
    """
    covered = "".join(page.get_textbox(fitz.Rect(r)) for r in rects)
    return fuzz.partial_ratio(_norm_nospace(quote), _norm_nospace(covered)) >= min_sim


def localize(doc, page_1based, quote, min_score, min_ratio):
    """Gibt einen Hit NUR zurueck, wenn (1) das Alignment eine Stelle findet UND
    (2) der Text unter den Bboxes das Zitat reproduziert (Geometrie-Beweis)."""
    page = doc[page_1based - 1]
    hit = aligner.locate(quote, page_tokens(page), min_score=min_score, min_len_ratio=min_ratio)
    if hit and not geometry_matches(page, hit["rects"], quote):
        hit = {**hit, "geometry_ok": False}
        return None  # Alignment-Score hoch, aber Bbox deckt falschen Text -> verwerfen
    return hit


# --- Harness ----------------------------------------------------------------


def evaluate(quotes, docs, args):
    rows = []
    for q in quotes:
        src = q["source"]
        if src not in SOURCE_MAP:
            continue
        if src not in docs:
            docs[src] = fitz.open(f"{LIT}/{SOURCE_MAP[src]}")
        doc = docs[src]
        page, char_score = find_page(doc, q["quote"])
        present = char_score >= args.present
        hit = None
        if page is not None:
            hit = localize(doc, page, q["quote"], args.min_score, args.min_ratio)
        rows.append(
            {
                "source": src[:26],
                "cite": q["page"],
                "page": page,
                "char_score": char_score,
                "present": present,
                "hit": hit,
                "quote": q["quote"][:52],
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=96.0)
    ap.add_argument("--min-ratio", type=float, default=0.9)
    ap.add_argument(
        "--present", type=float, default=97.0, help="Char-Score-Schwelle, ab der ein Zitat als verbatim-praesent gilt"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    docs = {}
    real = evaluate(load_quotes(os.path.join(HERE, "quotes.json")), docs, args)

    near_path = os.path.join(HERE, "near_miss.json")
    near = evaluate(load_quotes(near_path), docs, args) if os.path.exists(near_path) else []

    present = [r for r in real if r["present"]]
    absent = [r for r in real if not r["present"]]
    gaps = [r for r in present if not r["hit"]]

    print(f"=== Alignment-Spike (min_score={args.min_score} min_ratio={args.min_ratio} present>={args.present}) ===")
    print(
        f"Zitate gesamt: {len(real)}  |  verbatim-praesent: {len(present)}  "
        f"|  nicht-verbatim (upstream-Sidecar): {len(absent)}"
    )
    print(f"\nHIGHLIGHT-GEOMETRIE auf praesenten Zitaten:")
    print(f"  lokalisiert: {len(present) - len(gaps)}/{len(present)}")
    if present:
        print(f"  ECHTE LUECKE: {len(gaps)}/{len(present)} = {len(gaps) / len(present) * 100:.1f}%")

    if near:
        false_hits = [r for r in near if r["hit"]]
        print(f"\nNEAR-MISS (Falsch-Treffer-Test): {len(near)} Zitate")
        print(f"  vom Orakel als nicht-praesent verworfen: {sum(1 for r in near if not r['present'])}")
        print(f"  FALSCH-TREFFER (Alignment highlightet Falsches): {len(false_hits)}")
        for r in false_hits:
            h = r["hit"]
            print(
                f"    !! {r['source']} orakel-S.{r['page']} char={r['char_score']:.0f} "
                f"score={h['score']:.0f} ratio={h['len_ratio']:.2f} -- {r['quote']}"
            )

    print(f"\n--- Echte Luecken (praesent, aber nicht lokalisiert) ---")
    for r in gaps:
        print(f"  GAP char={r['char_score']:.0f} S.{r['page']} {r['source']} -- {r['quote']}")

    print(f"\n--- Nicht-verbatim (kein Highlight, ehrlich in Sidecar) ---")
    for r in absent:
        print(f"  char={r['char_score']:.0f} cite=S.{r['cite']} {r['source']} -- {r['quote']}")

    if args.verbose:
        print(f"\n--- Alle praesenten Treffer ---")
        for r in present:
            if r["hit"]:
                h = r["hit"]
                print(
                    f"  OK S.{r['page']:>3} score={h['score']:.0f} ratio={h['len_ratio']:.2f} "
                    f"words={len(h['word_indices'])} char={r['char_score']:.0f} -- {r['quote']}"
                )


if __name__ == "__main__":
    sys.exit(main())
