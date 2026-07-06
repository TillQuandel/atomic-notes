# -*- coding: utf-8 -*-
import json
import re
import fitz
from rapidfuzz import fuzz

LIT = r"C:/Users/tillq/OneDrive/Dokumente/Literatur"
SOURCE_MAP = {
    "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf": "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf",
    "Knowles - From Pedagogy to Andragogy.pdf": "Knowles - From Pedagogy to Andragogy.pdf",
    "Mahmood und University of the Punjab - 2016 - Do People Overestimate Their Information Literacy Skills A Systematic Review of Empirical Evidence.pdf": "Mahmood und University of the Punjab - 2016 - Do People Overestimate Their Information Literacy Skills A Systematic Review of Empirical Evidence.pdf",
    "Merrill - First principles of instruction.pdf": "Merrill - First principles of instruction.pdf",
    "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf": "Sühl-Strohmenger - 2008 - Informationsvermittlung. Neugier, Zweifel, Lehren, Lernen ….pdf",
    "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf": "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
}


def page_chars(page):
    """Liste von (char, fitz.Rect) in Lesereihenfolge aus rawdict, OHNE synthetische Spaces."""
    raw = page.get_text("rawdict")
    out = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    out.append((ch["c"], fitz.Rect(ch["bbox"])))
    return out


def normalize_char(c):
    return c.lower()


def char_level_match(page, quote, min_ratio=85):
    chars = page_chars(page)
    if len(chars) < 10:
        return None
    page_str = "".join(normalize_char(c) for c, r in chars)
    q_str = re.sub(r"\s+", "", quote.lower())
    q_str = q_str.replace("--", "").replace("–", "").replace("—", "")
    if len(q_str) < 15:
        return None

    # 1) exakter Substring-Versuch (whitespace-frei)
    idx = page_str.find(q_str)
    if idx != -1:
        matched = chars[idx : idx + len(q_str)]
        return matched, 100.0

    # 2) Fuzzy-Alignment per Sliding Window (Schrittweite 1, Fensterbreite = len(q_str) +-10%)
    win = len(q_str)
    best_ratio, best_start = 0, None
    step = max(1, win // 20)
    for start in range(0, max(1, len(page_str) - win + 1), step):
        window = page_str[start : start + win]
        r = fuzz.ratio(q_str, window)
        if r > best_ratio:
            best_ratio, best_start = r, start
    if best_start is not None and best_ratio >= min_ratio:
        matched = chars[best_start : best_start + win]
        return matched, best_ratio
    return None


def span_fill_rects(matched_chars):
    """Gruppiert matched chars nach Zeile (y-Bucket) und liefert pro Zeile eine
    durchgehende Rect-Hülle (kein 'löchriges' Einzelwort-Highlight)."""
    if not matched_chars:
        return []
    lines = {}
    for c, r in matched_chars:
        key = round(r.y0, 1)
        lines.setdefault(key, []).append(r)
    rects = []
    for key, rs in lines.items():
        x0 = min(r.x0 for r in rs)
        x1 = max(r.x1 for r in rs)
        y0 = min(r.y0 for r in rs)
        y1 = max(r.y1 for r in rs)
        rects.append(fitz.Rect(x0, y0, x1, y1))
    return rects


def main():
    with open(r"C:/Users/tillq/.claude/jobs/d40fcd3a/tmp/quotes.json", encoding="utf-8") as f:
        content = f.read().split("\n\nTOTAL:")[0]
    quotes = json.loads(content)

    docs = {}
    results = []
    for q in quotes:
        src = q["source"]
        if src not in SOURCE_MAP:
            continue
        if src not in docs:
            docs[src] = fitz.open(f"{LIT}/{SOURCE_MAP[src]}")
        doc = docs[src]
        page_claim = q["page"]
        if page_claim is None:
            continue
        page_idx = int(page_claim) - 1
        candidates = [p for p in [page_idx - 1, page_idx, page_idx + 1] if 0 <= p < doc.page_count]

        found = None
        for p in candidates:
            m = char_level_match(doc[p], q["quote"])
            if m:
                rects = span_fill_rects(m[0])
                found = {"page": p + 1, "ratio": m[1], "n_line_rects": len(rects)}
                break
        results.append(
            {
                "note": q["note"],
                "source": src,
                "claimed_page": page_claim,
                "quote_preview": q["quote"][:50],
                "char_level": found,
            }
        )

    n = len(results)
    hits = [r for r in results if r["char_level"]]
    print(f"Gemessen: {n}")
    print(f"Char-Level-Treffer: {len(hits)} ({len(hits) / n * 100:.1f}%)")
    print(f"Weiterhin NICHT gefunden: {n - len(hits)} ({(n - len(hits)) / n * 100:.1f}%)")
    print()
    for r in results:
        status = (
            f"OK ratio={r['char_level']['ratio']:.0f} lines={r['char_level']['n_line_rects']}"
            if r["char_level"]
            else "MISS"
        )
        print(f"[{status:30s}] {r['source'][:35]:35s} S.{r['claimed_page']:>3} {r['quote_preview']}")


if __name__ == "__main__":
    main()
