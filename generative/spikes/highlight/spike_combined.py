# -*- coding: utf-8 -*-
import json, re
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

def strip_editorial_brackets(q):
    # "[T]his" -> "This"; "[asynchronous discussions]" -> "asynchronous discussions"
    return re.sub(r"\[([^\]]*)\]", r"\1", q)

def try_search_for(page, quote):
    flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_INHIBIT_SPACES
    q = re.sub(r"\s+", " ", strip_editorial_brackets(quote)).strip()
    return page.search_for(q, flags=flags)

def sig_words(s):
    return [w.lower() for w in re.findall(r"[A-Za-zÄÖÜäöüß']{3,}", s)]

def try_wordlist_sequence(page, quote, min_ratio=80, min_coverage=0.7):
    words = page.get_text("words")
    page_words = [(w[4], fitz.Rect(w[0], w[1], w[2], w[3])) for w in words]
    q_sig = sig_words(strip_editorial_brackets(quote))
    if len(q_sig) < 3:
        return None
    pw_idx, matched = 0, 0
    for qw in q_sig:
        limit = min(len(page_words), pw_idx + 400)
        for j in range(pw_idx, limit):
            pw_text = re.sub(r"[^\w]", "", page_words[j][0]).lower()
            if pw_text and fuzz.ratio(qw, pw_text) >= min_ratio:
                pw_idx = j + 1
                matched += 1
                break
    cov = matched / len(q_sig)
    return cov if cov >= min_coverage else None

def page_chars(page):
    raw = page.get_text("rawdict")
    out = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    out.append(ch["c"].lower())
    return out

def try_char_level(page, quote, min_score=82):
    chars = page_chars(page)
    if len(chars) < 10:
        return None
    page_str = "".join(chars)
    q_str = re.sub(r"\s+", "", strip_editorial_brackets(quote).lower())
    q_str = q_str.replace("--", "").replace("–", "").replace("—", "").replace("…", "")
    if len(q_str) < 15:
        return None
    res = fuzz.partial_ratio_alignment(q_str, page_str)
    if res.score >= min_score:
        return res.score
    return None

def main():
    with open(r"C:/Users/tillq/.claude/jobs/d40fcd3a/tmp/quotes.json", encoding="utf-8") as f:
        content = f.read().split("\n\nTOTAL:")[0]
    quotes = json.loads(content)

    docs, results = {}, []
    for q in quotes:
        src = q["source"]
        if src not in SOURCE_MAP:
            continue
        if src not in docs:
            docs[src] = fitz.open(f"{LIT}/{SOURCE_MAP[src]}")
        doc = docs[src]
        if q["page"] is None:
            continue
        page_idx = int(q["page"]) - 1
        candidates = [p for p in [page_idx - 1, page_idx, page_idx + 1] if 0 <= p < doc.page_count]

        methods_hit = []
        for p in candidates:
            page = doc[p]
            if try_search_for(page, q["quote"]):
                methods_hit.append("search_for")
                break
        if not methods_hit:
            for p in candidates:
                if try_wordlist_sequence(doc[p], q["quote"]):
                    methods_hit.append("wordlist")
                    break
        if not methods_hit:
            for p in candidates:
                if try_char_level(doc[p], q["quote"]):
                    methods_hit.append("char_level")
                    break
        results.append({"note": q["note"], "source": src[:30], "page": q["page"],
                         "quote": q["quote"][:45], "hit": methods_hit[0] if methods_hit else "MISS"})

    n = len(results)
    from collections import Counter
    c = Counter(r["hit"] for r in results)
    print(f"Gemessen: {n}")
    for k, v in c.items():
        print(f"  {k}: {v} ({v/n*100:.1f}%)")
    miss = c.get("MISS", 0)
    print(f"\nGesamt lokalisierbar: {n-miss}/{n} = {(n-miss)/n*100:.1f}%")
    print(f"Gap-Rate: {miss/n*100:.1f}%\n")
    for r in results:
        if r["hit"] == "MISS":
            print(f"MISS: {r['source']} S.{r['page']} -- {r['quote']}")

if __name__ == "__main__":
    main()
