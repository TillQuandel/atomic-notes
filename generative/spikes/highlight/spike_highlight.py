# -*- coding: utf-8 -*-
import json, re, sys
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

def normalize(s):
    s = s.replace("--", " ").replace("–", " ").replace("—", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sig_words(s):
    words = re.findall(r"[A-Za-zÄÖÜäöüß']{3,}", s)
    return [w.lower() for w in words]

def try_search_for(page, quote):
    flags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_INHIBIT_SPACES
    hits = page.search_for(normalize(quote), flags=flags)
    return hits

def try_wordlist_sequence(page, quote, min_ratio=80, min_coverage=0.7):
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wno
    page_words = [(w[4], fitz.Rect(w[0], w[1], w[2], w[3])) for w in words]
    q_sig = sig_words(quote)
    if len(q_sig) < 3:
        return None
    matched_rects = []
    pw_idx = 0
    matched_count = 0
    search_window = 0
    for qw in q_sig:
        found_here = False
        # search forward within a reasonable window from last match to allow skips
        limit = min(len(page_words), pw_idx + 400)
        for j in range(pw_idx, limit):
            pw_text = re.sub(r"[^\w]", "", page_words[j][0]).lower()
            if not pw_text:
                continue
            if fuzz.ratio(qw, pw_text) >= min_ratio:
                matched_rects.append(page_words[j][1])
                pw_idx = j + 1
                matched_count += 1
                found_here = True
                break
        if not found_here:
            continue
    coverage = matched_count / len(q_sig)
    if coverage >= min_coverage and matched_rects:
        return matched_rects, coverage
    return None

def main():
    with open(r"C:/Users/tillq/.claude/jobs/d40fcd3a/tmp/quotes.json", encoding="utf-8") as f:
        content = f.read().split("\n\nTOTAL:")[0]
    quotes = json.loads(content)

    results = []
    docs = {}
    for q in quotes:
        src = q["source"]
        if src not in SOURCE_MAP:
            results.append({**q, "status": "SKIPPED_NO_PDF"})
            continue
        path = f"{LIT}/{SOURCE_MAP[src]}"
        if src not in docs:
            try:
                docs[src] = fitz.open(path)
            except Exception as e:
                results.append({**q, "status": f"PDF_OPEN_ERROR:{e}"})
                continue
        doc = docs[src]
        page_claim = q["page"]
        if page_claim is None:
            results.append({**q, "status": "NO_PAGE_CLAIM"})
            continue
        page_idx = int(page_claim) - 1  # best-effort, 0-based guess
        candidates = [page_idx - 1, page_idx, page_idx + 1]
        candidates = [p for p in candidates if 0 <= p < doc.page_count]

        found_stage = None
        found_page = None
        for p in candidates:
            page = doc[p]
            hits = try_search_for(page, q["quote"])
            if hits:
                found_stage = "search_for"
                found_page = p + 1
                break
        if not found_stage:
            for p in candidates:
                page = doc[p]
                wl = try_wordlist_sequence(page, q["quote"])
                if wl:
                    found_stage = f"wordlist(cov={wl[1]:.2f})"
                    found_page = p + 1
                    break
        results.append({
            "note": q["note"], "source": src, "claimed_page": page_claim,
            "status": found_stage or "NOT_FOUND", "found_page": found_page,
            "quote_preview": q["quote"][:60],
        })

    total = len(results)
    by_status_bucket = {"search_for": 0, "wordlist": 0, "not_found": 0, "skipped": 0}
    for r in results:
        st = r["status"]
        if st.startswith("search_for"):
            by_status_bucket["search_for"] += 1
        elif st.startswith("wordlist"):
            by_status_bucket["wordlist"] += 1
        elif st == "NOT_FOUND":
            by_status_bucket["not_found"] += 1
        else:
            by_status_bucket["skipped"] += 1

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n=== SUMMARY ===")
    print(f"Total quotes: {total}")
    for k, v in by_status_bucket.items():
        print(f"{k}: {v}")
    locatable = by_status_bucket["search_for"] + by_status_bucket["wordlist"]
    measured = total - by_status_bucket["skipped"]
    if measured:
        print(f"Locatable rate (of measured, n={measured}): {locatable/measured*100:.1f}%")
        print(f"Gap rate (of measured): {by_status_bucket['not_found']/measured*100:.1f}%")

if __name__ == "__main__":
    main()
