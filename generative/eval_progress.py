#!/usr/bin/env python3
"""eval_progress.py — Qualitätsentwicklung Atomic-Agent, genaue Daten pro Note.

Usage:
    python eval_progress.py              # Alle Runs
    python eval_progress.py --pdf Bates  # Filter auf PDF
    python eval_progress.py --version v0.3.6  # Filter auf Version
"""
from __future__ import annotations
import argparse
import json

from generative.config import CACHE_DIR

_HISTORY = CACHE_DIR / "quality_history.jsonl"


def load_history(pdf_filter: str | None = None,
                 version_filter: str | None = None) -> list[dict]:
    if not _HISTORY.exists():
        return []
    records = []
    for line in _HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if "error" in r:
                continue
            if pdf_filter and pdf_filter.lower() not in r.get("pdf", "").lower():
                continue
            if version_filter and version_filter != r.get("version", ""):
                continue
            records.append(r)
        except json.JSONDecodeError:
            continue
    return records


def print_table(records: list[dict]) -> None:
    if not records:
        print("Keine Daten.")
        return

    # Header
    print(f"\n{'Timestamp':<17} {'Ver':<8} {'EV':<5} {'Lang':>6} {'Note':<36} "
          f"{'H-%':>6} {'CovF%':>6} {'SrcCov':>7} {'Conf/Unc/Hal':>13}  {'Tok':>8} {'Zeit':>6}  PDF")
    print("-" * 145)

    for r in sorted(records, key=lambda x: (x.get("version",""), x.get("timestamp",""))):
        ts   = r.get("timestamp", "")[:16]
        v    = r.get("version", "?")
        ev   = r.get("eval_version", "?")
        lang = r.get("language", "?")
        note = r.get("note", "?").replace("vault__", "").replace(".md", "")[:35]
        conf = r.get("anchors_confirmed", 0)
        unc  = r.get("anchors_uncertain", 0)
        hall = r.get("anchors_hallucinated", 0)
        h_rate  = r.get("hallucination_rate", 0)
        cov_f   = r.get("coverage_factual", r.get("coverage_rate", 0))
        src_cov = r.get("source_coverage", 0)
        tok  = r.get("tokens_total", 0)
        zeit = r.get("wall_time_s", 0)
        pdf  = r.get("pdf", "?")[:22]
        warn = "!" if r.get("small_sample_warning") else " "
        tok_s  = f"{tok//1000}k" if tok >= 1000 else str(tok)
        zeit_s = f"{int(zeit)}s" if zeit else "-"

        print(f"{ts:<17} {v:<8} {ev:<5} {lang:>6} {note:<36} "
              f"{h_rate:>5.1%}{warn} {cov_f:>5.1%} {src_cov:>6.1%}  {conf:>3}/{unc:>3}/{hall:>3}  "
              f"{tok_s:>8} {zeit_s:>6}  {pdf}")

    print("-" * 145)
    # Summen-Zeile
    total_conf  = sum(r.get("anchors_confirmed", 0) for r in records)
    total_unc   = sum(r.get("anchors_uncertain", 0) for r in records)
    total_hall  = sum(r.get("anchors_hallucinated", 0) for r in records)
    total_parse = sum(r.get("anchors_parseable", 0) for r in records)
    total_fact  = sum(r.get("body_sentences_factual", r.get("body_sentences", 0)) for r in records)
    total_tok   = sum(r.get("tokens_total", 0) for r in records)
    overall_h   = total_hall / total_parse if total_parse else 0
    overall_cov = total_conf / total_fact if total_fact else 0
    tok_s = f"{total_tok//1000}k" if total_tok >= 1000 else str(total_tok)
    print(f"{'GESAMT':<17} {'':<8} {'':<5} {'':<6} {f'{len(records)} Notes':<36} "
          f"{overall_h:>5.1%}  {overall_cov:>5.1%} {'':>7}  {total_conf:>3}/{total_unc:>3}/{total_hall:>3}  "
          f"{tok_s:>8}")
    print()
    print("H-%=Halluzinationsrate | CovF%=Coverage faktische Saetze | SrcCov=PDF-Seitenabdeckung")
    print("Conf/Unc/Hal=Confirmed/Uncertain/Halluziniert | !=kleine Stichprobe | EV=Eval-Version")
    print("Sprache: PDF-Sprache->Note-Sprache  |  Tok=Tokens (gesamt, nicht gecacht)")
    print(f"\nQuelle: {_HISTORY}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitätsentwicklung Atomic-Agent")
    ap.add_argument("--pdf", default=None, help="Filter auf PDF-Name (Substring)")
    ap.add_argument("--version", default=None, help="Filter auf Pipeline-Version")
    args = ap.parse_args()

    records = load_history(pdf_filter=args.pdf, version_filter=args.version)
    print(f"\n=== Atomic-Agent Qualitätsentwicklung === ({len(records)} Einträge)")
    print_table(records)


if __name__ == "__main__":
    main()