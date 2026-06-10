"""Eval-Wrapper: orchestrator.py N-mal pro PDF laufen lassen, Vault/Inbox-Quoten
aggregieren, Per-Note-Stability auswerten.

Sampling-Drift im Confidence-Agent + Critic macht Single-Run-Vergleiche
unzuverlässig (siehe Pipeline-Doc v22). Mehrfach-Runs trennen Filter-Effekte
von LLM-Sampling-Noise.

Usage:
    python eval_repeat.py --version v22 --runs 3
    python eval_repeat.py --version v22 --runs 3 --pdfs schlebbe   # nur ein PDF

Output:
    .cache/eval/baseline/<stem>_<version>_run<i>.log         pro Run
    .cache/eval/baseline/<version>_summary.md                aggregierter Report
"""
from __future__ import annotations
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from generative.config import LITERATURE_DIR as _LIT  # noqa: E402

BASELINE_PDFS = {
    "bates":    _LIT / "Bates - 2017 - Information Behavior.pdf",
    "kuhlthau": _LIT / "Kuhlthau - INFORMATION SEARCH PROCESS.pdf",
    "schlebbe": _LIT / "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
}
EVAL_DIR = Path(__file__).parent / ".cache" / "eval" / "baseline"

_NOTE_RE = re.compile(r"^\s*\[DRY-RUN\] -> (Vault|Inbox)[^:]*: (.+?)\.md\b")


def run_one(pdf: Path, version: str, run_idx: int, key: str) -> Path:
    log = EVAL_DIR / f"{key}_{version}_run{run_idx}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(
            [sys.executable, "orchestrator.py", "--source", str(pdf), "--dry-run"],
            cwd=Path(__file__).parent,
            env=env,
            stdout=fh, stderr=subprocess.STDOUT,
            check=False,
        )
    return log


def parse_log(log: Path) -> dict[str, str]:
    """Returns {note_stem: 'Vault'|'Inbox'} aus DRY-RUN-Zeilen.

    Unterstützt zwei Formate:
    - Alt (v22): [DRY-RUN] -> Vault: slug.md
    - Neu (v0.3.6+): [DRY-RUN] -> Inbox: Titel.md  [Vault-Empf.]
    """
    notes: dict[str, str] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _NOTE_RE.match(line)
        if m:
            routing, stem = m.group(1), m.group(2)
            if routing == "Vault" or "[Vault-Empf.]" in line:
                notes[stem] = "Vault"
            else:
                notes[stem] = "Inbox"
    return notes


def aggregate(per_run: list[dict[str, str]]) -> dict:
    """{note_stem: (vault_count, inbox_count)} + Vault-Counts pro Run."""
    all_notes: set[str] = set()
    for r in per_run:
        all_notes.update(r.keys())
    stability: dict[str, tuple[int, int]] = {}
    for note in all_notes:
        v = sum(1 for r in per_run if r.get(note) == "Vault")
        i = sum(1 for r in per_run if r.get(note) == "Inbox")
        stability[note] = (v, i)
    vault_counts = [sum(1 for v in r.values() if v == "Vault") for r in per_run]
    total_counts = [len(r) for r in per_run]
    return {
        "stability": stability,
        "vault_counts": vault_counts,
        "total_counts": total_counts,
    }


_MDAD_WARNING_THRESHOLD = 25  # LLM-Micro-Benchmarking-Reliability: <25 Items → 51% pairwise falsch


def _stat_warning(total_notes: float, runs: int) -> str | None:
    if total_notes < _MDAD_WARNING_THRESHOLD:
        return (
            f"> **Stat. Warnung:** {total_notes:.0f} Notes < {_MDAD_WARNING_THRESHOLD} Schwelle "
            f"(MDAD, ICLR 2026). Pairwise-Vergleiche bei dieser Größe sind mit ~51% "
            f"Fehlerrate nicht belastbar. Nur Deltas ≥ 6 Notes sind sicher interpretierbar."
        )
    return None


def render_summary(version: str, results: dict[str, dict]) -> str:
    out = [f"# Eval-Summary {version}\n"]
    out.append("## Vault-Quote pro PDF (Median | Mean ± Stdev über Runs)\n")
    out.append("| PDF | Runs | Vault Median | Mean ± Stdev | Notes Mean | Range |")
    out.append("|---|---|---|---|---|---|")
    grand_v: list[float] = []
    grand_t: list[float] = []
    for key, agg in results.items():
        v = agg["vault_counts"]
        t = agg["total_counts"]
        v_median = statistics.median(v)
        v_mean = statistics.mean(v)
        v_std = statistics.stdev(v) if len(v) > 1 else 0.0
        t_mean = statistics.mean(t)
        rng = f"{min(v)}–{max(v)}/{min(t)}–{max(t)}"
        out.append(f"| {key} | {len(v)} | {v_median:.1f} | {v_mean:.1f} ± {v_std:.1f} | {t_mean:.1f} | {rng} |")
        grand_v.append(v_median)
        grand_t.append(t_mean)
    out.append(f"| **Total** | — | **{sum(grand_v):.1f}** | — | **{sum(grand_t):.1f}** | — |")

    total_notes = sum(grand_t)
    runs = len(next(iter(results.values()))["vault_counts"]) if results else 0
    warning = _stat_warning(total_notes, runs)
    if warning:
        out.append(f"\n{warning}")

    out.append("\n## Per-Note Stability\n")
    out.append("Stable = identisches Routing in allen Runs. Flaky = Vault/Inbox-Drift.\n")
    for key, agg in results.items():
        out.append(f"\n### {key}\n")
        stable_v = []
        stable_i = []
        flaky = []
        for note, (vc, ic) in sorted(agg["stability"].items()):
            n = vc + ic
            if vc == n:
                stable_v.append(note)
            elif ic == n:
                stable_i.append(note)
            else:
                flaky.append((note, vc, ic))
        out.append(f"- Stable Vault ({len(stable_v)}): {', '.join(stable_v) if stable_v else '—'}")
        out.append(f"- Stable Inbox ({len(stable_i)}): {', '.join(stable_i) if stable_i else '—'}")
        if flaky:
            out.append(f"- **Flaky ({len(flaky)})**:")
            for note, vc, ic in flaky:
                out.append(f"  - `{note}` — Vault {vc}× / Inbox {ic}×")
        else:
            out.append("- Flaky (0): —")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="z.B. v22")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pdfs", nargs="+", choices=list(BASELINE_PDFS), default=list(BASELINE_PDFS))
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for key in args.pdfs:
        pdf = BASELINE_PDFS[key]
        if not pdf.exists():
            print(f"  [skip] {key}: {pdf} not found", file=sys.stderr)
            continue
        print(f"\n=== {key} ({args.runs} runs) ===", file=sys.stderr)
        per_run: list[dict[str, str]] = []
        for i in range(1, args.runs + 1):
            print(f"  run {i}/{args.runs}…", file=sys.stderr)
            log = run_one(pdf, args.version, i, key)
            per_run.append(parse_log(log))
        results[key] = aggregate(per_run)

    summary = render_summary(args.version, results)
    summary_path = EVAL_DIR / f"{args.version}_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\nSummary written: {summary_path}", file=sys.stderr)
    print("\n" + summary)


if __name__ == "__main__":
    main()
