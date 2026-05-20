"""Paired-Comparison-Eval für Code-Versions-Vergleiche.

Pinnt Vault-Snapshot vor allen Runs, lässt Pipeline pro PDF × Config einmal
laufen, vergleicht Routing pro Note paired (Anthropic „Adding Error Bars to
Evals" 2411.00640).

Prämisse: Pipeline ist deterministisch bei (Code, Cache, Vault) konstant.
Wenn der Vault gepinnt ist und die Code-Version die einzige Variable ist,
genügt 1 Run pro Config — Multi-Run misst nur Cache-Hits, kein neues Signal
(siehe LLM-Pipeline-Eval-Methodologie im Wissenspool).

Usage:
    python eval_paired.py --version v22 --configs no-er v22
    python eval_paired.py --version v22 --configs v22 --pdfs schlebbe

Configs:
    v22       — ENABLE_ENTITY_RESOLUTION=1 (Default)
    no-er     — ENABLE_ENTITY_RESOLUTION=0

Output:
    .cache/eval/baseline/<key>_<version>_<config>.log
    .cache/eval/baseline/vault-snapshot-<version>.json
    .cache/eval/baseline/<version>_paired.md
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LITERATURE_DIR as _LIT  # noqa: E402

BASELINE_PDFS = {
    "bates":    _LIT / "Bates - 2017 - Information Behavior.pdf",
    "kuhlthau": _LIT / "Kuhlthau - INFORMATION SEARCH PROCESS.pdf",
    "schlebbe": _LIT / "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
}
EVAL_DIR = Path(__file__).parent / ".cache" / "eval" / "baseline"

CONFIGS = {
    "v22":    {"ENABLE_ENTITY_RESOLUTION": "1"},
    "no-er":  {"ENABLE_ENTITY_RESOLUTION": "0"},
}

_NOTE_RE = re.compile(r"^\s*\[DRY-RUN\] -> (Vault|Inbox)[^:]*: (.+?)\.md\b")
_MDAD_WARNING_THRESHOLD = 25  # LLM-Micro-Benchmarking-Reliability: <25 Items → 51% pairwise falsch


def run_one(pdf: Path, key: str, version: str, config: str, snapshot: Path) -> Path:
    log = EVAL_DIR / f"{key}_{version}_{config}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["EVAL_VAULT_SNAPSHOT"] = str(snapshot)
    env.update(CONFIGS[config])
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


def render_paired(version: str, configs: list[str], results: dict[str, dict[str, dict[str, str]]]) -> str:
    """results[pdf][config] = {note_stem: 'Vault'|'Inbox'}"""
    out = [f"# Paired-Eval {version}\n"]
    out.append(f"Configs: {', '.join(configs)}. Vault-Snapshot gepinnt vor allen Runs.\n")

    out.append("## Vault-Quote pro Config\n")
    out.append("| PDF | " + " | ".join(configs) + " |")
    out.append("|---|" + "---|" * len(configs))
    totals = {c: [0, 0] for c in configs}  # [vault, total]
    for pdf, cfg_map in results.items():
        cells = []
        for c in configs:
            notes = cfg_map.get(c, {})
            v = sum(1 for r in notes.values() if r == "Vault")
            t = len(notes)
            cells.append(f"{v}/{t}")
            totals[c][0] += v
            totals[c][1] += t
        out.append(f"| {pdf} | " + " | ".join(cells) + " |")
    out.append("| **Total** | " + " | ".join(f"**{totals[c][0]}/{totals[c][1]}**" for c in configs) + " |")

    total_notes = max(totals[c][1] for c in configs) if totals else 0
    if total_notes < _MDAD_WARNING_THRESHOLD:
        out.append(
            f"\n> **Stat. Warnung:** {total_notes} Notes < {_MDAD_WARNING_THRESHOLD} Schwelle "
            f"(MDAD, ICLR 2026). Pairwise-Deltas bei dieser Größe sind mit ~51% "
            f"Fehlerrate nicht belastbar. Nur Deltas ≥ 6 Notes sicher interpretierbar."
        )

    out.append("\n## Per-Note Routing-Diff\n")
    out.append("Notes mit unterschiedlichem Routing zwischen Configs (paired).\n")
    for pdf, cfg_map in results.items():
        all_notes = set().union(*(set(d.keys()) for d in cfg_map.values()))
        diffs = []
        for note in sorted(all_notes):
            routings = {c: cfg_map.get(c, {}).get(note, "—") for c in configs}
            if len(set(routings.values())) > 1:
                diffs.append((note, routings))
        out.append(f"\n### {pdf}\n")
        if not diffs:
            out.append("- keine Diffs (alle Configs identisch)")
            continue
        out.append("| Note | " + " | ".join(configs) + " |")
        out.append("|---|" + "---|" * len(configs))
        for note, routings in diffs:
            out.append(f"| `{note}` | " + " | ".join(routings[c] for c in configs) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), required=True)
    ap.add_argument("--pdfs", nargs="+", choices=list(BASELINE_PDFS), default=list(BASELINE_PDFS))
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = EVAL_DIR / f"vault-snapshot-{args.version}.json"
    if not snapshot.exists():
        print(f"[snapshot] erster Run baut Snapshot: {snapshot.name}", file=sys.stderr)
    else:
        print(f"[snapshot] re-use: {snapshot.name}", file=sys.stderr)

    results: dict[str, dict[str, dict[str, str]]] = {}
    for key in args.pdfs:
        pdf = BASELINE_PDFS[key]
        if not pdf.exists():
            print(f"  [skip] {key}: not found", file=sys.stderr)
            continue
        results[key] = {}
        for cfg in args.configs:
            print(f"\n=== {key} × {cfg} ===", file=sys.stderr)
            log = run_one(pdf, key, args.version, cfg, snapshot)
            results[key][cfg] = parse_log(log)

    md = render_paired(args.version, args.configs, results)
    out_path = EVAL_DIR / f"{args.version}_paired.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n[paired] {out_path}", file=sys.stderr)
    print("\n" + md)


if __name__ == "__main__":
    main()
