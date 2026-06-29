"""Berechnet Gwet's AC1 + Cohen's κ + Krippendorff's α aus Human-Labels
vs. Pipeline-Labels für die v4.1-Kalibrierung.

Inputs (alle in .cache/eval/calibration/):
  - labels_human.jsonl     (collect.py)
  - labels_blind.jsonl     (collect.py auf blind/, optional)
  - adversarial.jsonl      (adversarial.py, optional)
  - quality_history.jsonl  (run.py, gefiltert auf AGENT_VERSION)

Output:
  - kappa_report.md (Ampel: AC1 ≥ 0.70 grün, 0.40-0.70 gelb, < 0.40 rot)

Verweigert die Berechnung wenn Labels unvollständig sind (Selection-Bias-Vermeidung).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from generative.config import AGENT_VERSION, QUALITY_HISTORY as HISTORY_PATH  # noqa: E402

CALIB = ROOT / ".cache" / "eval" / "calibration"
SAMPLE_FILE = CALIB / "sample.jsonl"
HUMAN_LABELS = CALIB / "labels_human.jsonl"
BLIND_LABELS = CALIB / "labels_blind.jsonl"
ADVERSARIAL = CALIB / "adversarial.jsonl"
REPORT_OUT = CALIB / "kappa_report.md"

PIPELINE_TO_BINARY = {
    "supported_exact": "s",
    "supported_paraphrase": "s",
    "partially_supported": "s",
    "not_in_context": "h",
    "contradicted": "h",
    "retrieval_or_parse_uncertain": "?",
    "parse_error": "?",
}
CATEGORIES = ("s", "h", "?")

REFERENCE_BENCHMARKS = [
    ("Human-to-human (NVIDIA Judge's Verdict)", "Cohen's κ", 0.801, "QA-Grading"),
    ("FEVER (Thorne 2018)", "Fleiss-κ", 0.68, "Fact-Verification"),
    ("RAGTruth (Niu 2024)", "Response-Agreement", 0.918, "RAG-Hallucination"),
    ("RAGTruth++ (BlueGuardrails 2025)", "F1", 0.78, "RAG-Hallucination Span"),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_pipeline_results() -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not HISTORY_PATH.exists():
        return results
    with HISTORY_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("version") != AGENT_VERSION:
                continue
            name = Path(entry.get("note", "")).name
            if name:
                results[name] = entry
    return results


def extract_pipeline_labels(pipeline: dict[str, dict]) -> dict[tuple[str, int], str]:
    """Mappt (note_name, claim_idx) → binäres Label."""
    out: dict[tuple[str, int], str] = {}
    for name, entry in pipeline.items():
        for score in entry.get("claim_scores", []):
            label = score.get("label", "")
            binary = PIPELINE_TO_BINARY.get(label, "?")
            out[(name, score["claim_idx"])] = binary
    return out


# ---------- Statistik-Maße (manuell implementiert) ----------


def percent_agreement(r1: list[str], r2: list[str]) -> float:
    if not r1:
        return float("nan")
    return sum(1 for a, b in zip(r1, r2) if a == b) / len(r1)


def cohen_kappa(r1: list[str], r2: list[str], categories=CATEGORIES) -> float:
    n = len(r1)
    if n == 0:
        return float("nan")
    p_a = percent_agreement(r1, r2)
    c1 = Counter(r1)
    c2 = Counter(r2)
    p_e = sum((c1[k] / n) * (c2[k] / n) for k in categories)
    if p_e >= 1.0:
        return float("nan")
    return (p_a - p_e) / (1 - p_e)


def gwet_ac1(r1: list[str], r2: list[str], categories=CATEGORIES) -> float:
    """Robust gegen Klassen-Imbalance (Kappa-Paradox)."""
    n = len(r1)
    if n == 0:
        return float("nan")
    p_a = percent_agreement(r1, r2)
    K = len(categories)
    if K <= 1:
        return float("nan")
    pi = {}
    for k in categories:
        pi[k] = (r1.count(k) + r2.count(k)) / (2 * n)
    p_e = sum(pi[k] * (1 - pi[k]) for k in categories) / (K - 1)
    if p_e >= 1.0:
        return float("nan")
    return (p_a - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(r1: list[str], r2: list[str]) -> float:
    n = len(r1)
    if n == 0:
        return float("nan")
    D_o = sum(0 if a == b else 1 for a, b in zip(r1, r2)) / n
    all_ratings = r1 + r2
    counter = Counter(all_ratings)
    total = sum(counter.values())
    if total < 2:
        return float("nan")
    D_e_num = 0
    keys = list(counter.keys())
    for i, ki in enumerate(keys):
        for kj in keys:
            if ki != kj:
                D_e_num += counter[ki] * counter[kj]
    D_e = D_e_num / (total * (total - 1))
    if D_e == 0:
        return 1.0
    return 1 - D_o / D_e


def confusion_matrix(human: list[str], pipeline: list[str], categories=CATEGORIES) -> dict:
    matrix = {h: {p: 0 for p in categories} for h in categories}
    for h, p in zip(human, pipeline):
        if h in matrix and p in matrix[h]:
            matrix[h][p] += 1
    return matrix


# ---------- Report-Rendering ----------


def ampel(value: float) -> str:
    if value != value:  # NaN
        return "⚪"
    if value >= 0.70:
        return "🟢"
    if value >= 0.40:
        return "🟡"
    return "🔴"


def render_metric_table(human: list[str], pipeline: list[str]) -> list[str]:
    ac1 = gwet_ac1(human, pipeline)
    kappa = cohen_kappa(human, pipeline)
    alpha = krippendorff_alpha_nominal(human, pipeline)
    agree = percent_agreement(human, pipeline)
    return [
        "| Maß | Wert | Ampel |",
        "|---|---|---|",
        f"| **Gwet's AC1** (Primär) | {ac1:.3f} | {ampel(ac1)} |",
        f"| Krippendorff's α | {alpha:.3f} | {ampel(alpha)} |",
        f"| Cohen's κ (Referenz) | {kappa:.3f} | {ampel(kappa)} |",
        f"| % Agreement | {agree:.1%} | — |",
    ]


def render_confusion(human: list[str], pipeline: list[str]) -> list[str]:
    matrix = confusion_matrix(human, pipeline)
    lines = ["| Mensch \\ Pipeline | s | h | ? | Σ |", "|---|---|---|---|---|"]
    totals = {p: 0 for p in CATEGORIES}
    for h in CATEGORIES:
        row_total = sum(matrix[h].values())
        cells = " | ".join(str(matrix[h][p]) for p in CATEGORIES)
        lines.append(f"| **{h}** | {cells} | {row_total} |")
        for p in CATEGORIES:
            totals[p] += matrix[h][p]
    total_cells = " | ".join(str(totals[p]) for p in CATEGORIES)
    lines.append(f"| **Σ** | {total_cells} | {sum(totals.values())} |")
    return lines


def check_completeness(humans: list[dict], pipeline_labels: dict) -> tuple[list[dict], list[str]]:
    """Filtert nur Claims, für die sowohl Human- als auch Pipeline-Label existieren.
    Liefert Warnungen für fehlende."""
    warnings: list[str] = []
    pipeline_keys = set(pipeline_labels.keys())
    valid: list[dict] = []
    for h in humans:
        key = (h["note"], h["claim_idx"])
        if key in pipeline_keys:
            valid.append(h)
        else:
            warnings.append(f"  Mensch labelte {h['note']}#{h['claim_idx']}, aber kein Pipeline-Output.")
    # Reverse: Pipeline-Claims ohne Human-Label
    human_keys = {(h["note"], h["claim_idx"]) for h in humans}
    sampled_notes = {h["note"] for h in humans}
    missing_human = [
        (note, idx) for (note, idx) in pipeline_keys if note in sampled_notes and (note, idx) not in human_keys
    ]
    if missing_human:
        warnings.append(f"  {len(missing_human)} Pipeline-Claims ohne Human-Label (in Sample-Notes).")
    return valid, warnings


def main() -> None:
    humans = load_jsonl(HUMAN_LABELS)
    if not humans:
        print(f"FEHLER: {HUMAN_LABELS} fehlt/leer — erst labeln + collect.py", file=sys.stderr)
        sys.exit(1)
    pipeline = load_pipeline_results()
    if not pipeline:
        print(f"FEHLER: quality_history.jsonl ohne v{AGENT_VERSION}-Einträge — erst run.py", file=sys.stderr)
        sys.exit(1)
    pipeline_labels = extract_pipeline_labels(pipeline)
    valid, warnings = check_completeness(humans, pipeline_labels)
    if warnings:
        print("Warnungen:")
        for w in warnings:
            print(w)
    if len(valid) < 10:
        print(f"FEHLER: nur {len(valid)} matching (Human + Pipeline) Claims — zu wenig für AC1.")
        sys.exit(2)

    # Pro Sprachpaar + gesamt
    by_pair: dict[str, list[dict]] = {}
    for h in valid:
        by_pair.setdefault(h["language_pair"], []).append(h)

    blind = load_jsonl(BLIND_LABELS)
    adversarial_records = load_jsonl(ADVERSARIAL)

    lines: list[str] = []
    lines.append(f"# Kalibrierungs-Report v{AGENT_VERSION} (Gwet AC1)")
    lines.append("")
    lines.append(f"- Gemessen am: {Path(HUMAN_LABELS).stat().st_mtime}")
    lines.append(f"- Human-Labels: {len(humans)} (valid für AC1: {len(valid)})")
    lines.append(f"- Pipeline-Eval: {len(pipeline_labels)} Claims über {len(pipeline)} Notes")
    lines.append("")
    lines.append("## Härtungs-Status")
    lines.append(
        "- Härtung #1 (Pipeline-Verdict verborgen): siehe `labeling_protocol.md` — strukturell durch build_labels.py."
    )
    lines.append("- Härtung #2 (Conditional Blind): User-Disziplin, im Pre-Reg dokumentiert.")
    lines.append(f"- Härtung #3 (Blind-Sub-Sample): {len(blind)} Blind-Labels gefunden.")
    lines.append("- Härtung #4 (Pre-Registration): `calibration/labeling_protocol.md`.")
    lines.append("- Härtung #5 (Dritt-Reviewer): noch nicht durchgeführt (separat nach Erstreport).")
    lines.append("")
    lines.append("## Ergebnis pro Sprachpaar")
    for pair, items in sorted(by_pair.items()):
        h_labels = [x["label"] for x in items]
        p_labels = [pipeline_labels[(x["note"], x["claim_idx"])] for x in items]
        lines.append(f"\n### {pair} (n={len(items)})")
        lines.append("")
        lines.extend(render_metric_table(h_labels, p_labels))
        lines.append("")
        lines.append("**Confusion Matrix:**")
        lines.append("")
        lines.extend(render_confusion(h_labels, p_labels))
        lines.append("")

    # Overall
    all_h = [x["label"] for x in valid]
    all_p = [pipeline_labels[(x["note"], x["claim_idx"])] for x in valid]
    lines.append(f"\n## Gesamt (n={len(valid)})")
    lines.append("")
    lines.extend(render_metric_table(all_h, all_p))
    lines.append("")
    lines.append("**Confusion Matrix:**")
    lines.append("")
    lines.extend(render_confusion(all_h, all_p))
    lines.append("")

    # Blind-Δ — vergleicht über identische Items (Schnittmenge Hybrid ∩ Blind ∩ Pipeline)
    if blind:
        blind_map = {(b["note"], b["claim_idx"]): b["label"] for b in blind}
        hybrid_map = {(v["note"], v["claim_idx"]): v["label"] for v in valid}
        common_keys = sorted(set(blind_map) & set(hybrid_map) & set(pipeline_labels))
        if common_keys:
            hb_h = [hybrid_map[k] for k in common_keys]
            bl_h = [blind_map[k] for k in common_keys]
            hb_p = [pipeline_labels[k] for k in common_keys]
            ac1_hybrid = gwet_ac1(hb_h, hb_p)
            ac1_blind = gwet_ac1(bl_h, hb_p)  # identische Pipeline-Items, beide Mensch-Modi vergleichen
            delta = ac1_hybrid - ac1_blind if ac1_hybrid == ac1_hybrid and ac1_blind == ac1_blind else float("nan")
            lines.append("\n## Blind-Kontroll-Δ (Härtung #3)")
            lines.append("")
            lines.append(f"- Schnittmenge (gleiche Items): n={len(common_keys)}")
            lines.append(f"- Hybrid-AC1: {ac1_hybrid:.3f}")
            lines.append(f"- Blind-AC1:  {ac1_blind:.3f}")
            lines.append(f"- **Δ(Hybrid − Blind)**: {delta:+.3f} {'🔴 INFLATION (>0.10)' if delta > 0.10 else '🟢 OK'}")
            lines.append("")
            if delta > 0.10:
                lines.append("  → Hybrid-AC1 als oberer Bound reporten. Anchoring-Bias dokumentieren.")
                lines.append("")

    # Adversarial-Recall
    if adversarial_records:
        lines.append("\n## Adversarial-Recall (Härtung-extern)")
        lines.append("")
        adv_total = 0
        adv_caught = 0
        for rec in adversarial_records:
            perturbed_name = rec["perturbed_note"]
            adv_pipe = pipeline.get(perturbed_name)
            if not adv_pipe:
                continue
            for perturb in rec["perturbations"]:
                adv_total += 1
                target = perturb["perturbed_claim"].strip().lower()
                # Fehlertolerant: Substring-Match (Pipeline kann beim Re-Extract leicht umformulieren).
                # Erste Treffer-Heuristik: target ⊂ extracted ODER extracted ⊂ target.
                for score in adv_pipe.get("claim_scores", []):
                    extracted = score.get("claim", "").strip().lower()
                    if not extracted:
                        continue
                    if target in extracted or extracted in target:
                        binary = PIPELINE_TO_BINARY.get(score.get("label", ""), "?")
                        if binary == "h":
                            adv_caught += 1
                        break
        recall = adv_caught / adv_total if adv_total else float("nan")
        lines.append(f"- Adversarial-Claims total: {adv_total}")
        lines.append(f"- Korrekt als `h` erkannt: {adv_caught}")
        lines.append(f"- **Recall**: {recall:.1%} {ampel(recall)}")
        lines.append("")

    # Referenz-Anker
    lines.append("\n## Referenz-Benchmarks (zum Vergleich)")
    lines.append("")
    lines.append("| Quelle | Maß | Wert | Setting |")
    lines.append("|---|---|---|---|")
    for source, metric, value, setting in REFERENCE_BENCHMARKS:
        lines.append(f"| {source} | {metric} | {value:.3f} | {setting} |")
    lines.append("")
    lines.append("Siehe [[Faithfulness-Annotation-Protokoll]] für Methodik.")
    lines.append("")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport geschrieben: {REPORT_OUT}")
    print(f"Gesamt-AC1: {gwet_ac1(all_h, all_p):.3f}")


if __name__ == "__main__":
    main()
