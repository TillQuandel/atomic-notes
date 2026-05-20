"""Adversarial-Set: Gemini perturbiert ~2 supported-Claims pro 3 ausgewählten
Notes je Sprachpaar (= 6 Notes, ~12 perturbierte Claims, ~5-8 % der Claims).

Schritte:
  1. Wähle 3 Notes pro Sprachpaar deterministisch (Seed=42).
  2. Pro Note: lade Pipeline-Output, finde supported-Claims (label ∈ supported_*).
  3. Pro Claim: Gemini perturbiert (numeric_swap / entity_swap / negation).
  4. Schreibe perturbierte Note-Datei (Body-Replace Original → Perturbiert).
  5. Re-eval mit v4.1 → quality_history erhält Einträge mit must_detect=true.
  6. Ground-Truth → adversarial.jsonl.

Voraussetzung: run.py muss bereits gelaufen sein (sonst keine claim_scores zum Auswählen).
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import AGENT_VERSION  # noqa: E402
from eval_quality_v4 import (  # noqa: E402
    _QUALITY_HISTORY as HISTORY_PATH,
    eval_note,
    save_result,
)

CALIB = ROOT / ".cache" / "eval" / "calibration"
SAMPLE_FILE = CALIB / "sample.jsonl"
PERTURB_DIR = CALIB / "perturbed"
ADVERSARIAL_OUT = CALIB / "adversarial.jsonl"
VAULT_ROOT = ROOT.parent.parent.parent

NOTES_PER_PAIR = 3
CLAIMS_PER_NOTE = 2
PERTURBATION_TYPES = ("numeric_swap", "entity_swap", "negation")
SEED = 42
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_TIMEOUT_S = 240  # Gemini Pro Preview Latenz, insbesondere bei Cold-Start

SUPPORTED_LABELS = {"supported_exact", "supported_paraphrase", "partially_supported"}

PERTURB_PROMPT = """Aufgabe: Verändere genau einen Claim für einen Adversarial-Test.

Perturbationstyp: {ptype}
- numeric_swap: tausche eine Zahl/ein Jahr/einen Prozentwert (5→7, 1999→2003, 30%→70%).
- entity_swap: tausche eine Entität gegen eine plausible aber falsche ("Bates 2017"→"Wilson 2015").
- negation: drehe die Kernaussage um oder negiere explizit ("ist"→"ist nicht").

Regeln:
- plausibel klingen, grammatikalisch korrekt bleiben
- NUR die geänderte Aussage ausgeben, ohne Anführungszeichen, ohne Erklärung, ohne Präfix
- wenn keine passende Stelle vorhanden ist: gib exakt das Wort NONE aus

Original-Claim:
{claim}

Perturbierter Claim:"""


def load_sample() -> list[dict]:
    with SAMPLE_FILE.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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


_GEMINI_EXE: str | None = None


def _resolve_gemini() -> str | None:
    """Findet die gemini-Executable inkl. Windows-.CMD-Wrapper-Auflösung."""
    global _GEMINI_EXE
    if _GEMINI_EXE is not None:
        return _GEMINI_EXE
    path = shutil.which("gemini")
    if path is None:
        print("FEHLER: 'gemini' nicht in PATH gefunden", file=sys.stderr)
        return None
    _GEMINI_EXE = path
    return path


def call_gemini_perturb(claim: str, ptype: str) -> str | None:
    prompt = PERTURB_PROMPT.format(claim=claim, ptype=ptype)
    exe = _resolve_gemini()
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-m", GEMINI_MODEL, "-p", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GEMINI_TIMEOUT_S,
            shell=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"      Gemini-Fehler ({ptype}): {exc}")
        return None
    if proc.returncode != 0:
        print(f"      Gemini-Returncode {proc.returncode}: {proc.stderr[:200]}")
        return None
    out = proc.stdout.strip()
    # Erste nicht-leere Zeile, ohne Markdown-Marker
    for line in out.splitlines():
        line = line.strip().lstrip("- ").strip("\"'`*")
        if line and line.upper() != "NONE":
            return line
        if line.upper() == "NONE":
            return None
    return None


def perturb_note_body(body: str, original_claim: str, perturbed_claim: str) -> str | None:
    """Ersetzt das erste Vorkommen des Claims (toleriert leichte Whitespace-Variation)."""
    # exact match
    if original_claim in body:
        return body.replace(original_claim, perturbed_claim, 1)
    # whitespace-tolerant: re.escape verhält sich versionsabhängig (3.7–3.13 lässt Spaces
    # unescaped, 3.14 escapet sie wieder zu `\ `). Wir handlen beide Varianten.
    escaped = re.escape(original_claim)
    pattern = escaped.replace(r"\ ", r"\s+").replace(" ", r"\s+")
    match = re.search(pattern, body)
    if match:
        return body[: match.start()] + perturbed_claim + body[match.end():]
    return None


def main() -> None:
    samples = load_sample()
    pipeline = load_pipeline_results()
    if not pipeline:
        print(f"FEHLER: quality_history.jsonl enthält keine v{AGENT_VERSION}-Einträge — erst run.py", file=sys.stderr)
        sys.exit(1)

    # Notes mit Pipeline-Output + ≥CLAIMS_PER_NOTE supported-Claims
    by_pair: dict[str, list[dict]] = {"DE→DE": [], "EN→DE": []}
    for s in samples:
        result = pipeline.get(s["note_name"])
        if result is None:
            continue
        supported = [
            (i, c) for i, c in enumerate(result.get("claim_scores", []))
            if c.get("label") in SUPPORTED_LABELS
        ]
        if len(supported) < CLAIMS_PER_NOTE:
            continue
        by_pair[s["language_pair"]].append({**s, "_supported": supported, "_result": result})

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for pair, pool in by_pair.items():
        if len(pool) < NOTES_PER_PAIR:
            print(f"WARN {pair}: nur {len(pool)} qualifizierende Notes (Soll: {NOTES_PER_PAIR})")
            chosen.extend(pool)
        else:
            chosen.extend(rng.sample(pool, NOTES_PER_PAIR))

    print(f"Perturbiere {len(chosen)} Notes × {CLAIMS_PER_NOTE} Claims = {len(chosen) * CLAIMS_PER_NOTE} Claims")
    PERTURB_DIR.mkdir(parents=True, exist_ok=True)

    adversarial_records: list[dict] = []
    for idx, entry in enumerate(chosen, start=1):
        note_name = entry["note_name"]
        pair = entry["language_pair"]
        supported = entry["_supported"]
        picks = rng.sample(supported, CLAIMS_PER_NOTE)

        original_path = VAULT_ROOT / entry["note_path"]
        if not original_path.exists():
            print(f"  [{idx}] SKIP {note_name}: original fehlt")
            continue
        body = original_path.read_text(encoding="utf-8")
        perturbed_body = body
        perturbed_claims_info: list[dict] = []

        for claim_idx, claim_score in picks:
            original_claim = claim_score["claim"].strip()
            ptype = rng.choice(PERTURBATION_TYPES)
            print(f"  [{idx}] {note_name} claim_idx={claim_idx} ({ptype})")
            perturbed = call_gemini_perturb(original_claim, ptype)
            if not perturbed or perturbed.strip() == original_claim.strip():
                print(f"      → unverändert, skip")
                continue
            new_body = perturb_note_body(perturbed_body, original_claim, perturbed)
            if new_body is None:
                print(f"      → Original-Claim nicht im Body gefunden, skip")
                continue
            perturbed_body = new_body
            perturbed_claims_info.append({
                "original_claim_idx": claim_idx,
                "original_claim": original_claim,
                "perturbed_claim": perturbed,
                "perturbation_type": ptype,
            })

        if not perturbed_claims_info:
            print(f"  [{idx}] keine erfolgreichen Perturbationen, skip")
            continue

        perturbed_name = f"adv__{note_name}"
        perturbed_path = PERTURB_DIR / perturbed_name
        perturbed_path.write_text(perturbed_body, encoding="utf-8")

        # Re-eval
        pdf_path = Path(entry["pdf_path"])
        print(f"      Re-eval v{AGENT_VERSION}…")
        try:
            result = eval_note(perturbed_path, pdf_path, AGENT_VERSION)
            save_result(result)
        except Exception as exc:
            print(f"      FEHLER bei Re-eval: {exc}")
            continue

        adversarial_records.append({
            "note": note_name,
            "perturbed_note": perturbed_name,
            "perturbed_path": str(perturbed_path).replace("\\", "/"),
            "language_pair": pair,
            "pdf_path": str(pdf_path).replace("\\", "/"),
            "must_detect": True,
            "perturbations": perturbed_claims_info,
        })

    with ADVERSARIAL_OUT.open("w", encoding="utf-8") as fh:
        for rec in adversarial_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_perturbed = sum(len(r["perturbations"]) for r in adversarial_records)
    print(f"\n=== Done. {len(adversarial_records)} Notes, {total_perturbed} perturbierte Claims ===")
    print(f"Output: {ADVERSARIAL_OUT}")
    print(f"Perturbierte Notes: {PERTURB_DIR}")


if __name__ == "__main__":
    main()
