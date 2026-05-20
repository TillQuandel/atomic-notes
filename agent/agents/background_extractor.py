"""Stage-0.5: Background-Extractor — Trainingswissen pro Konzept vor Extractor-Phase.

Fragt das Modell nach allgemeinem Hintergrundwissen für jeden Konzept-Titel aus
dem Planner-Output — ohne Quellentext-Kontext. Output wird dem Extractor als
`background_context` übergeben, damit dieser [Trainingswissen]-Tags setzen kann.

Rationale: LLMs können nicht zwischen "ich kenne das aus der Quelle" und "ich kenne
das aus dem Training" diskriminieren (Yona et al. 2024 Discriminative Gap). Indem
wir das Hintergrundwissen in einem dedizierten Stage ohne Quellentext abfragen,
ist der Ursprung strukturell sauber: alles was hier rauskommt ist Trainingswissen.
"""
from __future__ import annotations
import re
import sys

from agents.base import call_claude
from schemas.atomic_note import ConceptPlan
from config import MODEL_HAIKU

# Max Claims pro Konzept — mehr als 4 wird selten benötigt und erhöht Token-Kosten
_MAX_CLAIMS = 4


def run(concept_plan: ConceptPlan) -> dict[str, list[str]]:
    """Gibt {concept_title: [claim, ...]} zurück.

    Ein Haiku-Call mit allen Konzept-Titeln als Batch.
    Non-fatal: bei Fehler → leeres Dict, Pipeline läuft ohne Hintergrundwissen weiter.
    """
    actionable = [c for c in concept_plan.concepts if c.action != "skip"]
    if not actionable:
        return {}

    titles_block = "\n".join(f"- {c.title}" for c in actionable)

    prompt = f"""Du bekommst eine Liste von Konzept-Titeln. Für jeden Titel: nenne bis zu {_MAX_CLAIMS} kurze, allgemein bekannte Fakten (1 Satz je) aus deinem Trainingswissen — OHNE Bezug auf eine spezifische Quelle.

Regeln:
- Nur breites Hintergrundwissen, keine quellenspezifischen Aussagen
- Wenn du zu einem Titel wenig sicheres Wissen hast: 0–1 Sätze, lieber weniger als halluzinieren
- Keine Wertungen, keine Vorhersagen

Konzepte:
{titles_block}

Format (exakt einhalten, ein Block pro Konzept):
<!--CONCEPT-->
title: <Titel exakt wie oben>
<!--CLAIMS-->
- <Satz 1>
- <Satz 2>
<!--END-->

Nur dieses Format, kein erklärender Text."""

    try:
        raw = call_claude(prompt, model=MODEL_HAIKU, agent="background-extractor")
    except Exception as e:
        print(f"      [background-extractor] Haiku-Call fehlgeschlagen: {e}", file=sys.stderr)
        return {}

    result = _parse(raw)
    print(
        f"      [background-extractor] {len(result)}/{len(actionable)} Konzepte mit Hintergrundwissen",
        file=sys.stderr,
    )
    return result


def _parse(raw: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for block in re.split(r"<!--CONCEPT-->", raw):
        block = block.strip()
        if not block:
            continue
        title_m = re.search(r"^title:\s*(.+)$", block, re.MULTILINE)
        claims_m = re.search(r"<!--CLAIMS-->(.*?)<!--END-->", block, re.DOTALL)
        if not title_m or not claims_m:
            continue
        title = title_m.group(1).strip()
        claims = [
            line.lstrip("- ").strip()
            for line in claims_m.group(1).splitlines()
            if line.strip().startswith("-") and line.strip() != "-"
        ]
        if claims:
            result[title] = claims
    return result
