"""Planner-Agent: TOC + Intro + Fazit + Relevanz-Profil → ConceptPlan."""
from __future__ import annotations

import re

# Generika-Blacklist (portiert aus extractive/gliner_planner.py, angepasst für LLM-Output).
# Fängt seltene LLM-"Ausrutscher" ab wenn trotz Prompt-Vorgabe abstrakte Einzel-Konzepte
# geplant werden. Normalisierung auf lowercase Pflicht (LLM gibt Title-Case aus).
_GENERIC_BLACKLIST: frozenset[str] = frozenset({
    "information", "system", "process", "method", "model", "data", "analysis",
    "management", "result", "approach", "aspect", "concept", "theory", "issue",
    "factor", "element", "component", "feature", "problem", "solution", "area",
    "level", "type", "form", "role", "ability", "use", "need", "way", "part",
    "point", "case", "end", "set",
    "methods", "models", "metrics", "factors", "systems", "concepts", "aspects",
    "results", "studies", "issues", "elements",
})

from agents.base import call_claude
from agents.cross_reference import _tokens  # Stoppwort-gefilterte Content-Tokens
from agents.structured_output import parse_planner_output
from config import MODEL_PLANNER
from schemas.atomic_note import ConceptPlan, ConceptItem

_PROMPT = """Du bist ein Wissensmanagement-Assistent, der Atomic Notes in Obsidian anlegt.

## Deine Aufgabe
Analysiere den untenstehenden Quellentext (Anfang + Ende eines Dokuments) und erstelle einen Konzept-Plan:
Welche Konzepte sollen als eigenständige Atomic Notes angelegt werden?

Bewerte ausschließlich aus dem Quellentext heraus, welche Konzepte substantiell behandelt werden — kein externer Themen-Bias. Jedes Buch/Paper bestimmt seine Konzepte selbst.

Sprache: Extrahiere Konzepte in der Hauptsprache des Dokuments. Etablierte englische Fachbegriffe in deutschen Texten (z.B. "Prompt Engineering", "Transfer Learning") sind erlaubt wenn sie im Text substantiell behandelt werden. Vermeide rein generische Begriffe wie "Method", "Model", "System" ohne spezifisches Qualifikator.

## Dein Scan-Prozess (zwei Pässe — Fix 1, Category-aware Planner)

### Pass 1 — Architektonische und konzeptuelle Konzepte
Scanne den Text auf Definitionen, Komponenten, Muster, Taxonomien, Modelle.
Priorität "high" = Kern-Konzept (TOC/Intro/Fazit, eigenes Kapitel)
Priorität "medium" = wichtige Sub-/Begleit-Konzepte mit Substanz

### Pass 2 — Operative Konzepte (systematisch übersehen ohne expliziten Scan)
Prüfe den Text gezielt auf diese Klassen — nur aufnehmen wenn SUBSTANTIELL behandelt:
- **State/Memory**: Zustandsverwaltung, Gedächtnis, Kontext über Schritte hinweg
- **Evaluation/Testing**: Evals, Metriken, Benchmarks, Baseline, Accuracy
- **Error Handling**: Retry, Fehlerkorrektur, Eskalation, Fallback, Abbruchbedingungen
- **Sicherheit**: Guardrails, Adversarial Inputs, Prompt Injection, Least Privilege
- **Deployment/Betrieb**: Latenz, Kosten, Monitoring, Logging

### Regeln für beide Pässe
- Nur Konzepte die SUBSTANTIELL im Text behandelt werden (nicht nur erwähnt)
- action "create" = neue Note | "extend" = existierende Note ergänzen | "skip" = zu allgemein oder nicht im Text
- **Selbst-Filter (kritisch):** Wenn ein Konzept dir aus deinem Trainingswissen bekannt ist, aber im bereitgestellten Text nicht substantiell behandelt wird → action="skip". Trainingswissen ersetzt NICHT den Textnachweis. Zweifel → skip.
- Konzept-Dedup: wenn zwei Konzepte inhaltlich dasselbe beschreiben → nur den präziseren Titel behalten
  (Beispiel: "LLM-Agent" + "Single-Agent-System" → nur eines)

## Primärautoren dieser Quelle
{primary_authors_block}

## Origin-Klassifikation (Pflicht für jedes Konzept)

Wessen Konzept ist das primär?

- `origin: primary` — Konzept stammt von den Primärautoren ({primary_authors_line}). Sie definieren, entwickeln oder präsentieren es als eigene Arbeit.
- `origin: extension` — Primärautoren analysieren, kritisieren oder erweitern eine fremde Theorie **substanziell** (>1 Absatz eigene Analyse). Die Note dokumentiert die Auseinandersetzung der Primärautoren.
- `origin: secondary_mention` — Konzept stammt von anderen Autoren (nicht {primary_authors_line}) und wird nur zitiert oder kurz erwähnt. Kein substanzieller Eigenanteil der Primärautoren.

**Survey-/Review-Papers:** `primary` = eigenes Framing, Taxonomie, Evaluationsergebnisse. Einzelne gelistete Methoden anderer Autoren → `secondary_mention`.
**Grenzfall:** Bei Unsicherheit zwischen extension und secondary_mention: >1 Absatz eigener Analyse → `extension`, sonst `secondary_mention`.
**Negativ-Beispiel `extension`:** Ein einzelner Satz wie „In Anlehnung an Smith (2020) adaptieren wir…" ist KEIN extension — das ist `secondary_mention`. Extension erfordert eigene inhaltliche Auseinandersetzung im Text.

`cited_authors`: kommagetrennte Nachnamen der Urheber des Konzepts (nicht der Primärautoren). Leer bei `origin: primary`.

## Vault-Dedup-Hilfe (NICHT relevanz-leitend)
Die folgende Liste ist eine reine Dedup-Hilfe — sie zeigt was im Vault schon existiert.
Sie darf NICHT beeinflussen, welche Konzepte du für substantiell hältst. Welche Konzepte
substantiell sind, entscheidest du ausschließlich aus dem Quellentext. Diese Liste ist
nur dafür da, bei Treffer "extend" oder "skip" zu wählen statt "create".

{existing_concepts}
- Wenn der Quelltext einzelne Phasen, Stages, Schritte oder Komponenten eines Prozesses individuell mit eigener Charakteristik beschreibt (z.B. affektiv/kognitiv/physisch pro Phase), plane separate Notes pro Phase statt einer aggregierten Listen-Note. Eine zusätzliche Übersichts/MoC-Note ist erlaubt, ersetzt aber NICHT die Einzelnotizen.

## Output — NUR dieses Format, kein erklärender Text, KEIN JSON:

source_title: Knapper Titel der Quelle
source_summary: 2 Sätze worum es insgesamt geht (einzeilig — keine Newlines)
<!--CONCEPT-->
title: Konzeptname
priority: high
chapter: Kapitel oder Abschnitt wo das Konzept hauptsächlich behandelt wird
action: create
extend_path:
category: conceptual
origin: primary
cited_authors:
<!--CONCEPT-->
title: Fremde Theorie kurz zitiert
priority: low
chapter: Background
action: create
extend_path:
category: conceptual
origin: secondary_mention
cited_authors: Smith, Jones
<!--END-->

**Format-Regeln (strikt):**
- Sentinels exakt `<!--CONCEPT-->` und `<!--END-->`. ALL_CAPS, kein Whitespace im Sentinel.
- `source_title` und `source_summary` als Header-Lines VOR dem ersten `<!--CONCEPT-->`. Beide einzeilig.
- Pro Konzept ein `<!--CONCEPT-->` Block mit **acht** Header-Lines (title, priority, chapter, action, extend_path, category, origin, cited_authors).
- `extend_path:` leer lassen wenn nicht zutreffend (entspricht null).
- `origin:` einer von `primary` | `extension` | `secondary_mention` (Pflicht).
- `cited_authors:` kommagetrennte Nachnamen bei secondary_mention/extension. Leer bei primary.
- `category:` einer von `architectural` | `operational` | `conceptual`:
  - `architectural` = Komponenten, Strukturen, Taxonomien, Modelle (Pass 1)
  - `conceptual` = Definitionen, Begriffe, Theorien (Pass 1)
  - `operational` = State/Memory, Evaluation, Error Handling, Security, Deployment (Pass 2)
- **EIN** finaler `<!--END-->` schließt den Output.

## Quellentext
{overview}
"""


def _bm25_rerank(concepts: list[str], query_text: str, top_n: int = 200) -> list[str]:
    """Sortiert Konzept-Titel per BM25 nach Relevanz zum Overview-Text.

    Fix für Lost-in-the-Middle (Liu et al. 2023): relevante Konzepte landen
    oben in der Liste → Planner sieht sie mit höherer Attention.
    Top-200 statt 1500 — alles darüber erhöht Rauschen ohne Mehrwert (Gemini-Review 2026-05-14).
    Fallback auf einfaches Slice bei fehlender rank_bm25-Dependency.
    """
    if not concepts:
        return []
    if not query_text or len(concepts) <= top_n:
        return concepts[:top_n]
    try:
        import numpy as np
        from rank_bm25 import BM25Okapi
        tokenized = [list(_tokens(c)) or ["_"] for c in concepts]
        bm25 = BM25Okapi(tokenized)
        query_toks = list(_tokens(query_text)) or ["_"]
        scores = np.array(bm25.get_scores(query_toks), dtype=float)
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        return [concepts[i] for i in ranked_idx]
    except Exception:
        return concepts[:top_n]


def run(overview: str, relevance_profile: dict,
        primary_authors: list[str] | None = None) -> ConceptPlan:
    # BM25-Reranking: relevante Konzepte zuerst, Top 200 statt 1500
    # (Fix 3 — Lost-in-the-Middle, Gemini/Nemotron-Review 2026-05-14).
    # Vorher [:1500] flach — Gemini-Finding G2 (2026-05-10) hatte Limit auf 1500
    # hochgezogen weil [:50] zu wenig war. Jetzt BM25-sortiert Top-200.
    all_existing = list(relevance_profile.get("existing_concepts", {}).keys())
    existing = _bm25_rerank(all_existing, overview, top_n=200)
    existing_str = "\n".join(f"- {c}" for c in existing) if existing else "(noch keine)"

    authors = primary_authors or []
    if authors:
        if len(authors) == 1:
            primary_authors_line = authors[0]
        else:
            primary_authors_line = ", ".join(authors[:-1]) + " & " + authors[-1]
        primary_authors_block = f"Primärautoren: {primary_authors_line}"
    else:
        primary_authors_line = "(unbekannt)"
        primary_authors_block = "Primärautoren: unbekannt — klassifiziere origin nach bestem Wissen aus dem Kontext."

    prompt = _PROMPT.format(
        existing_concepts=existing_str,
        overview=overview,
        primary_authors_line=primary_authors_line,
        primary_authors_block=primary_authors_block,
    )

    raw = call_claude(prompt, model=MODEL_PLANNER, agent="planner")
    data, parse_warnings = parse_planner_output(raw)
    if parse_warnings:
        import sys
        for w in parse_warnings:
            print(f"      [planner-warn] {w}", file=sys.stderr)

    concepts = [
        ConceptItem(
            title=c["title"],
            priority=c.get("priority", "medium"),
            chapter=c.get("chapter", ""),
            action=c.get("action", "create"),
            extend_path=c.get("extend_path"),
            category=c.get("category", "conceptual"),
            origin=c.get("origin", "primary"),
            cited_authors=c.get("cited_authors", []),
        )
        for c in data.get("concepts", [])
    ]

    # Kategorien-Verteilung loggen — macht operative Lücke sichtbar (Fix 1).
    if concepts:
        import sys
        cat_counts: dict[str, int] = {}
        for c in concepts:
            cat_counts[c.category] = cat_counts.get(c.category, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
        print(f"      [planner] Kategorien: {dist} (total={len(concepts)})", file=sys.stderr)

    return ConceptPlan(
        source_title=data.get("source_title", ""),
        source_summary=data.get("source_summary", ""),
        concepts=concepts,
    )




def filter_hallucinated(plan: ConceptPlan, full_text: str,
                        min_coverage: float = 0.5) -> tuple[ConceptPlan, list[str]]:
    """Verwirft Konzepte deren Titel-Tokens nur teilweise im PDF-Volltext vorkommen.

    Coverage-Filter: |Title-Tokens ∩ Text-Tokens| / |Title-Tokens| ≥ min_coverage

    Ergänzt durch Planner-Prompt-Instruktion (Self-Filter: action="skip" für
    konzepte die nur aus Trainingswissen bekannt sind). Kein domain-spezifischer
    Code-Filter — neutral für alle Themenbereiche.

    Beispiele (min_coverage=0.5):
    - "Maslow Bedürfnishierarchie" → Coverage 2/2 → kept
    - "Blockchain für Information Retrieval" → Coverage 1/3 < 0.5 → rejected

    Returns: (gefilterter Plan, Liste verworfener Konzept-Titel)
    """
    text_tokens = _tokens(full_text)
    full_text_lower = full_text.lower()
    rejected: list[str] = []
    kept: list[ConceptItem] = []
    for c in plan.concepts:
        title_tokens = _tokens(c.title)
        if not title_tokens:
            rejected.append(c.title)
            continue
        coverage = len(title_tokens & text_tokens) / len(title_tokens)
        if coverage < min_coverage:
            rejected.append(c.title)
            continue
        # Blacklist-Check: generische Einzel-Konzepte verwerfen (portiert aus extractive).
        # Normalisierung auf lowercase nötig da LLM Title-Case ausgibt.
        if c.title.strip().lower() in _GENERIC_BLACKLIST:
            rejected.append(c.title)
            continue
        kept.append(c)
    return ConceptPlan(
        source_title=plan.source_title,
        source_summary=plan.source_summary,
        concepts=kept,
    ), rejected
