# Plan: Claim-level Faithfulness-Gate (Phrasen-Attribution)

> Stand 2026-06-24, Cross-Model-überarbeitet (Codex Repo-read-only + Mistral). Status: geplant, nicht umgesetzt. M2-Roadmap ("trustworthy output").

## Problem
Der Verifier prüft nur Anker-**Zitat**-Existenz auf Seitenebene — nicht, ob jedes Detail einer **Paraphrase** gedeckt ist. Entgangene Fehler (Hrastinski-Lauf 2026-06-24, von Qwen gefangen, Pipeline-Eval meldete hall=0,000):
1. **Fehlattribution** von Sekundärzitaten (Aussage dem falschen zitierten Autor zugeordnet).
2. **Extrapolation** (plausibles Detail ergänzt, das nicht im Quelltext steht).

Beide sind LLM-Compliance-Verstöße gegen einen Extractor-Prompt, der sie bereits verbietet — kein deterministischer Bug. Forschungsbasis: [[Phrasen-Attribution-schlägt-Claim-Dekomposition]] (deep-research-lite, 2026-06-24).

## Ziel / Akzeptanzkriterium
Nachgelagertes, **seiteneffekt-freies** Gate (kein Body-Edit, analog #8 Body-Redundanz-Flag): pro High-Risk-Claim die stützende/widersprechende Quellphrase lokalisieren + Entailment prüfen. Nicht gedeckt → Routing Inbox + `quality_flag`.

Konkrete Metriken (statt Platzhalter): High-Risk-Recall priorisieren (beide Hrastinski-Fälle müssen gefangen werden); FPR auf sauberen Notes messen (Ziel < ~10 %, sonst Inbox-Flut); "abstain/kein Quellfenster" separat zählen (nicht als Fail).

## Architektur (dreistufig)
Kernbefund der Recherche: der Attribution-Schritt ist der Hebel, nicht die Dekomposition (CLATTER +2,29; reine FactScore/SAFE-Zerlegung marginal/negativ).

### 0. Page-Index (größte v1-Lücke — beide Reviewer HIGH)
Vor dem Gate: Page-Index aus `full_text` bauen (`[S. N]` → Seitentext-Mapping; selber Marker-Mechanismus wie der Anker-Fix in PR #67). Pro Claim die **echte verankerte Seite + Nachbarseite** als Quellfenster — NICHT das gekürzte `concept_text_window` (8000-char-Top-Fenster), sonst ist ein Claim-Fail evtl. nur ein Retrieval-Fail.

### 1. Claim-Dekomposition (MVP: nur High-Risk)
`decompose_claims(body) -> [Claim]` mit reichem Output: `text, anchor_page, anchor_span, risk_type, is_quote`. MVP-Schnitt: zuerst nur **High-Risk-Claims** — Autorzuschreibungen / `zit. n.`-Muster, Zahlen, Vergleiche, Kausalwörter. Direktzitate (Block-Quotes) sind schon anker-verifiziert → ausnehmen. Volle Satz-für-Satz-Dekomposition als spätere Ausbaustufe.

### 2. Attribution + Entailment (zweistufig — NLI nicht allein)
- **2a Attribution-Heuristik (Misattribution):** Autor/Jahr/`zit. n.`-Muster im Claim erkennen → prüfen, ob dieselbe Relation (Autor↔Aussage) im Seitenfenster steht. NLI allein zu subtil (Recherche-Caveat: ROC-AUC fällt auf ~74 % bei full-vs-partial-support).
- **2b Entailment (Extrapolation):** stützende Quellphrase via MiniLM-Retrieval Top-k Sätze → **mDeBERTa/XNLI**-Entailment (nicht `nli-deberta-v3-small` — falsch für DE-Note/EN-Quelle; mDeBERTa bereits in config). Über **shared NLI-Adapter** (Batching, Label-Mapping entail/neutral/contra) — existiert noch nicht (nur Config-Schalter + CrossRef-interner Helper) → sauber herauslösen.

### 3. Gate-Logik (δ-Anteil maskiert Einzelfehler — Codex MED)
NICHT reiner δ-Anteil: **jeder ungedeckte HIGH-RISK-Claim → harter Fail** (1/12 Fehlattribution muss blocken — genau der Hrastinski-Fall); δ-Anteil nur als Zusatz für Low-Risk-Rauschen.

## Eingriffspunkt (Codex HIGH)
**Nach** dem finalen Critic UND im **Refine-Pfad** (`orchestrator.py:666` ruft Verifier/CrossRef/Critic erneut auf) — sonst überschreibt der Critic das Signal. Gate setzt `faithfulness_fail`, das über `hard_gates_pass`/Writer-Routing wirkt (nicht nur `quality_flag`) und Hub-Ausnahmen überstimmt.

## Bewusst NICHT (Scope)
Kein Body-Rewrite; kein reiner Prompt-Eingriff als "Fix" (schwächster Hebel; Prompt verbietet die Fehler bereits); kein reines LLM-as-Judge (Recall 16–17 %); keine Cross-Lingualität neu erfinden (mDeBERTa deckt ab); MVP zuerst nur High-Risk-Claims.

## Risiken
- Gold-Set-Kalibrierung bei wenig realen Fällen → FPR-Risiko (Mistral HIGH). Gold-Set ~30–50 gelabelte Claims, inkl. der 2 Hrastinski-Fälle.
- NLI-Granularität bei subtiler Misattribution begrenzt → daher Attribution-Heuristik als 1. Filter.
- CLATTER-Evidenz ist Einzelstudie → Richtung, nicht Beweis.
- M2-Kaliber; TDD nur partiell (deterministisch: Page-Index, Dekomposition, Attribution-Heuristik, Adapter-Mechanik; nicht: NLI-Urteilsqualität → Gold-Set).

## Umsetzungsschritte
1. Gold-Set (~30–50 Claims) inkl. Hrastinski-Fälle.
2. `build_page_index(full_text)` + Claim→Seite-Mapping (TDD, deterministisch).
3. `decompose_claims(body)` mit risk_type/anchor_span/is_quote (TDD).
4. Shared NLI-Adapter (mDeBERTa/XNLI) + Attribution-Heuristik (TDD auf Mechanik).
5. Gate-Logik (any-high-risk-fail + δ-Low-Risk) + Kalibrierung gegen Gold-Set.
6. Orchestrator-Verdrahtung nach finalem Critic + Refine-Pfad; Routing über hard_gates_pass.
7. Cross-Model-Review + kanonische Suite + E2E Hrastinski (beide Fehler müssen flaggen) + FPR-Kontrolllauf auf sauberen Notes.

## Review-Historie
Cross-Model-Review 2026-06-24 (Codex + Mistral): Gesamturteil beider "tragfähig, Richtung stimmt, kein Pivot". 8 Findings (Page-Index, Eingriffspunkt nach Critic, NLI-Adapter/mDeBERTa, any-high-risk-Hard-Gate, Attribution-Heuristik, Routing über hard_gates_pass, Kalibrierung/Metriken, MVP-Schnitt) — alle in diese Fassung eingearbeitet.
