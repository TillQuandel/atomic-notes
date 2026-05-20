# Atomic-Agent — Projekt-Status

> Schnell-Übersicht für den Re-Entry. Letzte Aktualisierung: 2026-05-19.

## Version & Stand

**v0.3.40** — produktionsreif, aktiv in Entwicklung

Seit v0.3.6 erledigt: ER-Threshold 0.974 → 0.985, Chunk-Cap (MAX_CHUNKS_SHORT_DOC=10), Pre-Merge Source-Check (MVP via XSOURCE-MERGE-Stub), Merge-Stub-Tracking, **Fix 1 Category-aware Planner** (ConceptItem.category architectural/operational/conceptual + Logging-Verteilung), **Window-Builder Option D** (Sliding-Window Co-Occurrence Ranking ersetzt expansive ±cluster-Logik — fixt Substanz-Kapitel-Miss bei generischen Tokens).

**v0.3.38–v0.3.40 (2026-05-19): Origin-Field — Secondary Citation Handling**
ConceptItem erweitert um `origin: str` (primary/extension/secondary_mention) + `cited_authors: list[str]`. Planner klassifiziert wessen Konzept es ist, Orchestrator filtert `secondary_mention` vor Extractor-Call. Related Mentions als Kontext-Liste an Extractor weitergegeben. Begleit-Fix: DB-Write-Bug (17 vs 14 Spalten), Dashboard-Sentinel-Filter. 6 Commits, 252 Tests grün.

Messbares Ergebnis (Jaiswal 2020 RL-Paper): 7→5 Notes, 671s→326s, 2.5M→932k Tokens.

```
python orchestrator.py --source <pdf> --dry-run    # Test ohne Schreiben
python orchestrator.py --source <pdf>              # Live-Run → 00-inbox/
python orchestrator.py --source <pdf> --no-llm     # FOSS-only (Stage 6)
```

## Pipeline (7 Stages)

| Stage | Was | Status |
|---|---|---|
| 1 | PDF → Text + Chunks (Kapitel-Split) | stabil |
| 2 | Context-Builder: Vault-Scan → Relevanz-Profil | stabil |
| 3 | Quality-Agent: CrossRef/OpenAlex → QualityReport | stabil |
| 4 | Planner: overview → ConceptPlan (Opus) | operative Konzepte oft vergessen — Fix in Planung |
| 4.5 | Background-Extractor: Trainingswissen pro Konzept | DISABLED (senkt Vault-Quote) |
| 5 | Extractor × N: Konzept → Draft-Note (Opus, parallel) | stabil |
| 6 | Verifier + CrossRef + Critic pro Note (Haiku, parallel) | stabil, Rate-Limit-Retry |
| 7 | Vault-Writer → 00-inbox/ | stabil |

## Bekannte Probleme

| Problem | Status |
|---|---|
| Planner übersieht operative Konzepte (Evals, State, Error Handling) | DONE v0.3.14 — Pass 1/2 + category-Output, Verteilungs-Logging |
| Background-Extractor senkt Vault-Quote durch fehlende Anker | ENABLE_BACKGROUND_EXTRACTOR=0 (Default) |
| Schlebbe-Baseline-Test >30 min | Timeout auf 60 min erhöht |

## Offene Improvements (priorisiert nach Cross-Model-Review)

**Fix 3 — BM25-Reordering** (DONE 2026-05-14)
- existing_concepts-Liste per BM25 gegen Overview sortieren, Top 200 statt 1500
- Reduziert Lost-in-the-Middle beim Planner

**Fix 1 — Category-aware Planner** (DONE 2026-05-17, v0.3.14)
- Pass 1/Pass 2 im Prompt + `category:` Header pro Konzept im Output
- `ConceptItem.category`: architectural / operational / conceptual
- Verteilungs-Logging im Planner-Run macht operative Lücke sichtbar
- Tests: 5 neue Parser-Tests, 196 fast-Tests grün

**Fix 2 — LLM-Dedup für ambiguous Zone** (später, nur bei Bedarf)
- Paare Cosine 0.3-0.7: Batch-LLM (10 Paare/Prompt, gecacht)
- Schließt konzeptuelle Duplikate die ER verpasst

## Faithfulness-Eval v4.1 (Stand 2026-05-17)

Eigenständiger Strang neben der Extraktions-Pipeline. Ersetzt v1/v3.x als Hal-Rate-Messung. Details: [[Eval-v4-Stand]].

**Architektur:** `98-system/scripts/lib/decision_engine/` (domain-agnostic Rule-Pipeline, 4067 Property-Tests grün) + `eval_quality_v4.py` als Caller. 3-Phasen-Trennung Terminal → Mutator → Decider.

**Validierung 2026-05-17:** 4 manuelle Stichproben vs. Gemini-Ground-Truth — v4.1-Hal korreliert (10% wo Gemini 80-90% confirmed, 0% wo Gemini 100% confirmed). Davor v1: 80-92% Hal (kaputt).

**Gemini-Audit v4.1:** 0 Critical, 1 Suspicious (SYSTEM_LABELS-Vollständigkeit ohne static check), 2 Nitpicks.

**Quality-History:** `.cache/quality_history.jsonl` leer/bereit für v4.1-Baseline. Vor-v4-Werte in `quality_history_pre_v4.jsonl.bak` archiviert.

## Calibration-Workflow (Stand 2026-05-17)

Inter-Rater-Agreement Human vs. Pipeline auf Claim-Ebene (Cohen κ, Gwet AC1, Krippendorff α). Pre-Registration: Hybrid C++ nach [[Faithfulness-Annotation-Protokoll]], Härtungen #1-#5.

**Code-Pipeline:** `calibration/sample.py` → `build_labels.py` → manuell labeln → `collect.py` → `adversarial.py` → `kappa.py`. Alle Scripts vorhanden, kappa.py neu (2026-05-17).

**Phase 1 — Datenstand:**
- 30 v4.1-Evals durch (run.py, 25 + 5 nach Rate-Limit-Reset)
- 359 Claims total (Ø ~12/Note)
- 35 Label-Files unter `calibration/labels-active/` (30 Hybrid) + `calibration/labels-active-blind/` (5 Blind, Härtung #3)
- **Alle 30 Hybrid-Notes Status `to-label`** — manuelles Labeling offen (~3-5h)

**Nach Labeln:** `collect.py` → `adversarial.py` (Gemini-Perturbation) → `kappa.py`. Bei AC1 ≥ 0.70 → Phase 2 (Qualitätsziele auf v4.1-Skala neu formulieren in [[Atomic-Agent-Qualitaetsziele]]).

## Letzte Eval-Ergebnisse

### Baseline (pre-v0.3.6)
| PDF | Run | Vault/Total | Datum |
|---|---|---|---|
| OpenAI Guide (Practical Agents) | v0.3.6 Run4 | 8/13 | 2026-05-14 |
| Forster 2017 (Workplace IL) | v0.3.6 | 7/11 (64%) | 2026-05-13 |
| Kuhlthau ISP | v0.3.6-pre-verifier | 10/12 | 2026-05-15 |
| Bates 2017 | v0.3.6-pre-verifier | 1/6 | 2026-05-15 |

### Session 2026-05-15 (mit Artifact-Detector + Planner-Self-Filter)

_Token-Tracking ab v0.3.7 — alle Runs dieser Session liefen vor dem Fix → Zeit/Tokens n/a._

| PDF | Vault/Total | Vault-% | Zeit | Tokens | Anmerkung |
|---|---|---|---|---|---|
| Porst 2014 (Fragebogen) | 7/8 | 88% | n/a | n/a | Beste Quote; 2 Konzepte korrekt verworfen |
| Petersen 2014 | 6/7 | 86% | n/a | n/a | Starke Quote; Fragebogenmethodik |
| Hiatt ADKAR 2006 | 3/4 | 75% | n/a | n/a | Domain-Neutralität ✓ (Change Management) |
| Beutelspacher 2014 (IK-Messung) | 6/12 | 50% | n/a | n/a | 3 MERGE-Stubs; direkt BA-relevant |
| Lloyd 2010 (IL Landscapes) | 3/7 | 43% | n/a | n/a | 3 MERGE-Stubs; domain-neutral ✓ |
| Venkatesh UTAUT 2003 | 2/5 | 40% | n/a | n/a | Future-Self "neuer" Gate blockiert 3 Notes |
| Bühner 2011 (Fragebogenkonstruktion) | 4/11 | 36% | n/a | n/a | Nur 25-S.-Excerpt, schlechte Extraktion |
| Bates 2017 (mit Fixes) | 1/4 | 25% | n/a | n/a | Ellis-Artefakt weg; 2 MERGE-Stubs korrekt |
| Beutelspacher 2022 | — | — | — | — | Retry ausstehend (Timeout im Batch) |

**Lektion:** Max. 2-3 parallele Runs bei neuen PDFs (cache-miss) — mehr → Rate-Limit-Timeouts → Score 0/5.

## Schlüssel-Dateien

```
orchestrator.py             Einstiegspunkt + Pipeline
config.py                   Alle Flags, Thresholds, Feature-Flags
agents/planner.py           Stage 4 — Konzept-Plan
agents/extractor.py         Stage 5 — Draft-Note
agents/verifier.py          Stage 6a — Anker-Verifikation
agents/critic.py            Stage 6c — Score 0-5
pipeline/pdf_chunker.py     extract_overview() — Planner-Input
pipeline/vault_writer.py    Stage 7 — Rendering + Routing
tests/test_e2e_baseline.py  Slow-Tests: pytest -m slow -k schlebbe
eval_quality_v4.py          Faithfulness-Eval (v4.1) — Caller für decision_engine
calibration/run.py          v4.1-Eval-Batch über Sample
calibration/build_labels.py Hybrid- + Blind-Label-MDs für manuelles Labeln
calibration/collect.py      Labels einsammeln → labels.jsonl
calibration/adversarial.py  Gemini-Perturbation für Robustheits-Test
calibration/kappa.py        Inter-Rater-Agreement (Cohen κ / Gwet AC1 / Krippendorff α)
../lib/decision_engine/     Domain-agnostic Rule-Pipeline (Eval-Decisions)
```

## Feature-Flags (config.py)

| Flag | Default | Bedeutung |
|---|---|---|
| ENABLE_BACKGROUND_EXTRACTOR | False | Stage-0.5, experimentell |
| ENABLE_ENTITY_RESOLUTION | True | Embedding-Cluster-Dedup |
| ENABLE_NLI_VALIDATION | False | DeBERTa CrossRef-Validation |
| ENABLE_LLM | True | False = FOSS-only Stage 6 |

## Qualitätsziele (eval_dashboard.py — THRESH_*)

Quellen: Vectara Hallucination Leaderboard Mai 2026, RAGAS-Benchmarks, Industry-Werte für Knowledge Extraction.

| Metrik | Jetzt (Ø) | Kurzfristig | Langfristig | Richtung |
|---|---|---|---|---|
| **Fehlerquote** (Halluzinationsrate) | ~20 % | < 15 % | < 5 % | ↓ |
| **Inbox-Rate** (nicht akzeptiert) | ~30 % | < 20 % | < 10 % | ↓ |
| **Abdeckung** (Coverage factual) | ~35 % | > 50 % | > 80 % | ↑ |

Akzeptanzrate = 100 % − Inbox-Rate (Langzeitziel: > 90 %).

**Referenz:** Claude Sonnet-4.6 Baseline = 10.6 % Fehlerquote (Vectara Leaderboard Mai 2026, Summarization).
Gemessene ~20 % liegt darüber → Cross-Language-Verluste (DE Note ← EN PDF), nicht nur Modell-Problem.

**Leitprinzip:** Ziel ist vollständige Wissensextraktion, nicht Metrik-Optimierung. Abdeckung ist primäre Metrik, Akzeptanzrate sekundär.

**Merge-Stubs zählen als Erfolg** (nicht Inbox) — derzeit nicht separat erfasst → Mess-Artefakt.

**Bugs aus Cross-Model-Review Mai 2026 — Status:**
- ER-Threshold 0.974 → 0.985 (DONE in v0.3.13, `ER_BODY_COSINE_THRESHOLD` in config.py)
- Chunk-Deckel (DONE, `MAX_CHUNKS_SHORT_DOC=10` für Docs <50 Seiten)
- Pre-Merge Source-Check MVP (DONE, XSOURCE-MERGE-Stub-Prefix bei Quellen-Konflikt in vault_writer.py:631-643)
- Voller Pre-Merge LLM-Validation-Call (TODO v28, MVP-Source-Check ersetzt nur Trivial-Fall)

**Offen:** Cross-Language-Threshold unkalibriert · Abdeckungs-Ziel > 80 % nicht extern validiert.

**Schwellen anpassen:** `THRESH_ACCEPT`, `THRESH_HALL`, `THRESH_COV` oben in `eval_dashboard.py`.
**Dashboard:** `python eval_dashboard.py` → `.cache/eval/dashboard.html`

## Eval-Tools

| Script | Zweck | Usage |
|---|---|---|
| eval_quality.py | Deterministische Halluzinations-Messung (PyMuPDF + Fuzzy + Semantic) | python eval_quality.py --note <note.md> --pdf <source.pdf> --save |
| eval_paired.py | Version-Vergleich A vs B | python eval_paired.py |
| eval_repeat.py | Varianz über mehrere Runs | python eval_repeat.py |

**Limitierung eval_quality:** Cross-Language-Threshold unkalibriert (DE Note ← EN PDF).
Absolut-Werte sind Annäherungen; Versions-Vergleiche sind valide wenn Threshold konstant.

## Design-Prinzipien (nicht verhandelbar)

| Prinzip | Bedeutung |
|---|---|
| **Domain-Neutralität** | Kein Code darf domain-spezifische Wortlisten, Themenfelder oder Sprach-Annahmen hardcoden. Filter und Heuristiken müssen für IBI, Psychologie, Informatik, Medizin etc. gleich funktionieren. Domain-Wissen gehört in den Prompt, nicht in den Code. |
| **Downstream-Safety-Net** | Jede Filterebene hat ein Netz dahinter. Planner-Self-Filter → filter_hallucinated (Coverage) → _drop_artifacts → hard-gate fail (Critic). Kein einzelner Filter muss perfekt sein. |
| **Prompt vor Code** | Wenn ein LLM-Agent ein Problem verursacht (Halluzination, falsche Klassifikation), ist der Prompt die erste Lösung. Code-Filter nur für deterministisch erkennbare Fälle (Absence-Phrases, leere Outputs). |

## Quellen

*Eigene Erfahrung (2026-05-15)*
