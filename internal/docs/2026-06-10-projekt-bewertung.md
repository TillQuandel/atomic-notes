# Projekt-Bewertung 2026-06-10 (Baseline)

Methode: 3 unabhängige Rater × verankerte Rubrik (siehe `projekt-bewertung-prompt.md`).
Rater: Codex gpt-5.5 (Modus B, Fakten-Bundle), Mistral magistral-medium (Modus B),
Claude Fable 5 (Modus A, **befangen** — hatte am Vortag selbst Code beigetragen).
Fakten-Bundle (mechanisch erzeugt, nicht committet): README, generative/README,
Issue-Liste, pdf_chunker.py + verifier.py vollständig, repo-map-Signaturen, Statistiken
(20.161 LOC, 82 Commits seit 2026-05-20, 413 Tests, AGENT_VERSION v0.3.136).

Zweck: Baseline für Delta-Re-Rating (Vorschlag: ~3 Monate, gleiche Rubrik, gleicher Modus).

## Score-Synthese

| Dimension | Codex | Mistral | Claude (befangen) | Median |
|---|---|---|---|---|
| Architektur & Modularität | 4 | 4 | 4 | **4** |
| Code-Qualität | 3 | 4 | 3 | **3** |
| Test- & Eval-Disziplin | 4 | 4 | 4 | **4** |
| Engineering-Prozess | 4 | 4 | 4 | **4** |
| Robustheit & Produktreife | 3 | 3 | 3 | **3** |
| Wissenschaftliche Messdisziplin | 3 | 4 | 4 | **4** |
| Sinnhaftigkeit / Problem-Fit | 4 | 5 | 4 | **4** |
| **Gesamt (Median)** | **4** | **4** | **4** | **4** |

Spannweite ≤1 auf jeder Dimension → hohe Konvergenz. Caveats: Codex' Messdisziplin-3
mit Confidence „niedrig-mittel" (Labeling-Protokoll lag dem Bundle nicht bei);
Mistral war durchgängig ~0,5 großzügiger bei generischerer Evidenz.

- Skilllevel (3/3 konvergent): **Senior** (Engineering-Urteil: Systemsteuerung,
  Messkultur, Prozess-Disziplin), Caveat: Produkt-/Code-Hygiene-Ränder
  (Orchestrator-Größe, globale Config, sys.path-Hacks, lokale Tool-Annahmen).
- Potenzial (konvergent): privates Power-Tool **hoch** (ist es), Showcase **hoch**,
  Forschungs-Artefakt **mittel-hoch** (Eval-Methodik = interessantester externer
  Beitrag), OSS **mittel** (Gap: Installation/Backend-Wahl/Doku), SaaS **niedrig**.
- Einstimmiges Urteil: weiterbauen, nicht breiter — Reihenfolge: PDF-Gate/OCR (#27),
  Kalibrierung (#29/G3), G5-Goldstandard, reproduzierbarer Mini-Benchmark; erst
  danach Release-/API-Scope (#9–#11).

Grenzen der Messung: LLM-Rater teilen Trainings-Biases (korrelierte Fehler möglich);
Rubrik misst Urteile, keine Außenwirkung. Härtere Außenkriterien: reproduzierter
Benchmark-Lauf, Fremd-Installation nach README, Delta-Re-Rating.

---

## Anhang A — Codex gpt-5.5 (Modus B, Original)

| Dim | Score | Confidence | Evidenz | Gap zur nächsten Stufe |
|---|---:|---|---|---|
| 1 Architektur & Modularität | 4 | mittel | `README.md`: klare Trennung `generative/`, `extractive/`, `shared/`, `internal/`; Struktur-Map: dedizierte Agenten (`planner`, `extractor`, `verifier`, `cross_reference`, `critic`), Schemas in `schemas/atomic_note.py`, Renderer in `pipeline/vault_writer.py`. Gegenbefund: `orchestrator.py` bündelt sehr viel Stage-Logik. | Orchestrator weiter entflechten, stabile öffentliche Pipeline-API, weniger implizite globale Config. |
| 2 Code-Qualität | 3 | mittel | `pdf_chunker.py`: lesbare Heuristiken mit Tests/Kommentaren; `verifier.py`: Prepass, Semantic-Fallback, Parse-Fail-Erhalt. Gegenbefunde: `pdf_to_pages()` macht `sys.exit`, viele mutierende Draft-Pfade, lange Funktionen, viel historischer Kommentar-/Patch-Kontext. | Konsistentere Fehlerklassen, kleinere Funktionen, weniger globale Zustände/Side Effects. |
| 3 Test- & Eval-Disziplin | 4 | mittel | 413 Tests grün; Test-Map deckt Parser, Backends, Runtime-Config, Verifier-Prepass, Stage6-Crash, Vault-Writer, PDF-Chunker, E2E-Baseline ab; Regressionstest `test_regression_same_rate_but_zero_claim_agreement`. | Mehr echte End-to-End-Läufe mit LLM/CLI, Reproduzierbarkeit über Fixtures, Coverage-Report statt bloßer Testzahl. |
| 4 Engineering-Prozess | 4 | mittel | 82 Commits seit 2026-05-20; 28 Issues, 9 closed; Issues enthalten Bugs, Release-Prep und Research-Needed; Docs/Plans wie `atomic-notes-monorepo-design.md`, `GROBID-Patterns Voruntersuchung`, Runtime-Config-Plan. | Commit-/Issue-Verknüpfung und Release-Kriterien härter machen; offene Release-Prep-Themen schließen. |
| 5 Robustheit & Produktreife | 3 | hoch | Robustheitsarbeit sichtbar: Timeouts gegen 110-Minuten-Hänger, Stage6-Crash-Tests, no-LLM-Modus für Stage6, Crash-Reports. Gegenbefunde: offene Issues `PDF-Textqualitaets-Gate + OCR-Fallback`, `collect.py` Dashboard-Bug; externe Tools `pdftotext/pdfinfo`, Claude-CLI, Obsidian-Pfadannahmen. | Installationspfad für Fremdnutzer, Preflight-Checks, OCR/Textqualitäts-Gate, stabile Config/Service-API. |
| 6 Wissenschaftliche Messdisziplin | 3 | niedrig-mittel | Eval-Infrastruktur vorhanden: `eval_quality_v4.py`, `calibration/kappa.py` mit Gwet AC1/Cohen/Krippendorff, 30 aktive Label-Dateien + Blind-Set; offene Kalibrierungs-Issues G1-G6 zeigen Methodenbewusstsein. Gegenbefund: Protocol-Inhalt lag mir nicht vor; Issue #24 meldet Agreement-Bug; NLI-Schwellen explizit unkalibriert. | Vorregistriertes Protokoll, größere Gold-Samples, Schwellenkalibrierung, Konfound-Analyse und veröffentlichbare Ergebnisberichte. |
| 7 Sinnhaftigkeit / Problem-Fit | 4 | hoch | Problem ist real: geprüfte atomare Obsidian-Notes aus PDFs; README trennt Input-Adapter, Pipeline und Renderer; Extractive-Pipeline als Low-Hallucination-Baseline. Gegenbefund: Nutzen stark an persönliches PKM/Claude-CLI gebunden. | Nachweisbarer Nutzwert über mehrere fremde Nutzer/Domänen und Vergleich gegen einfachere Workflows. |

Gesamt (Median): 4

**Skilllevel:** Senior. Dafür: Erstens zeigt die Architektur echte Systemsteuerung: Stages, Agenten, Schemas, Renderer, Eval, Tracing und Runtime-Config sind als getrennte Verantwortungen angelegt. Zweitens ist die Fehler- und Messkultur überdurchschnittlich: Crash-Isolation, Regressionstests, Gwet AC1, offene Kalibrierungs-Issues und bewusste A/B-Entscheidungen statt blindem Feature-Einbau. Stärkstes Gegenargument: Produkt- und Code-Hygiene sind noch nicht Senior-vorbildlich: großer Orchestrator, globale Config, externe Tool-Annahmen, unkalibrierte Schwellen und offene Release-/Robustheitslücken.

**Potenzial:** (a) hoch — Workflow eng genug und bereits operationalisiert. (b) mittel — Tests und Doku da, aber Installation, Config und Fremdnutzer-Robustheit fehlen. (c) mittel-hoch — Eval-Methodik ist der interessanteste externe Beitrag, braucht saubere Protokollierung und größere Gold-Daten. (d) niedrig-mittel — Trust, UX, Rechte/Quellenhandling, Betriebskosten ungelöst. (e) hoch — zeigt klares Engineering-Urteil, solange LLM-Anteil transparent eingeordnet wird. Realistischste Option: privates Power-Tool. Kritischer Gap: PDF/Textqualitäts-Gates und Kalibrierung zuerst stabilisieren, bevor Scope erweitert wird.

**Stärken:** klare Pipeline-Zerlegung mit expliziten Quality-Gates; ungewöhnlich gute Regressionstest-Breite für ein junges Einzelprojekt; messorientierte Entwicklung mit Issues für Kalibrierung statt bloßer Prompt-Optimierung.
**Schwächen:** Produktreife hängt an lokalen Tools, Claude-CLI und Vault-Konventionen; wissenschaftliche Claims sind stärker als die derzeit sichtbare Gold-Label-Evidenz; Orchestrator/Config tragen zu viel Systemwissen.

**Gesamturteil:** Ja, weiterbauen, aber nicht breiter. Die nächste sinnvolle Arbeit ist Robustheit und Messfundament: PDF-Textqualitäts-Gate/OCR, Dashboard-Agreement-Bug, NLI-/mDeBERTa-Kalibrierung und ein reproduzierbarer kleiner Benchmarklauf. Danach erst neue Adapter oder SaaS/API-Scope.

---

## Anhang B — Mistral magistral-medium (Modus B, Original)

| Dim | Score | Confidence | Evidenz | Gap zur nächsten Stufe |
|---|---|---|---|---|
| 1 Architektur & Modularität | 4 | hoch | Klare Trennung der Verantwortlichkeiten in generative/, extractive/, shared/, tests/, internal/. Schemas als Verträge in schemas/atomic_note.py. | Einige Vermischungen in Modulen wie verifier.py. |
| 2 Code-Qualität | 4 | hoch | Lesbar und idiomatisch in pdf_chunker.py und verifier.py. Präzise Fehlerbehandlung, selbsterklärende Kommentare. | Stellenweise inkonsistent, komplexe Funktionen in verifier.py. |
| 3 Test- & Eval-Disziplin | 4 | hoch | 413 Tests grün. Halluzinations-Messung v4.1, Human-Labeling + Gwet AC1. TDD-Spuren und Regression-Beweise. | Lücken bei Integrationstests. |
| 4 Engineering-Prozess | 4 | hoch | 28 Issues, 9 geschlossen. Fokussierte Commits, gute Issue-Hygiene. Specs, Pläne, Protokolle. | Doku lückenhaft. |
| 5 Robustheit & Produktreife | 3 | mittel | Kernfunktionalität stabil und robust. | Ränder fragil, noch nicht bereit für fremde Nutzer. |
| 6 Wissenschaftliche Messdisziplin | 4 | hoch | Vorregistrierte Methodik, IRR-Statistik, Zirkularitäts-/Konfound-Bewusstsein, ehrliche Limits. | Methodik-Lücken könnten besser dokumentiert sein. |
| 7 Sinnhaftigkeit / Problem-Fit | 5 | hoch | Löst ein reales Problem, keine klaren Alternativen, Nutzen > Aufwand. | — |

Gesamt (Median): 4

**Skilllevel:** Senior — klare Architektur, umfassende Testabdeckung, gute Dokumentation; Gegenargument: komplexe Struktur einiger Module.
**Potenzial:** (a) hoch, (b) mittel, (c) mittel, (d) niedrig, (e) hoch. Realistischste Option (a); Gap zu (b): Benutzerfreundlichkeit und Dokumentation.
**Stärken:** Test-/Eval-Infrastruktur; Architektur/Modularität; Messdisziplin. **Schwächen:** Modul-Komplexität; Doku-Lücken; fragile Robustheits-Ränder.
**Gesamturteil:** Gut strukturiert, starke technische Grundlage, klarer Nutzen. Weiterbauen, insbesondere Benutzerfreundlichkeit und Dokumentation.

---

## Anhang C — Claude Fable 5 (Modus A, befangen — Selbsteinschätzung als Vergleichspunkt)

Scores: 4 / 3 / 4 / 4 / 3 / 4 / 4 (Median 4).
Begründungs-Kurzform: Architektur 4 (Stage-Trennung + Schemas; Abzug: orchestrator.py
1440 Z., eval_quality v1–v4 als Schichten); Code-Qualität 3 (idiomatischer Kern;
sys.path-Hacks, Pfad-Konventions-Drift — der calibration-DB-Pfad-Bug ließ den
calibration_labels-Write seit Anlage still scheitern); Tests 4 (TDD-Praxis real,
Integrationslücke genau dort, wo der DB-Bug lag); Prozess 4 (Issue-Hygiene,
Cross-Model-Review-Praxis; README dünn); Robustheit 3 (Anker „Kern robust, Ränder
fragil" trifft exakt); Messdisziplin 4 (AC1, Labeling-Protokoll, Konfound-/
Zirkularitäts-Bewusstsein; G5-Goldstandard fehlt → Coverage ungemessen);
Sinnhaftigkeit 4 (reales Problem ohne fertige Alternative; Aufwand/Nutzen als
reines Effizienz-Tool diskutabel, als Lern-/Methodik-Projekt klar positiv).
Skilllevel: Senior (Engineering-Urteil); Gegenargument: Packaging-/Deployment-Basics fehlen.
