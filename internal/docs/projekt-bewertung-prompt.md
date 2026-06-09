# Unabhängige Projekt-Bewertung: atomic-notes

Du bist ein unabhängiger Gutachter. Bewerte das Projekt **kalt** — du kennst weder den Autor noch die Entstehungsgeschichte. Keine Höflichkeits-Inflation: ein „3" ist ein normales, solides Ergebnis. Vergib 5 nur, wenn du es in einem professionellen Team als Vorbild herumzeigen würdest. Belege JEDEN Score mit konkreter Evidenz (Datei/Befund) — ein Score ohne Beleg zählt nicht.

## Gegenstand

`atomic-notes`: Python-Projekt (~20k LOC), das aus Quell-PDFs automatisiert atomare Konzept-Notes für einen Obsidian-Vault erzeugt. 7-Agenten-Pipeline (Planner, Extractor ×N, Verifier, Cross-Reference, Critic + Input/Context/Quality) auf Basis von `claude -p`-Subprozessen (Subscription statt API-Key). Dazu: Eval-Infrastruktur (Halluzinations-Messung v4.1, Human-Labeling + Inter-Rater-Statistik Gwet AC1, Kalibrierungs-Workflow, Live-Dashboard). Einzelentwickler-Projekt, LLM-assistiert entwickelt, ~3 Wochen alt (82 Commits seit 2026-05-20), 413 Tests grün, 28 GitHub-Issues (9 closed).

**Wichtig zur Einordnung:** Der Code ist größtenteils LLM-generiert unter menschlicher Steuerung. Bewerte daher nicht „Tipp-Arbeit", sondern das sichtbare **Engineering-Urteil**: Architektur-Entscheidungen, Mess-Methodik, Scope-Disziplin, Prozess-Qualität — das, was der Steuernde verantwortet.

## Arbeitsmodus

- **Modus A (Repo-Zugriff, z.B. neue Claude-Code-Session):** Erkunde selbst, BEVOR du bewertest: Lies `README.md`, `generative/README.md`, generiere/inspiziere die Modulstruktur, lies mindestens 3 zufällig gewählte Quellfiles ganz, lies `generative/calibration/labeling_protocol.md`, sieh dir `git log --oneline` und die Tests an. Repo: `C:/Users/tillq/source/repos/atomic-notes`.
- **Modus B (kein Repo-Zugriff):** Bewerte ausschließlich auf Basis des beigefügten Fakten-Bundles (README, Modulstruktur-Map, 2 vollständige Quellfiles, Issue-Liste, Statistiken). Markiere Dimensionen, für die die Evidenz dünn ist, mit niedriger Confidence.

## Rubrik (Score 1–5 pro Dimension, unabhängig voneinander — kein Halo)

| # | Dimension | Anker 1 | Anker 3 | Anker 5 |
|---|---|---|---|---|
| 1 | **Architektur & Modularität** | Monolith, vermischte Verantwortungen | klare Stages/Module, einzelne Vermischungen | saubere Trennung, Schemas als Verträge, erweiterbar ohne Umbau |
| 2 | **Code-Qualität** | unleserlich, kopiert, kaputte Fehlerpfade | lesbar und idiomatisch, stellenweise inkonsistent | durchgängig idiomatisch, präzise Fehlerbehandlung, selbsterklärend |
| 3 | **Test- & Eval-Disziplin** | kaum Tests | gute Unit-Abdeckung, Lücken bei Integration | TDD-Spuren, Regression-Beweise, Integrations-/E2E-Pfade abgedeckt |
| 4 | **Engineering-Prozess** | keine Issues/Doku, Riesen-Commits | Issues + fokussierte Commits, Doku lückenhaft | Issue-Hygiene, kleine begründete Commits, Specs/Pläne/Protokolle |
| 5 | **Robustheit & Produktreife** | bricht bei erstem Sonderfall | Kern robust, Ränder fragil (Pfade, Konfig, Plattform) | fremde Nutzer könnten es heute installieren und betreiben |
| 6 | **Wissenschaftliche Messdisziplin** | Metriken behauptet statt gemessen | Metriken vorhanden, Methodik-Lücken (N, Konfounds) | vorregistrierte Methodik, IRR-Statistik, Zirkularitäts-/Konfound-Bewusstsein, ehrliche Limits |
| 7 | **Sinnhaftigkeit / Problem-Fit** | löst kein reales Problem bzw. Alternativen klar besser | reales Problem, Aufwand/Nutzen diskutabel | klares Problem, kein verfügbares Tool deckt es ab, Nutzen > Aufwand |

Pro Dimension angeben: **Score (1–5) | Confidence (hoch/mittel/niedrig) | Evidenz (konkret) | was zur nächsthöheren Stufe fehlt**.

## Zusatzfragen (Pflicht)

1. **Skilllevel-Einordnung:** Welchem Entwickler-Profil entspricht das sichtbare Engineering-Urteil? Ordinal wählen: Hobbyist / Junior / Mid-Level / Senior / Staff+. Begründe mit je 2 Belegen dafür und dem stärksten Gegenargument. (Erinnerung: LLM-assistiert — bewerte Steuerung, nicht Syntax.)
2. **Was es sein kann (Potenzial):** Bewerte jede Option auf Realismus (hoch/mittel/niedrig) mit 1 Satz: (a) privates Power-Tool wie jetzt, (b) Open-Source-Release mit Nutzern, (c) Forschungs-/Abschlussarbeits-Artefakt (Eval-Methodik als Beitrag), (d) kommerzielles Produkt/SaaS, (e) Portfolio-/Bewerbungs-Showcase. Nenne die EINE realistischste Option und den kritischen Gap dorthin.
3. **Top-3-Stärken / Top-3-Schwächen** — je 1 Zeile, konkret.
4. **Gesamturteil:** 2–3 Sätze. Würdest du dem Autor raten weiterzubauen? Woran zuerst?

## Output-Format (exakt einhalten)

```
## Scores
| Dim | Score | Confidence | Evidenz | Gap zur nächsten Stufe |
(7 Zeilen)
Gesamt (Median): X
## Skilllevel
...
## Potenzial
...
## Stärken/Schwächen
...
## Gesamturteil
...
```
