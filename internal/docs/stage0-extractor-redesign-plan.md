# Stage-0-Extractor — Re-Design-Plan für belastbares A/B (Issue #3)

**Status:** AUSGEFÜHRT 2026-06-03. Follow-up-Entscheidung nach Qwen 3.7 Max + Claude:
**keinen pdfplumber-Adapter bauen.** AK (N≥10) nicht erreicht — Subscription-5h-Session-Limit
(~8 Läufe/Fenster). Hall%-Zahlen trennen die Tools nicht (N=1, Rauschen). Der relevante
Vergleich ist `pdftotext` vs. pdfplumber, nicht pdfplumber vs. GROBID; pdfplumber zeigte auf
dem zweispaltigen Bates-PDF eine Grounding-Regression (`Thelibraryisagrowing`, 5784 vs. 8846 W).
`pdftotext` bleibt Stage-0-Baseline. GROBID nur für #13/#14. Volldoku: Wissenspool
`[[Atomic-Agent-Pipeline]]` §Stand 2026-06-03, Issue #3.

## Warum dieser Plan existiert
Eine erste A/B-Voruntersuchung (2026-06-02, Bates + Porst-Auszug, je N=1–2) zeigte robust:
**strukturierte/gesäuberte Extraktion (GROBID-Body, pdfplumber) senkt die Halluzinationsrate
deutlich gegenüber rohem `pdftotext`** (Porst Ø 14,7 % → 6–8 %). Die *Tool-Wahl* (#5 vs #6) und der
*Mechanismus* (sauberer Text vs. bessere Segmentierung) konnten NICHT geklärt werden — ein
Cross-Model-Review (Codex/gpt-5.5) deckte vier Confounds auf. Dieser Plan behebt sie.

Befund-SSoT: Wissenspool `[[Atomic-Agent-Pipeline]]` §Stand 2026-06-03. GitHub: Issue #3 geschlossen; Reopen nur mit neuen Vergleichsdaten gegen `pdftotext`.

## Akzeptanzkriterium
Eine **statistisch trennscharfe** Aussage zu: (a) #5 vs #6 vs Status-quo bei Halluzination UND
Quell-Recall; (b) ob der Gewinn von Text-Bereinigung oder Segmentierung kommt.

## Confounds, die das Design beheben muss (aus Codex-Review)
1. **Mechanismus nicht isoliert (HIGH):** `extract_overview()` ruft `split_by_chapters()`. Wird
   letzteres global gepatcht, bekommt der **Planner-Overview** in jedem strukturierten Arm die
   Sections — unabhängig vom `chunk-cap`. „Segmentierung egal" wurde so nie getestet.
2. **pdfplumber-Arm inkonsistent (HIGH):** Frontmatter-Strip nur auf Text-, nicht auf Chunk-Pfad.
3. **Metrik verzerrt (HIGH):** Stage-8 wertet nur auto-written Notes; unterschiedliche Note-Zahlen
   (1 vs 5) machen Ø-Halluzination unvergleichbar.
4. **N zu klein (HIGH):** pro-Note-Halluzination streut empirisch 0–33 % → N=1–2 nicht trennscharf.

## Schritte

### 0. Phoenix-Tracing AN (Pflicht — Lehre aus der ersten Runde)
Bei JEDEM Re-Design-Lauf `ATOMIC_AGENT_TRACING=phoenix` setzen + Phoenix-Server (`.venv-phoenix`,
localhost:6006) starten. Grund: Run-Traces speichern nur `prompt_hash`, nicht den Input. Phoenix
speichert Prompt-Input + Output pro Call → der **Planner-Overview pro Arm** wird inspizierbar, was
Confound #1 erst direkt verifizierbar macht (Arm A vs B: sah der Planner verschiedene Inputs?).

### 1. Mechanismus isolieren (faktorielles Design)
Zwei Faktoren getrennt variieren statt gebündelt:
- **Text:** roh (`pdftotext`) vs. body-clean (GROBID-Body / pdfplumber)
- **Segmentierung:** Word-Count vs. echte Sections
4 Zellen statt 3 Arme. `extract_overview` muss im A/B die **gleiche** Overview-Quelle bekommen wie
der Chunk-Pfad — sonst leckt Segmentierung in den Planner. Konkret: Patch so umbauen, dass Text-
und Overview-/Chunk-Quelle pro Zelle konsistent gesetzt werden (nicht nur `split_by_chapters`
global ersetzen).

### 2. Fairness-Paritäten
- frontmatter-strip in BEIDEN Pfaden (Text + Chunks) jedes Extractors.
- `[S. N]`-Marker-Validierung pro Arm: Marker-Dichte, fehlende Seiten, monotone Folge,
  Anchor-Trefferquote — sicherstellen, dass der Verifier in allen Armen gleich arbeiten kann.

### 3. Planner-Varianz neutralisieren
Stage 4 (ConceptPlan) EINMAL einfrieren und per `--save-drafts`/`--load-drafts` in alle Arme
identisch einspeisen. So vergleicht das A/B nur die Extraktions-/Verifikations-Qualität, nicht die
zufällige Konzept-Auswahl (Note-Zahl schwankte 1–5 bei identischer Config).

### 4. Metrik erweitern
Nicht nur Per-Note-Halluzination der auto-written Notes. Zusätzlich reporten:
- **Quell-Recall/Coverage:** abgedeckte Quellkapitel / faktische Treffer pro PDF-Seite.
- getrennt: geplante Konzepte, erzeugte Drafts, auto-written Notes, Inbox-Notes.
- paired pro eingefrorenem Konzept statt aggregierter Ø.

### 5. Statistik
N≥10 pro Zelle, **paired/ABBA**-Reihenfolge gegen Drift. Bootstrap-CI über Notes×Runs statt
Punkt-Ø. Worst-Case (gedroppte/Inbox-Notes) einbeziehen.

## Dokumenttyp-Vorbehalt
GROBID war bei Bates (12 S., referenz-lastig) SCHLECHTER (1 Note, Recall-Verlust via `listBibl`-
Filter), bei Porst besser. Das Design muss ≥2 Dokumenttypen (Buch-Kapitel + kurzer/referenz-lastiger
Artikel) abdecken — „immer besser" ist widerlegt.

## Aufwand
~12+ h Pipeline-Wandzeit (N≥10 × ≥4 Zellen × ~25 min) + Quota + Runner-Erweiterung. Deshalb
kein weiterer Matrix-Lauf ohne neue konkrete Reopen-Frage. Ein lokaler Prototyp-Harness (Monkeypatch-Runner + GROBID-/pdfplumber-
Extractoren) existierte für die erste Runde, ist aber Wegwerf-Code außerhalb des Repos.

## Follow-up-Entscheidung 2026-06-03

Qwen 3.7 Max und Claude prüften die Option, pdfplumber als optionalen Stage-0-Adapter mit
Mehrspalten-/Textqualitäts-Guard zu bauen. Beide lehnten ab: Der Guard wäre selbst das
unbewiesene schwierige Feature und würde im Zweifel auf `pdftotext` zurückfallen. Damit ist
`pdftotext` weiter die einfachere und sicherere Baseline.

Reopen-Kriterien für pdfplumber:
- fokussierter Vergleich gegen `pdftotext`, nicht gegen GROBID;
- mindestens N≥10 oder ein kleineres, aber klar gepaartes Experiment mit stabiler Effektgröße;
- belegter Yield- oder Grounding-Gewinn über `pdftotext` jenseits von Run-Rauschen;
- keine Zweispalter-Regression bei Wortgrenzen, Seitenmarkern und Anchor-Treffern.

Nächster Kandidat ist kein Adapter-Build, sondern eine separate Messung zu Overlap-Chunking /
Planner-Recall bei fixem Extractor.

## Phase-A-Messung 2026-06-04 (LLM-frei, deterministisch)

Umgesetzt als `generative/eval_chunk_recall.py` (+ Tests). Zwei deterministische Metriken bei
fixem Extractor (pdftotext), gemessen auf Porst-2014, Bates-2017, Beutelspacher-2014; Konzept-
Referenz aus `calibration/labels-active` (menschlich, source-getaggt, N=10). Codex-gehärtet.

- **Boundary-Loss (`straddle_stats`, Modus `word_split`):** Sätze, die an Wort-Split-Chunk-
  Grenzen zerschnitten werden. ≤0,6 % bei `CHUNK_WORDS=3000`, durch Overlap=50 vollständig
  rückgeholt. **Caveat:** misst nur den `_split_by_words`-Fallback, NICHT den Default
  (`split_by_chapters` / `concept_text_window`, das schon 50 % Overlap hat). → Kein Hinweis,
  dass Overlap-Chunking nötig ist; Schluss gilt für einen von mehreren Chunking-Modi.
- **Overview-Coverage (`overview_coverage`):** token-basierte SCHWACHE Recall-Obergrenze.
  Sensitivitätskurve statt Einzelzahl: @0.5 = 10/10, @0.75 = 9/10, @1.0 = 8/10, exakter
  Phrasen-Match nur 2/10. Die ursprüngliche „100 %" war ein Schwellen-Artefakt von
  `min_coverage=0.5` (Codex-Review). Belastbar nur: in dieser kleinen Stichprobe kein großer
  Recall-Gap, aber die Metrik ist zu lasch/zirkulär (gleiche `_tokens`-Logik wie der Pipeline-
  Filter) für eine „kein Gap"-Behauptung. N=10 underpowered.

Volldoku + projektübergreifende Lehre: Wissenspool `[[Atomic-Agent-Pipeline]]` §Phase-A.
