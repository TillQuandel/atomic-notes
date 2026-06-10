# Review-Auftrag: Eval-Dashboard

Auftrag: Reviewe das Eval-Dashboard und liefere **priorisierte Verbesserungs-Empfehlungen.
KEINE Implementierung** — Deliverable ist der Befund, umgesetzt wird nach Freigabe.

## Gegenstand

- `generative/eval_dashboard.py` (~1800 Zeilen — Daten-Funktionen + HTML-Generierung)
- `generative/eval_dashboard_server.py` (~895 Zeilen — HTTP-Server Port 8051, Daten-Endpunkte)
- `internal/dashboard/eval_dashboard.html` (~1394 Zeilen — Frontend/JS)
- Datenquellen (alle READ-ONLY): `generative/.cache/quality_history.jsonl` (~14 MB),
  `.cache/atomic_analytics.db` (note_evals, calibration_labels, pipeline_runs),
  `generative/.cache/runs/*.jsonl`

## Pflicht-Vorarbeit (vor jedem Urteil)

1. **Design-SSoT lesen:** `C:/Users/tillq/Obsidian_Vault/05-llm-wiki/Dashboard-Filter-Refactor.md`
   — enthält das akzeptierte Variant-H-Ziel-Layout, die Chart-für-Chart-Entscheidungen,
   die Cross-Review-Korrekturen K1–K5 (u. a. Chart.js statt ECharts, Goal-Marker erst ab
   N≥20, Drill-Down per Inline-Drawer) und den Status „teilweise umgesetzt".
   **Bereits entschiedene oder bewusst verworfene Punkte nicht neu aufrollen** — gegen
   den Plan abgleichen statt ihn zu ersetzen. Vertiefung optional:
   `LLM-Eval-Dashboard-Design.md` + `Versions-Vergleich-Visualisierung.md` (gleicher Ordner).
2. **Bekannte Vorgeschichte einpreisen:**
   - `avg_agree` zeigte bis 2026-06-10 „0 %" statt „–" bei fehlenden Agreement-Daten (gefixt).
   - Der `calibration_labels`-DB-Write in `collect.py` war bis 2026-06-10 end-to-end tot
     (falscher DB-Pfad) — die Kalibrierungs-View hat live also **nie echte Daten gezeigt**.
     Prüfen, ob sie mit gefüllter Tabelle korrekt rendert.
   - Packaging-Umbau 2026-06-10 (pyproject, Entry-Points, sys.path-Hacks entfernt) —
     prüfen, ob Dashboard-Start (`eval_dashboard_server.py`) das überlebt hat.

## Review-Dimensionen

1. **Daten-Korrektheit:** Mindestens 3 angezeigte Aggregationen per Hand gegen die
   Rohdaten nachrechnen (SQL auf die DB / jq-artig auf die JSONL). Präzedenzfälle waren
   genau diese Klasse: stille Falsch-Aggregation statt Crash. Auch: Version-Sortierung,
   Median- vs. Mittelwert-Konsistenz, None-/Leer-Behandlung in jedem KPI.
2. **Ehrlichkeit der Darstellung:** Aggregate bei kleinem N (N-Annotation vorhanden?),
   Goal-Marker-Regel (erst ab N≥20), Mischung inkompatibler Skalen/eval_versions,
   irreführende Farb-Semantik.
3. **Filter-Logik:** Default = höchste eval_version? Leere Filter-Kombinationen →
   klare „keine Daten"-Anzeige statt 0-Werte? Komponieren Filter mit AND wie geplant?
4. **Performance:** Was wird pro Reload geparst (14-MB-JSONL voll?)? Caching?
   Spürbare Latenz beim Filterwechsel?
5. **Code-Architektur:** Daten/Render-Trennung (1800-Zeilen-Mischfile?), Duplikation
   zwischen server/dashboard/html, Testbarkeit der Aggregations-Funktionen (Vorbild:
   `_avg_agreement` wurde 2026-06-10 als pure Funktion extrahiert + getestet),
   Einhaltung der Read-Only-Regel (kein Write auf Eval-Artefakte aus Dashboard-Code).
6. **UX gegen Variant-H-Spec:** Was vom akzeptierten Layout ist umgesetzt, was fehlt
   (Filter-Top-Bar + URL-State sind laut Plan offen), was hat sich seit 2026-05-17
   überholt? Demo-Referenz falls noch vorhanden: `c:/tmp/dashboard-demo/variant-h-sidebar.html`.

## Methode

- Server lokal starten, Daten-Endpunkte abrufen und gegen Rohdaten gegenprüfen;
  HTML/JS vollständig lesen (nicht nur grep).
- Alles READ-ONLY: keine Mutation von quality_history.jsonl, runs/, DB, Eval-Code
  (`eval_quality_v4.py`, `decision_engine/` sind tabu — harte Regel aus dem Design-Doc).
- Vor dem Befund-Finale: Cross-Model-Pass (codex exec) auf die Befundliste — Fakten
  inline mitgeben, nicht nur Pfade.

## Output-Format (Übersicht vor Tiefe)

1. **Befund-Tabelle:** | # | Severity 🔴🟡🟢 | Datei:Zeile | Befund | Beleg (nachgerechnet/zitiert) |
2. **Plan-Abgleich:** | Variant-H-/K1–K5-Punkt | Status (umgesetzt / offen / überholt) | Anmerkung |
3. **Empfehlungen:** Top 5–8, priorisiert, je: Nutzen, Aufwand (S/M/L), Abhängigkeit,
   Mapping auf bestehende Plan-Phase bzw. Issue. Quick-Wins explizit markieren.
4. **Nicht tun:** Liste der Dinge, die naheliegend wirken, aber laut Design-SSoT bereits
   entschieden/verworfen sind (mit Quelle).

Danach stoppen und auf Freigabe warten.
