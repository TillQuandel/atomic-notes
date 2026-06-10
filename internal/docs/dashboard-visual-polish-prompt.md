# Auftrag: Eval-Dashboard visuell aufwerten

Repo: `C:/Users/tillq/source/repos/atomic-notes`. Gegenstand: das Eval-Dashboard
(`generative/eval_dashboard.py` — HTML-Generierung, `generative/eval_dashboard_server.py`,
`internal/dashboard/eval_dashboard.html`). Es wurde gerade funktional überarbeitet
(Branch `fix/dashboard-quick-wins`); jetzt geht es um die **visuelle Qualität**:
Typografie, Spacing, Card-Design, Sidebar, Zustände, Darkmode — produktreif statt
Entwickler-Provisorium.

Nutze den `frontend-design`-Skill (distinctive, production-grade, keine generische
KI-Ästhetik) — aber innerhalb der bindenden Vorgaben unten.

## Vorbedingungen (in dieser Reihenfolge)

1. Prüfe `git status` + aktuellen Branch. Falls `fix/dashboard-quick-wins` noch nicht
   in master ist: Merge-Status mit mir klären BEVOR du beginnst. Eigenen Branch
   `feat/dashboard-visual-polish` von der aktuellsten Basis ziehen. Falls der Working
   Tree fremde uncommitted Änderungen enthält: STOPP, nachfragen (heute lief bereits
   eine Session parallel im selben Checkout).
2. Design-SSoT lesen: `C:/Users/tillq/Obsidian_Vault/05-llm-wiki/Dashboard-Filter-Refactor.md`
   — Variant-H-Layout ist die akzeptierte Spec. Vertiefung: `LLM-Eval-Dashboard-Design.md`
   (Anti-Slop-Patterns) im selben Ordner.
3. Ist-Zustand erfassen: Server starten (`python generative/eval_dashboard_server.py`),
   Dashboard im Browser ansehen/screenshotten (falls Tooling da), HTML/CSS vollständig lesen.

## Bindende Design-Vorgaben (NICHT verhandelbar — aus der akzeptierten Spec)

- **Layout bleibt Variant H**: Sidebar (220px ↔ 56px collapsible, Icon-Modus mit
  Tooltips, localStorage-State), sticky Filter-Bar, Sektionsreihenfolge Übersicht →
  Qualität → Kosten → Trade-off → Vergleich. Kein Layout-Redesign.
- **Palette**: warme Pastell-Palette — PDF-Encoding Coral/Teal/Amber,
  Pipeline-Versionen als Teal-Triple, Ampel als gedeckter Mint/Mute/Dust-Red.
  Max. 3 Akzentfarben + Grayscale. Semantische Farben NUR an Threshold-Stellen
  (Hallucination 5 %/15 %, Cache-Hit ≥50 %, Inbox-Rate <10 %).
- **Ehrlichkeits-Regeln**: N-Annotation an jedem KPI (`Coverage 35 % (n=4)`);
  Goal-Marker erst ab N≥20 pro Gruppe; aspirationale Ziele als „aspirational" gelabelt,
  keine Rot/Grün-Logik dafür; keine Gauges/Donuts, keine Dual-Axis-Charts.
- **Stack**: Chart.js + `chartjs-plugin-annotation`. KEIN Bibliothekswechsel, keine
  Build-Pipeline, kein Framework — es bleibt eine lokal servierte Seite.
- **Read-only**: keine Mutation von quality_history.jsonl, runs/, DB;
  `eval_quality_v4.py` + `decision_engine/` tabu. Daten-Funktionen nur anfassen,
  wenn rein additiv für die Darstellung nötig.

## Arbeitspakete (Pflicht-Scope aus der Spec, §Phase-3-Polish)

1. **Darkmode**: `prefers-color-scheme: dark` + manueller Toggle (System/Hell/Dunkel)
   in der Sidebar-Fußzeile. CSS-Custom-Properties (`--c-bg`, `--c-card`, `--c-ink`, …)
   existieren — Dark-Werte via `[data-theme="dark"]` überschreiben. Akzentfarben bleiben;
   Mint/Dust-Red ggf. 5–10 % heller für Kontrast. **WCAG 2.2 AA in BEIDEN Modi prüfen.**
2. **Visuelle Hierarchie & Konsistenz**: Typo-Skala (max. 3 Größenstufen + Mono für
   Zahlen), konsistentes Spacing-System, Card-Design (Radius/Border/Schatten einheitlich),
   Filter-Bar und Sidebar aufräumen, Tabellen-Lesbarkeit (Zebra/Alignment/Tabular-Nums).
3. **Zustände**: Loading-Skeleton statt Blank-Screen beim Reload; klare
   „keine Daten"-Anzeige bei leeren Filter-Kombinationen; Hover-/Focus-Zustände
   (sichtbarer Fokus-Ring, Tastatur-Bedienbarkeit).
4. **Insight-Headlines** (falls Datenlage es trägt): auto-generierte Befund-Zeile pro
   Sektion im Reuters/FT-Stil — konkrete Aussage mit Werten aus echten Daten, mit
   N-abhängiger Vorsicht (kein „X ist besser" bei N=4). Sonst als TODO dokumentieren.

## Checkpoint-Pflicht

Nach der Ist-Aufnahme und VOR der Implementierung: zeige mir einen kompakten Vorschlag
(was änderst du konkret pro Arbeitspaket, 1 Zeile je Punkt + ggf. Screenshot/Mockup)
und warte auf mein OK. Geschmacksfragen entscheide ich.

## Technische Pflicht

- Async-Chart-Init-Pattern anwenden (SSoT: Vault-Note `Async-Chart-Init-Pattern`):
  `window.load` + doppeltes `requestAnimationFrame` vor erstem Embed, zentrale
  Embed-Funktion mit View-Registry und `.catch()`, Re-Render bei Sidebar-Toggle und
  Window-Resize — sonst rendern Charts im CSS-Grid mit collapsible Sidebar leer.
- Volle Test-Suite vor und nach den Änderungen grün; Cross-Model-Review (codex exec)
  vor dem finalen Commit.
- Diesen Prompt als `internal/docs/dashboard-visual-polish-prompt.md` mit einchecken.

## Akzeptanz

Dashboard rendert in Hell- UND Dunkel-Modus korrekt (inkl. Charts nach
Sidebar-Toggle/Resize), WCAG-AA-Kontraste belegt (Stichproben mit Werten), Loading-
und Leer-Zustände sichtbar implementiert, keine Verletzung der Palette-/Ehrlichkeits-
Regeln, Suite grün. Vorher/Nachher-Screenshots (oder genaue Beschreibung, falls kein
Screenshot-Tooling) im Abschlussbericht.
