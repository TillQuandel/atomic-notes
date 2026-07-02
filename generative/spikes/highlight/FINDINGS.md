# PDF-Highlight-Spike — Ergebnis (2026-07-01)

Validierungs-Spike für die geplante Highlight-Erweiterung (Ansatz A: Sequence-
Alignment gegen den `get_text("words")`-Token-Strom). Gate VOR jedem Feature-Code.

## Akzeptanzkriterium — ERFÜLLT

| Kriterium | Schwelle | Ergebnis |
| --- | --- | --- |
| Echte Lückenquote | < 10 % | **0,0 %** (36/36 präsente Zitate lokalisiert) |
| Falsch-Treffer | 0 | **0** (12/12 Near-Miss-Zitate verworfen) |
| Geometrie korrekt | — | 7/7 Stichprobe (sim 98–100), + visueller Render |

Reproduzieren: `python -m pytest test_aligner.py -q && python spike_align.py`

## Zentraler Befund, der das Design präzisiert

**Zitier-Seite ≠ sauber highlightbare Seite.** Die `page`-Felder in `quotes.json`
stammen aus den Note-Zitationen (z. B. Merrill „S. 3"), nicht von der tatsächlichen
Fundstelle. Beispiel Merrill: die Prinzipien stehen auf der Zitier-Seite 3 nur
**fragmentiert in einer Figur/Textbox** (Lesereihenfolge zerrissen, Alignment-Score
~77), zusammenhängend aber als Prosa auf Seite 1/7/8 (Score 100). Der Verifier
(`verify_citations.py`) findet die wahren Seiten (search_for) mit Offsets bis +5.

→ Konsequenz für das Feature: Die Highlight-Stage muss die **Verifier-Fundseite**
nutzen, nicht die Note-Zitierseite. Das deckt sich mit dem Design („Alignment nur
gegen die words der Verifier-Seite"), macht aber explizit, dass Page-Finding Sache
des bestehenden Verifiers ist und die Highlight-Stage nur noch **Geometrie auf der
bekannten Seite** liefert. Das ursprüngliche ±1-Fenster um die Zitierseite hätte die
Merrill-Zitate systematisch verfehlt (18,4 % Lücke — schlechter als Baseline).

## Architektur des Spikes (zwei getrennte Belange)

1. **Page-Orakel** (`find_page`, simuliert den Verifier): Char-Level-
   `partial_ratio`-Scan über alle Seiten → wahre Fundseite + Präsenz-Score.
   Präsenz-Schwelle 97 (Char-Level, Whitespace-strippt, Bracket/Silbentrennung
   normalisiert).
2. **Word-Alignment** (`aligner.locate`, der eigentliche Spike): `rapidfuzz.
   partial_ratio_alignment` des normalisierten Zitats gegen den
   `get_text("words")`-Token-Strom **genau der Orakel-Seite** (harter Page-
   Constraint, window=0). Char-Span → Wort-Index → Bbox. Guardrails: **Score ≥ 96**
   UND Längen-Ratio ≥ 0,9.

`column_boxes()` als Stufe 0 bleibt drin (no-op bei einspaltig), war aber auf den
sauberen Prosa-Fundseiten nicht nötig — die native `words`-Reihenfolge ist dort
korrekt, und die per-Wort-Bboxes respektieren Spaltengrenzen von sich aus (visuell
belegt an Knowles S. 6, zweispaltige Tabelle: Highlight bleibt in der Spalte).

## Schwellwert-Kalibrierung (Score)

Saubere Trennung zwischen präsent und Near-Miss:

- 36 präsente Zitate: Word-Score **97,4 … 100** (Minimum 97,4)
- 12 Near-Miss-Zitate: Word-Score **≤ 94,9** (höchster: bedeutungs-umgekehrte
  Variante „andragogy **toward**" statt „**away from**")

→ `min_score = 96` trennt mit ~1,4 Punkten Margin auf jeder Seite. Der Ellipsen-
Negativfall (Knowles „The first,… the goal-oriented", char 79) und ein echtes
Fehlzitat („People **are** ready" statt „People **become** ready", char 96) werden
schon vom Orakel als nicht-präsent verworfen — beide korrekt in den Sidecar.

## Residual (kein Tooling-Mangel)

- **Ellipsen-Zitat** (Knowles S. 3): strukturell unmöglich als ein zusammenhängender
  Block — bleibt ehrlich im Sidecar (wie im Design vorgesehen).
- **Fehlzitat** „People are ready…": die Note zitiert leicht falsch; die korrekte
  Variante „People become ready…" ist separat präsent und wird korrekt highlightet.

## Offen fürs Feature (aus dem Spike gelernt, nicht Teil des Gates)

- Per-Wort-Rects ergeben leicht „boxige" Highlights mit Mini-Lücken zwischen Wörtern
  → pro Zeile zu einer durchgehenden Rect-Hülle mergen (y-Bucket, vgl. altes
  `span_fill_rects`) für saubere Optik.
- Score-Margin (94,9 ↔ 97,4) ist real, aber schmal. In Produktion fängt der Verifier
  bedeutungs-umgekehrte/Fehlzitate ohnehin upstream ab (found:false), bevor sie die
  Highlight-Stage erreichen — die 96er-Schwelle ist die zweite Verteidigungslinie.

## Cross-Model-Review (2026-07-01)

Codex am Usage-Limit → Zweit-Reviewer **Mistral magistral-medium** (🟢 Code).
Konvergenz mit der eigenen Skepsis auf **einem** substanziellen Punkt:

- **Zirkularität (hoch, berechtigt):** `find_page` (Char-Orakel) bestimmte Präsenz
  UND Seite, auf der `locate` dann testete — beide partial_ratio, „0 % Lücke" wirkte
  dadurch geschönt. **Fix umgesetzt:** `geometry_matches()` als orakel-**unabhängige**
  Erfolgsbedingung — ein Zitat zählt nur als lokalisiert, wenn der per `get_textbox`
  extrahierte Text UNTER den Bboxes das Zitat reproduziert (≥95 partial_ratio). Das
  entkoppelt die Metrik vom Alignment-Score. **Ergebnis hält:** 36/36, 0 Falsch-Treffer.
- **Schwellwert-Overfit (mittel):** min_score=96 an nur 12 Near-Miss kalibriert, Margin
  schmal. Ehrlich stehen gelassen; der Geometrie-Check ist jetzt die **zweite
  Verteidigungslinie** neben dem Score (nicht mehr alleiniger Schutz).
- **Mistral-Code-Bugs WIDERLEGT** (gegen Repo verifiziert, nicht blind gefolgt):
  (a) 3-fach-Silbentrennung klebt korrekt („architecture", Regressions-Test ergänzt);
  (b) kein `dest_end`-Off-by-one (`min(dest_end, len(char_to_word))` war schon drin).

## Dateien

- `aligner.py` — reine Alignment-Logik (TDD, 10 Tests in `test_aligner.py`)
- `spike_align.py` — Mess-Harness (Orakel + Word-Alignment + Near-Miss)
- `render_highlight.py` — erzeugt Beleg-PDF + PNG-Render (Original nie mutiert)
- `quotes.json` — 46 Zitate (38 messbar aus 6 PDFs)
- `near_miss.json` — 12 Falsch-Treffer-Testfälle
- `multi_column.py` — vendored `column_boxes` (PyMuPDF-Utilities, self-contained)
