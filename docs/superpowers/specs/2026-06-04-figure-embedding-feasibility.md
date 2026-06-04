# Feasibility: Figur+Caption-Einbettung in atomic notes ("Variante A")

**Datum:** 2026-06-04
**Status:** Vektor-Pfad widerlegt & pausiert; Caption-Overmatch refutiert (keine Härtung nötig); Raster-only-Minimal (~17 %, saubere Eingabe) offen — reine User-Entscheidung Bau vs. Pause
**Methode:** Brainstorm + iteratives Cross-Model-Review (Codex via `codex exec`, Qwen 3.7 max) + zwei empirische Wegwerf-Spikes.

## Ziel

Aus PDFs Figur+Caption **deterministisch** (kein Vision-Modell, kein LLM) extrahieren und in die von der Pipeline geschriebenen atomic notes einbetten. Precision-first: im Zweifel skippen + Manifest, nie eine falsche Figur in eine Note.

## Festgelegte Design-Entscheidungen (vor dem Spike)

1. **End-State:** Bild+Caption in den Note-Body + PNG-Asset in `98-system/attachments/figures/`.
2. **Precision-first:** einbetten nur bei eindeutiger 1:1-Bindung Figur→Note; sonst skip+Manifest.
3. **Architektur A** (nicht B): `figure_embedder` mutiert `AtomicNoteDraft`-Objekte **vor** dem Markdown-Render; `vault_writer` rendert. Kein Markdown-String-Editieren. Fresh-run only.
4. **Bindungs-Key:** `AtomicNoteDraft.source_anchors` (verifiziert, am Objekt) — nicht Chunk-Range (transient), nicht Markdown-Footnotes (fragil). Nur `page` (exakt), nie `fuzzy_page`. Genau ein `create`-Draft mit Seite P → binden.

## Empirische Funde

### 1. Klassifikator-Kollaps (gefixt, committed `b297be0`)
`classify_page_signals` klassifizierte jede Seite als `vector_or_composite`, weil `get_drawings()` auf 100 % der Seiten >0 liefert (Layout-Linien/Rahmen/Bullets). Umbau auf 4-Klassen-Taxonomie nur über Caption+Raster; Vektor degradiert zu Roh-Diagnostik. Caption-Recall über 5 PDFs gegengeprüft: Separator-Guard verwirft keine echten Captions (in diesem Sample).

### 2. Messung Raster vs. Vektor (7 PDFs, caption-führende Seiten)
| Quelle | Caption-Seiten | raster-deckbar | vektor-only |
|---|---:|---:|---:|
| Klaus 2016 (Paper) | 1 | 1 | 0 |
| Reibel-Felten 2022 (Report) | 2 | 2 | 0 |
| Varian (Lehrbuch) | 213 | 14 | 199 |
| Blankart (Lehrbuch) | 109 | 41 | 68 |
| Felsmann (Fachbuch) | 107 | 15 | 92 |
| **Gesamt** | **432** | **73 (17 %)** | **359 (83 %)** |

Befund: in Lehrbuch-/Akademik-PDFs sind ~83 % der caption-führenden Seiten vektor-only (kein qualifizierendes Raster). → Ein Vektor-Pfad wäre nötig, um die Mehrheit zu erfassen. Die Messung nutzte „genau eine Caption pro Seite" (`== 1`), Verzeichnis-/Mehr-Caption-Seiten sind also ausgeschlossen.

### 3. Design-Pivot (Codex + Qwen unabhängig konvergent)
Vektor-**Pfade clustern** wurde verworfen (das tote/fragile CV-Regelwerk). Stattdessen **caption-anchored ROI extraction**: Figur = Negativraum zwischen Caption und nächstem Textblock, spaltenbegrenzt; ROI via `get_pixmap(clip=…)` rendern; per Weißraum/Textzeilen/Grid-Check validieren. Begründung: Text-Geometrie ist exporter-stabil, der Renderer löst Raster/Vektor einheitlich auf.

### 4. Spike-Widerlegung des ROI-Ansatzes (entscheidend)
Wegwerf-Spike über mehrere hundert vektor-only Caption-Seiten (Varian/Blankart/Felsmann):

- **Caption-Lokalisierung scheitert** (`page.search_for(label)` „nicht gefunden") auf der Mehrheit; Felsmann ~100 %. Ursache: Engine-Mixing — pdftotext (Caption-Detektion) und PyMuPDF (Geometrie) sehen unterschiedliche Glyphen (CID-Fonts, Ligaturen, Custom-Subsets).
- **ROI kollabiert auf ~6 pt**, wo search_for klappt: figur-**interner** Text (Achsenlabels) wird als Textblock erkannt → „nächster Textblock oberhalb" liegt in der Figur → kein Gap.
- **Gerenderte Crops** waren ausnahmslos **Fließtext-Absätze oder Weißraum** — keine echte Figur (visuell verifiziert: Blankart p87, p202).
- **Recovery-Rate echter Vektor-Figuren im Sample: ~0 %.**

### 5. Cross-Model-Verdikt zur Vektor-Machbarkeit (Codex + Qwen, hohe Konfidenz)
Deterministische caption-verankerte Vektor-Extraktion ist **konzeptionell widerlegt**, nicht „unoptimiert": PDF speichert bei Lehrbuch-Vektorfiguren keine semantische Einheit „Figur"; Linien/Labels/Fließtext sind nur Zeichenoperationen. Der „Gap" ist eine optische Illusion. Engine-Mixing ist nicht deterministisch lösbar. **Ohne Vision-Modell oder Tagged-PDF-Strukturmetadaten (`/Figure`, MCIDs) nicht generisch machbar.**

## Entscheidung

1. **Vektor-Pfad: pausiert + dokumentiert.** Nicht weiterbauen, bis entweder PDF-Strukturmetadaten genutzt werden oder ein Vision-/Segmentierungs-Ansatz explizit akzeptiert ist.
2. **Caption-Härtung gegen Overmatch: nicht nötig** (Hypothese refutiert, s. o.). Die Captions auf Einzel-Caption-Seiten sind sauber. (Verzeichnis-Seiten sind durch die `== 1`-Regel schon raus.)
3. **Raster-only-Minimalfeature**: reine Bau-vs-Pause-Entscheidung. Eingabe ist sauber (~17 % Coverage, echte Captions + echte Raster). Kosten = die Embedder-Maschinerie (Bindung über `source_anchors`, Asset-Export via `get_pixmap`, Render-Helper in `vault_writer`, Idempotenz, Manifest) — auch raster-only nicht trivial. Nutzen = manchmal eine perfekte Figur, nie Müll.

### Caption-Overmatch-Hypothese: verifiziert → REFUTIERT (Schritt 2a)
Die Modell-Hypothese (Caption-Regex matcht In-Text-Referenzen) wurde gegen die echten gematchten Caption-Zeilen geprüft: Auf **Einzel-Caption-Seiten** (genau die, die in die 83 %-Messung eingehen) sind die Treffer durchweg **echte Captions** (Label + Beschreibung, z. B. „Abbildung 1.1: Altruisten, Egoisten und Bösartige"). Die einzige Kontamination sind **Abbildungsverzeichnis-Seiten** (viele Caption-Labels auf einer Seite) — die sind durch die `== 1`-Regel bereits ausgeschlossen. **Folge:** Keine Caption-Härtung gegen Overmatch nötig. Die 83 %-Vektor-Zahl ist robust; die 17 %-Raster-Pages haben saubere Captions + echte Raster-Figuren.

### Resultierende Entscheidung
Es gibt keine Caption-Kontamination zu fixen. Damit ist die Wahl rein: **Raster-only-Minimalfeature bauen (~17 % Coverage, saubere Eingabe)** vs. **Feature ganz pausieren**. Der Vektor-Pfad bleibt unabhängig davon tot (Vision/Tagged-PDF nötig). Offen für den User.

## Salvage / Wert dieser Arbeit
- Klassifikator-Fix committed (`b297be0`), unabhängig nützlich.
- Probe + Mess-Skripte zeigen die Raster/Vektor-Verteilung deterministisch.
- Klare Grenze dokumentiert: deterministische Figur-Extraktion endet bei Vektor-Lehrbuchfiguren; alles darüber braucht Vision/Tagged-PDF.
