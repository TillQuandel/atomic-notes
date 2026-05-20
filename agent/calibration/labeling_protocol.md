# Labeling-Protokoll — Pre-Registration (v4.1 Kalibrierung)

Pre-registered 2026-05-17, vor erstem Human-Label. Basiert auf [[Faithfulness-Annotation-Protokoll]] Hybrid C++.

## Kontext

- **Pipeline-Version unter Test**: atomic-agent v0.3.13 + eval_quality_v4 (v4.1)
- **Stichprobe**: 30 Notes, stratifiziert (15 DE→DE + 15 EN→DE)
- **Adversarial-Set**: ~10 % Gemini-perturbierte Claims (separate Datei, später)
- **Ziel-Maß**: Gwet\'s AC1 ≥ 0.70 pro Sprachpaar, Δ(Hybrid − Blind) ≤ 0.10

## Label-Schema

### Primär-Label (für AC1, Pflicht)

Pro Claim **genau eines** der folgenden:

| Label | Definition |
|---|---|
| `s` (supported) | Der Claim wird im gezeigten Kontext **oder** (nach Voll-PDF-Check, siehe Härtung #2) im PDF wörtlich, paraphrasiert oder teilweise belegt |
| `h` (hallucinated) | Der Claim wird **nicht** im Kontext belegt und auch **nicht** nach Voll-PDF-Check im PDF gefunden, ODER er widerspricht dem Quelltext |
| `?` (uncertain) | Genuine Mehrdeutigkeit (Übersetzungs-Frage, Synthese aus Mehrfach-Stellen unklar attribuierbar, technisches Vokabular nicht eindeutig). **NICHT** für Zeit-Druck oder Bequemlichkeit |

### Sekundär-Tag (optional, für Fehleranalyse — RAGTruth-Schema)

Nur bei `h` setzen, sonst leer:

| Tag | Definition |
|---|---|
| `evident_conflict` | Direkter Widerspruch zur Quelle (z.B. Zahl gedreht, Negation) |
| `evident_baseless` | Information klar nicht in der Quelle (frei erfunden) |
| `subtle_conflict` | Subtiler / indirekter Widerspruch (Bedeutung verschoben) |
| `subtle_baseless` | Plausibel klingend, aber unbelegt |

## Entscheidungs-Härtungen (Pflicht-Regeln)

### Härtung #1 — Kein Pipeline-Verdict sichtbar

Im Labeling-Markdown ist die `label`-Spalte der Pipeline-Entscheidung **versteckt** (entweder garnicht angezeigt oder in einer Separat-Datei). Du siehst nur: Claim + Kontext-Snippet + Seitenzahl. **Niemals** vor dem Final-Label die Pipeline-Entscheidung nachschlagen.

### Härtung #2 — Conditional Blind bei „nicht-supported"-Tendenz

Wenn der gezeigte Kontext (±500 chars) den Claim **nicht** stützt:
1. **STOP** — nicht direkt `h` labeln.
2. Öffne die PDF auf der angegebenen Seite (Page-Link folgen).
3. Suche mit ≥2 Keywords aus dem Claim im PDF (Ctrl+F).
4. Erst nach erfolgloser Voll-PDF-Suche → `h`. Falls Beleg im PDF gefunden → `s` (und im Notes-Feld vermerken: „Retrieval-Miss").

### Härtung #3 — Blind-Doppel-Annotation (15-20 % Sub-Sample)

5 Notes (zufällig aus den 30) werden zusätzlich **blind** gelabelt: nur Claim + komplette PDF (kein Kontext-Snippet sichtbar). Diese Labels gehen in separate Datei `labels_blind.jsonl` und werden für Δκ-Kontrolle gegen Hybrid-Labels gerechnet.

### Härtung #4 — Pre-Registration (dieses Dokument)

Dieses Dokument ist die Pre-Registration. **Änderungen daran nach Label-Start sind als Amendment zu markieren (Datum + Begründung + welche Labels davon betroffen).**

### Härtung #5 — Dritt-Reviewer für Disagreements

Bei Disagreements (Mensch ↔ Pipeline) wird **nach Abschluss aller Labels** ein Gemini-Tiebreaker-Lauf gestartet. Disagreements bleiben dokumentiert (kein Override deines Labels), Gemini-Vote dient nur der Fehler-Analyse („wer hat öfter recht").

## Praktische Hinweise

- **Reihenfolge**: Notes nach Sprachpaar gemischt labeln (nicht erst alle DE→DE), um Domain-Drift im Urteil zu mitteln.
- **Pause-Regel**: alle 30-45 min Labeling 5 min Pause — Annotator-Fatigue erzeugt Drift Richtung „supported" (Bequemlichkeits-Bias).
- **Notizen-Spalte**: Bei jedem `?` und bei jedem `h` mit Retrieval-Miss kurzer Vermerk (1 Satz) — hilft Phase-2-Ziel-Reformulierung.
- **Zeit-Estimate**: ~1 min/Claim bei klarem Hybrid, ~3-5 min/Claim bei Conditional-Blind-Trigger. Gesamt: 3-5 h.

## Operationalisierung der „uncertain"-Regel (Pre-Registered)

`?` ist erlaubt **nur** in genau einem dieser Fälle:

1. **Übersetzungs-Mehrdeutigkeit**: DE-Note ↔ EN-PDF, Begriff hat keine 1:1-Übersetzung (z.B. „Information Behavior" vs. „Informationsverhalten" mit Konnotations-Drift).
2. **Synthese-Attribution unklar**: Claim ist Aggregation aus 2+ PDF-Stellen, einzelne Stellen partial belegen, Gesamtaussage nicht eindeutig zuweisbar.
3. **Technisches Vokabular**: Domain-Term unbekannt, kann nicht entscheiden ob Belegstelle dasselbe meint (z.B. statistisches Maß, dessen genaue Definition variiert).

**Nicht** für: „weiß nicht genau", „bin müde", „zu viel Text". In diesen Fällen Pause machen.

## Output-Files

| File | Wann erzeugt | Inhalt |
|---|---|---|
| `labels_human.jsonl` | Beim Labeln (build_labels.py append-mode) | Eine Zeile pro Claim: `{note, claim_idx, label, tag, notes}` |
| `labels_blind.jsonl` | Beim Labeln des Blind-Sub-Sample | Wie oben, für 5 Notes |
| `labels_pipeline.jsonl` | Automatisch aus quality_history.jsonl extrahiert | Eine Zeile pro Claim: `{note, claim_idx, label}` |
| `kappa_report.md` | Step D | AC1 pro Sprachpaar + Δ-Kontrolle + Confusion-Matrix + Adversarial-Recall |
