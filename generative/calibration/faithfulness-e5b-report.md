# Faithfulness-Gate E5b — Kalibrierungs-Report (Pre-Label-Stand)

> Stand 2026-07-05, Branch `feat/faithfulness-e5b-calibration`. Pre-Labels von Claude
> (jeder Claim gegen das Quell-PDF verifiziert); Tills Human-Label-Pass folgt nach der
> BA (#123) über die Label-Dateien in `faithfulness-goldset/`.

## Ergebnis-Übersicht

| Metrik | Wert | Ziel | Status |
| --- | --- | --- | --- |
| Recall Inhalts-Fehler (`e`) | **2/2** | 2/2 Pflichtfälle | ⚠ siehe Pflichtfall 2 |
| Recall Anker-Fehler (`m`) | **5/5** | — (Bonus-Klasse) | ✅ |
| FPR auf sauberen Claims (`s` → failed) | **8/51 = 15,7 %** | < 10 % | ❌ verfehlt, Klassen-Analyse unten |
| Abstain-Quote auf `s`-Claims | 9/51 = 17,6 % | separat ausweisen | ✅ nie blockierend |
| Kanonische Suite | 1177 passed | grün | ✅ |

**Gold-Set:** 59 Claims aus 20 Notes / 6 Quellen (Hrastinski ×4, Merrill ×4, Knowles-alt ×7,
Mahmood ×1, Assfalg/KSS ×1 [0 High-Risk-Claims], knowles-e3a ×4 als FPR-Sauber-Set).
Pre-Labels: 51 `s` / 2 `e` / 5 `m` / 1 `?` (Pipeline-Artefakt, vom FPR ausgenommen).
Reproduktion: `python -m generative.calibration.build_faithfulness_goldset <pdf> --notes … --gate`
(Aufrufe je Quelle siehe `faithfulness-goldset/INDEX`-Kopfzeilen der Label-Dateien; Assfalg-Kapitel
aus dem KSS-Handbuch via `pypdf` extrahieren, `set_page_label(0, 12, style='/D', start=159)`,
physische Seiten 179–191).

## Kalibrierungs-Fixes dieser Etappe (Runden 2 + 3)

| Fix | Wurzel | Wirkung (gemessen) |
| --- | --- | --- |
| Prefix-Konkat-Premises (`_claim_premises`) | Voll-Konkatenation aller Top-5 verwässert legitime 2-Satz-Synthese | FP-A: e=0.149 → **0.998** (top1..2), FP-B: 0.164 → 0.988; direkt gemessen: top1..3=0.993, top1..4=0.392, top1..5=0.149 |
| `abstain_unverifiable_numbers` (Schritt 4b) | pdftotext zerlegt Tabellen in zuordnungslose Fragmente (`369 (99%)`); Synthese-Claims über Absätze entailen satz-basiert nicht (perfekter Stütz-Satz → e=0.000) | Tabellen-FP-Klasse: 4 Fails → 0 Fails (2 supported via Prefix-Konkat, 2 abstain); erfundene Zahlen failen weiter |
| Listen-Marker-Guard (`claims.py`) | `1. Ein Klima …` spaltete Junk-Claim „1." ab | 7 Junk-Claims im Knowles-Set eliminiert |
| Ordinal-Jahrhundert-Guard (`claims.py`) | „im 18. und 19. Jahrhundert" zerriss Sätze → zerrissene Teil-Claims failten | Splitter-FPs eliminiert, Claims wieder ganz |
| AUTHOR_YEAR-Stopwords (`attribution.py`) | „Zwischen 1929 und 1948 …" extrahierte „Zwischen" als Fremd-Autor → author_missing-FP | Journal-Claim: failed_attribution → NLI-Pfad |

**Empirisch verworfen — Tabellen-Zeilen als Pseudo-Satz-Premises (Option a):** Der linearisierte
Tabellen-Block macht mDeBERTa bidirektional falsch: wahrer 99-%-Claim → Contradiction 0.996,
*vertauschte* Aussagen („sync 99 %", „kleinere Klasse n=19") → Entailment 0.90–0.99. Hätte
falsche Fails durch falsche Supports ersetzt.

## Pflichtfall-Bilanz

1. **Zeitzonen-Extrapolation** (Hrastinski, Detail-Erfindung „beruflich/familiär/zeitzonenbedingt"):
   **gefangen** — failed_entailment e=0.000, über alle Fix-Runden stabil.
2. **Sekundärzitat-Fehlattribution** (Hrastinski: „Motivation" Kock statt Robert & Dennis
   zugeschrieben, Asynchronous-Note [^8] — der Kandidat in der Cognitive-Note erwies sich am PDF
   als korrekt attribuiert, nur die „zit. n."-Kennzeichnung fehlte): **strukturell nicht fangbar**
   mit dem aktuellen Baukasten, dreifach belegt:
   - Der Satz wird nie Claim — „nach <Name>s <X>-Hypothese" matcht kein E2-Attributions-Muster.
   - Als Claim: Namens-Präsenz-Check passt (Kock steht im Fenster); General-NLI entailt die
     Fehlattribution mit e=0.995–0.999 (Premise = Faktensack, keine Sprecher-Zuordnung).
   - Autor-gescopte Premises trennen nicht: mDeBERTa entailt „erhöht Motivation" aus „increases
     psychological *arousal*" (e=0.998) und lässt korrekte Zuordnungen fallen (R&D→Motivation
     e=0.003, weil Tabelle 2 den Trägersatz zerreißt; korrekter Kock-Claim e=0.001).
   - **Konsequenz:** Aussage↔Autor-Zuordnung braucht einen anderen Kanal (LLM-Judge nur für
     Attributions-Claims oder strukturiertes Parsing) — Design-Entscheidung nach #123.
   - Note-Level-Einordnung (ehrlich als Ko-Lokations-Glück markiert): die Fehlattribution liegt
     in der Async-Note, die wegen des Zeitzonen-Falls ohnehin blockt → Note-Level 2/2, Claim-Level 1/2.
3. **Bonus — Merrill-Richtungsdreher** (natürlich vorkommender dritter Fall, beim Pre-Labeling
   entdeckt): Note kehrt „examples in addition to practice > practice alone" zu „Übung ergänzt
   Info+Beispiele" um → **gefangen** (failed_entailment e=0.001).

## FPR-Klassen-Analyse (8 False-Positive-Fails auf 51 sauberen Claims)

| # | Claim (Kern) | e | Klasse |
| --- | --- | --- | --- |
| 1 | Merrill: Prinzipien-Verstoß → Lerneinbußen | 0.004 | abstrakte Synthese („präskriptive Gestaltungsregeln") |
| 2 | Knowles: Kinder fachzentriert, da Fach-Logik | 0.032 | Kausal-Synthese über 2 Fenster-Sätze |
| 3 | Knowles: Kompetenzkategorien statt Sachgebiete | **0.464** | einziger Grenzfall — Evidence nahezu wörtlich |
| 4 | e3a: Lindeman „kooperativ, nicht-autoritär" | 0.015 | Paraphrase eines langen historischen Zitats |
| 5 | e3a: Whitehead-Fußnoten-Argument | 0.007 | Meta-Rahmung („argumentiert in einer Fußnote, …") + Inhalt |
| 6 | e3a: Journal of Adult Education 1929–1948 | 0.004 | Meta-Rahmung („Empirisch stützt sich diese Prämisse auf …") |
| 7 | e3a: Panacea-superior-Frage | 0.012 | verschachtelte Reported-Speech-Paraphrase |
| 8 | e3a: ideologische Bindung Lehrender | 0.048 | verschachtelte Reported-Speech-Paraphrase |

Gemeinsame Signatur: lange, verschachtelte deutsche Synthese-/Rahmungs-Sätze gegen englische
Quelle; 7/8 mit e ≤ 0.05 (nicht durch Schwellen-Tuning rettbar), alle Inhalte am PDF verifiziert.
Kandidaten-Gegenmaßnahme für eine spätere Runde (bewusst NICHT mehr in E5b — Overfitting-Gefahr
auf n=8): Hypothesen-Zerlegung an Semikolon/Meta-Rahmung. Entscheidung nach Human-Pass #123.

**Note-Level-FPR** (relevant für E6-Routing, any-fail pro Note): 6 der 19 sauberen Notes würden
fälschlich blocken (32 %) — deutliches Argument dafür, `ENABLE_FAITHFULNESS_GATE` bis nach dem
Human-Pass **default aus** zu lassen (Verdicts erscheinen als quality_flags, blocken aber nicht).

## Schwellen-Empfehlung

**`MDEBERTA_THRESHOLD_CONFIRMED = 0.7 beibehalten.`** Die Score-Verteilung ist bimodal: alle
korrekten Supports ≥ 0.711, alle echten Fänge ≤ 0.001, 7/8 FPs ≤ 0.048. Eine Absenkung auf 0.45
würde genau einen FP (Grenzfall 0.464) retten und den Sicherheitsabstand halbieren — Feintuning
auf einem einzelnen Grenzfall ist der dokumentierte n=1-Kalibrierungsfehler (Ebner-Lehre).
`contra_threshold` bleibt ungenutzt dokumentiert: die Kontra-Ausnahme wurde empirisch geprüft und
verworfen (mDeBERTa ist entailment-seitig zahlenblind — „600 Wörter ≈ 6 *Stunden*" wird mit
e=0.995 aus der 6-*Minuten*-Prosa entailt; Contradiction-Signale sind bei Themen-Nachbarschaft
spurious, z. B. c=0.998 des Sync-Prosa-Satzes gegen den wahren Async-99-%-Claim).

## Bekannte Modell-Grenzen (mDeBERTa-XNLI, für #123 und E6-Doku)

- **Zahlen-/Einheiten-Blindheit in Entailment-Richtung:** Einheiten-Verzerrungen (Minuten→Stunden
  e=0.995) und n-Vertauschungen (e≈0.97) passieren das Gate als supported.
- **Keine Sprecher-Zuordnung:** Premise wird als Faktensack gelesen (Pflichtfall 2).
- **Tabellen-Layout:** von pdftotext zerstört; `abstain_unverifiable_numbers` ist die ehrliche
  Antwort (Human-Review-Flag, nie Block).
- Ziffern in Produktnamen („4-MAT") triggern die Zahlen-Abstain-Regel — harmlos (abstain), notiert.

## Konsequenzen für E6

1. `ENABLE_FAITHFULNESS_GATE` default **aus**; Verdicts als quality_flags (Plan unverändert).
2. Merge-Stub-Notes (`MERGE - *.md`) vom Gate ausnehmen — Verwaltungstext erzeugt Junk-Claims
   („Pipeline v28" → wäre failed_entailment).
3. Abstain-Stati (inkl. `abstain_unverifiable_numbers`) nie als Fail werten (Vertrag bestätigt).
