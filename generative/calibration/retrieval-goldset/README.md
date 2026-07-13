# Retrieval-Goldset (#232)

Abnahme-Grundlage fuer den #232-Retrieval-Fix (PR-B). Dieses Goldset selbst
ist additiv (PR-A) -- es aendert **keinen** Retrieval-/Chunking-Code, sondern
dokumentiert den aktuellen Bug-Zustand messbar und reproduzierbar.

## Hintergrund (Root Cause #232)

Das Eval-Retrieval (`generative/eval_quality_v4.py::_retrieve_claim_contexts`)
rankt Claims gegen **Chunk**-Embeddings (100-180 Token, `_chunks_from_sentences`
in `generative/eval_common.py`). Der tatsaechlich stuetzende **Satz** hat oft
hohe Cosine-Aehnlichkeit zum Claim, sein Chunk landet aber wegen der groben
Chunk-Granularitaet weit hinten in der Rangliste -- `adaptive_k` (meist 2,
selten bis 5) liefert ihn nie an den Judge. Ergebnis: `not_in_context` obwohl
der Claim im Quelltext steht (falsch-positive Halluzination) bzw. in einem
Fall ein echter Attributionsfehler, der zufaellig durchs Retrieval "gerettet"
wurde und deshalb korrekt als `contradicted` erkannt wurde (idx8, siehe unten).

## Format

`anchors.jsonl`, ein JSON-Record pro Zeile:

```json
{
  "id": "hrastinski-2008__sync__idx6",
  "source_pdf": "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf",
  "claim": "<Claim-Text exakt wie extract_claims() ihn produziert>",
  "expected_label": "supported_paraphrase",
  "evidence_quote": "<englischer/deutscher Quell-Satz, pdftotext-verifiziert>",
  "evidence_page": 54,
  "adjudication": "false_positive_retrieval_miss",
  "adjudicated_by": "human",
  "run_id": "20260712-215118",
  "eval_version_at_adjudication": "4.1"
}
```

Feld-Hinweise:

- **`claim`**: bewusst der **rohe** `extract_claims()`-Output (Markdown-Formatierung
  entfernt, z.B. "E-Learning" -> "E Learning"), NICHT die huebsche Note-Prosa --
  das ist exakt der String, den die Pipeline embedded und an `_retrieve_claim_contexts`
  uebergibt. Der Test-Harness muss dieselbe Eingabe verwenden wie Produktion,
  sonst waere der gemessene Recall nicht uebertragbar.
- **`evidence_quote`**: der woertliche Quellsatz (bzw. bei Tabellen ein bewusst
  kurzes, zusammenhaengendes Fragment -- siehe idx8-Hinweis unten), verifiziert
  per `pdftotext -layout` gegen die Original-PDF UND gegen den tatsaechlichen
  `Chunk.text`-Output der Pipeline (PyMuPDF-Block-Extraktion kann von
  `pdftotext` abweichen, z.B. bei zweispaltigen Tabellen).
- **`evidence_page`**: Druckseite bzw. PDF-Position, wie sie `anchor_page_numbers`
  liefert (identisch zum Namespace, den `Chunk.pages`/`best_page` nutzen).
- **`adjudication`** ∈ `{false_positive_retrieval_miss, true_hallucination, true_support}`.
- **`adjudicated_by`**: `"human"` -- alle Anker stammen aus Till Quandels
  handadjudizierten Diagnose-Laeufen (Sonnet-Subagent + `pdftotext`-Gegenprobe,
  Session-Notizen unter `OneDrive/Dokumente/Claude/Projects/Atomic notes notizen/`);
  die konkrete Zuordnung Claim-Text <-> Report-Befund <-> PDF-Zitat fuer Bates/
  Suehl-Strohmenger (siehe Provenienz-Tabelle) wurde in dieser PR-A-Session
  zusaetzlich unabhaengig nachverifiziert (erneutes `pdftotext` + reale
  Pipeline-Retrieval-Rekonstruktion, nicht nur Report-Zitat vertraut).

## Adjudikations-Legende

| Wert | Bedeutung |
|---|---|
| `false_positive_retrieval_miss` | Claim ist im Quelltext belegt, aber der stuetzende Chunk wurde vom Retrieval nicht in den Top-`adaptive_k` geliefert -- Eval-Artefakt, keine echte Halluzination. |
| `true_hallucination` | Echter inhaltlicher Fehler (z.B. Attributions-/Kontext-Verwechslung) -- KEIN Retrieval-Problem. Muss nach dem #232-Fix weiterhin als Fehler erkannt werden. |
| `true_support` | (im aktuellen Set nicht vorhanden) Claim korrekt belegt UND korrekt vom Retrieval gefunden -- fuer spaetere Erweiterung reserviert. |

## Anker-Uebersicht (11 Stueck)

| Quelle | Note (Lauf) | run_id | # Anker | Adjudikation |
|---|---|---|---|---|
| Hrastinski 2008 | "Synchronous E-Learning" (Lauf 5) | `20260712-215118` | 6 | 5x `false_positive_retrieval_miss`, 1x `true_hallucination` (idx8) |
| Bates 2017 | "Disziplinaere Ausweitung der IB-Forschung" (Lauf 4) | `20260712-213647` | 2 | `false_positive_retrieval_miss` |
| Suehl-Strohmenger 2008 | "Learning Library" (Lauf 3) | `20260712-205627` | 3 | `false_positive_retrieval_miss` |

**Cross-Lingual-Faelle (DE-Claim / EN-Quelle) mit niedriger Presence-Cosine
0,64-0,67** (reale, deterministisch nachgerechnete Werte aus
`generative.embeddings`, `paraphrase-multilingual-MiniLM-L12-v2`, max. Satz-
Cosine ueber alle PDF-Saetze): **`hrastinski-2008__sync__idx4`** (0,666) und
**`hrastinski-2008__sync__idx13`** (0,640). Diese zwei sind bewusst NICHT die
"leichten" Faelle mit hoher Cross-Lingual-Aehnlichkeit -- sie stellen sicher,
dass ein kuenftiger Fix nicht nur auf einfach zu findende Ankertexte kalibriert
wird. Die drei Suehl-Strohmenger-Anker sind zusaetzlich ein DE-Quelle/DE-Claim-
Kontrastfall: sie zeigen, dass das Problem an der Chunk-**Granularitaet**
liegt, nicht primaer an Cross-Lingual-Embedding-Schwaeche.

**Negativ-Kontrolle**: `hrastinski-2008__sync__idx8` (Table 3 der Quelle
attribuiert "When synchronous meetings cannot be scheduled" der ASYNC-Spalte,
der extrahierte Claim schreibt es faelschlich der SYNC-Empfehlung zu -- ein
echter Attributionsfehler, per `audit_override` korrekt als `contradicted`
erkannt). Das Zitat ist bewusst kurz gehalten ("When synchronous meetings
cannot be scheduled", ohne Fortsetzung): Table 3 ist zweispaltig, PyMuPDFs
Block-Extraktion liest beide Spalten zeilenweise interleaved
("... n Getting acquainted because of work, family ..."), ein laengeres
woertliches Zitat ueber die Spaltengrenze hinweg waere im echten `Chunk.text`
gar nicht als zusammenhaengender String vorhanden.

## Reale Zahlen aus der Retrieval-Rekonstruktion (master, 2026-07-13)

Deterministisch reproduziert via `generative.eval_quality_v4._retrieve_claim_contexts`
gegen die echten Quell-PDFs (kein LLM). Die Chunk-Top-Cosine deckt sich bei
Hrastinski exakt mit den im Diagnose-Bericht protokollierten Werten (siehe
`LAUF-INFO.md` der Lauf-5-Session) -- bestaetigt, dass diese Rekonstruktion
bit-fuer-bit dieselbe Pipeline-Logik nutzt:

| id | chunk_top_cosine | home_chunk_rank | home_chunk_cosine |
|---|---|---|---|
| hrastinski...idx4 | 0.614 | 9 | 0.413 |
| hrastinski...idx6 | 0.756 | 14 | 0.485 |
| hrastinski...idx7 | 0.812 | 16 | 0.449 |
| hrastinski...idx8 | 0.774 | -- (direkt retrieved, kein Miss) | -- |
| hrastinski...idx11 | 0.819 | 2 | 0.737 |
| hrastinski...idx13 | 0.503 | 22 | 0.256 |
| bates...idx6 | 0.560 | 15 | 0.485 |
| bates...idx14 | 0.520 | 44 | 0.370 |
| suehl...idx2 | 0.492 | 5 | 0.378 |
| suehl...idx4 | 0.729 | 12 | 0.589 |
| suehl...idx9 | 0.776 | 12 | 0.581 |

**Evidence-in-Pool-Recall (batched, wie Produktion -- alle Claims einer Note
teilen einen Kontext-Pool)**: **2/10 = 20,0 %** ueber alle
`false_positive_retrieval_miss`-Anker (siehe `generative/tests/test_retrieval_goldset.py`).
Die 2 Treffer (idx4, idx7) sind KEIN Widerspruch zum Bug, sondern zeigen genau
seine Zufaelligkeit: idx4s Evidenz liegt direkt in Chunk 0, der von idx6/idx7/
idx11 (nicht von idx4 selbst!) retrieved und dadurch in den gemeinsamen Pool
gezogen wird; idx7s Evidenz liegt in Chunk 17, der nicht direkt retrieved wird,
aber als Nachbar von Chunk 16 (idx8s eigenem Retrieval-Treffer) per
`_expand_context` (±1 Chunk) mit in den Pool expandiert. Bei isolierter
Einzel-Claim-Retrieval (ohne Batch-Pooling anderer Claims derselben Note) waere
der Recall 0/10 -- der Batch-Effekt ist also eine reale, aber vom jeweiligen
Note-Claim-Mix abhaengige Zufallsrettung, kein verlaesslicher Fix.

## Wie erweitern

1. Neue Diagnose-Laeufe mit Anker-Forensik-Tabelle (`| # | Claim | Label |
   Cosine | Befund |`) via `build_from_reports.py <LAUF-INFO.md>` parsen (Schritt 1).
2. Jeden Anker per `pdftotext -layout` gegen die Quell-PDF verifizieren
   (Schritt 2, PFLICHT -- niemals Report-Zitat blind uebernehmen). Bei Tabellen/
   mehrspaltigem Layout: Zitat gegen den ECHTEN `Chunk.text`
   (`generative.eval_common._chunks_from_sentences`) pruefen, nicht nur gegen
   `pdftotext`-Rohtext -- PyMuPDFs Block-Reihenfolge kann abweichen.
3. `claim`-Feld aus `generative.eval_common.extract_claims(note_path)` uebernehmen
   (nicht die Markdown-Prosa von Hand abschreiben).
4. Neue Zeile an `anchors.jsonl` anhaengen, `adjudication` setzen.
5. `pytest -m slow generative/tests/test_retrieval_goldset.py -s` laufen lassen
   und den ausgegebenen Recall gegen die README-Tabelle oben aktualisieren.

## Verworfene Kandidaten (nicht ins Goldset aufgenommen)

- **Bates Lauf 4, Claim idx16** ("Vielzahl empirischer Einzelstudien... von
  ethnografischen Beobachtungen... bis Zitationsanalysen"): Der Claim
  synthetisiert ueber mehrere, nicht-benachbarte Textstellen (ethnografische
  Studie auf PDF-S. 4, Fallstudie/Zitationsanalyse auf PDF-S. 8) -- kein
  einzelnes zusammenhaengendes Quellzitat extrahierbar, das den GESAMTEN Claim
  traegt. Adjudikation waere Spekulation gewesen (Prinzip "niemals raten").
- **Suehl-Strohmenger Lauf 3, "Frontalunterricht" Claim idx2**
  ("Bibliothekswissenschaftler Sühl-Strohmenger..."): Der Fehler selbst ist
  real und pdftotext-verifiziert (Quelle S. 1 nennt den Autor
  "promovierter Erziehungswissenschaftler", nicht "Bibliothekswissenschaftler"
  -- echte Attributionsverwechslung, analog zu idx8). Die Retrieval-Forensik
  ergab hier aber ein uneindeutiges Bild: der im Bericht genannte Wert
  ("cos 0,70, Page 1 gefunden") passt nicht sauber zur eigenen Nachrechnung
  (Top-Cosine 0.703 gehoert zu einem Chunk auf PDF-S. 9-10, nicht S. 1; der
  Chunk mit dem exakten Beleg-Satz auf S. 1 rankt erst auf Platz 8, cos 0.468,
  ausserhalb `adaptive_k=3`). Ohne zweifelsfreie Rekonstruktion, WELCHEN
  Kontext der urspruengliche Judge tatsaechlich sah, waere ein
  `true_hallucination`-Label hier eine Vermutung -- deshalb verworfen statt
  spekulativ aufgenommen.
- **Sühl-Strohmenger Lauf 3, "Learning Library" 4. Anker** und die aggregierten
  Halluzinations-Ausreisser aus den `endreview-messreihe.md`-Berichten (Bates
  M7/M8, Suehl M6, Hrastinski M9): Diese Berichte nennen nur Themen-Stichworte
  ("~5 markiert -- alle Inhalte auf S. 2 auffindbar") ohne Claim-Index oder
  Claim-Text -- eine eindeutige Zuordnung zu einem konkreten
  `extract_claims()`-Satz waere Raten gewesen.
