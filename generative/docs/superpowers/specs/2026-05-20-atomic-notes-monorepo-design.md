# atomic-notes — Design Spec v2 (2026-05-20)

> Monorepo mit zwei Pipelines: `agent/` (LLM) + `foss/` (kein API)

## Struktur

```
atomic-notes/         # GitHub-Repo (TillQuandel/atomic-notes)
├── agent/            # LLM-Pipeline (aktueller atomic-agent-Code)
├── foss/             # FOSS-Pipeline (kein API-Key noetig)
├── shared/           # SSOT: embeddings, stage6, schemas, eval
├── dashboard/        # eval_dashboard_server.py
├── compare.py        # Vergleich: beide Pipelines auf gleicher PDF
└── README.md
```

## Ziel

Monorepo: zwei PDF-zu-Atomic-Notes-Pipelines unter einem Dach.

**agent/** — LLM-Pipeline: Claude API oder Subscription, hohe Qualitaet, Synthese
**foss/** — FOSS-Pipeline: kein API-Key, offline, 0% Halluzination, Baseline-Vergleich

**Scope v1:** Englischsprachige PDFs. Deutsch in v2.

## Abgrenzung zum atomic-agent

| | atomic-agent | foss-atomic |
|---|---|---|
| Planner | Claude Sonnet (LLM) | GLiNER (Zero-Shot NER) |
| Extractor | Claude Sonnet (Synthese) | Sentence-Cluster + LexRank |
| Stage 6 | Haiku (LLM) | FOSS (shared via git submodule) |
| Output | Synthetisierte Notes | Extrakt-basierte Notes |
| Halluzination | ~7-10% | 0% (per Konstruktion: 100% aus PDF) |
| Kosten | API-Kosten | 0 |

## Architektur

```
PDF
 +-- pdf_chunker (pdfplumber (Font-Size/Bold-Flag fuer Header-Erkennung))         [aus atomic-agent]
      +-- GLiNER (Zero-Shot NER, en)                 [NEU]
           | Typen: ["Theory","Concept","Method",
           |          "Metric","Model","Framework"]
           +-- Sentence-Cluster-Extractor            [NEU - kein QA]
                | Pro Konzept:
                | 1. Alle Saetze mit Konzept-Erwaehnung
                | 2. + je 1-2 Kontext-Saetze
                | 3. LexRank -> Top 3-5 Saetze
                +-- FOSS Stage 6                    [git submodule]
                     | . BM25+Embedding+RRF (CrossRef)
                     | . NLI-DeBERTa (Widersprueche)
                     | . Regex Critic
                     +-- Pydantic AtomicNote         [NEU]
                          +-- Jinja2 Adapter         [NEU]
                               +-- obsidian.md.jinja2
                               +-- generic.md.jinja2
                               +-- note.json.jinja2
```

## Komponenten

### 1. PDF-Extraktion (pdf_chunker)
Aus atomic-agent (git submodule). pdfplumber (Font-Size/Bold-Flag fuer Header-Erkennung) + Schwartz-Hearst-Akronyme.
Bewusst nicht in v1: marker/Nougat (zu schwer). v2: --extractor marker.

### 2. Planner: GLiNER
- Modell: urchade/gliner_medium-v2.1 (English, ~200MB, Apache 2.0)
- Typen: ["Theory", "Concept", "Method", "Metric", "Model", "Framework"]
- Dedup: rapidfuzz >= 90 auf Konzeptnamen
- Fallback: KeyBERT wenn GLiNER < 3 Konzepte findet

### 3. Extractor: Sentence-Cluster + LexRank
**KEIN SQuAD QA** — SQuAD-Modelle liefern kurze Spans (1-10W), keine Saetze.

Algorithmus pro Konzept:
1. Sentence-Tokenize Volltext (nltk sent_tokenize)
2. Finde alle Saetze die den Konzeptnamen enthalten (rapidfuzz >= 85)
3. Erweitere: +1 Satz vor und nach jedem Treffer (Kontext-Fenster)
4. Dedupliziere ueberlappende Cluster
5. LexRank (sumy) ueber den Cluster -> Top 3-5 Saetze
6. Jeder Satz behaelt seinen (S. N)-Anker aus dem Quelltext

Output: 3-8 Saetze, 100% wortwoertlich aus PDF, keine Halluzination.

### 4. FOSS Stage 6 (shared/)
Kein Submodule. Gemeinsamer Code liegt in shared/ im Monorepo.
Import: `from shared.embeddings import ...`

Wiederverwendet:
- pipeline/embeddings.py (MiniLM, English)
- agents/verifier.py (ENABLE_LLM=0)
- agents/cross_reference.py (ENABLE_LLM=0)
- agents/critic.py (ENABLE_LLM=0)
- agents/confidence.py (bereits FOSS)

### 5. Pydantic Core-Model

```python
class AtomicNote(BaseModel):
    title: str
    concept_type: str
    extracted_body: list[str]   # Saetze aus PDF, mit (S. N)
    source_anchors: list[dict]  # [{page, quote, rapidfuzz_score}]
    related: list[str]
    tags: list[str]
    source_file: str
    pipeline: str = "foss-atomic"
    hallucination_rate: float = 0.0
    extraction_coverage: float  # Anteil Saetze im Original verifizierbar
```

Verifikation 0%-Halluzination: Jeder Satz aus extracted_body muss
rapidfuzz.partial_ratio(satz, pdf_volltext) >= 95 erfuellen.
Vor dem Match: (S. N)-Anker via Regex strippen.

### 6. Jinja2 Adapter
Templates (templates/):
- obsidian.md.jinja2  (Frontmatter, Wikilinks, Callouts)
- generic.md.jinja2   (Reines Markdown)
- note.json.jinja2    (JSON)

User kann Templates ueberschreiben via --template obsidian /path/custom.jinja2

### 7. FOSS-Eval + JSONL-Vergleich
Deterministische Eval (kein decision_engine):
- Anchor-Rate: Anteil Saetze mit (S. N)-Anker
- BM25-Coverage: Score der Saetze gegen PDF-Chunks
- Concept-Recall: Anteil GLiNER-Konzepte mit >= 3 Saetzen im Cluster
- 0%-Halluzination: rapidfuzz >= 95 fuer alle Saetze vs. PDF-Text

Output: identisch zu atomic-agent quality_history.jsonl.

## Projektstruktur

```
foss-atomic/
+-- pipeline/
|   +-- pdf_chunker.py         [via vendor/]
|   +-- gliner_planner.py
|   +-- sentence_extractor.py  [LexRank-basiert]
|   +-- adapter.py
+-- shared/               [SSOT - kein Submodule]
+-- eval/
|   +-- foss_eval.py
+-- templates/
|   +-- obsidian.md.jinja2
|   +-- generic.md.jinja2
|   +-- note.json.jinja2
+-- schemas/
|   +-- atomic_note.py
+-- orchestrator.py
+-- requirements.txt
+-- README.md
```

## CLI

```bash
python orchestrator.py \
  --source paper.pdf \
  --output obsidian \
  --out-dir ./notes \
  --device cpu \
  --eval-jsonl ./quality.jsonl
```

## Dependencies (requirements.txt)

```
gliner>=0.2
sentence-transformers>=3.0  # MiniLM (Stage 6)
sumy>=0.12                  # LexRank
nltk>=3.8                   # Sentence Tokenization
rank-bm25>=0.2
rapidfuzz>=3.0
jinja2>=3.1
pydantic>=2.0
transformers>=4.40          # NLI-DeBERTa (Stage 6)
# Fallbacks:
keybert>=0.8
```

Keine GPU-Pflicht. CPU-Profil (4 Cores, 8GB RAM):
- GLiNER: ~30-60s pro PDF
- LexRank: ~5-10s
- Stage 6: ~30-60s
- Gesamt: ~2-5 Minuten fuer typisches Paper (15-30 Seiten)

## Akzeptanzkriterien

- [ ] Pipeline laeuft auf Bates 2017 (EN, 12 S.) in unter 10 Minuten auf CPU
- [ ] Output: >= 3 Atomic Notes im Obsidian-Format
- [ ] 0%-Halluzination: alle Saetze rapidfuzz >= 95 vs. PDF-Text
- [ ] Anchor-Rate: >= 80% der Saetze haben (S. N)-Anker
- [ ] JSONL-Output kompatibel mit atomic-agent Dashboard
- [ ] Warnung wenn Nicht-EN-PDF erkannt (langdetect)
- [ ] README Quickstart: pip install + erster Run in < 5 Minuten

## Risiken

1. GLiNER < 3 Konzepte -> Fallback KeyBERT
2. Sentence-Cluster leer (Konzept zu abstrakt) -> Fallback LexRank global
3. Notes kuerzer/weniger informativ als LLM-Notes -> erwartet, Zweck ist Baseline
4. git submodule Komplexitaet -> Dokumentation in README

## Bewusst nicht in v1

- Coreference Resolution (fastcoref) -> v2
- Deutsch/multilingual -> v2 (xlm-roberta Modelle)
- marker/Nougat PDF-Parsing -> v2
- Concept-Hierarchie -> v2
- Streaming/Async -> v2
- Web-UI -> nicht geplant


## Quellen

*Eigene Konzeption 2026-05-20, basierend auf [[FOSS-NLP-Alternativen-Atomic-Agent]] und [[FOSS-Pipeline]]*
*Gemini 3.1 Pro Review 2026-05-20 (intern)*
