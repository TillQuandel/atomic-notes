# Architecture

A short map for contributors: how the repository is laid out, how a source flows
through the pipeline, and why there are two pipelines. For usage see
[README.md](README.md); for backend details see
[generative/README.md](generative/README.md).

## The contract

The product is **input- and output-independent**. Adapters normalize a source
into a common representation, a pipeline turns it into atomic notes, and a
renderer decides where the notes go.

```text
source input  ->  normalized source  ->  atomic-note pipeline  ->  output renderer
   (PDF)              (text + meta)         (generative/extractive)     (Obsidian MD)
```

PDF input and Obsidian-style Markdown are the first supported path, not the whole
product.

## Module map

| Path | Role |
|------|------|
| `generative/` | LLM-based synthesis pipeline (planner, extractor, verifier, critic, quality gates), the CLI entry point (`generative.cli:main`), and the FastAPI GUI (`generative/gui/`). |
| `extractive/` | Local sentence-extraction pipeline (GLiNER + LexRank). No free generation — notes are built from source sentences. |
| `shared/` | Schemas, the SQLite DB schema, and cross-pipeline utilities used by both pipelines. |
| `lib/decision_engine/` | Aggregation + decision rules (packaged as `decision_engine`). |
| `tests/`, `generative/tests/`, `extractive/tests/` | Test suites. The canonical CI suite is `generative` + `lib/decision_engine/tests` (LLM-free). |
| `examples/` | Bundled `zettelkasten-primer.pdf` for the quickstart and the demo script. |

## Generative pipeline stages

A run (`atomic-notes run --source <pdf>`) moves through the pipeline; each stage
logs a `[n/7]` banner:

1. **PDF extract + chunk** — `pdftotext` → text → chapter chunks. (`[0/7]` source
   enrichment fills author/year metadata when derivable.)
2. **Context builder** — scans the target vault so later stages can dedupe and link.
3. **Quality agent** — assesses source text quality (warns on scanned/thin PDFs; fail-open).
4. **Planner** — builds the concept plan (which atomic notes to attempt).
5. **Extractor** — synthesizes the notes (concepts processed in parallel).
6. **Verifier + cross-reference + critic** — checks claims against the source,
   resolves siblings/duplicates (4-stage entity resolution), scores each note.
7. **Vault writer** — writes notes (or, with `--dry-run`, previews a diff).

Post-pipeline (optional, logged as `[8/8]`): **quality eval** — faithfulness eval
that can be deferred to `reeval_baseline.py`.

```mermaid
flowchart LR
    PDF[PDF] --> X[extract + chunk]
    X --> C[context builder]
    C --> Q[quality agent]
    Q --> P[planner]
    P --> E[extractor]
    E --> V[verifier + cross-ref + critic]
    V --> W[vault writer]
    W --> Notes[Atomic Notes]
```

## ADR: why two pipelines (and why they are not merged)

`generative/` and `extractive/` are kept **separate on purpose**. They are not a
duplication to consolidate:

- **Generative** synthesizes prose with LLM stages — the higher-quality path when
  synthesis is useful and model access is acceptable.
- **Extractive** builds notes from source sentences only, local-first, no free
  generation — a **privacy-preserving baseline** and a **low-hallucination
  comparison path** (it never invents text, so it bounds what "good" looks like).

Merging them would collapse the comparison baseline into the thing it measures and
remove the local-only path. They share `shared/` (schemas, DB) but stay distinct
at the pipeline level by design.

## Setup & dependencies

Dependencies are managed with **uv** and pinned in `uv.lock` (resolved for Windows,
Linux, and macOS). `torch` is declared directly; on Windows/Linux it is mapped to
the CPU wheel index (no large CUDA wheels), on macOS it resolves from PyPI (CPU-only
there). CI runs the suite on all three OSes. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the dev setup and a GPU override.
