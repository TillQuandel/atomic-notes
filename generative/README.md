# generative pipeline

The generative pipeline creates synthesized atomic notes from source material.

It is the high-quality path for turning PDFs and later other rich sources into standalone knowledge units. It uses model calls for concept planning, extraction, verification, cross-reference checks, and critique.

## What It Does

- Plans candidate concepts from a source
- Extracts standalone atomic-note drafts
- Verifies anchors against the source text
- Checks related concepts and possible conflicts
- Scores notes with quality gates before writing/exporting
- Tracks run data for evaluation

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env for local paths and model/backend settings.
python orchestrator.py --source "path/to/paper.pdf" --dry-run
```

## Requirements

- Python 3.11+
- Claude CLI or a configured API backend
- A writable output target for generated notes

Obsidian output is currently the best-tested target, but the pipeline should not be treated as Obsidian-specific long term.

## Relationship To `extractive/`

`extractive/` is the local sentence-extraction pipeline. It is useful for privacy-preserving runs, low-hallucination baselines, and comparisons against generated notes.
