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
pip install -e .   # from the repository root
cp .env.example .env
# Edit .env for local paths and model/backend settings.
atomic-notes run --source "path/to/paper.pdf" --dry-run
```

## Requirements

- Python 3.12+
- `pdftotext`/`pdfinfo` (poppler-utils) on PATH
- An LLM backend (see below)
- A writable output target for generated notes

## Backends

| Backend | Set via | Needs | Notes |
|---|---|---|---|
| `subscription` (default) | `ATOMIC_AGENT_BACKEND=subscription` | Claude Pro/Max subscription + installed, logged-in Claude Code CLI (`claude`) | No API key, no extra cost beyond the subscription. Headless `claude -p` is an officially documented CLI mode. Subject to the subscription's 5-hour rate window — roughly 8 full pipeline runs per window, then HTTP 429 until reset. |
| `litellm` | `ATOMIC_AGENT_BACKEND=litellm` | Provider API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or local Ollama) | Pay-as-you-go; no rate-window coupling. Model IDs in `.env` use litellm naming. |

Model IDs are configured once in litellm format (e.g. `anthropic/claude-sonnet-4-6`);
the subscription backend maps them to CLI shorthands internally.

Obsidian output is currently the best-tested target, but the pipeline should not be treated as Obsidian-specific long term.

## Relationship To `extractive/`

`extractive/` is the local sentence-extraction pipeline. It is useful for privacy-preserving runs, low-hallucination baselines, and comparisons against generated notes.
