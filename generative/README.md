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
# all commands from the repository root
pip install -e .
cp generative/.env.example generative/.env
# Edit generative/.env for local paths and model/backend settings.
atomic-notes doctor   # preflight: poppler, backend login, vault path
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

### litellm model strings

litellm uses `provider/model` strings. Set them in `.env`, or leave them commented
to inherit the defaults from `config.py` (sonnet for the heavy slot, haiku for the
light stages):

| Provider | `ATOMIC_AGENT_MODEL_OPUS` (heavy slot) example | Extra setup |
|----------|-----------------------------------------------|-------------|
| Anthropic | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Ollama (local) | `ollama/mistral` | `OLLAMA_API_BASE=http://localhost:11434` |

See the [litellm provider list](https://docs.litellm.ai/docs/providers) for the
full set of supported providers and exact model strings.

Failure behavior: if the `claude` CLI is missing, not logged in, or the rate window
is exhausted (429), the pipeline fails fast with an actionable message (install/login
hint, window reset, litellm alternative) instead of retrying or dumping a traceback.
`atomic-notes doctor` checks the full setup up front.

Obsidian output is currently the best-tested target, but the pipeline should not be treated as Obsidian-specific long term.

## Export Formats

Beyond writing notes into the vault/inbox, a run can additionally render its notes into portable file formats — for archiving, sharing, or reading outside Obsidian. Opt-in via `--export-format` (CLI) or the run settings (GUI); without it, nothing changes.

Core set:

| Format | Purpose |
|---|---|
| `json` | Canonical machine-readable export (schema in `pipeline/note_json.py`) — for external tooling. |
| `obsidian-md` | Copy of the `.md` files already written to the vault/inbox — no new rendering, just gathered in the export folder. |
| `portable-md` | Standard CommonMark + footnotes (no Obsidian wikilinks/callouts) — readable in any Markdown viewer. |
| `docx` | Microsoft Word, via pandoc. |
| `pdf` | PDF, via pandoc + typst. |
| `html` | Standalone HTML page, via pandoc. |

Optional, additionally selectable: `odt` (OpenDocument Text), `epub` (EPUB e-book) — both via pandoc.

`docx`/`pdf`/`html`/`odt`/`epub` need the optional `export` dependency group (pandoc+typst, pip-only — no system pandoc/typst binary required):

```bash
pip install "atomic-notes[export]"
```

`json`/`obsidian-md`/`portable-md` work without it.

CLI:

```bash
atomic-notes run --source "path/to/paper.pdf" --dry-run \
  --export-format json,portable-md,docx,pdf,html \
  --export-dir path/to/exports/
```

Formats are comma-separated, case-insensitive, deduplicated. `--export-dir` defaults to `generative/.cache/exports/<pdf-stem>/`.

GUI: pick formats in the run settings ("Export-Formate") before starting a run — a "mehr Formate" toggle exposes `odt`/`epub`. Exported files appear under "Exporte" in the results section once the run finishes, downloadable individually.

Planned, not yet selectable: `rtf`, `latex`, `typst`, `mediawiki`, `rst`, `epub3` (decision 2026-07-05). Requesting one of these raises a "planned, not yet enabled" error instead of a generic "unknown format" one.

## Relationship To `extractive/`

`extractive/` is the local sentence-extraction pipeline. It is useful for privacy-preserving runs, low-hallucination baselines, and comparisons against generated notes.
