# atomic-notes

[![CI](https://github.com/TillQuandel/atomic-notes/actions/workflows/ci.yml/badge.svg)](https://github.com/TillQuandel/atomic-notes/actions/workflows/ci.yml)

Atomic Notes transforms rich sources into verified atomic knowledge units.

The current implementation starts with PDFs, but the project is meant to be input- and output-independent: source adapters normalize different media into a common source representation, pipelines create atomic notes, and renderers/exporters decide where those notes go.

```text
source input -> normalized source -> atomic-note pipeline -> output renderer
```

PDF input and Obsidian-style Markdown are the first supported path, not the whole product.

## Status

**generative v0.3.136** · **extractive v0.2.0** · last updated: 2026-06-10

413 tests passing. An independent multi-rater project assessment (2026-06-10) and the
resulting roadmap live in `internal/docs/` — see `2026-06-10-projekt-bewertung.md`
and `m1-installierbarkeit-plan.md`.

## Roadmap

1. **M1 — installable by strangers**: packaging (`pyproject.toml`, entry points), CI,
   preflight `doctor`, quickstart docs. Plan: `internal/docs/m1-installierbarkeit-plan.md`.
2. **M2 — trustworthy output**: gold-standard coverage measurement, threshold
   calibration, PDF text-quality gate + OCR fallback, a small reproducible benchmark.
3. **M3 — staying power**: configurable note conventions beyond Obsidian, REST/API
   layer (issues #9–#11).

## Repository Layout

```text
generative/     LLM-based synthesis pipeline with verifier, critic, and quality gates
extractive/     Local extractive pipeline; no free generation, source sentences only
shared/         Shared schemas, database schema, and cross-pipeline utilities
tests/          Repository-level tests, currently focused on the extractive pipeline
internal/       Internal evaluation assets and development notes, not user-facing product
```

`internal/dashboard/` is used for evaluation and debugging while developing the pipelines. It is not part of the public user workflow.

## Pipelines

### Generative

The generative pipeline synthesizes standalone atomic notes from source material. It uses LLM stages for planning, extraction, verification, cross-reference checks, and critique. This is the higher-quality path when synthesis is useful and API/model access is acceptable.

No API key is required: the default backend drives the Claude Code CLI, so a Claude
Pro/Max subscription plus a logged-in CLI is enough. An API-based backend (litellm:
Anthropic, OpenAI, Ollama, …) is available via `ATOMIC_AGENT_BACKEND=litellm`. See
`generative/README.md` for details and limits.

```bash
pip install -e .
atomic-notes run --source <pdf> --dry-run
atomic-notes run --source <pdf>
```

### Extractive

The extractive pipeline builds notes from source sentences. It is local-first and does not freely generate prose, so it is useful as a privacy-preserving baseline and as a low-hallucination comparison path.

```bash
python extractive/orchestrator.py --source <pdf> --output obsidian --out-dir ./notes
python extractive/orchestrator.py --source <pdf> --output json --out-dir ./notes
```

## Output Direction

The long-term output contract is a structured atomic note: title, body, source anchors, source metadata, quality status, and optional links/tags. Obsidian Markdown is one renderer. Plain Markdown, JSON, ZIP exports, and other PKM formats should be renderer concerns rather than pipeline assumptions.

## Input Direction

PDF is the first adapter. Future adapters should normalize HTML/articles, RSS items, transcripts, podcasts, videos, and other concept-rich sources into the same source model before the pipeline runs.

Current Stage-0 baseline is `pdftotext`. A June 2026 A/B probe evaluated pdfplumber and GROBID but did not show a robust advantage over `pdftotext`; pdfplumber also regressed on a two-column PDF through glued words and lower word yield. The pdfplumber adapter is therefore parked until a focused comparison shows a yield or grounding gain over `pdftotext` beyond run noise.

## Development Notes

The project is still early and carries some historical naming in older internal docs and eval data. New code and public documentation should use:

- `generative` for the LLM synthesis pipeline
- `extractive` for the local sentence-extraction pipeline
- `internal` for dashboards, calibration, and development-only tooling

## License

Apache 2.0
