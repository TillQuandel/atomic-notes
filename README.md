# atomic-notes

Atomic Notes transforms rich sources into verified atomic knowledge units.

The current implementation starts with PDFs, but the project is meant to be input- and output-independent: source adapters normalize different media into a common source representation, pipelines create atomic notes, and renderers/exporters decide where those notes go.

```text
source input -> normalized source -> atomic-note pipeline -> output renderer
```

PDF input and Obsidian-style Markdown are the first supported path, not the whole product.

## Status

**generative v0.3.79** · **extractive v0.2.0** · last updated: 2026-05-29

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

```bash
cd generative
python orchestrator.py --source <pdf> --dry-run
python orchestrator.py --source <pdf>
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

## Development Notes

The project is still early and carries some historical naming in older internal docs and eval data. New code and public documentation should use:

- `generative` for the LLM synthesis pipeline
- `extractive` for the local sentence-extraction pipeline
- `internal` for dashboards, calibration, and development-only tooling

## License

Apache 2.0
