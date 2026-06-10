# extractive pipeline

The extractive pipeline creates atomic-note drafts from source sentences without free-text generation.

It is local-first: text is extracted from the PDF, concepts are detected, and note bodies are assembled from sentences found in the source. That makes the output less polished than the generative pipeline, but easier to audit and useful as a low-hallucination baseline.

## Quick Start

```bash
git clone https://github.com/TillQuandel/atomic-notes
cd atomic-notes
pip install -e ".[extractive]"
python -m extractive.orchestrator --source paper.pdf --output obsidian --out-dir ./notes
```

## How It Works

1. `pdfplumber` extracts text with layout detection.
2. GLiNER finds candidate concepts via zero-shot NER.
3. LexRank selects relevant sentences per concept.
4. Renderers write Obsidian Markdown, plain Markdown, or JSON.

## Output Formats

| Flag | Format |
|---|---|
| `--output obsidian` | Obsidian-style Markdown with frontmatter |
| `--output md` | Plain Markdown |
| `--output json` | JSON |

## Limits

- English PDFs are the current primary target.
- Output is extractive, not synthesized.
- First run may download the GLiNER model.

## Compare With `generative/`

`generative/` synthesizes better standalone notes and performs richer verification and critique. `extractive/` trades polish for local execution and direct source grounding.
