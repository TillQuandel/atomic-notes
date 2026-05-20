# foss-atomic

PDF to Atomic Notes without any API key. Pure FOSS, runs offline.

## Quickstart

    git clone https://github.com/TillQuandel/atomic-notes
    cd atomic-notes
    pip install -r foss/requirements.txt
    python foss/orchestrator.py --source paper.pdf --output obsidian --out-dir ./notes

## How it works

1. **pdfplumber** extracts text with layout detection (no poppler needed)
2. **GLiNER** finds concepts via Zero-Shot NER — no API, no training
3. **LexRank** (sumy) selects the most relevant sentences per concept
4. Notes written in Obsidian format, plain Markdown, or JSON

## Hallucination rate: always 0%

Every sentence is extracted verbatim from the source PDF — no synthesis, no generation.

## Output formats

| Flag | Format |
|---|---|
| --output obsidian | Obsidian Markdown with frontmatter |
| --output md | Plain Markdown |
| --output json | JSON |

## Compare with LLM pipeline (agent/)

Both pipelines share the same eval format. The dashboard shows both side by side.

## Limitations (v1)

- English PDFs only (German support in v2)
- Extractive notes — less readable than LLM-synthesized notes
- First run downloads GLiNER model (~200MB)
