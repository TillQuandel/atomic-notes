# Atomic Agent

> Notebook LM gives you a conversation. Atomic Agent gives you a card catalog.

Turns your PDF library into a verified Obsidian card catalog —
atomic notes with anchor verification, citation fidelity, and tier classification.
Open-source, your API key, your data.

## Status

Early development — CLI tool, Obsidian plugin coming.

## What it does

- Extracts atomic concept notes from PDFs
- Verifies every citation anchor against the original text
- Classifies sources (CrossRef, OpenAlex, Retraction Watch)
- Writes directly into your Obsidian vault with wikilinks and PDF jump links
- Model-agnostic: Claude, GPT-4o, local models via Ollama

## Quick start

    pip install -r requirements.txt
    cp .env.example .env
    # Edit .env: set ATOMIC_AGENT_VAULT_PATH and ATOMIC_AGENT_PDF_BASE
    python orchestrator.py --source "path/to/paper.pdf" --dry-run

## Requirements

- Python 3.11+
- [Claude CLI](https://claude.ai/download) (Pro/Max subscription) or Anthropic API key
- Obsidian vault

## License

Apache 2.0
