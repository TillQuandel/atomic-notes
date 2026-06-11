# examples/

## zettelkasten-primer.pdf

A self-authored introductory text on atomic note-taking (concepts: atomic note,
Zettelkasten, progressive summarization, linking, emergence). It is an original
work, licensed under Apache 2.0, and safe to redistribute as part of this repo.

Running the pipeline on this file produces a set of atomic notes in your
configured output directory (the Obsidian vault path set via
`ATOMIC_AGENT_VAULT_PATH` in `generative/.env`).

### Regenerating the PDF

The PDF is generated from the Python source in `generate_example_pdf.py` using
[fpdf2](https://py-pdf.github.io/fpdf2/). fpdf2 is not a project dependency;
install it just for regeneration:

```bash
pip install fpdf2
python examples/generate_example_pdf.py
```

### Showcase notes

Example output notes produced by running the pipeline on this file are not yet
included. They will be added in a future milestone so that new contributors can
see what the pipeline produces without running it themselves.
