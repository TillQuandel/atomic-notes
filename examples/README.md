# examples/

## zettelkasten-primer.pdf

A self-authored introductory text on atomic note-taking (concepts: atomic note,
Zettelkasten, progressive summarization, linking, emergence). It is an original
work created for this repository (SPDX-License-Identifier: Apache-2.0, same
copyright holder as the project), and safe to redistribute as part of this repo.

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

An example note produced by running the pipeline on this file is shown in the
top-level `README.md` under "Example output", so new contributors can see what
the pipeline produces without running it themselves.
