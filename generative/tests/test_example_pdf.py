"""Smoke tests for the bundled example PDF.

Verifies that examples/zettelkasten-primer.pdf:
  1. exists and is larger than 5 KB
  2. contains more than 400 words of extractable text (via pdftotext)
  3. the extracted text contains the word 'atomic'

pdftotext is required (poppler-utils). It is available in CI (ubuntu + windows)
and is a documented prerequisite for running the pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Repo root is two levels above this file:
#   generative/tests/test_example_pdf.py -> generative/tests -> generative -> repo root
REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLE_PDF = REPO_ROOT / "examples" / "zettelkasten-primer.pdf"

# CI installs poppler on both OS; local contributors without it get a skip
# instead of a hard fail (pdftotext is a pipeline prerequisite, not a test bug).
needs_pdftotext = pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext (poppler-utils) not on PATH",
)


def test_example_pdf_exists_and_size():
    """The bundled PDF must exist and exceed 5 KB."""
    assert EXAMPLE_PDF.exists(), f"PDF not found: {EXAMPLE_PDF}"
    size = EXAMPLE_PDF.stat().st_size
    assert size > 5 * 1024, f"PDF is only {size} bytes; expected > 5120"


@needs_pdftotext
def test_example_pdf_word_count():
    """pdftotext must extract more than 400 words from the PDF."""
    result = subprocess.run(
        ["pdftotext", str(EXAMPLE_PDF), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    word_count = len(result.stdout.split())
    assert word_count > 400, f"pdftotext extracted only {word_count} words; expected > 400"


@needs_pdftotext
def test_example_pdf_contains_atomic():
    """The extracted text must contain the word 'atomic'."""
    result = subprocess.run(
        ["pdftotext", str(EXAMPLE_PDF), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "atomic" in result.stdout.lower(), "The word 'atomic' was not found in the extracted PDF text"
