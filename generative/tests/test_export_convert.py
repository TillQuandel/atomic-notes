"""Tests für pipeline.export_convert — portables Markdown (F2) → docx/pdf/html/
odt/epub via pandoc+typst (Output-Projekt F3). Reine Konvertierungs-Schicht,
kein Pipeline-/CLI-/GUI-Wiring (folgt in F4).

Alle Tests außer der reinen Format-Validierung brauchen echtes pandoc+typst
(via `pypandoc_binary`+`typst`, optionale `[export]`-Gruppe) — Modul-Level-
Guard, damit ohne Installation sauber geskippt statt fehlgeschlagen wird.
"""

import re
import zipfile

import pytest

pytest.importorskip("pypandoc")
pytest.importorskip("typst")

from generative.pipeline.export_convert import (
    EXPORT_FORMATS,
    _build_typst_source,
    convert_portable_md,
    export_available,
)
from generative.pipeline.portable_md import gfm_anchor_slug


def test_export_formats_are_exactly_the_five_core_formats():
    assert EXPORT_FORMATS == ("docx", "pdf", "html", "odt", "epub")


def test_unknown_format_raises_value_error_no_pandoc_needed(tmp_path):
    with pytest.raises(ValueError) as exc:
        convert_portable_md("# H\n\nAbsatz.", "rtf", tmp_path / "out.rtf")
    msg = str(exc.value)
    for fmt in EXPORT_FORMATS:
        assert fmt in msg


def test_export_available_true_with_installed_deps():
    ok, detail = export_available()
    assert ok is True
    assert detail


def test_html_smoke_umlauts_footnote_title(tmp_path):
    md = (
        "# Überschrift mit „Anführungszeichen“\n\n"
        "Ein Absatz mit Umlauten (äöüß) und einer Fußnote.[^1]\n\n"
        "[^1]: Beleg-Quelle, S. 1.\n"
    )
    out = tmp_path / "out.html"
    result = convert_portable_md(md, "html", out, title="Testtitel", lang="de")
    assert result == out
    html = out.read_text(encoding="utf-8")
    assert "Anführungszeichen" in html
    assert "„" in html and "“" in html
    assert "äöüß" in html
    assert "Fußnote" in html
    assert re.search(r'id="fn1"', html)
    assert "<title>Testtitel</title>" in html


def test_anchor_slugs_match_gfm_anchor_slug_for_realistic_headings(tmp_path):
    headings = [
        "Atomic Notes: Eine Note enthält genau eine Idee",
        "Übersicht über Größen",
        "3 Regeln für 2026",
    ]
    md = "\n\n".join(f"## {h}\n\nAbsatz zu {h}." for h in headings)
    out = tmp_path / "out.html"
    convert_portable_md(md, "html", out)
    html = out.read_text(encoding="utf-8")
    ids = re.findall(r'<h2 id="([^"]+)"', html)
    assert len(ids) == len(headings)
    for heading, pandoc_id in zip(headings, ids):
        assert pandoc_id == gfm_anchor_slug(heading), (
            f"Anker-Mismatch für {heading!r}: pandoc={pandoc_id!r} vs. gfm_anchor_slug={gfm_anchor_slug(heading)!r}"
        )


def test_docx_smoke_zip_content_and_core_properties(tmp_path):
    md = "# Belegtitel\n\nEin Beleg-Absatz mit Umlauten: Prüfungsfähig.\n"
    out = tmp_path / "out.docx"
    convert_portable_md(md, "docx", out, title="Meta-Titel", author="Meta-Autor", lang="de")
    assert out.exists()
    assert zipfile.is_zipfile(out)
    with zipfile.ZipFile(out) as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")
        assert "Prüfungsfähig" in doc_xml
        core_xml = z.read("docProps/core.xml").decode("utf-8")
        assert "Meta-Titel" in core_xml
        assert "Meta-Autor" in core_xml


def test_pdf_smoke_magic_bytes_and_size(tmp_path):
    md = "# Titelüberschrift\n\nEin laengerer Absatz als Fuellstoff fuer den PDF-Smoke-Test.\n"
    out = tmp_path / "out.pdf"
    result = convert_portable_md(md, "pdf", out, title="PDF-Test", lang="de")
    assert result == out
    data = out.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 1024


def test_typst_intermediate_source_has_lang_and_hyphenate():
    """Injektions-/Template-Logik direkt testen, ohne PDF-Compile (Deliverable-2-Spec)."""
    src = _build_typst_source("# H\n\nAbsatz.", lang="de")
    assert 'lang: "de"' in src
    assert "hyphenate: true" in src


def test_internal_anchor_link_is_not_dead(tmp_path):
    md = "# Erster Abschnitt\n\nSiehe [Zweiter Abschnitt](#zweiter-abschnitt).\n\n## Zweiter Abschnitt\n\nZielinhalt.\n"
    out = tmp_path / "out.html"
    convert_portable_md(md, "html", out)
    html = out.read_text(encoding="utf-8")
    assert 'href="#zweiter-abschnitt"' in html
    assert re.search(r'id="zweiter-abschnitt"', html)
