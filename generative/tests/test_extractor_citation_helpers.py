"""Tests für die bisher ungetesteten Extractor-Prompt-Helfer nach der
CitationMeta-Umstellung (#96 E3a): `_short_author`, `_format_source_meta`,
`_clean_source_file_display`. Vorher lasen diese Helfer aus einem lose
typisierten `dict[str, str]` — jetzt aus `CitationMeta` (kanonisch)."""

from generative.agents.extractor import _short_author, _format_source_meta, _clean_source_file_display
from generative.schemas.citation import CitationMeta


def _cm(author=None, year=None, title=None, source_file="quelle.pdf") -> CitationMeta:
    return CitationMeta(author=author, year=year, title=title, doi=None, source_file=source_file)


class TestShortAuthor:
    def test_single_author(self):
        assert _short_author(_cm(author="Bertram")) == "Bertram"

    def test_two_authors_semicolon(self):
        assert _short_author(_cm(author="Schlebbe; Greifeneder")) == "Schlebbe & Greifeneder"

    def test_two_authors_und(self):
        assert _short_author(_cm(author="Schlebbe und Greifeneder")) == "Schlebbe & Greifeneder"

    def test_three_authors_et_al(self):
        assert _short_author(_cm(author="Gross; Latham; Folk")) == "Gross et al."

    def test_crossref_lastname_comma_format(self):
        assert _short_author(_cm(author="Bertram, Jutta")) == "Bertram"

    def test_no_author_falls_back_to_placeholder(self):
        assert _short_author(_cm(author=None)) == "Autor"

    def test_empty_author_falls_back_to_placeholder(self):
        assert _short_author(_cm(author="")) == "Autor"


class TestCleanSourceFileDisplay:
    def test_zotero_filename_with_year(self):
        out = _clean_source_file_display(_cm(source_file="Bertram - 2019 - Angeleitetes Lernen.pdf"))
        assert out == "Bertram - 2019 - Angeleitetes Lernen.pdf"

    def test_affiliation_coauthor_dropped(self):
        src = (
            "Mahmood und University of the Punjab - 2016 - Do People Overestimate Their Information Literacy Skills.pdf"
        )
        out = _clean_source_file_display(_cm(source_file=src))
        assert "University of the Punjab" not in out
        assert "Mahmood" in out

    def test_unparsable_filename_passes_through_unchanged(self):
        src = "randomfile123.pdf"
        assert _clean_source_file_display(_cm(source_file=src)) == src


class TestFormatSourceMeta:
    def test_full_meta_all_lines_present(self):
        out = _format_source_meta(_cm(author="Bertram", year="2019", title="Angeleitetes Lernen"))
        assert "Autor: Bertram" in out
        assert "Titel: Angeleitetes Lernen" in out
        assert "Jahr: 2019" in out
        assert "Datei:" in out

    def test_missing_year_shows_o_j_marker_and_hint(self):
        out = _format_source_meta(_cm(author="Bertram", year=None, title="Angeleitetes Lernen"))
        assert "Jahr: [o. J.]" in out
        # LLM-Hinweis: exakt "[o. J.]" verwenden, kein Jahr erfinden
        assert "o. J." in out
        assert "erfind" in out.lower()

    def test_present_year_has_no_o_j_hint(self):
        out = _format_source_meta(_cm(author="Bertram", year="2019", title="X"))
        assert "erfind" not in out.lower()

    def test_missing_author_and_title_omit_those_lines(self):
        out = _format_source_meta(_cm(author=None, year=None, title=None))
        assert "Autor:" not in out
        assert "Titel:" not in out
        assert "Jahr: [o. J.]" in out
