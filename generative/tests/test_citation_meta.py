"""Tests für generative.schemas.citation (CitationMeta, #96 E3a).

CitationMeta ist die Single Source of Truth für Zitations-Metadaten (Autor/Jahr/
Titel/DOI) — ersetzt die vorherigen ZWEI dict-Welten `pdf_meta` (Extractor/
Planner, ohne CrossRef-Korrekturen) und `enriched_meta` (nur Vault-Writer, mit
CrossRef-Korrekturen). Deckt ab: Factory-Logik (CrossRef-Override-Blocklogik,
Filename-Year-Vorrang), `[o. J.]`-Regel, `short_label`, Golden-Äquivalenz zum
Vor-Refactor-Rendering.
"""

from datetime import date as _date
from unittest.mock import patch

from generative.schemas.citation import CitationMeta, build_citation_meta, crossref_override_blocked
from generative.schemas.atomic_note import QualityReport, AtomicNoteDraft, TextAnchor
from generative.pipeline import vault_writer


def _qr(**kw) -> QualityReport:
    defaults = dict(peer_reviewed=None, citation_count=None, retracted=False, flags=[])
    defaults.update(kw)
    return QualityReport(**defaults)


# ---- display_year / short_label ------------------------------------------------


class TestDisplayYear:
    def test_year_present(self):
        c = CitationMeta(author="Bertram", year="2019", title="T", doi=None, source_file="x.pdf")
        assert c.display_year == "2019"

    def test_year_missing(self):
        c = CitationMeta(author="Bertram", year=None, title="T", doi=None, source_file="x.pdf")
        assert c.display_year == "[o. J.]"

    def test_year_empty_string_treated_as_missing(self):
        c = CitationMeta(author="Bertram", year="", title="T", doi=None, source_file="x.pdf")
        assert c.display_year == "[o. J.]"


class TestShortLabel:
    def test_with_year(self):
        c = CitationMeta(author="Bertram", year="2019", title="T", doi=None, source_file="x.pdf")
        assert c.short_label == "Bertram 2019"

    def test_without_year_uses_o_j_marker(self):
        c = CitationMeta(author="Bertram", year=None, title="T", doi=None, source_file="x.pdf")
        assert c.short_label == "Bertram [o. J.]"

    def test_multi_author_et_al(self):
        c = CitationMeta(author="Schlebbe und Greifeneder", year="2020", title="T", doi=None, source_file="x.pdf")
        assert c.short_label == "Schlebbe et al. 2020"

    def test_crossref_format_lastname_comma_firstname(self):
        c = CitationMeta(author="Bertram, Jutta", year="2019", title="T", doi=None, source_file="x.pdf")
        assert c.short_label == "Bertram 2019"

    def test_no_author_falls_back_to_filename_stem(self):
        c = CitationMeta(author=None, year="2019", title="T", doi=None, source_file="unknown-source.pdf")
        assert c.short_label == "unknown-source"


class TestAsMetaDict:
    def test_full(self):
        c = CitationMeta(author="Bertram", year="2019", title="T", doi="10.1/x", source_file="x.pdf")
        assert c.as_meta_dict() == {"Author": "Bertram", "Year": "2019", "Title": "T"}

    def test_partial_omits_missing_keys(self):
        c = CitationMeta(author=None, year="2019", title=None, doi=None, source_file="x.pdf")
        assert c.as_meta_dict() == {"Year": "2019"}


# ---- crossref_override_blocked / build_citation_meta ---------------------------


class TestCrossrefOverrideBlocked:
    def test_no_title_match_at_all_not_blocked(self):
        qr = _qr(doi_from_title_match=False, crossref_title=None)
        assert crossref_override_blocked(qr, "Some Title") is False

    def test_confident_title_match_not_blocked(self):
        qr = _qr(doi_from_title_match=True, crossref_title="Angeleitetes Lernen im Fernstudium")
        assert crossref_override_blocked(qr, "Angeleitetes Lernen im Fernstudium") is False

    def test_weak_title_match_blocked(self):
        qr = _qr(doi_from_title_match=True, crossref_title="Ein völlig anderes Werk über Biologie")
        assert crossref_override_blocked(qr, "Angeleitetes Lernen im Fernstudium") is True

    def test_hard_doi_match_never_blocked_even_if_titles_differ(self):
        # doi_from_title_match=False -> DOI kam aus --doi/hartem Enrichment, kein Raten
        qr = _qr(doi_from_title_match=False, crossref_title="Ein anderer Titel")
        assert crossref_override_blocked(qr, "Angeleitetes Lernen") is False


class TestBuildCitationMeta:
    def test_no_crossref_data_keeps_pdf_meta(self):
        pdf_meta = {"Author": "Bertram", "Year": "2019", "Title": "Angeleitetes Lernen"}
        qr = _qr()
        c = build_citation_meta(pdf_meta, qr, "Angeleitetes Lernen", "Bertram - 2019 - X.pdf")
        assert c.author == "Bertram"
        assert c.year == "2019"
        assert c.title == "Angeleitetes Lernen"

    def test_confident_crossref_overrides_title_and_author(self):
        pdf_meta = {"Author": "Bertram", "Year": "2019", "Title": "Angeleitetes Lernen"}
        qr = _qr(
            doi_from_title_match=True,
            crossref_title="Angeleitetes Lernen im Fernstudium",
            crossref_author="Bertram, Jutta",
        )
        c = build_citation_meta(pdf_meta, qr, "Angeleitetes Lernen", "Bertram - 2019 - X.pdf")
        assert c.title == "Angeleitetes Lernen im Fernstudium"
        assert c.author == "Bertram, Jutta"

    def test_weak_crossref_match_blocked_keeps_pdf_meta(self):
        pdf_meta = {"Author": "Bertram", "Year": "2019", "Title": "Angeleitetes Lernen"}
        qr = _qr(
            doi_from_title_match=True,
            crossref_title="Ein völlig anderes Werk über Biologie",
            crossref_author="Müller, Hans",
        )
        c = build_citation_meta(pdf_meta, qr, "Angeleitetes Lernen", "Bertram - 2019 - X.pdf")
        assert c.title == "Angeleitetes Lernen"
        assert c.author == "Bertram"

    def test_filename_year_has_precedence_over_crossref_year(self):
        # Dateiname "... - 2006 - ..." -> fb_year="2006" ist gesetzt -> CrossRef
        # darf das Jahr NICHT überschreiben (v28-Regel), selbst wenn confident.
        pdf_meta = {"Author": "Hiatt", "Year": "2006", "Title": "ADKAR"}
        qr = _qr(doi_from_title_match=False, crossref_year="2023")
        c = build_citation_meta(pdf_meta, qr, "ADKAR", "Hiatt - 2006 - ADKAR.pdf")
        assert c.year == "2006"

    def test_crossref_year_fills_when_no_filename_year(self):
        # Dateiname ohne Jahressegment -> fb_year fehlt -> CrossRef darf das Jahr setzen.
        pdf_meta = {"Author": "Hiatt", "Title": "ADKAR"}
        qr = _qr(doi_from_title_match=False, crossref_year="2006")
        c = build_citation_meta(pdf_meta, qr, "ADKAR", "Hiatt - ADKAR.pdf")
        assert c.year == "2006"

    def test_missing_year_everywhere_yields_o_j_display(self):
        pdf_meta = {"Author": "Unbekannt", "Title": "X"}
        qr = _qr()
        c = build_citation_meta(pdf_meta, qr, "X", "Unbekannt - X.pdf")
        assert c.year is None
        assert c.display_year == "[o. J.]"

    def test_doi_defensively_read_from_pdf_meta(self):
        # pdf_meta traegt heute nie einen DOI-Key (kein Producer befuellt ihn) —
        # CitationMeta.doi ist defensiv fuer zukuenftige Producer vorbereitet.
        pdf_meta = {"Author": "X", "Year": "2020", "Title": "Y", "DOI": "10.1/abc"}
        qr = _qr()
        c = build_citation_meta(pdf_meta, qr, "Y", "X - 2020 - Y.pdf")
        assert c.doi == "10.1/abc"

    def test_source_file_carried_through(self):
        pdf_meta = {"Author": "X", "Year": "2020", "Title": "Y"}
        qr = _qr()
        c = build_citation_meta(pdf_meta, qr, "Y", "X - 2020 - Y.pdf")
        assert c.source_file == "X - 2020 - Y.pdf"


# ---- Golden-Äquivalenz: render_note mit CitationMeta == Vor-Refactor-Dict-Pfad --


class _FixedDate(_date):
    @classmethod
    def today(cls):
        return _date(2026, 1, 1)


# Auf master (e7fee98) mit `source_meta={"Author": "Bertram, J.", "Year": "2019",
# "Title": "Angeleitetes Lernen"}` erzeugtes render_note()-Ergebnis, eingefroren
# VOR dem CitationMeta-Refactor (siehe PR #96 E3a). Beweist Verhaltensneutralität
# für den Mit-Jahr-Fall.
_GOLDEN_RENDER_NOTE = (
    '---\npipeline-content-hash: 93ab85e0d0540bfa\ntitle: "Testkonzept Bertram"\naliases:\n'
    '  - "Bertram-Modell"\ntype: atomic\nsynthesis-confidence: medium\n'
    'confidence-rationale: "Nur eine Quelle, keine Vault-Korroboration."\n'
    'source-file: "Bertram - 2019 - Angeleitetes Lernen.pdf"\nclaude-generated: true\n'
    'quality-flags:\n  - "⚠️ niedrige Zitationsanzahl (n=2)"\ncreated: 2026-01-01\ntags:\n'
    '  - uni/ibi/konzept\nrelated:\n  - "[[Anderes Konzept]]"\n---\n'
    "# Testkonzept Bertram: Kerncharakteristik in einem Satz\n\n"
    "Erster Absatz mit einer Aussage[^1]. Zweite Aussage auf derselben Seite[^2].\n\n"
    "> [!quote]- Bertram 2019, S. 7\n> „Originalzitat aus der Quelle.“\n\n"
    "Weitere Substanz-Aussage mit Anker[^3].\n\n"
    "[^1]: Bertram 2019, S. 5.\n[^2]: Bertram 2019, S. 5.\n[^3]: Bertram 2019, S. 12.\n\n"
    "## Quellen\n\n*Quelle: Bertram 2019: Angeleitetes Lernen, S. 5, 12*\n"
)


def _golden_draft() -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title="Testkonzept Bertram",
        body=(
            "# Testkonzept Bertram: Kerncharakteristik in einem Satz\n\n"
            "Erster Absatz mit einer Aussage (S. 5). Zweite Aussage auf derselben Seite (S. 5).\n\n"
            "> [!quote]- Bertram 2019, S. 7\n> „Originalzitat aus der Quelle.“\n\n"
            "Weitere Substanz-Aussage mit Anker (S. 12)."
        ),
        source_anchors=[TextAnchor(quote="q1", page="S. 5"), TextAnchor(quote="q2", page="S. 12")],
        related=["[[Anderes Konzept]]"],
        tags=["uni/ibi/konzept"],
        synthesis_confidence="medium",
        aliases=["Bertram-Modell"],
        quality_flags=["⚠️ niedrige Zitationsanzahl (n=2)"],
        action="create",
        critic_score=5,
        hard_gates_pass=True,
        confidence_reasoning="Nur eine Quelle, keine Vault-Korroboration.",
    )


class TestGoldenEquivalence:
    def test_render_note_with_citation_meta_matches_frozen_dict_output(self):
        source_file = "Bertram - 2019 - Angeleitetes Lernen.pdf"
        citation = CitationMeta(
            author="Bertram, J.", year="2019", title="Angeleitetes Lernen", doi=None, source_file=source_file
        )
        with patch("generative.pipeline.vault_writer.date", _FixedDate):
            out = vault_writer.render_note(_golden_draft(), source_file, citation=citation)
        assert out == _GOLDEN_RENDER_NOTE
