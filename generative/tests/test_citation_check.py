"""Tests für das Validierungsnetz gegen CitationMeta (#96, Etappe E3b).

Deterministischer, seiteneffekt-freier Check: LLM-generierte Autor-/Jahr-
Attributionen im Note-Body werden gegen die kanonische CitationMeta geprüft.
Mismatch → quality_flags-Eintrag (Review-Hinweis) — KEIN Body-Edit, KEIN
Routing-Eingriff (analog zu `flag_redundant_siblings`, #8).

Regressionstest: Issue #96 Fall 2 ("Landry 2019" statt "Knowles").
"""

from __future__ import annotations

from generative.pipeline.citation_check import apply_citation_check, validate_citation_attributions
from generative.schemas.atomic_note import AtomicNoteDraft
from generative.schemas.citation import CitationMeta


def _cm(**kw) -> CitationMeta:
    defaults = dict(author=None, year=None, title=None, doi=None, source_file="primaerquelle.pdf")
    defaults.update(kw)
    return CitationMeta(**defaults)


def _draft(body: str) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title="T",
        body=body,
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
    )


class TestRegression96:
    def test_landry_foreign_author_without_zit_n_flagged(self):
        # Issue #96 Fall 2: LLM konfabuliert "Landry 2019" statt Knowles (o. J. im Dateiname).
        body = "Der Erwachsenenbildungsforscher Pierre Landry legt dieses Modell vor (Landry 2019, S. 24)."
        citation = _cm(author="Knowles", year=None)
        flags = validate_citation_attributions(body, citation)
        assert len(flags) == 1
        assert "Landry" in flags[0]


class TestSecondaryCitationException:
    def test_legitimate_secondary_citation_no_flag(self):
        body = "Haythornthwaite betont X (zit. n. Hrastinski, S. 2)."
        citation = _cm(author="Hrastinski", year="2008")
        assert validate_citation_attributions(body, citation) == []

    def test_secondary_citation_wrong_primary_source_flagged(self):
        body = "Kock argumentiert Y (zit. n. Mueller, S. 3)."
        citation = _cm(author="Hrastinski", year="2008")
        flags = validate_citation_attributions(body, citation)
        assert len(flags) == 1
        assert "Sekundärzitat" in flags[0]


class TestPrimaryAttributionYearCheck:
    def test_wrong_year_flagged(self):
        body = "Merrill (2006) formuliert das Modell."
        citation = _cm(author="Merrill", year="2002")
        flags = validate_citation_attributions(body, citation)
        assert len(flags) == 1
        assert "ungedecktem Jahr" in flags[0]

    def test_wrong_year_flagged_when_citation_year_none(self):
        body = "Merrill (2006) formuliert das Modell."
        citation = _cm(author="Merrill", year=None)
        flags = validate_citation_attributions(body, citation)
        assert len(flags) == 1
        assert "ungedecktem Jahr" in flags[0]

    def test_correct_year_no_flag(self):
        body = "Merrill (2006) formuliert das Modell."
        citation = _cm(author="Merrill", year="2006")
        assert validate_citation_attributions(body, citation) == []


class TestOJTolerance:
    def test_bracketed_o_j_no_flag(self):
        body = "Knowles [o. J.] beschreibt das Modell."
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions(body, citation) == []

    def test_unbracketed_o_j_no_flag(self):
        body = "Knowles o. J. beschreibt das Modell."
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions(body, citation) == []


class TestMultiAuthor:
    def test_both_surnames_legitimate(self):
        body = "Ebner (2020) und Gegenfurtner (2020) beschreiben das Modell gemeinsam."
        citation = _cm(author="Ebner; Gegenfurtner", year="2020")
        assert validate_citation_attributions(body, citation) == []


class TestIgnoredRegions:
    def test_blockquote_ignored(self):
        body = "> Landry (2019) sagt etwas Falsches, das hier nicht zaehlt."
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions(body, citation) == []

    def test_quellen_section_ignored(self):
        body = "Text ohne Risiko.\n\n## Quellen\n\n*Quelle: Landry 2019: X*"
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions(body, citation) == []

    def test_footnote_definition_ignored(self):
        body = "[^1]: Landry 2019, S. 5."
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions(body, citation) == []


class TestEmptyInputs:
    def test_empty_body_no_flags(self):
        citation = _cm(author="Knowles", year=None)
        assert validate_citation_attributions("", citation) == []

    def test_author_none_no_flags_no_crash(self):
        body = "Landry (2019) behauptet etwas."
        citation = _cm(author=None, year=None)
        assert validate_citation_attributions(body, citation) == []


class TestIdempotentOrchestratorWiring:
    def test_double_invocation_no_duplicate_flags(self):
        body = "Der Erwachsenenbildungsforscher Pierre Landry legt dieses Modell vor (Landry 2019, S. 24)."
        citation = _cm(author="Knowles", year=None)
        draft = _draft(body)

        apply_citation_check([draft], citation)
        apply_citation_check([draft], citation)

        landry_flags = [f for f in draft.quality_flags if "Landry" in f]
        assert len(landry_flags) == 1


# ---- Real-Check-Fixup: Callout-Header sind LLM-generiert und pruefpflichtig ----


def _cit_knowles():
    return CitationMeta(
        author="Knowles",
        year=None,
        title="T",
        doi=None,
        source_file="Knowles - From Pedagogy to Andragogy.pdf",
    )


def test_callout_header_with_foreign_author_is_flagged():
    # Historischer #96-Fall: gerenderte/gedraftete Note traegt die Fehl-Attribution
    # im Callout-Header, nicht im Fliesstext.
    body = '> [!quote]- Landry 2019, S. 24\n> "Woertliches Zitat."'
    flags = validate_citation_attributions(body, _cit_knowles())
    assert len(flags) == 1
    assert "Landry 2019" in flags[0]


def test_callout_header_with_correct_author_year_not_flagged():
    cit = CitationMeta(author="Hrastinski", year="2008", title="T", doi=None, source_file="Hrastinski - 2008 - X.pdf")
    body = '> [!quote]- Hrastinski 2008, S. 2\n> "Zitat."'
    assert validate_citation_attributions(body, cit) == []


def test_quote_content_with_foreign_name_not_flagged():
    # Woertliche Zitate duerfen fremde Autor-Jahr-Nennungen enthalten.
    body = '> [!quote]- Knowles [o. J.], S. 21\n> "Wie Thorndike (1928) zeigte, lernen Erwachsene."'
    assert validate_citation_attributions(body, _cit_knowles()) == []


def test_zit_n_primary_reference_needs_word_boundary():
    # "Berg" via Substring in "Bergbau" darf NICHT als Primaerquellen-Nennung gelten.
    cit = CitationMeta(author="Berg", year="2010", title="T", doi=None, source_file="Berg - 2010 - X.pdf")
    body = "Mueller betont Y (zit. n. Bergbau-Studie, S. 3)."
    flags = validate_citation_attributions(body, cit)
    assert len(flags) == 1 and "Sekund" in flags[0]
    body_ok = "Mueller betont Y (zit. n. Berg, S. 3)."
    assert validate_citation_attributions(body_ok, cit) == []
