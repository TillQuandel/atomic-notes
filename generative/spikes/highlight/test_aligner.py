# -*- coding: utf-8 -*-
"""TDD-Tests fuer die reine Zitat->Geometrie-Alignment-Logik (kein PDF-I/O)."""
import aligner


def test_strip_editorial_brackets_removes_capitalization_bracket():
    assert aligner.strip_editorial_brackets("[T]his process") == "This process"


def test_strip_editorial_brackets_removes_inserted_referent():
    assert (
        aligner.strip_editorial_brackets("in [asynchronous discussions] we")
        == "in asynchronous discussions we"
    )


def test_normalize_resolves_fi_ligature():
    # U+FB01 LATIN SMALL LIGATURE FI
    assert aligner.normalize_text("ﬁrst") == "first"


def test_normalize_collapses_whitespace_and_lowercases():
    assert aligner.normalize_text("The   QUICK\n brown") == "the quick brown"


def test_build_page_string_dehyphenates_linebreak_split_word():
    tokens = [("ar-", (0, 0, 10, 8)), ("chitecture", (0, 10, 40, 18))]
    page_str, char_to_word = aligner.build_page_string(tokens)
    assert "architecture" in page_str
    assert "ar- chitecture" not in page_str


def test_build_page_string_maps_each_char_to_source_word():
    tokens = [("Learning", (0, 0, 30, 8)), ("is", (32, 0, 40, 8))]
    page_str, char_to_word = aligner.build_page_string(tokens)
    assert len(char_to_word) == len(page_str)
    # first char belongs to word 0, the 'i' of "is" belongs to word 1
    assert char_to_word[0] == 0
    assert char_to_word[page_str.index("is")] == 1


def test_locate_returns_bboxes_of_verbatim_quote():
    tokens = [
        ("Learning", (0, 0, 30, 8)),
        ("is", (32, 0, 40, 8)),
        ("promoted", (42, 0, 80, 8)),
        ("when", (82, 0, 100, 8)),
    ]
    hit = aligner.locate("Learning is promoted", tokens)
    assert hit is not None
    assert hit["word_indices"] == [0, 1, 2]
    assert (0, 0, 30, 8) in hit["rects"]
    assert (82, 0, 100, 8) not in hit["rects"]


def test_build_page_string_dehyphenates_word_split_across_three_lines():
    # Gegen Mistral-Cross-Review-Behauptung: 3-fach-Split muss zusammenkleben.
    tokens = [("ar-", (0, 0, 5, 8)), ("chitec-", (0, 10, 20, 18)),
              ("ture", (0, 20, 30, 28))]
    page_str, _ = aligner.build_page_string(tokens)
    assert page_str == "architecture"


def test_locate_absorbs_hyphenation_across_line_break():
    tokens = [
        ("modern", (0, 0, 30, 8)),
        ("ar-", (32, 0, 45, 8)),
        ("chitecture", (0, 10, 50, 18)),
        ("wins", (52, 10, 80, 18)),
    ]
    hit = aligner.locate("modern architecture", tokens)
    assert hit is not None
    assert 1 in hit["word_indices"] and 2 in hit["word_indices"]


def test_locate_rejects_absent_phrase_below_threshold():
    tokens = [
        ("Learning", (0, 0, 30, 8)),
        ("is", (32, 0, 40, 8)),
        ("promoted", (42, 0, 80, 8)),
    ]
    # phrase shares no meaningful run with the page
    assert aligner.locate("quantum entanglement dynamics", tokens) is None


def test_locate_rejects_when_matched_span_too_short():
    # quote longer than anything present -> length-ratio guardrail must veto
    tokens = [("Learning", (0, 0, 30, 8)), ("is", (32, 0, 40, 8))]
    quote = "Learning is promoted when learners solve real world problems repeatedly"
    assert aligner.locate(quote, tokens) is None
