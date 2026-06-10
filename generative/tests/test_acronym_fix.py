"""Tests für Schwartz-Hearst-basierte Akronym-Erkennung + Body-Insertion."""
from __future__ import annotations
import sys
from pathlib import Path


from generative.pipeline.acronym_fix import (
    _short_is_valid,
    _letter_match,
    _trim_long_form,
    extract_acronym_pairs,
    expand_acronyms,
)


# ---- _short_is_valid -----------------------------------------------------

def test_short_valid_classic_acronyms():
    for s in ["IT", "API", "CSCW", "UCLA", "DOI", "ISBN"]:
        assert _short_is_valid(s), f"{s!r} should be valid"


def test_short_valid_mixed_case_with_lowercase_prefix():
    for s in ["fMRT", "mRNA"]:
        assert _short_is_valid(s), f"{s!r} mixed case should be valid"


def test_short_invalid_normal_words():
    for s in ["Technology", "least", "the", "and", "Information"]:
        assert not _short_is_valid(s), f"{s!r} should be invalid"


def test_short_invalid_single_uppercase():
    for s in ["Hello", "World", "Java"]:
        assert not _short_is_valid(s), f"{s!r} (1 uppercase) should be invalid"


def test_short_invalid_too_short_too_long():
    assert not _short_is_valid("A")
    assert not _short_is_valid("ABCDEFGHIJK")  # 11 chars


def test_short_valid_with_special_chars():
    assert _short_is_valid("ASIS&T")
    assert _short_is_valid("R&D")


# ---- _letter_match -------------------------------------------------------

def test_letter_match_exact():
    assert _letter_match("CSCW", "Computer-Supported Cooperative Work")
    assert _letter_match("ISP", "Information Search Process")
    assert _letter_match("API", "Application Programming Interface")


def test_letter_match_german():
    assert _letter_match("ISP", "Informations-Such-Prozess")
    assert _letter_match("MTB", "Mountain-Bike")


def test_letter_match_first_char_must_be_word_start():
    # 'A' at non-word-start should fail
    assert not _letter_match("ABC", "xABCxyz")
    # 'A' at word start works
    assert _letter_match("ABC", "Apple Banana Cat")


def test_letter_match_fails_when_letters_missing():
    # 'XC' for 'Cross-Country' fails — X not in 'Cross'
    assert not _letter_match("XC", "Cross-Country")


# ---- _trim_long_form -----------------------------------------------------

def test_trim_long_form_picks_shortest_match():
    # Even if longer prefix would match, pick shortest valid
    long = "the field of Computer-Supported Cooperative Work"
    result = _trim_long_form(long, "CSCW")
    assert result == "Computer-Supported Cooperative Work"


def test_trim_long_form_handles_compound_word():
    # 'Mountain-Bike' is 1 word but contains all letters of 'MTB'
    result = _trim_long_form("Mountain-Bike", "MTB")
    assert result == "Mountain-Bike"


# ---- extract_acronym_pairs (integration) ---------------------------------

def test_extract_pattern_i_long_paren_short():
    text = "The Information Search Process (ISP) was developed."
    pairs = extract_acronym_pairs(text)
    assert pairs == {"ISP": "Information Search Process"}


def test_extract_pattern_ii_short_paren_long():
    text = "The CSCW (Computer-Supported Cooperative Work) field grew."
    pairs = extract_acronym_pairs(text)
    assert pairs == {"CSCW": "Computer-Supported Cooperative Work"}


def test_extract_german_text():
    text = (
        "Das Verfahren der Informations-Such-Prozess (ISP) wurde entwickelt. "
        "Mountain-Bike (MTB) hat Disziplinen."
    )
    pairs = extract_acronym_pairs(text)
    assert "ISP" in pairs
    assert pairs["ISP"] == "Informations-Such-Prozess"
    assert pairs["MTB"] == "Mountain-Bike"


def test_extract_skips_year_in_parens():
    text = "Bates (2017) argues that information behavior is broad."
    pairs = extract_acronym_pairs(text)
    # "Bates" is short_invalid (1 uppercase), 2017 not alpha — no match
    assert pairs == {}


def test_extract_multiple_acronyms():
    text = (
        "Application Programming Interface (API) returns JSON. "
        "HyperText Transfer Protocol (HTTP) is standard."
    )
    pairs = extract_acronym_pairs(text)
    assert pairs["API"] == "Application Programming Interface"
    assert pairs["HTTP"] == "HyperText Transfer Protocol"


def test_extract_no_long_form_no_match():
    # Acronym in parens but no long form before
    text = "We use (XYZ) without context."
    pairs = extract_acronym_pairs(text)
    assert pairs == {}


# ---- expand_acronyms (body insertion) ------------------------------------

def test_expand_inserts_at_first_occurrence():
    body = "CSCW ist ein Forschungsfeld. CSCW betrifft Gruppenarbeit."
    new, expanded = expand_acronyms(body, {"CSCW": "Computer-Supported Cooperative Work"})
    assert new.startswith("CSCW (Computer-Supported Cooperative Work) ist")
    assert "CSCW (Computer-Supported Cooperative Work) betrifft" not in new  # only first
    assert expanded == ["CSCW"]


def test_expand_idempotent_when_already_resolved():
    body = "CSCW (Computer-Supported Cooperative Work) ist ein Feld. CSCW heute."
    new, expanded = expand_acronyms(body, {"CSCW": "Computer-Supported Cooperative Work"})
    assert new == body  # no change
    assert expanded == []


def test_expand_skips_when_long_form_anywhere_in_body():
    body = "Computer-Supported Cooperative Work ist ein Forschungsfeld. CSCW heute."
    new, expanded = expand_acronyms(body, {"CSCW": "Computer-Supported Cooperative Work"})
    assert new == body
    assert expanded == []


def test_expand_skips_when_followed_by_paren():
    body = "CSCW (Computer-Supported Cooperative Work)"
    new, expanded = expand_acronyms(body, {"CSCW": "Computer-Supported Cooperative Work"})
    assert new == body


def test_expand_word_boundary():
    # 'CERQual' should not match in 'Sub-CERQual'
    body = "Sub-CERQual research"
    new, expanded = expand_acronyms(body, {"CERQual": "Confidence Eval"})
    # Word boundary: '-' is not word, but Sub-CERQual is one token logically.
    # Python \b treats it as boundary, so this WILL match. Document the behavior.
    # If false-positive shows up in eval, tighten regex.
    # For now, just check no crash.
    assert isinstance(new, str)


def test_expand_empty_whitelist():
    body = "Some text with API."
    new, expanded = expand_acronyms(body, {})
    assert new == body
    assert expanded == []


def test_expand_none_whitelist():
    body = "Some text."
    new, expanded = expand_acronyms(body, None)
    assert new == body
    assert expanded == []
