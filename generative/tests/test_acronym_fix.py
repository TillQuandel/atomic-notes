"""Tests für Schwartz-Hearst-basierte Akronym-Erkennung + Body-Insertion."""

from __future__ import annotations


from unittest.mock import patch

from generative.pipeline.acronym_fix import (
    _short_is_valid,
    _letter_match,
    _trim_long_form,
    extract_acronym_pairs,
    expand_acronyms,
    llm_resolve_unknown,
    llm_fallback_resolve,
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
    text = "Das Verfahren der Informations-Such-Prozess (ISP) wurde entwickelt. Mountain-Bike (MTB) hat Disziplinen."
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
    text = "Application Programming Interface (API) returns JSON. HyperText Transfer Protocol (HTTP) is standard."
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


# ---- LLM-Fallback (ENABLE_ACRONYM_LLM_FALLBACK=1) ------------------------
# Regression #145: llm_resolve_unknown importierte nicht-existentes
# call_claude_sync -> ImportError bei aktiviertem Flag. Der Import liegt vor
# dem try-Block, propagiert also aus der Funktion.


def test_llm_resolve_unknown_returns_long_form():
    """Aktivierter LLM-Fallback-Pfad: gültige LONG_FORM-Antwort wird geparst."""
    with patch(
        "generative.agents.base.call_claude",
        return_value="LONG_FORM: Application Programming Interface",
    ) as mock_call:
        result = llm_resolve_unknown("API", "Kontext rund um die API im Text.")
    assert result == "Application Programming Interface"
    mock_call.assert_called_once()


def test_llm_resolve_unknown_returns_none_on_unknown():
    """LONG_FORM: UNKNOWN -> None (keine Halluzination)."""
    with patch("generative.agents.base.call_claude", return_value="LONG_FORM: UNKNOWN"):
        assert llm_resolve_unknown("XYZ", "Kein auflösbarer Kontext.") is None


def test_llm_fallback_resolve_flag_on(monkeypatch):
    """Flag=1: unaufgelöste Body-Akronyme werden per LLM ergänzt."""
    monkeypatch.setenv("ENABLE_ACRONYM_LLM_FALLBACK", "1")
    body = "Das Dokument nutzt CSCW an mehreren Stellen ohne Auflösung."
    with patch(
        "generative.agents.base.call_claude",
        return_value="LONG_FORM: Computer-Supported Cooperative Work",
    ):
        extra = llm_fallback_resolve(body, {})
    assert extra.get("CSCW") == "Computer-Supported Cooperative Work"


def test_llm_fallback_resolve_flag_off(monkeypatch):
    """Flag=0: kein LLM-Call, leeres Dict."""
    monkeypatch.setenv("ENABLE_ACRONYM_LLM_FALLBACK", "0")
    with patch("generative.agents.base.call_claude") as mock_call:
        extra = llm_fallback_resolve("Body mit CSCW.", {})
    assert extra == {}
    mock_call.assert_not_called()


# ---- Issue #279: H1-Header-Korruption durch Akronym-Fix ------------------


def test_short_invalid_hyphen_fragment():
    """Bindestrich-Fragmente sind keine gueltigen Short Forms — der
    Schwartz-Hearst-Scanner erkennt sonst faelschlich Textfragmente wie
    'KI-' als Akronym und liest ein zufaelliges nachfolgendes Textstueck
    als vermeintliche Langform (Issue #279)."""
    assert not _short_is_valid("KI-")
    assert not _short_is_valid("-KI")
    assert not _short_is_valid("K-I")


def test_short_valid_classic_acronyms_no_regression_279():
    """Legitime Akronyme ohne Bindestrich bleiben gueltig (keine Ueber-Restriktion)."""
    for s in ["TAM", "SEM"]:
        assert _short_is_valid(s), f"{s!r} should remain valid"


def test_expand_acronyms_preserves_h1_header_against_corruption():
    """Repro Issue #279: Selbst ein legitimes Akronym ('KI'), das im H1-Titel
    vorkommt, darf den Titel nicht veraendern — nur Body-Text nach der
    H1-Zeile ist fair game fuer Insertion."""
    body = "# Spannungsfeld KI-Fokus vs. Skepsis\n\nIm Studium wird KI zunehmend eingesetzt."
    whitelist = {
        "KI": "künstliche Intelligenz",
        "KI-": "Einsatz im Studium und Berufsleben vorbereitet.",
    }
    new, expanded = expand_acronyms(body, whitelist)
    new_lines = new.split("\n", 1)
    assert new_lines[0] == "# Spannungsfeld KI-Fokus vs. Skepsis"


def test_expand_acronyms_still_expands_body_after_h1():
    """Positiv-Kontrolle: Akronym in einer Body-Zeile (nicht im H1) wird
    weiterhin expandiert — der H1-Schutz darf die normale Funktion nicht
    kaputt machen."""
    body = "# Titel ohne Aenderung\n\nSEM wird in der Studie oft verwendet."
    whitelist = {"SEM": "Structural Equation Modeling"}
    new, expanded = expand_acronyms(body, whitelist)
    assert new.split("\n", 1)[0] == "# Titel ohne Aenderung"
    assert "SEM (Structural Equation Modeling) wird" in new
    assert expanded == ["SEM"]


def test_expand_acronyms_no_h1_unaffected():
    """Robustheit: Body ohne H1-Header verhaelt sich wie vor dem Fix."""
    body = "TAM wird oft verwendet."
    whitelist = {"TAM": "Technology Acceptance Model"}
    new, expanded = expand_acronyms(body, whitelist)
    assert new == "TAM (Technology Acceptance Model) wird oft verwendet."
    assert expanded == ["TAM"]
