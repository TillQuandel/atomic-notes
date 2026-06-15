"""Tests für abgestufte Retraction-Klassifikation in quality.check_quality().

Differenzierung zwischen Full-Retraction, Expression of Concern, Erratum/Korrektur
und Withdrawal — alle landen als separate Quality-Flags. Nur retraction/withdrawal
setzen `retracted=True` (Hard-Signal für Pipeline).
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch


from generative.agents import quality


def _fake_crossref(update_types: list[str] | None = None, ctype: str = "journal-article") -> dict:
    return {
        "type": ctype,
        "is-referenced-by-count": 42,
        "update-to": [{"type": t} for t in (update_types or [])],
        "title": ["Some Paper"],
        "author": [{"family": "Mustermann", "given": "Max"}],
        "issued": {"date-parts": [[2020]]},
    }


def _run_check(crossref_meta: dict, openalex: dict | None = None):
    with patch.object(quality, "_crossref_meta", return_value=crossref_meta), \
         patch.object(quality, "_openalex_work", return_value=openalex):
        return quality.check_quality(doi="10.1/test")


def test_retraction_sets_retracted_true():
    rep = _run_check(_fake_crossref(["retraction"]))
    assert rep.retracted is True
    assert any("ZURÜCKGEZOGEN" in f for f in rep.flags)


def test_withdrawal_sets_retracted_true():
    rep = _run_check(_fake_crossref(["withdrawal"]))
    assert rep.retracted is True


def test_expression_of_concern_flag_but_not_retracted():
    rep = _run_check(_fake_crossref(["expression-of-concern"]))
    assert rep.retracted is False
    assert any("Expression of Concern" in f for f in rep.flags)
    assert not any("ZURÜCKGEZOGEN" in f for f in rep.flags)


def test_erratum_flag_but_not_retracted():
    rep = _run_check(_fake_crossref(["erratum"]))
    assert rep.retracted is False
    assert any("Erratum" in f or "Korrektur" in f for f in rep.flags)


def test_correction_flag_but_not_retracted():
    rep = _run_check(_fake_crossref(["correction"]))
    assert rep.retracted is False
    assert any("Erratum" in f or "Korrektur" in f for f in rep.flags)


def test_no_update_no_extra_flags():
    rep = _run_check(_fake_crossref([]))
    assert rep.retracted is False
    assert not any("Expression of Concern" in f for f in rep.flags)
    assert not any("Erratum" in f for f in rep.flags)


def test_openalex_is_retracted_still_works_without_crossref_signal():
    # Regression: OpenAlex-Fallback bleibt wirksam wenn CrossRef nichts hat
    rep = _run_check(_fake_crossref([]), openalex={"is_retracted": True, "cited_by_count": 10})
    assert rep.retracted is True


def test_multiple_update_types_combined():
    rep = _run_check(_fake_crossref(["correction", "expression-of-concern"]))
    assert rep.retracted is False
    assert any("Expression of Concern" in f for f in rep.flags)
    assert any("Erratum" in f or "Korrektur" in f for f in rep.flags)


def test_defensive_against_non_dict_update_entries():
    # CrossRef-Antworten können malformed sein (z.B. None oder String statt Dict)
    crossref = _fake_crossref(["retraction"])
    crossref["update-to"] = [None, "garbage", {"type": "retraction"}, 42]
    rep = _run_check(crossref)
    assert rep.retracted is True, "Non-Dict-Einträge dürfen nicht crashen, Retraction muss erkannt bleiben"


def test_case_insensitive_update_type():
    # CrossRef-Antworten haben gelegentlich uppercase — Lowering muss greifen
    rep = _run_check(_fake_crossref(["Retraction"]))
    assert rep.retracted is True


def test_doi_from_title_match_flagged_true():
    """Per Title geratener DOI wird als unsicher markiert (kein harter ID-Match)."""
    with patch.object(quality, "_crossref_doi_lookup", return_value="10.9999/guessed"), \
         patch.object(quality, "_crossref_meta", return_value=None), \
         patch.object(quality, "_openalex_work", return_value=None):
        rep = quality.check_quality(title="Some Sufficiently Long Title")
    assert rep.doi_from_title_match is True


def test_explicit_doi_not_title_match():
    """Explizit übergebener DOI ist ID-basiert -> kein Title-Match-Flag."""
    rep = _run_check(_fake_crossref())
    assert rep.doi_from_title_match is False
