"""Tests für die deterministische Attribution-Heuristik (Faithfulness-Gate E4b, #69).

Reine Präsenz-Prüfung: prüft ob attribuierte Fremd-Nachnamen aus dem Claim-Text
im Seitenfenster VORKOMMEN — keine semantische Aussage-Zuordnung (das macht
später NLI, Etappe E5). Hrastinski/Haythornthwaite-Fixture spiegelt das
"zit. n."-Szenario aus `test_citation_check.py` (Issue #96): der dokumentierte
Fehlerfall ist eine Aussage, die dem falschen zitierten Autor zugeordnet wird.
"""

from __future__ import annotations

from generative.pipeline.attribution import check_attribution
from generative.pipeline.claims import Claim


def _claim(text: str, risk_types=("attribution",)) -> Claim:
    return Claim(
        text=text,
        anchor_page=2,
        anchor_span=(0, len(text)),
        risk_types=list(risk_types),
        is_quote=False,
    )


# zit.-n.-Satz: nur das zit.-n.-Ziel (Hrastinski, Primärautor) wird extrahiert —
# der berichtete Autor davor bewusst nicht mehr (False-Positive-Klasse bei
# deutschen Substantiven vor der Klammer, siehe Modul-Docstring attribution.py).
HRASTINSKI_CLAIM = _claim(
    "Drei Kommunikationstypen unterscheiden sich in Synchronität, führt Haythornthwaite aus (zit. n. Hrastinski, S. 2)."
)

# Fremd-Autor über die verbleibenden Extraktionswege (laut/AUTHOR_YEAR_RE) —
# trägt die Fenster-Check-Tests nach dem Rückbau der Vor-Klammer-Extraktion.
LAUT_CLAIM = _claim("Laut Haythornthwaite sind drei Kommunikationstypen zentral (S. 2).")


class TestHrastinskiFixture:
    def test_supported_when_window_contains_reported_author(self):
        window = "[S. 2]\nHaythornthwaite unterscheidet drei Kommunikationstypen nach Synchronität."
        assert check_attribution(LAUT_CLAIM, window, primary_surnames=["Hrastinski"]) == "supported"

    def test_zit_n_claim_with_primary_target_is_supported_without_window(self):
        # Nach dem Rückbau der Vor-Klammer-Extraktion nennt der zit.-n.-Claim
        # nur noch Hrastinski (Primärautor) → Ausnahme greift, kein Fenster nötig.
        assert check_attribution(HRASTINSKI_CLAIM, None, primary_surnames=["Hrastinski"]) == "supported"

    def test_author_missing_when_window_lacks_reported_author(self):
        # Fehlerfall über den laut-Weg: Fenster nennt Haythornthwaite nicht.
        window = "[S. 2]\nDrei Kommunikationstypen werden anhand ihrer Synchronität unterschieden."
        assert check_attribution(LAUT_CLAIM, window, primary_surnames=["Hrastinski"]) == "author_missing"

    def test_no_window_when_source_window_is_none(self):
        assert check_attribution(LAUT_CLAIM, None, primary_surnames=["Hrastinski"]) == "no_window"

    def test_no_window_when_source_window_is_empty_string(self):
        assert check_attribution(LAUT_CLAIM, "", primary_surnames=["Hrastinski"]) == "no_window"

    def test_german_noun_before_zit_n_is_not_treated_as_author(self):
        # FP-Klasse vom echten PDF (Review 2026-07-04): deutsches Substantiv vor
        # der zit.-n.-Klammer darf kein author_missing gegen ein englisches
        # Fenster erzeugen.
        claim = _claim("Der dritte Typ umfasst soziale Unterstützung (zit. n. Hrastinski, S. 2).")
        window = "[S. 2]\nThree types of communication are distinguished by synchronicity."
        assert check_attribution(claim, window, primary_surnames=["Hrastinski"]) == "supported"


class TestNotApplicable:
    def test_claim_without_attribution_risk(self):
        claim = _claim("Die Zahl stieg um 40 Prozent.", risk_types=("number",))
        assert check_attribution(claim, "irrelevantes Fenster", primary_surnames=[]) == "not_applicable"

    def test_attribution_risk_without_recognizable_name(self):
        # ET_AL_RE loest attribution_risk() aus, liefert aber selbst keinen
        # extrahierbaren Nachnamen (keine AUTHOR_YEAR/zit.n./laut/zufolge-Form).
        claim = _claim("Diverse Autoren et al. bestätigen den Befund.", risk_types=("attribution",))
        assert check_attribution(claim, "irrelevantes Fenster", primary_surnames=[]) == "not_applicable"


class TestPrimaryAuthorException:
    def test_primary_only_attribution_is_supported_without_window_presence(self):
        claim = _claim("Hrastinski (2008) differenziert drei Kommunikationstypen.")
        window = "[S. 2]\nEin Fenstertext, der den Autorennamen selbst nicht wiederholt."
        assert check_attribution(claim, window, primary_surnames=["Hrastinski"]) == "supported"

    def test_primary_only_attribution_is_supported_even_without_any_window(self):
        claim = _claim("Hrastinski (2008) differenziert drei Kommunikationstypen.")
        assert check_attribution(claim, None, primary_surnames=["Hrastinski"]) == "supported"


class TestWordBoundaries:
    def test_berg_not_falsely_matched_inside_bergbau(self):
        claim = _claim("Berg (2015) beschreibt das Phänomen.")
        window = "[S. 2]\nDer Bergbau prägte die Region."
        assert check_attribution(claim, window, primary_surnames=[]) == "author_missing"

    def test_bergbau_not_falsely_matched_by_berg(self):
        claim = _claim("Bergbau (2015) beschreibt das Phänomen.")
        window = "[S. 2]\nDer Berg war hoch."
        assert check_attribution(claim, window, primary_surnames=[]) == "author_missing"

    def test_exact_word_match_supported(self):
        claim = _claim("Berg (2015) beschreibt das Phänomen.")
        window = "[S. 2]\nBerg beschreibt in seinem Aufsatz das Phänomen genau."
        assert check_attribution(claim, window, primary_surnames=[]) == "supported"


class TestLautAndZufolge:
    def test_laut_form_extracts_name(self):
        claim = _claim("Laut Meyer sinkt die Fehlerquote deutlich.")
        window = "[S. 2]\nText ohne den genannten Namen."
        assert check_attribution(claim, window, primary_surnames=[]) == "author_missing"

    def test_zufolge_form_extracts_name(self):
        claim = _claim("Meyer zufolge sinkt die Fehlerquote deutlich.")
        window = "[S. 2]\nMeyer zeigt in seiner Studie einen Rückgang."
        assert check_attribution(claim, window, primary_surnames=[]) == "supported"


class TestMultiAuthor:
    def test_co_authors_both_checked(self):
        claim = _claim("Schlebbe & Greifeneder (2020) zeigen einen Effekt.")
        window = "[S. 2]\nSchlebbe berichtet von einem Effekt, Greifeneder bestätigt dies."
        assert check_attribution(claim, window, primary_surnames=[]) == "supported"

    def test_co_authors_one_missing_flagged(self):
        claim = _claim("Schlebbe & Greifeneder (2020) zeigen einen Effekt.")
        window = "[S. 2]\nSchlebbe berichtet von einem Effekt."
        assert check_attribution(claim, window, primary_surnames=[]) == "author_missing"


class TestGermanNonNameWordsBeforeYear:
    """Kalibrierungs-Fund E5b (Knowles-Gold-Set): Satzanfangs-Funktionswörter
    vor Jahreszahlen („Zwischen 1929 und 1948 …") matchen AUTHOR_YEAR_RE und
    wurden als Fremd-Autor extrahiert → author_missing-FP am englischen
    Quellfenster. Großschreibung ist kein Eigennamen-Signal."""

    def test_zwischen_before_year_is_not_an_author(self):
        claim = _claim("Zwischen 1929 und 1948 veröffentlichte das Journal Praxisberichte (S. 3).")
        window = "[S. 3]\nBetween 1929 and 1948 the Journal of Adult Education carried articles."
        assert check_attribution(claim, window, primary_surnames=[]) == "not_applicable"

    def test_seit_before_year_is_not_an_author(self):
        claim = _claim("Seit 2005 wächst das Feld stetig (S. 4).")
        window = "[S. 4]\nThe field has grown since 2005."
        assert check_attribution(claim, window, primary_surnames=[]) == "not_applicable"

    def test_real_surname_before_year_still_checked(self):
        claim = _claim("Houle (1961) identifizierte drei Lerntypen (S. 3).")
        window = "[S. 3]\nText ohne den Namen."
        assert check_attribution(claim, window, primary_surnames=[]) == "author_missing"
