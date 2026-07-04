"""Tests für High-Risk-Claim-Dekomposition (Faithfulness-Gate E2, #69).

Pure, deterministische Zerlegung eines Draft-Bodys (Markdown ohne Frontmatter,
Seitenanker inline als `(S. N)`/`(zit. n. Autor, S. N)`, Blockquotes mit
`> `-Prefix) in risiko-markierte Claims. Kein LLM, keine Embeddings.

Fixtures unten sind reale Sätze aus Pipeline-Läufen (siehe Issue #69 E2-Auftrag).
"""

from __future__ import annotations

from generative.pipeline.claims import (
    Claim,
    attribution_risk,
    causal_risk,
    comparison_risk,
    decompose_claims,
    number_risk,
)


# ---- decompose_claims: Pflicht-Fälle aus echten Pipeline-Läufen -----------


def test_attribution_with_zit_n_and_page_anchor():
    body = (
        "Drei Kommunikationstypen sind zentral für den Aufbau und Erhalt von "
        "E-Learning-Gemeinschaften: inhaltsbezogene Kommunikation, Aufgabenplanung "
        "und soziale Unterstützung, führt Haythornthwaite aus (zit. n. Hrastinski, S. 2)."
    )
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "attribution" in claims[0].risk_types
    assert claims[0].anchor_page == 2
    assert claims[0].is_quote is False


def test_attribution_author_year_does_not_trigger_number_risk():
    body = (
        "Van Merrienboer (1997) unterschied verschiedene Aufgabenformate und "
        "belegte, dass Worked-out Examples wichtige erste Schritte einer "
        "Instruktionssequenz sind (zit. n. Merrill, S. 6)."
    )
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "attribution" in claims[0].risk_types
    assert claims[0].anchor_page == 6
    # Das Jahr 1997 ist kein number-Risk (reines Jahr, kein sonstiger Ziffern-Token)
    assert "number" not in claims[0].risk_types


def test_number_risk_effect_size_page_anchor_number_not_counted():
    body = "Die Effektstärke lag bei r = 0,59 (S. 12)."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "number" in claims[0].risk_types
    assert claims[0].anchor_page == 12


def test_page_anchor_alone_without_risk_pattern_is_no_claim():
    body = "Das Konzept stammt aus der Bibliothekswissenschaft (S. 3)."
    assert decompose_claims(body) == []


def test_causal_sentence_synthetic():
    body = "Die fehlende Rückmeldung verursacht Frustration bei den Lernenden."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert claims[0].risk_types == ["causal"]


def test_comparison_sentence_synthetic():
    body = "Blended Learning ist effektiver als reines Präsenzlernen."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert claims[0].risk_types == ["comparison"]


def test_combined_number_and_comparison_risk_types():
    body = "Die Rücklaufquote war mit 65 % höher als im Vorjahr."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "number" in claims[0].risk_types
    assert "comparison" in claims[0].risk_types


def test_blockquote_sentence_marked_as_quote():
    body = '> „Wörtliches Zitat mit Zahl 42." (S. 7)'
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert claims[0].is_quote is True
    assert "number" in claims[0].risk_types
    assert claims[0].anchor_page == 7


def test_headings_footnote_defs_and_quellen_section_skipped():
    body = "\n".join(
        [
            "# Überschrift",
            "",
            "Ein harmloser Satz ohne jedes Muster.",
            "",
            "[^1]: Eine Fußnotendefinition mit Zahl 42 und (S. 9).",
            "",
            "## Quellen",
            "",
            "Hrastinski (2009). Studie mit r = 0,80 (S. 5).",
        ]
    )
    assert decompose_claims(body) == []


def test_anchor_span_contains_claim_text():
    claim_sentence = "Die Effektstärke lag bei r = 0,59 (S. 12)."
    body = f"Einleitungssatz ohne Risiko.\n{claim_sentence}\nAbschlusssatz ohne Risiko."
    claims = decompose_claims(body)
    assert len(claims) == 1
    start, end = claims[0].anchor_span
    assert claims[0].text in body[start:end]


def test_empty_body_returns_empty_list():
    assert decompose_claims("") == []


def test_harmless_sentence_without_pattern_not_in_result():
    body = "Ein völlig harmloser Satz ohne jedes Risiko-Muster."
    assert decompose_claims(body) == []


def test_claim_is_dataclass_with_expected_fields():
    body = "Die Effektstärke lag bei r = 0,59 (S. 12)."
    claim = decompose_claims(body)[0]
    assert isinstance(claim, Claim)
    assert isinstance(claim.text, str)
    assert isinstance(claim.risk_types, list)
    assert isinstance(claim.anchor_span, tuple)


# ---- Risk-Muster einzeln testbar (auch von #96 CitationMeta-Validierung
# wiederverwendbar) --------------------------------------------------------


def test_attribution_risk_zit_n():
    assert attribution_risk("Aussage (zit. n. Müller, S. 3).")


def test_attribution_risk_author_year():
    assert attribution_risk("Gross & Latham (2012) zeigten das.")


def test_attribution_risk_et_al():
    assert attribution_risk("Smith et al. fanden ähnliche Effekte.")


def test_attribution_risk_laut():
    assert attribution_risk("Laut Meyer ist das umstritten.")


def test_attribution_risk_zufolge():
    assert attribution_risk("Meyer zufolge ist das umstritten.")


def test_attribution_risk_nach_ansicht_von():
    assert attribution_risk("Nach Ansicht von Meyer ist das umstritten.")


def test_attribution_risk_false_for_plain_sentence():
    assert not attribution_risk("Ein Satz ganz ohne Attribution.")


def test_number_risk_true_for_percentage():
    assert number_risk("Die Quote lag bei 83 %.")


def test_number_risk_false_for_page_anchor_only():
    assert not number_risk("Ein Satz (S. 5).")


def test_number_risk_false_for_year_only():
    assert not number_risk("Das war im Jahr (1999).")


def test_comparison_risk_true():
    assert comparison_risk("Das Ergebnis war deutlich größer als erwartet.")


def test_comparison_risk_false():
    assert not comparison_risk("Ein neutraler Satz ohne Vergleich.")


def test_causal_risk_true():
    assert causal_risk("Das führt zu einer Verbesserung.")


def test_causal_risk_false():
    assert not causal_risk("Ein neutraler Satz ohne Kausalitaet.")


# ---- Review-Fixups (Qwen-Review + Real-Check auf echter Pipeline-Note) ------


def test_page_ref_ff_does_not_split_sentence():
    body = "Die Studie belegt den Effekt ausfuehrlich (S. 5ff.) und nennt r = 0,3 als Wert."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert claims[0].text == body
    assert claims[0].anchor_page == 5


def test_bzw_does_not_split_sentence():
    body = "Es nahmen 19 Teilnehmende teil, Durchschnittsalter 38 bzw. 43 Jahre."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "bzw. 43" in claims[0].text


def test_footnote_marker_digit_is_not_number_risk():
    assert not number_risk("Der erste Typ bildet den Kern[^4].")
    assert number_risk("Der Wert lag bei 0,42[^4].")


def test_callout_header_is_skipped():
    body = '> [!quote]- Hrastinski 2008, S. 2\n> "Ein Zitat mit Zahl 42." (S. 2)'
    claims = decompose_claims(body)
    assert all("[!quote]" not in c.text for c in claims)
    assert any(c.is_quote and "42" in c.text for c in claims)


def test_literatur_heading_stops_claims():
    body = "Ein Satz mit Zahl 42 davor.\n\n## Literaturverzeichnis\n\nMeyer (2001): Titel mit 99 Seiten."
    claims = decompose_claims(body)
    assert len(claims) == 1
    assert "42" in claims[0].text


def test_inflected_comparison_and_causal_matched():
    assert comparison_risk("Die Gruppe erzielte hoehere Werte.".replace("oe", "ö"))
    assert causal_risk("Das Training fuehrte zu besseren Ergebnissen.".replace("ue", "ü"))
    assert not causal_risk("Die Führung zu Ende bringen.")
