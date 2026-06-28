# -*- coding: utf-8 -*-
"""Fix A: Metatext-Vorfilter für die Eval-Claim-Extraktion.

Belegt durch die Cross-Model-Fehleranalyse 2026-06-28: von 14 als „halluziniert"
geflaggten Ankern waren 7 gar keine Inhalts-Claims (Merge-Stub-Marker, Wiki-Link-
Pointer, reine Zitat-Fragmente), die fälschlich gegen das PDF geprüft wurden und
die hallucination_rate künstlich hochtrieben.
"""
from generative.eval_quality_v2 import filter_meta_claims, extract_claims


# Echte KEIN_CLAIM-Beispiele aus den v0.3.142-Läufen → müssen verworfen werden.
META = [
    # Merge-Stub-Marker (Batch 1, #1)
    "Merge Stub: Forschungsströme der Information Behavior Forschung Pipeline hat "
    "das Konzept Forschungsströme der Information Behavior extrahiert",
    # Vault-Meta-Hinweis auf existierende Note (Batch 1, #2)
    "Eine bestehende Note existiert bereits: [[IBI Forschungsbereich Information "
    "Behavior]] (01 studium/informations und bibliothekswissenschaft)",
    # Reiner Wiki-Link-Pointer + Nummer (Batch 1, #4)
    "[[IBI Forschungsbereich Information Behavior]] 2.",
    # Reine Zitat-Fragmente (Batch 2, #4/#5/#6)
    "Schlebbe & Greifeneder, S. 1).",
    "Schlebbe & Greifeneder, S. 6).",
    "Schlebbe & Greifeneder, S. 5).",
]

# Echte Inhalts-Claims (verifiziert wörtlich im PDF) → müssen erhalten bleiben.
REAL = [
    # Wiki-Link-PRÄFIX auf echtem Claim (Batch 1, #5) — Link strippen, Claim behalten
    "[[Information Seeking]] Die Informationswissenschaftlerinnen Schlebbe & "
    "Greifeneder fassen in ihrer Überblicksanalyse mehrere Jahrzehnte Forschung zusammen.",
    # Bates-Inhaltsclaims (Batch 2, #1/#2/#3) — wörtlich im PDF belegt
    "Den mengenmäßig größten Anteil an aufgenommener Information stellt Bates "
    "zufolge passiv rezipierte Information dar.",
    "Sozialinformatik und sozialwissenschaftliche Studien zur Informationstechnologie "
    "haben zum Forschungsfeld beigetragen.",
]


def test_meta_claims_are_dropped():
    assert filter_meta_claims(META) == []


def test_real_claims_are_kept():
    out = filter_meta_claims(REAL)
    assert len(out) == 3


def test_wikilink_markup_stripped_from_kept_claim():
    out = filter_meta_claims([REAL[0]])
    assert len(out) == 1
    assert "[[" not in out[0] and "]]" not in out[0]
    # Der inhaltliche Kern bleibt erhalten
    assert "Schlebbe" in out[0]


def test_empty_input():
    assert filter_meta_claims([]) == []


def test_real_claim_ending_in_page_cite_is_kept():
    # Cross-Model-Review (Qwen 2026-06-28): echter Satz, der zufällig mit „, S. N."
    # endet, darf NICHT als Zitat-Fragment verworfen werden.
    out = filter_meta_claims(["Das Konzept ist widerlegt, S. 12."])
    assert out == ["Das Konzept ist widerlegt, S. 12."]


def test_wikilink_subject_with_short_predicate_is_kept():
    # Subjekt lebt nur im Wiki-Link → Innentext muss erhalten bleiben, nicht gelöscht.
    out = filter_meta_claims(["[[Information Seeking]] ist ein zielgerichteter Prozess."])
    assert len(out) == 1
    assert "Information Seeking" in out[0]


def test_alias_wikilink_keeps_display_text():
    # [[Ziel|Anzeigetext]] → der Anzeigetext ist der im Claim sichtbare Begriff.
    out = filter_meta_claims(["[[Paper|wichtige Studie]] zeigt signifikante Ergebnisse hier."])
    assert len(out) == 1
    assert "wichtige Studie" in out[0]


def test_pure_pointer_still_dropped():
    assert filter_meta_claims(["[[IBI Forschungsbereich Information Behavior]] 2."]) == []


# Re-Review Runde 2 (Qwen+Mistral konvergent, verifiziert): lange Zitat-Fragmente
# mit Jahr / et al. / Komma-Autoren / Adelspartikeln rutschten durch (≥30 Zeichen,
# also nicht vom Längen-Gate gefangen).
def test_long_citation_fragments_with_year_and_etal_dropped():
    frags = [
        "Schlebbe und Greifeneder 2022, S. 145.",
        "Müller, Schmidt und Becker et al. 2019, S. 7.",
        "Smith, Jones, Williams und Brown, S. 312.",
        "van der Berg und de la Cruz 2020, S. 3.",
    ]
    assert filter_meta_claims(frags) == []


def test_real_sentence_with_verb_ending_in_page_cite_is_kept():
    # Verb (lowercase, kein Adelspartikel) bricht die Namens-Kette → echter Claim bleibt.
    claims = [
        "Das Konzept ist widerlegt, S. 12.",
        "Bates zeigt dies deutlich anhand mehrerer Beispiele, S. 5.",
    ]
    assert filter_meta_claims(claims) == claims


def test_extract_claims_applies_meta_filter(tmp_path):
    note = tmp_path / "n.md"
    note.write_text(
        "Merge Stub: Pipeline hat das Konzept Forschungsstroeme extrahiert und abgelegt.\n\n"
        "Den groessten Anteil an aufgenommener Information stellt passiv rezipierte Information dar.\n",
        encoding="utf-8",
    )
    claims = extract_claims(note)
    assert not any("Merge Stub" in c for c in claims)
    assert any("passiv rezipierte" in c for c in claims)
