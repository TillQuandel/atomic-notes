"""Tests für Edition-Verifikation (Layer 1 + Layer 3).

Wurzel-Fix gegen stille Edition-Verwechslung: ein Kapitel-Extrakt trägt keine
Impressum-Seite, die Pipeline leitet Jahr/Edition allein aus dem Dateinamen ab.
Ohne DOI-Verifikation kann sie nicht wissen, welche Auflage vorliegt (z.B. KSS-6
2013 S.172 vs. KSS-7 2022 S.147). Layer 1 macht das laut + blockt auto-vault;
Layer 3 löst die Kapitel-DOI über den Seitenbereich auf (der die Edition beweist).
"""

from __future__ import annotations


# ---- Layer 1: is_edition_unverified -------------------------------------


def test_edition_unverified_when_excerpt_and_no_doi():
    from generative.pipeline.routing_report import is_edition_unverified

    # Auszug (erste Druckseite 172 > 1) + keine DOI-Verifikation → unverifiziert
    assert is_edition_unverified(doi_verified=False, first_print_page=172) is True


def test_edition_verified_when_doi_present():
    from generative.pipeline.routing_report import is_edition_unverified

    # DOI verifiziert (CrossRef-Auflösung) → Edition belegt, kein Flag
    assert is_edition_unverified(doi_verified=True, first_print_page=172) is False


def test_not_excerpt_when_starts_at_page_one():
    from generative.pipeline.routing_report import is_edition_unverified

    # Standalone-Dokument ab Seite 1 → kein Auszug, Edition-Risiko entfällt
    assert is_edition_unverified(doi_verified=False, first_print_page=1) is False


def test_no_print_page_info_is_not_flagged():
    from generative.pipeline.routing_report import is_edition_unverified

    # Keine numerischen PageLabels (normales Paper) → kein Auszug-Signal
    assert is_edition_unverified(doi_verified=False, first_print_page=None) is False


# ---- Layer 3: pick_chapter_doi (Seitenbereich beweist Edition) ----------

# Reale CrossRef-Form (gekürzt): zwei Auflagen desselben Kapitels von Reimer.
_KSS6 = {
    "DOI": "10.1515/9783110258264.172",
    "page": "172-182",
    "title": ["B 3 Wissensorganisation"],
    "issued": {"date-parts": [[2013, 3, 14]]},
}
_KSS7 = {
    "DOI": "10.1515/9783110769043-013",
    "page": "147-160",
    "title": ["B 1 Einführung in die Wissensorganisation"],
    "issued": {"date-parts": [[2023]]},
}


def test_pick_chapter_doi_disambiguates_by_start_page():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Startseite 172 → muss die 2013er (KSS-6) wählen, NICHT die 2023er
    hit = pick_chapter_doi([_KSS7, _KSS6], start_page=172, year="2013")
    assert hit is not None
    assert hit["DOI"] == "10.1515/9783110258264.172"


def test_pick_chapter_doi_other_edition_by_start_page():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Startseite 147 → 2022/2023er Ausgabe
    hit = pick_chapter_doi([_KSS6, _KSS7], start_page=147, year="2023")
    assert hit is not None
    assert hit["DOI"] == "10.1515/9783110769043-013"


def test_pick_chapter_doi_none_when_no_page_match():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Startseite 999 passt zu keinem Treffer → ehrlich None statt Raten
    assert pick_chapter_doi([_KSS6, _KSS7], start_page=999, year="2013") is None


def test_pick_chapter_doi_year_must_match_when_given():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Seitenbereich passt, aber Jahr widerspricht → kein Match (fail-closed)
    assert pick_chapter_doi([_KSS6], start_page=172, year="2022") is None


def test_pick_chapter_doi_ignores_year_when_not_given():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    hit = pick_chapter_doi([_KSS6], start_page=172, year=None)
    assert hit is not None and hit["DOI"] == "10.1515/9783110258264.172"


def test_pick_chapter_doi_ambiguous_returns_none():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Zwei DOI-Treffer mit gleicher Startseite+Jahr → mehrdeutig → fail-closed None
    # (CrossRef-Dublette o.ä.; lieber kein DOI als willkürlich raten). (Codex-Review.)
    dup = {"DOI": "10.1515/other", "page": "172-180", "issued": {"date-parts": [[2013]]}}
    assert pick_chapter_doi([_KSS6, dup], start_page=172, year="2013") is None


def test_pick_chapter_doi_skips_items_without_doi():
    from generative.tools.resolve_chapter_doi import pick_chapter_doi

    # Treffer mit passender Seite/Jahr aber ohne DOI-Key → ignorieren (nicht pinbar,
    # kein KeyError); der valide DOI-Treffer gewinnt. (Codex-Review.)
    no_doi = {"page": "172-182", "issued": {"date-parts": [[2013]]}}
    hit = pick_chapter_doi([no_doi, _KSS6], start_page=172, year="2013")
    assert hit is not None and hit["DOI"] == "10.1515/9783110258264.172"


# ---- Layer 1: auto-vault-Block + Framing --------------------------------


def _draft(**kw):
    from generative.schemas.atomic_note import AtomicNoteDraft

    base = dict(
        title="T",
        body="b",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
        action="create",
        critic_score=5,
        hard_gates_pass=True,
    )
    base.update(kw)
    return AtomicNoteDraft(**base)


def test_auto_write_blocks_edition_unverified_even_with_score5():
    from generative.pipeline.vault_writer import auto_write_decision

    note = _draft(source_status="edition-unverified")  # sonst Vault-tauglich
    auto, reason = auto_write_decision(note)
    assert auto is False
    assert "edition" in reason.lower()


def test_auto_write_allows_when_edition_resolved():
    from generative.pipeline.vault_writer import auto_write_decision

    note = _draft(source_status=None)
    assert auto_write_decision(note) == (True, "ok")


def test_framing_for_edition_unverified_mentions_doi_pin():
    from generative.pipeline.routing_report import source_status_framing

    line = source_status_framing("edition-unverified", "Reimer - 2013 - X.pdf")
    assert line is not None and "--doi" in line


def test_framing_none_for_resolved_source():
    from generative.pipeline.routing_report import source_status_framing

    assert source_status_framing(None, "x.pdf") is None
