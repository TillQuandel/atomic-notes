"""Tests für shared.author_norm.drop_institutional_coauthors + Integration in die
beiden Zotero-Dateiname-Parser.

Bug-Klasse (Mahmood-Lauf 2026-06-25): Zotero exportiert die Affiliation als
zweiten "Autor" (`Mahmood und University of the Punjab`). Die Pipeline behandelte
"University of the Punjab" als Koautor → Body-Zitate "Mahmood & Punjab",
`_extract_primary_authors` verlor den echten Autor ganz (-> ['Punjab']),
`_short_label` erzeugte falsches "et al.". Wurzel: ungereinigter Autor-String aus
ZWEI Dateiname-Parsern (pdf_enrich._parse_filename_dynamic Kanal 1 + vault_writer.
_parse_filename_fallback Kanal 2).
"""
from pathlib import Path

from shared.author_norm import drop_institutional_coauthors
from generative.tools.pdf_enrich import _parse_filename_dynamic
from generative.pipeline.vault_writer import _parse_filename_fallback


# --- Kern-Helper: Person + Institution gemischt -> Institution droppen ---

def test_german_und_affiliation_dropped():
    assert drop_institutional_coauthors("Mahmood und University of the Punjab") == "Mahmood"


def test_english_and_affiliation_dropped():
    assert drop_institutional_coauthors("Smith and University of California") == "Smith"


def test_semicolon_institute_dropped():
    assert drop_institutional_coauthors("Müller; Institut für Bildungsforschung") == "Müller"


# --- Regressions-Schutz: legitime Fälle dürfen NICHT verändert werden ---

def test_two_persons_unchanged():
    # KRITISCH: legitime Zwei-Autoren bleiben unverändert.
    assert drop_institutional_coauthors("Schlebbe und Greifeneder") == "Schlebbe und Greifeneder"


def test_sole_corporate_author_preserved():
    # Reiner Korporativ-Autor (alle Segmente institutionell) bleibt erhalten.
    assert drop_institutional_coauthors("World Health Organization") == "World Health Organization"


def test_single_person_unchanged():
    assert drop_institutional_coauthors("Mahmood") == "Mahmood"


def test_empty_unchanged():
    assert drop_institutional_coauthors("") == ""


def test_three_persons_unchanged():
    assert drop_institutional_coauthors("Gross and Latham and Folk") == "Gross and Latham and Folk"


# --- Review-Härtung (Qwen/Codex 2026-06-25) ---

def test_uppercase_separator_still_strips():
    # HIGH 1 (Qwen): 'UND'/'AND' aus manuellem Rename muss auch greifen.
    assert drop_institutional_coauthors("Mahmood UND University of the Punjab") == "Mahmood"
    assert drop_institutional_coauthors("Smith AND University of California") == "Smith"


def test_single_token_surname_collision_preserved():
    # HIGH 2 (Qwen) / LOW (Codex): 1-Wort-Nachname, der zufällig ein Marker-Wort
    # ist, darf NICHT als Institution gestrippt werden (≥2-Token-Guard).
    assert drop_institutional_coauthors("Smith und Hospital") == "Smith und Hospital"
    assert drop_institutional_coauthors("Bureau und Center") == "Bureau und Center"


def test_markerless_affiliation_passes_through():
    # Akzeptiertes Residual (Codex MED): verkürzte Affiliation ohne Marker kann
    # nicht erkannt werden — dokumentiert das bewusste Limit.
    assert drop_institutional_coauthors("Mahmood und Punjab") == "Mahmood und Punjab"


# --- Integration: beide Dateiname-Parser liefern den gereinigten Autor ---

_MAHMOOD = ("Mahmood und University of the Punjab - 2016 - "
            "Do People Overestimate Their Information Literacy Skills.pdf")


def test_parse_filename_dynamic_cleans_affiliation():
    meta = _parse_filename_dynamic(Path(_MAHMOOD))
    assert meta is not None
    assert meta["author"] == "Mahmood"


def test_parse_filename_fallback_cleans_affiliation():
    fb = _parse_filename_fallback(_MAHMOOD)
    assert fb["Author"] == "Mahmood"


def test_extractor_source_meta_datei_line_drops_affiliation():
    """Dritter Kanal: die 'Datei:'-Zeile im Extractor-Prompt zeigte den rohen
    Zotero-Dateinamen → der Affiliations-Koautor leakte trotz gesäubertem
    Autor-Feld in LLM-Sekundärzitate ('zit. n. Mahmood & Punjab')."""
    from generative.agents.extractor import _format_source_meta
    out = _format_source_meta(
        {"Author": "Mahmood", "Year": "2016",
         "Title": "Do People Overestimate Their Information Literacy Skills"},
        _MAHMOOD,
    )
    assert "University of the Punjab" not in out
    assert "Mahmood" in out
