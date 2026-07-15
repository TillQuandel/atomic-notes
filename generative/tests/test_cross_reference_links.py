"""Regression: cross_reference darf keine doppelt geklammerten Wikilinks erzeugen.

Ebner-Run 2026-06-22: das LLM lieferte duplicate_path als "[[Titel]]"; der Code
machte Path(dup_path).stem (strippt nur ".md", nicht Klammern) und wrappte erneut
→ related-Eintrag "[[[[Titel]]]]" + verklammertes Duplikat-Flag. Der nachträgliche
insert in data["related"] umging den _WIKILINK_RE-Validator des Parsers.
"""

from generative.agents import cross_reference as cr


def test_clean_wikilink_strips_nested_brackets():
    assert cr._clean_wikilink("[[[[Kirkpatrick Level 1→2 Zusammenhang]]]]") == "Kirkpatrick Level 1→2 Zusammenhang"
    assert cr._clean_wikilink("[[Foo]]") == "Foo"
    assert cr._clean_wikilink("Foo") == "Foo"
    assert cr._clean_wikilink("  [[Foo]]  ") == "Foo"


def test_clean_wikilink_keeps_alias_pipe():
    # Alias-Wikilinks bleiben erhalten (nur Klammern werden normalisiert).
    assert cr._clean_wikilink("[[A|alias]]") == "A|alias"


def test_clean_wikilink_handles_empty():
    assert cr._clean_wikilink("") == ""
    assert cr._clean_wikilink(None) == ""


def test_clean_wikilink_does_not_mangle_inner_brackets():
    # strip("[]") zerstörte Titel mit Klammern an einem Ende; jetzt nur paarweise
    # von außen strippen, wenn beide Enden Klammern tragen (Qwen-Review HIGH).
    assert cr._clean_wikilink("[2024] Projekt X") == "[2024] Projekt X"
    assert cr._clean_wikilink("Array [1]") == "Array [1]"
    assert cr._clean_wikilink("[Titel](path.md)") == "[Titel](path.md)"


def test_rewrapped_link_is_single_pair():
    # Der Code baut dup_link als f"[[{_clean_wikilink(x)}]]" — nie doppelt geklammert.
    raw = "[[Kirkpatrick Level 1→2 Zusammenhang]]"
    dup_link = f"[[{cr._clean_wikilink(raw)}]]"
    assert dup_link == "[[Kirkpatrick Level 1→2 Zusammenhang]]"
    assert "[[[[" not in dup_link


# Knowles-Run 2026-06-25: das LLM lieferte duplicate_path als EINE kommaseparierte
# Liste von 4 Titeln → related-Eintrag wurde zu einem kaputten Sammel-Wikilink
# "[[A, B, C, D]]". duplicate_path ist als EIN Ziel modelliert; Mehrfachziele
# müssen in saubere Einzelziele zerlegt werden, nie als ein Link gerendert.
def test_clean_dup_targets_splits_comma_list():
    raw = (
        "Readiness to Learn (Andragogy), Self-directed Learning, "
        "Experience as Learning Resource, Problem-centered Learning Orientation"
    )
    assert cr._clean_dup_targets(raw) == [
        "Readiness to Learn (Andragogy)",
        "Self-directed Learning",
        "Experience as Learning Resource",
        "Problem-centered Learning Orientation",
    ]


def test_clean_dup_targets_single_target():
    assert cr._clean_dup_targets("[[Foo]]") == ["Foo"]
    assert cr._clean_dup_targets("Foo|alias") == ["Foo"]
    assert cr._clean_dup_targets("path/to/Foo.md") == ["Foo"]


def test_clean_dup_targets_empty():
    assert cr._clean_dup_targets("") == []
    assert cr._clean_dup_targets(None) == []


# #75: Ein einzelner Wikilink-Block, der ein Komma im Titel trägt, ist EIN Ziel —
# der Komma-Split zerlegte ihn zuvor in Fake-Targets ('Smith' + 'John (2020)').
def test_clean_dup_targets_single_wikilink_with_comma_not_split():
    assert cr._clean_dup_targets("[[Smith, John (2020)]]") == ["Smith, John (2020)"]
    assert cr._clean_dup_targets("[[Daten, Information, Wissen]]") == ["Daten, Information, Wissen"]


# Mehrere explizite Wikilink-Blöcke bleiben getrennte Ziele (der eigentliche
# Multi-Target-Fall, den das Komma-Signal treffen soll).
def test_clean_dup_targets_multiple_wikilink_blocks_split():
    assert cr._clean_dup_targets("[[A]], [[B]]") == ["A", "B"]
    assert cr._clean_dup_targets("[[A]], [[B]], [[C]]") == ["A", "B", "C"]


# #285: ein klammerloser duplicate_path-Titel (kein Wikilink-Block, keine LLM-
# Roh-Liste) mit einem EIGENEN Komma in Klammern wurde vom naiven s.split(",")
# in zwei Fake-Targets zerschnitten ("... (Mann" + "Mozart & Molekuel)").
# Kommas innerhalb runder Klammern gehören zum Titel, nicht als Trenner.
def test_clean_dup_targets_bare_title_with_parenthetical_comma_not_split():
    assert cr._clean_dup_targets("Wissenskulturen im Vergleich (Mann, Mozart & Molekuel)") == [
        "Wissenskulturen im Vergleich (Mann, Mozart & Molekuel)"
    ]


# #285: Path(inner).stem interpretiert den letzten Punkt im String als Datei-
# Endung und schneidet alles danach ab — bricht Titel mit Abkürzungspunkten
# ("vs.", "z. B.") die keine echte .md-Datei-Endung tragen.
def test_clean_dup_targets_bare_title_with_abbreviation_dot_not_truncated():
    assert cr._clean_dup_targets("Eigenstaendiger KI-Kompetenzrahmen vs. Framework-Integration") == [
        "Eigenstaendiger KI-Kompetenzrahmen vs. Framework-Integration"
    ]
    assert cr._clean_dup_targets("Fallbeispiel z. B. Bibliothekskatalog") == ["Fallbeispiel z. B. Bibliothekskatalog"]


# Positiv-Kontrolle: echte .md-Vault-Pfade (auch mit Verzeichnis-Präfix) werden
# weiterhin korrekt zu ihrem Datei-Stem aufgelöst — die .md-Endungs-Erkennung
# darf durch den #285-Fix nicht verloren gehen.
def test_clean_dup_targets_real_md_path_still_resolves_to_stem():
    assert cr._clean_dup_targets("05-llm-wiki/some-note.md") == ["some-note"]
    assert cr._clean_dup_targets("Foo.md, Bar.md") == ["Foo", "Bar"]
