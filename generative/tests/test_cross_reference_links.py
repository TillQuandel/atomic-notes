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


def test_rewrapped_link_is_single_pair():
    # Der Code baut dup_link als f"[[{_clean_wikilink(x)}]]" — nie doppelt geklammert.
    raw = "[[Kirkpatrick Level 1→2 Zusammenhang]]"
    dup_link = f"[[{cr._clean_wikilink(raw)}]]"
    assert dup_link == "[[Kirkpatrick Level 1→2 Zusammenhang]]"
    assert "[[[[" not in dup_link
