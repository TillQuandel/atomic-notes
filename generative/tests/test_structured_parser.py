"""Tests für das Structured-Output-Format (HTML-Kommentar-Sentinels).

Edge-Cases aus 4 Review-Runden — Format-Spec gegen pathologische Outputs härten.
Ohne diese Suite wiederholen wir das v16-Silent-Fail-Pattern (anchor_patterns).
"""
from __future__ import annotations
import unittest

# Direct-Import-Pattern (wie in den anderen Modulen) — Sys-Path-Patch

from generative.agents.structured_output import (
    _normalize_lines,
    _split_sentinels,
    _parse_header_line,
    _parse_bool,
    _parse_int,
    parse_headers,
    parse_extractor_output,
    parse_canonicalizer_output,
    parse_verifier_output,
    parse_critic_output,
    parse_planner_output,
    parse_cross_reference_output,
)


# ---------- Low-level helpers ----------

class TestNormalizeLines(unittest.TestCase):
    def test_lf(self):
        self.assertEqual(_normalize_lines("a\nb\n"), ["a", "b", ""])

    def test_crlf(self):
        self.assertEqual(_normalize_lines("a\r\nb\r\n"), ["a", "b", ""])

    def test_mixed(self):
        self.assertEqual(_normalize_lines("a\r\nb\nc"), ["a", "b", "c"])


class TestSplitSentinels(unittest.TestCase):
    def test_basic(self):
        text = "<!--NOTE-->\ntitle: Foo\n<!--BODY-->\nbody\n<!--END-->"
        sections = _split_sentinels(_normalize_lines(text))
        names = [n for n, _ in sections]
        self.assertEqual(names, [None, "NOTE", "BODY", "END"])

    def test_indented_sentinel_tolerated(self):
        text = "  <!--BODY-->\nbody"
        sections = _split_sentinels(_normalize_lines(text))
        self.assertIn("BODY", [n for n, _ in sections])

    def test_lowercase_html_comment_is_content(self):
        text = "<!--BODY-->\n<!--todo: fix-->\nactual body\n<!--END-->"
        sections = _split_sentinels(_normalize_lines(text))
        body_lines = next(c for n, c in sections if n == "BODY")
        self.assertIn("<!--todo: fix-->", body_lines)
        self.assertIn("actual body", body_lines)

    def test_unknown_uppercase_sentinel_is_content(self):
        # FOOBAR ist nicht in der Enum — bleibt als Body-Content
        text = "<!--BODY-->\n<!--FOOBAR-->\ntext\n<!--END-->"
        sections = _split_sentinels(_normalize_lines(text))
        body_lines = next(c for n, c in sections if n == "BODY")
        self.assertIn("<!--FOOBAR-->", body_lines)


# ---------- Header parsing ----------

class TestParseHeader(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_parse_header_line("title: Foo"), ("title", "Foo"))

    def test_colon_in_value(self):
        # Split nur auf erstem Doppelpunkt
        self.assertEqual(
            _parse_header_line("title: Wilson 1981: A Theory"),
            ("title", "Wilson 1981: A Theory"),
        )

    def test_explanatory_text_no_match(self):
        # „Hier kommt..." matcht nicht das key-Pattern
        self.assertIsNone(_parse_header_line("Hier kommt die Note:"))

    def test_uppercase_key_no_match(self):
        # Strict lowercase key
        self.assertIsNone(_parse_header_line("Title: Foo"))

    def test_empty_value(self):
        self.assertEqual(_parse_header_line("extend_path:"), ("extend_path", ""))

    def test_trailing_whitespace(self):
        self.assertEqual(_parse_header_line("title: Foo   "), ("title", "Foo"))


class TestParseHeadersBlock(unittest.TestCase):
    def test_explanatory_text_ignored(self):
        lines = ["Hier kommt Note 1:", "title: Foo", "Dieser Text ignoriert"]
        h, ignored = parse_headers(lines)
        self.assertEqual(h, {"title": "Foo"})
        self.assertEqual(len(ignored), 2)

    def test_null_variants(self):
        lines = ["extend_path: null", "foo:", "bar: none", "baz: -"]
        h, _ = parse_headers(lines)
        self.assertEqual(h, {"extend_path": None, "foo": None, "bar": None, "baz": None})


# ---------- Value parsers ----------

class TestParseBool(unittest.TestCase):
    def test_truthy(self):
        for v in ["true", "TRUE", "True", "yes", "1", "t", "y"]:
            self.assertTrue(_parse_bool(v), f"Expected truthy for {v!r}")

    def test_falsy(self):
        for v in ["false", "no", "0", "", "null", "None", None]:
            self.assertFalse(_parse_bool(v), f"Expected falsy for {v!r}")


class TestParseInt(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_parse_int("4"), 4)

    def test_with_suffix(self):
        self.assertEqual(_parse_int("4 (good)"), 4)

    def test_with_slash(self):
        self.assertEqual(_parse_int("4/5"), 4)

    def test_no_digit(self):
        self.assertEqual(_parse_int("score"), 0)

    def test_empty(self):
        self.assertEqual(_parse_int(""), 0)

    def test_none(self):
        self.assertEqual(_parse_int(None), 0)


# ---------- Extractor ----------

EXTRACTOR_TWO_NOTES = """<!--NOTE-->
title: First
aliases: A, B
tags: t1
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
First body with --- HR
and "quotes" and `code`.

Multi-paragraph.
<!--ANCHOR-->
page: S. 5
<!--QUOTE-->
First quote with "embedded" quotes
<!--NOTE-->
title: Second
aliases:
tags: t2
synthesis_confidence: medium
action: create
extend_path:
<!--BODY-->
Second body
<!--ANCHOR-->
page: S. 10
<!--QUOTE-->
Second quote
<!--END-->
"""


class TestExtractorParse(unittest.TestCase):
    def test_two_notes(self):
        notes, warnings = parse_extractor_output(EXTRACTOR_TWO_NOTES)
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]["title"], "First")
        self.assertEqual(notes[1]["title"], "Second")

    def test_body_preserves_hr_and_quotes(self):
        notes, _ = parse_extractor_output(EXTRACTOR_TWO_NOTES)
        body = notes[0]["body"]
        self.assertIn("---", body)
        self.assertIn('"quotes"', body)
        self.assertIn("`code`", body)
        self.assertIn("Multi-paragraph", body)

    def test_aliases_split(self):
        notes, _ = parse_extractor_output(EXTRACTOR_TWO_NOTES)
        self.assertEqual(notes[0]["aliases"], ["A", "B"])
        self.assertEqual(notes[1]["aliases"], [])

    def test_anchors_extracted(self):
        notes, _ = parse_extractor_output(EXTRACTOR_TWO_NOTES)
        anchors = notes[0]["source_anchors"]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["page"], "S. 5")
        self.assertEqual(anchors[0]["quote"], 'First quote with "embedded" quotes')

    def test_extend_path_null(self):
        notes, _ = parse_extractor_output(EXTRACTOR_TWO_NOTES)
        self.assertIsNone(notes[0]["extend_path"])

    def test_anchor_without_quote_dropped(self):
        text = """<!--NOTE-->
title: T
<!--BODY-->
b
<!--ANCHOR-->
page: S. 5
<!--ANCHOR-->
page: S. 10
<!--QUOTE-->
q
<!--END-->
"""
        notes, warnings = parse_extractor_output(text)
        # Erster Anchor hat nur page → dropped. Zweiter ist komplett.
        self.assertEqual(len(notes[0]["source_anchors"]), 1)
        self.assertEqual(notes[0]["source_anchors"][0]["page"], "S. 10")
        self.assertTrue(any("ANCHOR ohne QUOTE" in w for w in warnings))

    def test_note_without_title_skipped(self):
        text = "<!--NOTE-->\n<!--NOTE-->\ntitle: Real\n<!--BODY-->\nbody\n<!--END-->\n"
        notes, warnings = parse_extractor_output(text)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["title"], "Real")
        self.assertTrue(any("title" in w.lower() for w in warnings))

    def test_duplicate_body_last_wins(self):
        text = """<!--NOTE-->
title: T
<!--BODY-->
first body
<!--BODY-->
second body
<!--END-->
"""
        notes, warnings = parse_extractor_output(text)
        self.assertEqual(notes[0]["body"], "second body")
        self.assertTrue(any("Duplicate" in w or "duplicate" in w.lower() for w in warnings))

    def test_crlf_line_endings(self):
        text = EXTRACTOR_TWO_NOTES.replace("\n", "\r\n")
        notes, _ = parse_extractor_output(text)
        self.assertEqual(len(notes), 2)

    def test_explanatory_preamble_ignored(self):
        text = "Hier ist mein Output:\n\n" + EXTRACTOR_TWO_NOTES
        notes, _ = parse_extractor_output(text)
        self.assertEqual(len(notes), 2)

    def test_empty_input(self):
        notes, _ = parse_extractor_output("")
        self.assertEqual(notes, [])

    def test_only_end_sentinel(self):
        notes, _ = parse_extractor_output("<!--END-->\n")
        self.assertEqual(notes, [])


# ---------- Canonicalizer ----------

class TestCanonicalizerParse(unittest.TestCase):
    def test_single_note(self):
        text = """<!--NOTE-->
title: Merged
aliases: A, B, C
tags: t1, t2
<!--BODY-->
Merged body with "all" content
<!--END-->
"""
        data, warnings = parse_canonicalizer_output(text)
        self.assertEqual(data["title"], "Merged")
        self.assertEqual(data["aliases"], ["A", "B", "C"])
        self.assertEqual(data["tags"], ["t1", "t2"])
        self.assertIn('"all"', data["body"])


# ---------- Verifier ----------

class TestVerifierParse(unittest.TestCase):
    def test_two_anchors(self):
        text = """all_verified: false
<!--ANCHOR-->
page: S. 5
verified: true
<!--QUOTE-->
verified quote
<!--ANCHOR-->
page:
verified: false
<!--QUOTE-->
unverified quote
<!--END-->
"""
        data, _ = parse_verifier_output(text)
        self.assertFalse(data["all_verified"])
        self.assertEqual(len(data["anchors"]), 2)
        self.assertTrue(data["anchors"][0]["verified"])
        self.assertEqual(data["anchors"][0]["page"], "S. 5")
        self.assertEqual(data["anchors"][0]["quote"], "verified quote")
        self.assertIsNone(data["anchors"][1]["page"])
        self.assertFalse(data["anchors"][1]["verified"])


# ---------- Critic ----------

class TestCriticParse(unittest.TestCase):
    def test_full_critic(self):
        text = """title_test: true
glance_test: true
future_self_test: false
quellen_test: true
deletion_test: true
score: 4
<!--REVISION_HINT-->
Improve the "title" — it has two ideas.
<!--END-->
"""
        data, _ = parse_critic_output(text)
        self.assertTrue(data["title_test"])
        self.assertFalse(data["future_self_test"])
        self.assertEqual(data["score"], 4)
        self.assertIn('"title"', data["revision_hint"])

    def test_score_with_suffix(self):
        text = "title_test: true\nscore: 4 (out of 5)\n<!--END-->\n"
        data, _ = parse_critic_output(text)
        self.assertEqual(data["score"], 4)

    def test_missing_revision_hint_is_none(self):
        text = "title_test: true\nscore: 5\n<!--END-->\n"
        data, _ = parse_critic_output(text)
        self.assertIsNone(data["revision_hint"])

    def test_score_clamped(self):
        text = "score: 99\n<!--END-->\n"
        data, _ = parse_critic_output(text)
        self.assertEqual(data["score"], 5)
        text = "score: -3\n<!--END-->\n"
        data, _ = parse_critic_output(text)
        self.assertEqual(data["score"], 0)


# ---------- Planner ----------

class TestPlannerParse(unittest.TestCase):
    def test_two_concepts(self):
        text = """source_title: Wilson 1981
source_summary: Two-sentence summary with "quotes" and: colons.
<!--CONCEPT-->
title: Information Need
priority: high
chapter: Kap. 2
action: create
extend_path:
category: conceptual
<!--CONCEPT-->
title: Berrypicking
priority: medium
chapter: Kap. 4
action: extend
extend_path: 04-wissen/berrypicking.md
category: architectural
<!--END-->
"""
        data, warnings = parse_planner_output(text)
        self.assertEqual(data["source_title"], "Wilson 1981")
        self.assertIn('"quotes"', data["source_summary"])
        self.assertEqual(len(data["concepts"]), 2)
        self.assertEqual(data["concepts"][0]["title"], "Information Need")
        self.assertEqual(data["concepts"][0]["priority"], "high")
        self.assertIsNone(data["concepts"][0]["extend_path"])
        self.assertEqual(data["concepts"][0]["category"], "conceptual")
        self.assertEqual(data["concepts"][1]["extend_path"], "04-wissen/berrypicking.md")
        self.assertEqual(data["concepts"][1]["category"], "architectural")

    def test_category_operational(self):
        text = """<!--CONCEPT-->
title: Eval-Pipeline
priority: high
chapter: Kap. 5
action: create
extend_path:
category: operational
<!--END-->
"""
        data, _ = parse_planner_output(text)
        self.assertEqual(data["concepts"][0]["category"], "operational")

    def test_category_default_conceptual_when_missing(self):
        text = "<!--CONCEPT-->\ntitle: Foo\n<!--END-->\n"
        data, _ = parse_planner_output(text)
        self.assertEqual(data["concepts"][0]["category"], "conceptual")

    def test_category_invalid_falls_back_with_warning(self):
        text = """<!--CONCEPT-->
title: Foo
category: weird-value
<!--END-->
"""
        data, warnings = parse_planner_output(text)
        self.assertEqual(data["concepts"][0]["category"], "conceptual")
        self.assertTrue(any("category" in w.lower() for w in warnings))

    def test_category_case_insensitive(self):
        text = """<!--CONCEPT-->
title: Foo
category: Operational
<!--END-->
"""
        data, _ = parse_planner_output(text)
        self.assertEqual(data["concepts"][0]["category"], "operational")

    def test_concept_without_title_skipped(self):
        text = """<!--CONCEPT-->
priority: low
<!--CONCEPT-->
title: Real
priority: high
chapter: x
action: create
extend_path:
<!--END-->
"""
        data, warnings = parse_planner_output(text)
        self.assertEqual(len(data["concepts"]), 1)
        self.assertEqual(data["concepts"][0]["title"], "Real")
        self.assertTrue(any("title" in w.lower() for w in warnings))

    def test_default_values(self):
        text = "<!--CONCEPT-->\ntitle: Foo\n<!--END-->\n"
        data, _ = parse_planner_output(text)
        c = data["concepts"][0]
        self.assertEqual(c["priority"], "medium")
        self.assertEqual(c["action"], "create")
        self.assertEqual(c["chapter"], "")

    def test_empty(self):
        data, _ = parse_planner_output("")
        self.assertEqual(data["concepts"], [])
        self.assertEqual(data["source_title"], "")


# ---------- Cross-Reference ----------

class TestCrossReferenceParse(unittest.TestCase):
    def test_full(self):
        text = """duplicate_risk: high
duplicate_path: 04-wissen/foo.md
<!--CONTRADICTION-->
Note A sagt X, neue Note sagt das Gegenteil: "nicht X" (S. 5).
<!--CONTRADICTION-->
Zweiter Widerspruch.
<!--RELATED-->
[[Note A]]
[[Note B]]
<!--END-->
"""
        data, _ = parse_cross_reference_output(text)
        self.assertEqual(data["duplicate_risk"], "high")
        self.assertEqual(data["duplicate_path"], "04-wissen/foo.md")
        self.assertEqual(len(data["contradictions"]), 2)
        self.assertIn('"nicht X"', data["contradictions"][0])
        self.assertEqual(data["related"], ["[[Note A]]", "[[Note B]]"])

    def test_no_contradictions_no_related(self):
        text = "duplicate_risk: none\nduplicate_path:\n<!--END-->\n"
        data, _ = parse_cross_reference_output(text)
        self.assertEqual(data["duplicate_risk"], "none")
        self.assertIsNone(data["duplicate_path"])
        self.assertEqual(data["contradictions"], [])
        self.assertEqual(data["related"], [])

    def test_related_only_wikilinks(self):
        text = """duplicate_risk: low
<!--RELATED-->
[[Valid]]
not a wikilink
[[Another]]
<!--END-->
"""
        data, warnings = parse_cross_reference_output(text)
        self.assertEqual(data["related"], ["[[Valid]]", "[[Another]]"])
        self.assertTrue(any("Wikilink" in w for w in warnings))

    def test_default_risk(self):
        data, _ = parse_cross_reference_output("<!--END-->\n")
        self.assertEqual(data["duplicate_risk"], "none")

    def test_related_rejects_double_brackets(self):
        # Modell-Halluzination `[[[[Note]]]]` darf nicht durchrutschen
        text = """duplicate_risk: low
<!--RELATED-->
[[[[Note]]]]
[[Sane]]
<!--END-->
"""
        data, warnings = parse_cross_reference_output(text)
        self.assertEqual(data["related"], ["[[Sane]]"])
        self.assertTrue(any("Wikilink" in w for w in warnings))

    def test_related_rejects_two_links_per_line(self):
        # `[[A]] [[B]]` auf einer Zeile darf nicht als ein Link gezählt werden
        text = """duplicate_risk: low
<!--RELATED-->
[[A]] [[B]]
[[C]]
<!--END-->
"""
        data, warnings = parse_cross_reference_output(text)
        self.assertEqual(data["related"], ["[[C]]"])
        self.assertTrue(any("Wikilink" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
