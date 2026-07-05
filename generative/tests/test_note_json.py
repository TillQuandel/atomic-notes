"""Tests für pipeline.note_json — kanonisches Export-Schema für AtomicNoteDraft
(Output-Projekt F1). Schützt den externen Contract: Vollständigkeit aller Draft-
Felder, Nicht-Mutation, Routing-Frische, Body-Legacy-Strip, JSON-Encoding.
"""

import copy
import dataclasses
import json
import unittest

from generative.pipeline.note_json import (
    SCHEMA_VERSION,
    dumps,
    note_to_json_dict,
    run_to_json_dict,
)
from generative.pipeline.vault_writer import strip_legacy_sections
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor
from generative.schemas.citation import CitationMeta


def _draft(**overrides) -> AtomicNoteDraft:
    base = dict(
        title="Zettelkasten-Methode",
        body="# Zettelkasten-Methode\n\nEin Zettelkasten ist ein Notizsystem (S. 4).",
        source_anchors=[TextAnchor(quote="Ein Zettelkasten ist ein Notizsystem", page="S. 4")],
        related=["[[Luhmann]]"],
        tags=["#konzept"],
        synthesis_confidence="high",
        quality_flags=["⚠️ Faithfulness: failed_entailment e=0.00 — Zeitangabe nicht belegt"],
        critic_score=5,
        hard_gates_pass=True,
    )
    base.update(overrides)
    return AtomicNoteDraft(**base)


def _citation(**overrides) -> CitationMeta:
    base = dict(author="Luhmann, Niklas", year="1992", title="Zettelkasten", doi=None, source_file="luhmann.pdf")
    base.update(overrides)
    return CitationMeta(**base)


class TestTopLevelStructure(unittest.TestCase):
    def test_top_level_keys_exact(self):
        result = note_to_json_dict(_draft(), _citation())
        self.assertEqual(
            set(result.keys()), {"schema_version", "generated_at", "agent_version", "source", "note", "routing"}
        )

    def test_schema_version_is_1(self):
        result = note_to_json_dict(_draft(), _citation())
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(SCHEMA_VERSION, 1)

    def test_generated_at_default_is_iso_date(self):
        result = note_to_json_dict(_draft(), _citation())
        # ISO-Datum: YYYY-MM-DD, parsebar
        from datetime import date

        date.fromisoformat(result["generated_at"])

    def test_generated_at_override(self):
        result = note_to_json_dict(_draft(), _citation(), generated_at="2020-01-01")
        self.assertEqual(result["generated_at"], "2020-01-01")

    def test_agent_version_matches_config(self):
        from generative import config

        result = note_to_json_dict(_draft(), _citation())
        self.assertEqual(result["agent_version"], config.AGENT_VERSION)


class TestNoteCompleteness(unittest.TestCase):
    def test_all_draft_fields_present_dynamically(self):
        draft = _draft()
        result = note_to_json_dict(draft, _citation())
        expected_keys = {f.name for f in dataclasses.fields(AtomicNoteDraft)}
        self.assertEqual(set(result["note"].keys()), expected_keys)


class TestSourceAnchors(unittest.TestCase):
    def test_source_anchors_structured(self):
        draft = _draft(
            source_anchors=[
                TextAnchor(quote="Zitat A", page="S. 4"),
                TextAnchor(quote="Zitat B", page=None, fuzzy_page="S. 7"),
            ]
        )
        result = note_to_json_dict(draft, _citation())
        anchors = result["note"]["source_anchors"]
        self.assertEqual(len(anchors), 2)
        self.assertEqual(anchors[0], {"quote": "Zitat A", "page": "S. 4", "fuzzy_page": None})
        self.assertEqual(anchors[1], {"quote": "Zitat B", "page": None, "fuzzy_page": "S. 7"})


class TestBodyStrip(unittest.TestCase):
    def test_legacy_quellen_section_stripped_inline_anchor_preserved(self):
        body = "# Titel\n\nText mit Anker (S. 4).\n\n## Quellen\n\n*Quelle: [[foo.pdf]]: Foo*\n"
        draft = _draft(body=body)
        result = note_to_json_dict(draft, _citation())
        rendered_body = result["note"]["body"]
        self.assertNotIn("## Quellen", rendered_body)
        self.assertIn("(S. 4)", rendered_body)
        self.assertNotIn("[^", rendered_body)  # keine Footnote-Konvertierung


class TestNoMutation(unittest.TestCase):
    def test_draft_unchanged_after_call(self):
        draft = _draft()
        before = copy.deepcopy(draft)
        note_to_json_dict(draft, _citation())
        self.assertEqual(draft, before)
        self.assertEqual(draft.quality_flags, before.quality_flags)
        self.assertEqual(draft.body, before.body)


class TestQualityFlagsPreserved(unittest.TestCase):
    def test_faithfulness_verdict_string_preserved(self):
        flag = "⚠️ Faithfulness: failed_entailment e=0.00 — Zeitangabe nicht durch Quelle belegt"
        draft = _draft(quality_flags=[flag])
        result = note_to_json_dict(draft, _citation())
        self.assertEqual(result["note"]["quality_flags"], [flag])


class TestRouting(unittest.TestCase):
    def test_faithfulness_fail_blocks_auto_vault(self):
        draft = _draft(faithfulness_fail=True, critic_score=5, hard_gates_pass=True)
        result = note_to_json_dict(draft, _citation())
        self.assertFalse(result["routing"]["auto_vault_recommended"])
        self.assertIn("Faithfulness", result["routing"]["reason"])

    def test_high_score_hard_gates_pass_routes_true(self):
        draft = _draft(critic_score=5, hard_gates_pass=True, faithfulness_fail=False)
        result = note_to_json_dict(draft, _citation())
        self.assertTrue(result["routing"]["auto_vault_recommended"])
        self.assertEqual(result["routing"]["reason"], "ok")

    def test_routing_is_freshly_computed_not_from_draft_field(self):
        # Draft-Feld auto_vault_recommended widerspricht der frischen Berechnung —
        # routing.auto_vault_recommended muss die frische Berechnung liefern.
        draft = _draft(critic_score=5, hard_gates_pass=True, faithfulness_fail=False, auto_vault_recommended=False)
        result = note_to_json_dict(draft, _citation())
        self.assertTrue(result["routing"]["auto_vault_recommended"])
        self.assertFalse(result["note"]["auto_vault_recommended"])


class TestCitation(unittest.TestCase):
    def test_short_label_author_with_year(self):
        result = note_to_json_dict(_draft(), _citation(author="Luhmann", year="1992"))
        self.assertEqual(result["source"]["citation"]["short_label"], "Luhmann 1992")

    def test_no_year_yields_o_j_display_and_label(self):
        result = note_to_json_dict(_draft(), _citation(author="Luhmann", year=None))
        self.assertEqual(result["source"]["citation"]["display_year"], "[o. J.]")
        self.assertTrue(result["source"]["citation"]["short_label"].endswith("[o. J.]"))

    def test_citation_fields_1to1_null_when_none(self):
        result = note_to_json_dict(_draft(), _citation(author=None, year=None, title=None, doi=None))
        c = result["source"]["citation"]
        self.assertIsNone(c["author"])
        self.assertIsNone(c["year"])
        self.assertIsNone(c["title"])
        self.assertIsNone(c["doi"])

    def test_source_file_present(self):
        result = note_to_json_dict(_draft(), _citation(source_file="test.pdf"))
        self.assertEqual(result["source"]["file"], "test.pdf")


class TestDumps(unittest.TestCase):
    def test_umlauts_and_special_chars_literal(self):
        result = note_to_json_dict(_draft(title="Übermäßig große Ähnlichkeit"), _citation())
        text = dumps(result)
        self.assertIn("Übermäßig", text)
        self.assertNotIn("\\u00", text)

    def test_ends_with_newline(self):
        result = note_to_json_dict(_draft(), _citation())
        text = dumps(result)
        self.assertTrue(text.endswith("\n"))

    def test_roundtrip(self):
        result = note_to_json_dict(_draft(), _citation())
        text = dumps(result)
        self.assertEqual(json.loads(text), result)


class TestRunToJsonDict(unittest.TestCase):
    def test_notes_list_with_note_and_routing_entries_no_duplicate_source(self):
        drafts = [_draft(title="Note A"), _draft(title="Note B", faithfulness_fail=True)]
        result = run_to_json_dict(drafts, _citation())
        self.assertEqual(set(result.keys()), {"schema_version", "generated_at", "agent_version", "source", "notes"})
        self.assertEqual(len(result["notes"]), 2)
        for entry in result["notes"]:
            self.assertEqual(set(entry.keys()), {"note", "routing"})
            self.assertNotIn("source", entry)
        self.assertEqual(result["notes"][0]["note"]["title"], "Note A")
        self.assertFalse(result["notes"][1]["routing"]["auto_vault_recommended"])


class TestSingleAndRunFormsEquivalent(unittest.TestCase):
    def test_run_entry_matches_single_form(self):
        # Mistral-Review MED (PR #133): pinnt die Helper-Garantie — wer künftig
        # eine der beiden Formen am `_note_and_routing`-Helper vorbei baut,
        # bricht diesen Test statt still zu driften.
        draft = _draft()
        citation = _citation()
        single = note_to_json_dict(draft, citation, generated_at="2020-01-01")
        run = run_to_json_dict([draft], citation, generated_at="2020-01-01")
        self.assertEqual(run["notes"][0]["note"], single["note"])
        self.assertEqual(run["notes"][0]["routing"], single["routing"])
        self.assertEqual(run["source"], single["source"])


class TestStripLegacySections(unittest.TestCase):
    def test_quellen_section_removed(self):
        body = "Text.\n\n## Quellen\n\n*Quelle: foo*\n"
        self.assertNotIn("## Quellen", strip_legacy_sections(body))

    def test_confidence_notiz_removed(self):
        body = "Text.\n\n## Confidence-Notiz\n\nBla bla.\n"
        self.assertNotIn("## Confidence-Notiz", strip_legacy_sections(body))

    def test_normal_heading_preserved(self):
        body = "Text.\n\n## Komponenten\n\n1. [[A]]\n"
        self.assertIn("## Komponenten", strip_legacy_sections(body))


if __name__ == "__main__":
    unittest.main()
