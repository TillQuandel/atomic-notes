"""Tests für den Gold-Set-Generator (Faithfulness-Gate E5b, #69).

Alles pure: Footnote→Seite-Mapping, Inline-Anker-Ersetzung, Offset-Detektion,
lexikalisches Kontext-Snippet, Gold-Set-Bau über injizierten page_index —
kein PDF, kein ML-Load in Tests.
"""

from __future__ import annotations

import pytest

from generative.calibration import build_faithfulness_goldset as gs


class TestFootnotePageMap:
    def test_parses_label_and_page_from_defs(self):
        body = "Text[^1] mehr[^2].\n\n[^1]: Hrastinski 2008, S. 4.\n[^2]: Knowles [o. J.], S. 21.\n"
        assert gs.footnote_page_map(body) == {"1": 4, "2": 21}

    def test_def_without_page_is_skipped(self):
        body = "[^1]: Nur ein Label ohne Seite.\n[^2]: Merrill, S. 7.\n"
        assert gs.footnote_page_map(body) == {"2": 7}


class TestInlinePageAnchors:
    def test_replaces_inline_markers_and_keeps_defs(self):
        body = "Aussage eins[^1].\nAussage zwei[^2].\n\n[^1]: Hrastinski 2008, S. 3.\n[^2]: Hrastinski 2008, S. 5.\n"
        result = gs.inline_page_anchors(body, {"1": 3, "2": 5}, offset=50)

        assert "Aussage eins (S. 53)." in result
        assert "Aussage zwei (S. 55)." in result
        # Definitionszeilen bleiben unangetastet (Auftrag: nur Inline-Marker)
        assert "[^1]: Hrastinski 2008, S. 3." in result
        assert "[^2]: Hrastinski 2008, S. 5." in result

    def test_unknown_marker_stays_verbatim(self):
        body = "Aussage[^9].\n\n[^1]: X, S. 2.\n"
        result = gs.inline_page_anchors(body, {"1": 2}, offset=0)
        assert "Aussage[^9]." in result


class TestDetectPageOffset:
    def test_note_pages_subset_of_index_keys_means_zero(self):
        assert gs.detect_page_offset({18, 21}, [1, 2, 18, 21, 25]) == 0

    def test_formfeed_notes_below_page_labels_get_min_minus_one(self):
        # Alt-Notes zaehlen Form-Feed 1..5, PageLabels liefern 51..55 → +50
        assert gs.detect_page_offset({3, 4, 5}, [51, 52, 53, 54, 55]) == 50

    def test_ambiguous_mapping_raises_instead_of_guessing(self):
        with pytest.raises(ValueError):
            gs.detect_page_offset({3, 60}, [51, 52, 53])

    def test_no_note_pages_means_zero(self):
        assert gs.detect_page_offset(set(), [51, 52]) == 0


class TestBestSnippet:
    def test_picks_region_with_most_claim_tokens(self):
        page_text = (
            ("Fuellsatz ohne Bezug. " * 30)
            + "Die Studie fand 99 Prozent inhaltsbezogene Saetze in kleinen Gruppen. "
            + ("Noch mehr Fuelltext. " * 30)
        )
        snippet = gs.best_snippet("99 Prozent der Saetze waren inhaltsbezogen", page_text, width=200)
        assert "99 Prozent inhaltsbezogene" in snippet
        assert len(snippet) <= 260  # width + Ellipsen-Toleranz


class TestBuildGoldset:
    def test_end_to_end_over_injected_page_index(self):
        raw = "---\ntitle: Testnote\n---\n# Testnote\n\nDie Quote lag bei 42 Prozent[^1].\n\n[^1]: Quelle 2008, S. 3.\n"
        page_index = {51: "Erste Seite.", 52: "Zweite Seite.", 53: "Der Anteil betrug 42 Prozent in der Stichprobe."}

        files, claims = gs.build_goldset([("Testnote.md", raw)], page_index, source_label="Quelle 2008", start_index=7)

        assert len(claims) == 1
        c = claims[0]
        assert c["note"] == "Testnote.md"
        assert c["anchor_page"] == 53  # Offset +50 automatisch erkannt
        assert "42" in c["text"]
        assert c["risk_types"] == ["number"]

        (fname, content) = next(iter(files.items()))
        assert fname.startswith("07__")
        assert "### Claim 1" in content
        assert "<!--claim_idx=0-->" in content
        # Kontext-Snippet aus der Anker-Seite ist enthalten
        assert "42 Prozent" in content

    def test_blockquote_claims_are_excluded(self):
        raw = "---\nt: x\n---\n> Zitat mit Zahl 42 (S. 3).\n\nEchte Aussage mit 7 Prozent[^1].\n\n[^1]: Q, S. 3.\n"
        page_index = {3: "7 Prozent und 42 stehen hier."}
        _files, claims = gs.build_goldset([("N.md", raw)], page_index, source_label="Q", start_index=1)
        assert len(claims) == 1
        assert "7 Prozent" in claims[0]["text"]


class TestGateVerdicts:
    def test_verdicts_are_mapped_per_claim_idx(self, monkeypatch):
        raw = "---\nt: x\n---\nQuote lag bei 42 Prozent[^1]. Anteil sank auf 7 Prozent[^2].\n\n[^1]: Q, S. 3.\n[^2]: Q, S. 3.\n"
        page_index = {3: "42 Prozent und 7 Prozent stehen hier."}

        class _V:
            def __init__(self, status, entailment, evidence):
                self.status = status
                self.entailment = entailment
                self.evidence = evidence

        class _R:
            verdicts = [_V("supported", 0.9, "Beleg A"), _V("failed_entailment", 0.1, "Beleg B")]

        monkeypatch.setattr(gs, "_run_gate", lambda body, page_index, citation: _R())

        records = gs.gate_verdicts([("N.md", raw)], page_index, author="Autor")

        assert len(records) == 2
        assert records[0] == {
            "note": "N.md",
            "claim_idx": 0,
            "status": "supported",
            "entailment": 0.9,
            "evidence": "Beleg A",
        }
        assert records[1]["claim_idx"] == 1
        assert records[1]["status"] == "failed_entailment"
