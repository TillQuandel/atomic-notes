"""Tests für die Trust-/Routing-Sichtbarkeit (#45).

Pure Helper, die das Trust-Rückgrat (fail-closed, Critic-Routing) im Lauf
sichtbar machen: Routing-Grund inkl. konkreter Flags, Final-Report-Aggregat,
fail-closed-NL-Zeile + source-status.
"""

from generative.schemas.atomic_note import AtomicNoteDraft


def _draft(
    title="T",
    critic_score=5,
    hard_gates_pass=True,
    quality_flags=None,
    source_status=None,
    action="create",
    hub_subconcepts=None,
):
    return AtomicNoteDraft(
        title=title,
        body="b",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
        quality_flags=quality_flags or [],
        action=action,
        hub_subconcepts=hub_subconcepts or [],
        critic_score=critic_score,
        hard_gates_pass=hard_gates_pass,
        source_status=source_status,
    )


class TestSummarizeRouting:
    def test_counts_vault_and_inbox(self):
        from generative.pipeline.routing_report import summarize_routing

        drafts = [
            _draft(critic_score=5, hard_gates_pass=True),  # vault
            _draft(critic_score=3, hard_gates_pass=True),  # inbox: score
            _draft(critic_score=5, hard_gates_pass=False),  # inbox: gate
        ]
        s = summarize_routing(drafts)
        assert s["vault"] == 1
        assert s["inbox"] == 2

    def test_breaks_down_inbox_by_reason(self):
        from generative.pipeline.routing_report import summarize_routing

        drafts = [
            _draft(critic_score=3, hard_gates_pass=True),  # score
            _draft(critic_score=5, hard_gates_pass=False),  # gate
            _draft(critic_score=5, hard_gates_pass=True, source_status="unresolved"),
        ]
        s = summarize_routing(drafts)
        assert s["inbox_score_gates"] == 2
        assert s["source_unresolved"] == 1

    def test_pdfs_modified_always_zero(self):
        from generative.pipeline.routing_report import summarize_routing

        s = summarize_routing([_draft()])
        assert s["pdfs_modified"] == 0


class TestRoutingStatusLine:
    def test_vault_note_shows_ok(self):
        from generative.pipeline.routing_report import routing_status_line

        line = routing_status_line(_draft(critic_score=5, hard_gates_pass=True))
        assert "Vault" in line
        assert "5/5" in line

    def test_inbox_note_shows_reason(self):
        from generative.pipeline.routing_report import routing_status_line

        line = routing_status_line(_draft(critic_score=3, hard_gates_pass=True))
        assert "Inbox" in line
        assert "score 3<4" in line

    def test_includes_concrete_flags_in_live_run(self):
        from generative.pipeline.routing_report import routing_status_line

        line = routing_status_line(
            _draft(critic_score=3, hard_gates_pass=True, quality_flags=["⚠️ nur 1 Quelle", "⚠️ nicht peer-reviewed"])
        )
        assert "nur 1 Quelle" in line
        assert "nicht peer-reviewed" in line


class TestFinalReportAggregateLine:
    def test_aggregate_line_mentions_distribution(self):
        from generative.pipeline.routing_report import final_report_lines

        drafts = [
            _draft(critic_score=5, hard_gates_pass=True),
            _draft(critic_score=3, hard_gates_pass=True),
            _draft(critic_score=5, hard_gates_pass=True, source_status="unresolved"),
        ]
        lines = final_report_lines(drafts)
        text = "\n".join(lines)
        assert "Vault" in text
        assert "Inbox" in text
        assert "Quelle offen" in text
        assert "0 PDFs" in text or "0 PDF" in text

    def test_source_unresolved_is_its_own_line_not_inside_inbox(self):
        # source_unresolved ist orthogonal zum Routing (eine Vault-Note kann
        # unresolved sein) → eigene Zeile, nicht als Inbox-Unterposten (sonst
        # Doppelzählung X+Y>Inbox).
        from generative.pipeline.routing_report import final_report_lines

        drafts = [_draft(critic_score=5, hard_gates_pass=True, source_status="unresolved")]
        lines = final_report_lines(drafts)
        inbox_line = next(l for l in lines if "Inbox" in l)
        assert "Quelle offen" not in inbox_line
        # und die Vault-Note IST trotzdem als Quelle-offen gezählt
        assert any("Quelle offen" in l and "1" in l for l in lines)


class TestSourceStatusFraming:
    def test_unresolved_produces_first_person_nl_line(self):
        from generative.pipeline.routing_report import source_status_framing

        line = source_status_framing("unresolved", "Beispiel.pdf")
        assert line is not None
        # first-person, ehrliche Rahmung — kein leeres Metadatum
        assert "ich" in line.lower()
        assert "Beispiel.pdf" in line
        # nennt das gesetzte Flag, behauptet aber kein Routing das es nicht garantiert
        assert "source-status" in line
        assert "Inbox gelegt" not in line

    def test_resolved_or_none_produces_no_line(self):
        from generative.pipeline.routing_report import source_status_framing

        assert source_status_framing(None, "x.pdf") is None
        assert source_status_framing("resolved", "x.pdf") is None


class TestIsSourceUnresolved:
    def test_resolved_from_enriched_meta(self):
        from generative.pipeline.routing_report import is_source_unresolved

        assert is_source_unresolved({"Author": "Müller", "Year": "2020"}, {}, False) is False

    def test_resolved_from_filename_fallback(self):
        # fb["Author"] landet nicht in enriched_meta — darf trotzdem nicht
        # fälschlich als unresolved gelten (Zotero-Schema).
        from generative.pipeline.routing_report import is_source_unresolved

        assert is_source_unresolved({}, {"Author": "Müller", "Year": "2020"}, False) is False

    def test_unresolved_when_author_or_year_missing(self):
        from generative.pipeline.routing_report import is_source_unresolved

        assert is_source_unresolved({"Author": "Müller"}, {}, False) is True
        assert is_source_unresolved({"Year": "2020"}, {}, False) is True
        assert is_source_unresolved({}, {}, False) is True

    def test_unresolved_when_crossref_override_blocked(self):
        # selbst bei aufgelöstem Autor/Jahr: verworfener CrossRef-Override = unsicher
        from generative.pipeline.routing_report import is_source_unresolved

        assert is_source_unresolved({"Author": "Müller", "Year": "2020"}, {}, True) is True


class TestSourceStatusFrontmatter:
    def test_unresolved_rendered_as_frontmatter_flag(self):
        from generative.pipeline.vault_writer import render_note

        note = _draft(source_status="unresolved")
        out = render_note(note, "Beispiel.pdf")
        assert "source-status: unresolved" in out

    def test_no_flag_when_status_none(self):
        from generative.pipeline.vault_writer import render_note

        note = _draft(source_status=None)
        out = render_note(note, "Beispiel.pdf")
        assert "source-status:" not in out
