"""Tests fuer den stdout-Parser der Live-GUI.

Der Parser uebersetzt die rohen stdout-Zeilen eines orchestrator-Laufs in
strukturierte Events (Stage-Stepper, Pro-Note-Fortschritt, Dry-Run-Preview,
Abschluss). Pure Logik, keine I/O.
"""

import re
from pathlib import Path

from generative.gui.run_parser import RunParser, STAGES


def _events(lines):
    """Hilfsfunktion: alle Zeilen durchfuettern + finalen flush, Events sammeln."""
    p = RunParser()
    out = []
    for ln in lines:
        out.extend(p.feed(ln))
    out.extend(p.flush())
    return out


def test_stage_marker_emits_stage_event():
    p = RunParser()
    evs = p.feed("[1/7] PDF extrahieren und chunken…")
    assert evs == [{"type": "stage", "num": 1, "total": 7, "label": "PDF extrahieren und chunken…"}]


def test_eval_stage_uses_total_8():
    p = RunParser()
    evs = p.feed("[8/8] Qualitäts-Eval…")
    assert evs[0]["type"] == "stage"
    assert evs[0]["num"] == 8
    assert evs[0]["total"] == 8


def test_fractional_stage_marker_floors_to_int():
    p = RunParser()
    evs = p.feed("[4.5/7] Background-Extractor: Trainingswissen pro Konzept…")
    assert evs[0]["type"] == "stage"
    assert evs[0]["num"] == 4


def test_per_note_progress_event():
    # Zwei fuehrende Leerzeichen + arbitraerer Nenner = Pro-Note (Stage 6),
    # NICHT Stage-Marker.
    p = RunParser()
    evs = p.feed("  [3/10] Zettelkasten als Denkwerkzeug")
    assert evs == [{"type": "note_progress", "index": 3, "total": 10, "title": "Zettelkasten als Denkwerkzeug"}]


def test_stage_marker_has_no_leading_whitespace_per_note_has():
    # Disambiguierung: gleicher Zaehler/Nenner, aber Einrueckung entscheidet.
    p = RunParser()
    stage = p.feed("[1/7] PDF extrahieren…")
    note = p.feed("  [1/7] Eine Note")
    assert stage[0]["type"] == "stage"
    assert note[0]["type"] == "note_progress"


def test_dryrun_vault_recommended_preview():
    lines = [
        "  [DRY-RUN] -> Inbox: zettelkasten-denkwerkzeug.md  [Vault-Empf.]",
        "    Score: 5/5 | Hard-Gates: pass | Confidence: high",
    ]
    evs = _events(lines)
    previews = [e for e in evs if e["type"] == "preview"]
    assert len(previews) == 1
    pv = previews[0]
    assert pv["name"] == "zettelkasten-denkwerkzeug.md"
    assert pv["routing"] == "vault"
    assert pv["score"] == 5
    assert pv["hard_gates"] is True
    assert pv["confidence"] == "high"
    assert pv["flags"] == ""


def test_dryrun_inbox_review_preview_with_reason_and_flags():
    # Die Flags-Zeile wird quellseitig (vault_writer) mit ", " gejoint UND
    # ASCII-safe gedruckt; einzelne Flags (Critic-Hints) enthalten selbst Kommas.
    # Es gibt also keinen verlässlichen Delimiter → der Parser hält die Flags als
    # Roh-String, statt fälschlich an Kommas zu zersplittern.
    lines = [
        "  [DRY-RUN] -> Inbox: schwache-note.md  [Inbox-Review: critic-score 2 < 4]",
        "    Score: 2/5 | Hard-Gates: fail | Confidence: low",
        "    Flags: ?? retracted, Critic: Titel zu generisch, sollte praeziser sein",
    ]
    evs = _events(lines)
    pv = [e for e in evs if e["type"] == "preview"][0]
    assert pv["routing"] == "inbox"
    assert pv["reason"] == "critic-score 2 < 4"
    assert pv["score"] == 2
    assert pv["hard_gates"] is False
    assert pv["confidence"] == "low"
    # Roh-String, NICHT an den eingebetteten Kommas zersplittert.
    assert pv["flags"] == "?? retracted, Critic: Titel zu generisch, sollte praeziser sein"


def test_dryrun_merge_stub_preview():
    lines = [
        "  [DRY-RUN] -> Inbox: vorhandenes-konzept.md  [Merge-Stub -> 04-wissen/Vorhandenes Konzept.md]",
        "    Score: 4/5 | Hard-Gates: pass | Confidence: medium",
    ]
    evs = _events(lines)
    pv = [e for e in evs if e["type"] == "preview"][0]
    assert pv["routing"] == "merge"
    assert pv["merge_target"] == "04-wissen/Vorhandenes Konzept.md"


def test_two_consecutive_previews_both_emitted():
    lines = [
        "  [DRY-RUN] -> Inbox: a.md  [Vault-Empf.]",
        "    Score: 5/5 | Hard-Gates: pass | Confidence: high",
        "  [DRY-RUN] -> Inbox: b.md  [Inbox-Review: x]",
        "    Score: 3/5 | Hard-Gates: pass | Confidence: medium",
    ]
    evs = _events(lines)
    previews = [e for e in evs if e["type"] == "preview"]
    assert [p["name"] for p in previews] == ["a.md", "b.md"]


def test_done_dry_run():
    p = RunParser()
    evs = p.feed("=== Fertig: 4 Notes (dry-run) ===")
    assert evs == [{"type": "done", "written": 4, "dry_run": True}]


def test_done_written():
    p = RunParser()
    evs = p.feed("=== Fertig: 7 Notes geschrieben ===")
    assert evs == [{"type": "done", "written": 7, "dry_run": False}]


def test_plain_line_is_log_event():
    p = RunParser()
    evs = p.feed("[runtime-config] profile=balanced inline_eval=True")
    assert len(evs) == 1
    assert evs[0]["type"] == "log"
    assert "runtime-config" in evs[0]["text"]


def test_blank_line_emits_nothing():
    p = RunParser()
    assert p.feed("") == []
    assert p.feed("   ") == []


def test_stages_table_covers_1_to_8():
    # Der Stepper braucht stabile Labels fuer alle 8 Stufen.
    assert [s["num"] for s in STAGES] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(s["label"] for s in STAGES)


def test_golden_real_stdout_sample_parses_full_run():
    # Kopplungstest: ein Sample ECHTER Orchestrator-stdout (fixtures/run_stdout_sample.txt,
    # bei Print-Format-Änderungen aus einem realen Lauf neu erzeugen) muss die erwartete
    # Eventfolge liefern. Fängt stillen Format-Drift, den synthetische Einzeltests nicht sehen.
    from pathlib import Path

    sample = (Path(__file__).parent / "fixtures" / "run_stdout_sample.txt").read_text(encoding="utf-8")
    evs = _events(sample.splitlines())
    stages = sorted({e["num"] for e in evs if e["type"] == "stage"})
    assert stages == [1, 2, 3, 4, 5, 6, 7, 8]  # alle Stufen erkannt
    notes = [e["title"] for e in evs if e["type"] == "note_progress"]
    assert notes == ["Atomic Note", "Zettelkasten", "Progressive Summarization", "Link as Claim"]
    previews = [e for e in evs if e["type"] == "preview"]
    assert [p["routing"] for p in previews] == ["vault", "inbox", "merge"]
    assert previews[1]["score"] == 2 and previews[1]["confidence"] == "low"
    assert previews[2]["merge_target"] == "04-wissen/Atomic Notes.md"
    done = [e for e in evs if e["type"] == "done"]
    assert done and done[0]["written"] == 4 and done[0]["dry_run"] is True
    summaries = [e for e in evs if e["type"] == "run_summary"]
    assert summaries == [
        {
            "type": "run_summary",
            "duration_s": 12.4,
            "tokens": {"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0},
        }
    ]


def test_error_hint_for_known_backend_failures():
    p = RunParser()
    login = p.feed("  claude-CLI nicht eingeloggt oder Session abgelaufen — einmal `claude` starten")
    assert any(e["type"] == "error_hint" for e in login)
    p2 = RunParser()
    rate = p2.feed("  [subscription] Rate-Limit (429) erreicht — 5-Stunden-Fenster")
    assert any(e["type"] == "error_hint" for e in rate)
    p3 = RunParser()
    pop = p3.feed("  pdftotext nicht gefunden — poppler installieren → doctor")
    assert any(e["type"] == "error_hint" for e in pop)


def test_normal_line_no_error_hint():
    p = RunParser()
    evs = p.feed("      57 existierende Konzepte gefunden")
    assert not any(e["type"] == "error_hint" for e in evs)


def test_enrichment_stage_zero_marker():
    # [0/7] = optionales PDF-Enrichment (Vor-Stufe) → stage num=0.
    p = RunParser()
    evs = p.feed("[0/7] PDF-Enrichment — keine Metadaten im Dateinamen erkannt…")
    assert evs[0]["type"] == "stage" and evs[0]["num"] == 0


def test_done_written_real_run():
    # dry_run=False: „geschrieben" statt „(dry-run)".
    p = RunParser()
    evs = p.feed("=== Fertig: 3 Notes geschrieben ===")
    assert evs == [{"type": "done", "written": 3, "dry_run": False}]


# --- note_written (P3: Schreib-Lauf-Pfade aus vault_writer.py) -------------


def test_write_mode_inbox_vault_recommended_emits_note_written():
    p = RunParser()
    evs = p.feed("  [Inbox] 00-inbox/Zettelkasten.md  (vault-empfohlen)")
    assert evs == [{"type": "note_written", "path": "00-inbox/Zettelkasten.md", "routing": "vault"}]


def test_write_mode_inbox_review_emits_note_written():
    p = RunParser()
    evs = p.feed("  [Inbox] 00-inbox/schwache-note.md  (review)")
    assert evs == [{"type": "note_written", "path": "00-inbox/schwache-note.md", "routing": "inbox"}]


def test_write_mode_merge_stub_emits_note_written_with_merge_target():
    p = RunParser()
    evs = p.feed("  [Merge-Stub] 00-inbox/MERGE - Atomic Note.md  -> 04-wissen/Atomic Notes.md")
    assert evs == [
        {
            "type": "note_written",
            "path": "00-inbox/MERGE - Atomic Note.md",
            "routing": "merge",
            "merge_target": "04-wissen/Atomic Notes.md",
        }
    ]


def test_write_mode_lines_do_not_leave_pending_preview_state():
    # Kein DRY-RUN-Block in echten Schreib-Laeufen — trotzdem defensiv: eine
    # note_written-Zeile darf keinen offenen Preview-Block "auffressen".
    p = RunParser()
    evs = p.feed("  [Inbox] 00-inbox/a.md  (vault-empfohlen)")
    assert evs[0]["type"] == "note_written"
    assert p.flush() == []


def test_golden_write_mode_stdout_sample_parses_full_run():
    # Konstruiertes Fixture (kein echter Pipeline-Lauf erlaubt) aus den exakten
    # Print-Formaten in generative/pipeline/vault_writer.py — abgesichert durch
    # den Kopplungstest unten, der die Formate gegen den Quelltext prueft.
    sample = (Path(__file__).parent / "fixtures" / "run_stdout_write_sample.txt").read_text(encoding="utf-8")
    evs = _events(sample.splitlines())
    stages = sorted({e["num"] for e in evs if e["type"] == "stage"})
    assert stages == [1, 2, 3, 4, 5, 6, 7, 8]
    written = [e for e in evs if e["type"] == "note_written"]
    assert [w["routing"] for w in written] == ["vault", "inbox", "merge"]
    assert written[0]["path"] == "00-inbox/Zettelkasten.md"
    assert written[1]["path"] == "00-inbox/schwache-note.md"
    assert written[2]["merge_target"] == "04-wissen/Atomic Notes.md"
    # Schreib-Modus druckt kein Score/Confidence — keine preview-Events.
    assert not [e for e in evs if e["type"] == "preview"]
    done = [e for e in evs if e["type"] == "done"]
    assert done and done[0]["written"] == 4 and done[0]["dry_run"] is False
    summaries = [e for e in evs if e["type"] == "run_summary"]
    assert summaries == [
        {
            "type": "run_summary",
            "duration_s": 9.7,
            "tokens": {"total": 6530, "input": 5000, "output": 1530, "cache_read": 200, "cache_create": 0},
        }
    ]


# --- run_summary (P5: Zeit/Tokens/Quelle-Block nach '=== Fertig… ===') -----


def test_run_summary_full_block_emits_single_event_after_quelle_line():
    lines = [
        "   -> Zeit:   12.4s",
        "   -> Tokens: 18,432 (In:14,200 Out:4,232 Cache-R:0 Cache-C:0)",
        "   -> Quelle: zettelkasten-primer.pdf",
    ]
    evs = _events(lines)
    assert evs == [
        {
            "type": "run_summary",
            "duration_s": 12.4,
            "tokens": {"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0},
        }
    ]


def test_run_summary_zeit_line_alone_emits_nothing_yet():
    p = RunParser()
    evs = p.feed("   -> Zeit:   12.4s")
    assert evs == []


def test_run_summary_fallback_line_when_tokens_unavailable():
    p = RunParser()
    evs = p.feed("   -> Zeit:   3.2s  |  Tokens: n/a  |  Quelle: broken.pdf")
    assert evs == [{"type": "run_summary", "duration_s": 3.2}]


def test_incomplete_run_summary_block_emits_nothing_if_stream_ends():
    # Stream bricht nach der Zeit-Zeile ab (z.B. Prozess-Crash) — kein
    # erfundenes run_summary-Event ohne Tokens/Quelle-Bestaetigung (L5).
    p = RunParser()
    p.feed("   -> Zeit:   12.4s")
    assert p.flush() == []


def test_routing_report_quelle_offen_line_is_not_mistaken_for_summary_quelle():
    # routing_report.final_report_lines() druckt "-> Quelle offen: N (...)" —
    # muss vom "-> Quelle: <name>"-Abschluss der Summary unterschieden werden.
    p = RunParser()
    evs = p.feed("   -> Quelle offen: 0 (source-status: unresolved)")
    assert evs == [{"type": "log", "text": "   -> Quelle offen: 0 (source-status: unresolved)"}]


def test_run_summary_print_formats_match_orchestrator_source():
    """Kopplungstest (analog test_write_mode_print_formats_match_vault_writer_source):
    schlaegt an, wenn orchestrator.py die Zeit/Tokens/Quelle-Print-Formate
    aendert, ohne dass Parser/Fixtures hier nachgezogen werden."""
    src = (Path(__file__).parents[2] / "orchestrator.py").read_text(encoding="utf-8")
    assert re.search(r"""print\(f"   -> Zeit:   \{_wall_s_early\}s"\)""", src), (
        "orchestrator.py Zeit-Print-Format geaendert — Parser/Fixture nachziehen"
    )
    assert re.search(
        r"""f"   -> Tokens: \{_pipe\['total'\]:,\} \(In:\{_pipe\['input'\]:,\} """
        r"""Out:\{_pipe\['output'\]:,\} Cache-R:\{_pipe\['cache_read'\]:,\} Cache-C:\{_pipe\['cache_create'\]:,\}\)\"""",
        src,
    ), "orchestrator.py Tokens-Print-Format geaendert — Parser/Fixture nachziehen"
    assert re.search(r"""print\(f"   -> Quelle: \{source_path.name\}"\)""", src), (
        "orchestrator.py Quelle-Print-Format geaendert — Parser/Fixture nachziehen"
    )
    assert re.search(
        r"""print\(f"   -> Zeit:   \{_wall_s_early\}s  \|  Tokens: n/a  \|  Quelle: \{source_path.name\}"\)""", src
    ), "orchestrator.py Fallback-Zeile geaendert — Parser/Fixture nachziehen"


def test_write_mode_print_formats_match_vault_writer_source():
    """Kopplungstest (Plan P3 Schritt 4): schlaegt an, wenn vault_writer.py die
    [Inbox]/[Merge-Stub]-Print-Formate aendert, ohne dass Parser + Fixture hier
    nachgezogen werden (Format-Drift-Detektor, siehe run_parser.py-Docstring —
    stdout-Parser-Kopplung ist Bruchstelle Nr. 1, Plan §8)."""
    src = (Path(__file__).parents[2] / "pipeline" / "vault_writer.py").read_text(encoding="utf-8")
    assert re.search(
        r"""print\(f"  \[Inbox\] \{_display\(target\)\}  \(\{'vault-empfohlen' if auto else 'review'\}\)"\)""",
        src,
    ), "vault_writer.py [Inbox]-Print-Format geaendert — Parser/Fixture in test_run_parser.py nachziehen"
    assert re.search(
        r"""print\(f"  \[Merge-Stub\] \{_display\(target\)\}  -> \{_display\(existing_vault\)\}"\)""",
        src,
    ), "vault_writer.py [Merge-Stub]-Print-Format geaendert — Parser/Fixture in test_run_parser.py nachziehen"
