"""Tests fuer den stdout-Parser der Live-GUI.

Der Parser uebersetzt die rohen stdout-Zeilen eines orchestrator-Laufs in
strukturierte Events (Stage-Stepper, Pro-Note-Fortschritt, Dry-Run-Preview,
Abschluss). Pure Logik, keine I/O.
"""
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
    assert evs == [{"type": "stage", "num": 1, "total": 7,
                    "label": "PDF extrahieren und chunken…"}]


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
    assert evs == [{"type": "note_progress", "index": 3, "total": 10,
                    "title": "Zettelkasten als Denkwerkzeug"}]


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
