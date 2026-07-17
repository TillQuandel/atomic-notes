"""Issue #330: 0-Notes-Laeufe (Konzeptmangel/Totalverlust) hinterliessen bisher
KEINE pipeline_runs-Zeile -- Run-Zaehlung, Token-/Kosten-Tracking und die Frage
"wie oft produziert die Pipeline nichts?" waren dadurch systematisch
untererfasst (2 Belege aus der Coverage-Serie 2: Lauf 1 dbv-Framework 198s/
15.325 Tokens, Lauf 4 Witt 268s/51.272 Tokens, beide Exit 0 ohne DB-Zeile).

Deckt beide Fruehausstiege in orchestrator.main() ab, die VOR dem
Erfolgspfad-Insert (#198 P1) zurueckkehren:

  1. `if not drafts:` direkt nach der Extraktion -- legitimer Konzeptmangel
     (0 Konzepte / alle secondary_mention / alle action=skip) ODER stiller
     Totalverlust (#281, >=1 Konzept versucht, 0 ueberlebt).
  2. `if not drafts:` nach `_drop_artifacts()` -- alle verbliebenen Drafts als
     Abwesenheits-Artefakte verworfen.

Harness: `--load-drafts` + monkeypatch(_load_draft_state) -- identisch zu
test_orchestrator_total_loss_exit_code.py -- kombiniert mit der autouse-
Fixture `isolate_pipeline_side_effects` (DB in tmp). Die pipeline_runs-Zeile
wird direkt per sqlite3 gegen `isolate_pipeline_side_effects.db_path` geprueft.

RED auf master-Stand: kein insert_run()-Aufruf in beiden Fruehausstiegen ->
0 pipeline_runs-Zeilen nach jedem dieser Laeufe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from generative import orchestrator
from generative.config import AGENT_VERSION
from generative.schemas.atomic_note import AtomicNoteDraft, QualityReport
from generative.schemas.citation import CitationMeta
from generative.schemas.run_context import RunContext


def _fake_ctx(
    *,
    dropped_total: int = 0,
    related_mentions: list[str] | None = None,
    drafts: list | None = None,
) -> RunContext:
    return RunContext(
        drafts=drafts or [],
        concept_map={},
        existing_concepts={},
        concept_links={},
        text="Etwas Text.",
        chunks=[],
        acronym_dict={},
        quality_report=QualityReport(peer_reviewed=None, citation_count=None, retracted=False, flags=[]),
        pdf_meta={},
        source_path=Path("fake.pdf"),
        tag_whitelist=[],
        background_map={},
        fb_year=None,
        dropped_total=dropped_total,
        word_count=42,
        related_mentions=related_mentions or [],
        q_title=None,
        citation=CitationMeta(author=None, year=None, title=None, doi=None, source_file="fake.pdf"),
        extractor_failures=[],  # KEIN Exception -- stille Drops (#280) wie in #281-Tests
    )


def _artifact_draft(title: str) -> AtomicNoteDraft:
    """Draft, den `_drop_artifacts()` als Abwesenheits-Artefakt verwirft (#1223ff.)."""
    return AtomicNoteDraft(
        title=title,
        body="Dieses Konzept wird im Quelltext nicht behandelt.",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="low",
    )


def _run_main_load_drafts(monkeypatch, ctx: RunContext) -> int:
    monkeypatch.setattr(orchestrator, "_load_draft_state", lambda _path: ctx)
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    return orchestrator.main(["--load-drafts", "irrelevant.json"])


def _pipeline_run_rows(db_path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT run_id, pipeline_version, n_generated, n_extracted, n_dropped, "
            "duration_s, tokens_total, abort_reason FROM pipeline_runs"
        ).fetchall()
    finally:
        conn.close()


def test_no_concepts_run_persists_with_abort_reason(monkeypatch, isolate_pipeline_side_effects):
    """0 versucht, 0 final, keine Sekundaer-Erwaehnungen -> abort_reason='no_concepts'."""
    rc = _run_main_load_drafts(monkeypatch, _fake_ctx(dropped_total=0, related_mentions=[]))

    assert rc == 0
    rows = _pipeline_run_rows(isolate_pipeline_side_effects.db_path)
    assert len(rows) == 1, f"Erwartet genau 1 pipeline_runs-Zeile, gefunden: {rows}"
    run_id, pipeline_version, n_generated, n_extracted, n_dropped, duration_s, tokens_total, abort_reason = rows[0]
    assert pipeline_version == AGENT_VERSION
    assert n_generated == 0
    assert n_extracted == 0
    assert n_dropped == 0
    assert duration_s is not None and duration_s >= 0
    assert abort_reason == "no_concepts"


def test_all_secondary_mention_run_persists_with_distinct_abort_reason(monkeypatch, isolate_pipeline_side_effects):
    """0 versucht, aber Sekundaer-Erwaehnungen vorhanden -> eigener, unterscheidbarer
    Abbruchgrund (Issue-Beispiel: 'all_secondary_mentions')."""
    rc = _run_main_load_drafts(monkeypatch, _fake_ctx(dropped_total=0, related_mentions=["Nebenkonzept"]))

    assert rc == 0
    rows = _pipeline_run_rows(isolate_pipeline_side_effects.db_path)
    assert len(rows) == 1
    assert rows[0][-1] == "all_secondary_mentions"


def test_extraction_total_loss_run_persists_with_abort_reason(monkeypatch, isolate_pipeline_side_effects):
    """#281-Fall (stiller Drop, dropped_total=1, Exit=_EXIT_TOTAL_LOSS) bekommt
    jetzt ZUSAETZLICH eine DB-Zeile -- der bestehende Exit-Code-Vertrag aus
    test_orchestrator_total_loss_exit_code.py bleibt dabei unveraendert."""
    rc = _run_main_load_drafts(monkeypatch, _fake_ctx(dropped_total=1, related_mentions=[]))

    assert rc == orchestrator._EXIT_TOTAL_LOSS
    rows = _pipeline_run_rows(isolate_pipeline_side_effects.db_path)
    assert len(rows) == 1
    assert rows[0][4] == 1  # n_dropped
    assert rows[0][-1] == "extraction_total_loss"


def test_all_artifacts_dropped_run_persists_with_abort_reason(monkeypatch, isolate_pipeline_side_effects):
    """Konzept wurde extrahiert (n_extracted=1), landete aber komplett als
    Abwesenheits-Artefakt -> zweiter Fruehausstieg, eigener Abbruchgrund."""
    ctx = _fake_ctx(drafts=[_artifact_draft("Nur-Artefakt-Konzept")])
    rc = _run_main_load_drafts(monkeypatch, ctx)

    assert rc == orchestrator._EXIT_TOTAL_LOSS
    rows = _pipeline_run_rows(isolate_pipeline_side_effects.db_path)
    assert len(rows) == 1
    assert rows[0][3] == 1  # n_extracted zaehlt den verworfenen Artefakt-Draft mit
    assert rows[0][2] == 0  # n_generated bleibt 0 -- keine Note geschrieben
    assert rows[0][-1] == "all_artifacts"
