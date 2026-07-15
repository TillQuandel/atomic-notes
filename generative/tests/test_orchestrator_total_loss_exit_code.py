"""Issue #281: ein Lauf mit Totalverlust (0 Notes trotz >=1 versuchtem Konzept)
darf nicht wie ein erfolgreicher Lauf mit Exit-Code 0 enden.

Befund (Testlauf-Serie 2026-07-14, Lauf 1): 2x claude-CLI-Timeout verlor das
einzige geplante Konzept -> 0 Notes, Pipeline endete trotzdem mit "Fertig."
und Exit-Code 0 -- das einzige Fehlersignal stand nur in stderr. Live-Gap im
aktuellen Code: `extractor_failure_exit_code()` wertet ausschliesslich
`ctx.extractor_failures` (Exceptions) -- der stille `<!--END-->`-Drop (#280,
`run_per_concept` gibt `None` OHNE Exception zurueck) zaehlt in
`dropped_total` mit, faellt aber durch dieses Sieb und liefert weiterhin
Exit-Code 0, obwohl >=1 Konzept versucht und 0 Notes erzeugt wurden --
ununterscheidbar von einem legitimen 0-Konzepte-Lauf (Konzeptmangel).

Fix: `total_loss_exit_code()` erweitert den Exit-Code um einen eigenen,
unterscheidbaren Wert (_EXIT_TOTAL_LOSS) fuer genau diesen Fall + eine
prominente stdout-Warnung (nicht nur stderr).
"""

from __future__ import annotations

from pathlib import Path

from generative import orchestrator
from generative.schemas.atomic_note import QualityReport
from generative.schemas.citation import CitationMeta
from generative.schemas.run_context import RunContext


# --- Pure Helper: total_loss_exit_code ---------------------------------------


def test_total_loss_exit_code_legit_zero_concept_run_stays_zero():
    """0 versucht, 0 final -> legitimer Konzeptmangel-Lauf, Exit 0."""
    assert orchestrator.total_loss_exit_code(0, n_attempted=0, n_final=0) == 0


def test_total_loss_exit_code_partial_success_stays_at_base():
    """>=1 versucht, aber mind. 1 Note ueberlebt -> kein Totalverlust."""
    assert orchestrator.total_loss_exit_code(0, n_attempted=2, n_final=1) == 0
    assert orchestrator.total_loss_exit_code(3, n_attempted=2, n_final=1) == 3


def test_total_loss_exit_code_silent_drop_without_exception_is_flagged():
    """Genau der #281-Luecken-Fall: base_exit_code=0 (keine Exception in
    extractor_failures), aber 0 finale Notes trotz >=1 Versuch -> muss auf
    einen unterscheidbaren Code hoch, NICHT bei 0 bleiben."""
    assert orchestrator.total_loss_exit_code(0, n_attempted=1, n_final=0) == orchestrator._EXIT_TOTAL_LOSS
    assert orchestrator._EXIT_TOTAL_LOSS != 0


def test_total_loss_exit_code_total_loss_takes_precedence_over_partial_code():
    """Auch wenn bereits Exceptions vorlagen (base=3): Totalverlust ist
    strenger als Teilverlust -> _EXIT_TOTAL_LOSS gewinnt."""
    assert orchestrator.total_loss_exit_code(3, n_attempted=2, n_final=0) == orchestrator._EXIT_TOTAL_LOSS


# --- Integration: main() ueber den --load-drafts-Pfad ------------------------


def _fake_ctx(dropped_total: int) -> RunContext:
    return RunContext(
        drafts=[],
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
        word_count=2,
        related_mentions=[],
        q_title=None,
        citation=CitationMeta(author=None, year=None, title=None, doi=None, source_file="fake.pdf"),
        extractor_failures=[],  # KEIN Exception -- der stille #280-Fall
    )


def test_main_returns_total_loss_code_with_stdout_warning(monkeypatch, capsys):
    """>=1 Konzept versucht (dropped_total=1), 0 Notes final -> main() muss
    einen unterscheidbaren Exit-Code liefern UND eine prominente stdout-Warnung
    ausgeben (nicht nur stderr, s. Fix-Richtung Issue #281)."""
    monkeypatch.setattr(orchestrator, "_load_draft_state", lambda _path: _fake_ctx(dropped_total=1))
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")

    rc = orchestrator.main(["--load-drafts", "irrelevant.json"])

    out = capsys.readouterr().out
    assert rc == orchestrator._EXIT_TOTAL_LOSS
    assert rc != 0
    assert "TOTALVERLUST" in out


def test_main_legit_zero_concept_run_still_exits_zero(monkeypatch, capsys):
    """Gegenprobe: 0 versucht (dropped_total=0), 0 final -> normaler
    Konzeptmangel-Lauf, Exit 0 bleibt unveraendert (kein False-Positive)."""
    monkeypatch.setattr(orchestrator, "_load_draft_state", lambda _path: _fake_ctx(dropped_total=0))
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")

    rc = orchestrator.main(["--load-drafts", "irrelevant.json"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "TOTALVERLUST" not in out
