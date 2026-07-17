"""#323: Scatter-Legende (ch2) zeigt Roh-PDF-Keys statt kanonisierter Labels.

Befund (UX-Review 16.07.): `_chart_scatter_versioned` baute die PDF-Legende
aus dem ROHEN `pdf`-String als Dict-Key -- ohne vorherige Kanonisierung ueber
`D._pdf_group_key()`. Zwei Rohvarianten derselben Quelle (z. B. die belegte
Bates-Drift-Aliase, `_PDF_GROUP_ALIASES`) landeten dadurch als zwei separate
Legenden-Eintraege in zwei Farben. Fix: Dict-Key ueber `D._pdf_group_key(pdf)`
statt Rohstring -- derselbe Kanonisierungs-Pfad wie `_calc_pdf_table`/#311.
"""

from __future__ import annotations

from generative.eval_dashboard_server import _chart_scatter_versioned


def _row(note, pdf, hall, cov, ver="v0.3.144", run_id="r1"):
    return {
        "run_id": run_id,
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "coverage_factual": cov,
        "pdf": pdf,
    }


def test_scatter_collapses_raw_pdf_variants_of_same_source_into_one_legend_entry():
    """ "Bates - 2017 - Information Behavior.pdf" und "bates-2017" bezeichnen
    dieselbe Quelle (_pdf_group_key kollabiert beide auf "bates-2017") -- die
    Scatter-Legende darf dafuer nur EINEN Eintrag zeigen, nicht zwei."""
    rows = [
        _row("n1", "Bates - 2017 - Information Behavior.pdf", 0.1, 0.5, run_id="r1"),
        _row("n2", "bates-2017", 0.2, 0.6, run_id="r2"),
    ]
    data = _chart_scatter_versioned(rows)
    assert len(data["pdfs"]) == 1
    assert data["pdfs"][0]["raw"] == "bates-2017"


def test_scatter_points_pdf_field_matches_canonical_legend_key():
    """Frontend gruppiert Punkte in Datasets ueber `pt.pdf === legendEntry.raw`
    (renderScatter in eval_dashboard.html) -- beide Felder muessen nach der
    Kanonisierung uebereinstimmen, sonst faellt jeder Punkt aus seiner
    Legenden-Gruppe."""
    rows = [
        _row("n1", "Bates - 2017 - Information Behavior.pdf", 0.1, 0.5, run_id="r1"),
        _row("n2", "bates-2017", 0.2, 0.6, run_id="r2"),
    ]
    data = _chart_scatter_versioned(rows)
    raw_keys = {p["raw"] for p in data["pdfs"]}
    for pt in data["points"]:
        assert pt["pdf"] in raw_keys
        assert pt["pdf"] == "bates-2017"
        assert pt["pdf_label"] == data["pdfs"][0]["label"]


def test_scatter_keeps_distinct_sources_separate():
    """Regressions-Waechter: unterschiedliche Quellen (verschiedene Autor-Jahr-
    Keys) duerfen NICHT zusammengefasst werden -- nur belegte Drift-Varianten
    derselben Quelle."""
    rows = [
        _row("n1", "Bates - 2017 - Information Behavior.pdf", 0.1, 0.5, run_id="r1"),
        _row("n2", "Beutelspacher - 2022 - Kryptographie.pdf", 0.2, 0.6, run_id="r2"),
    ]
    data = _chart_scatter_versioned(rows)
    assert len(data["pdfs"]) == 2
    assert {p["raw"] for p in data["pdfs"]} == {"bates-2017", "beutelspacher-2022"}
