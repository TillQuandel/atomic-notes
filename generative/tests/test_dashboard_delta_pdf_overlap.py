"""Tests für den Corpus-Overlap-Guard bei Versions-Deltas (Statistik-Review 2026-07-15).

Befund (3 unabhängige Opus-Statistiker, konvergent + adversarial bestätigt):
`version_delta()` markierte ein Delta als "reliable" allein ab n>=20 in beiden
Versionen (`_DELTA_MIN_N`) — unabhängig davon, ob die beiden Versionen
überhaupt denselben Corpus (PDF-Quellen) evaluieren. Produktionsbeleg
v0.3.140 -> v0.3.143: beide n>=20 (40/22 distinct Notes), aber nur 3 von 9/5
PDFs geteilt — von den 22 Notes der neueren Version stammen nur 8 (36 %) aus
einer PDF, die auch in v0.3.140 vorkommt. Das +2,7pp-Hall-Delta ist damit
größtenteils ein PDF-Mix-Artefakt, kein echter Versions-Effekt, wurde aber
grün/rot als belastbar angezeigt.

Fix: `reliable` zusätzlich an den Notes-Anteil der neueren Version gekoppelt,
dessen PDF auch in der Vergleichsversion vorkommt (>= `_DELTA_MIN_PDF_OVERLAP`
= 50 %). Unter der Schwelle bleibt das Delta sichtbar, aber `reliable=False`
mit `reason="pdf_mix"` statt `"n_lt_20"` — der Client kann die beiden Fälle
im Tooltip unterscheiden ("nicht vergleichbar (PDF-Mix)" statt "n<20").
"""

from __future__ import annotations

import pytest

from generative.eval_dashboard import version_delta

# Produktionsbeleg v0.3.140 -> v0.3.143 (dedupliziert, s. test_dashboard_reeval_dedup.py):
# pdf_notes = {pdf_group_key: distinct-Notes-Zahl} je Version.
_PROD_PDF_NOTES = [
    {
        "assfalg-2013": 2,
        "ebner-und-gegenfurtner-2019": 7,
        "hrastinski-2008": 6,
        "knowles-from-pedagogy-to-andragogy": 9,
        "mahmood-und-university-of-the-punjab-2016": 4,
        "merrill-first-principles-of-instruction": 5,
        "reimer-2013": 1,
        "schlebbe-und-greifeneder-2022": 1,
        "zettelkasten-primer": 5,
    },
    {
        "bates-information-behavior": 4,
        "hrastinski-2008": 2,
        "s-hl-strohmenger-2008": 10,
        "schlebbe-und-greifeneder-2022": 2,
        "zettelkasten-primer": 4,
    },
]


def _kpi_trend(**over):
    base = {
        "versions": ["v0.3.140", "v0.3.143"],
        "hall": [9.46, 12.01],
        "n": [40, 22],
        "pdf_notes": [dict(d) for d in _PROD_PDF_NOTES],
    }
    base.update(over)
    return base


def test_production_delta_140_143_not_reliable_due_to_pdf_mix():
    # n>=20 in beiden Versionen (40/22), aber nur 8/22 = 36 % Notes-Overlap
    # -> unter der 50%-Schwelle, nicht belastbar trotz ausreichendem n.
    d = version_delta(_kpi_trend(), "hall")
    assert d["reliable"] is False
    assert d["reason"] == "pdf_mix"
    assert d["pdf_overlap"] == pytest.approx(8 / 22, abs=0.001)


def test_full_overlap_stays_reliable():
    trend = _kpi_trend(pdf_notes=[{"a": 20}, {"a": 20}])  # 100 % Overlap
    d = version_delta(trend, "hall")
    assert d["reliable"] is True
    assert d["reason"] is None
    assert d["pdf_overlap"] == 1.0


def test_overlap_exactly_at_threshold_is_reliable():
    trend = _kpi_trend(
        pdf_notes=[
            {"a": 10, "b": 10},  # prev
            {"a": 10, "c": 10},  # latest: a geteilt (10), c nicht (10) -> 50 %
        ]
    )
    d = version_delta(trend, "hall")
    assert d["pdf_overlap"] == 0.5
    assert d["reliable"] is True  # genau an der Schwelle gilt als belastbar


def test_disjoint_corpus_zero_overlap():
    trend = _kpi_trend(pdf_notes=[{"a": 30}, {"b": 30}])
    d = version_delta(trend, "hall")
    assert d["pdf_overlap"] == 0.0
    assert d["reliable"] is False
    assert d["reason"] == "pdf_mix"


def test_n_guard_reason_takes_priority_over_pdf_mix():
    # latest zu klein (n<20) UND PDF-Mix -> reason meldet den n-Guard, nicht pdf_mix
    trend = _kpi_trend(n=[40, 5], pdf_notes=[{"a": 30}, {"b": 5}])
    d = version_delta(trend, "hall")
    assert d["reliable"] is False
    assert d["reason"] == "n_lt_20"


def test_missing_pdf_notes_key_skips_overlap_guard_backward_compat():
    # Ohne "pdf_notes" (ältere Aufrufer/Tests) bleibt das reine n>=20-Verhalten
    # unverändert -- der Guard greift nur, wenn der Server die Daten liefert.
    trend = {
        "versions": ["v1", "v2"],
        "hall": [12.0, 9.7],
        "n": [25, 22],
    }
    d = version_delta(trend, "hall")
    assert d["reliable"] is True
    assert d["pdf_overlap"] is None
    assert d["reason"] is None


def test_overlap_guard_applies_to_fallback_prev_too():
    # Kein früherer n>=20-Vergleichspunkt -> Fallback auf direkte Vorversion
    # (bisheriges Verhalten, reliable bleibt False) -- pdf_overlap wird
    # trotzdem gegen den Fallback-prev berechnet, nicht gecrasht.
    trend = {
        "versions": ["v1", "v2"],
        "hall": [5.0, 9.7],
        "n": [3, 25],
        "pdf_notes": [{"a": 3}, {"a": 25}],
    }
    d = version_delta(trend, "hall")
    assert d["reliable"] is False
    assert d["reason"] == "n_lt_20"


# ── Server-Integration: build_data() verdrahtet pdf_notes bis in kpi_trend ──


def _eval(note, ver, pdf, hall, ts, total=10, hallucinated=0):
    return {
        "run_id": f"r-{note}",
        "note_path": note,
        "pipeline_version": ver,
        "version": ver,
        "hallucination_rate": hall,
        "anchors_total": total,
        "anchors_hallucinated": hallucinated,
        "coverage_factual": 0.5,
        "pdf": pdf,
        "eval_version": "4.1",
        "timestamp": ts,
    }


def test_build_data_flags_pdf_mix_delta_via_kpi_trend(monkeypatch):
    """End-to-End: build_data() -> quality_by_version -> kpi_trend["pdf_notes"]
    -> version_delta() erkennt einen weitgehend ausgetauschten Corpus, obwohl
    n>=20 in beiden Versionen (Produktionsmuster v0.3.140 -> v0.3.143)."""
    from generative import config as _cfg
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    evals = []
    # v0.3.140: 25 Notes, davon 20 auf "Shared.pdf", 5 auf "Only140.pdf".
    for i in range(20):
        evals.append(_eval(f"shared-{i}.md", "v0.3.140", "Shared.pdf", 0.0, f"2026-06-01T00:00:{i:02d}"))
    for i in range(5):
        evals.append(_eval(f"only140-{i}.md", "v0.3.140", "Only140.pdf", 0.0, f"2026-06-02T00:00:{i:02d}"))
    # v0.3.143: 22 Notes, nur 3 teilen "Shared.pdf" mit v0.3.140 -> 3/22 = 13.6 % Overlap.
    for i in range(3):
        evals.append(_eval(f"shared2-{i}.md", "v0.3.143", "Shared.pdf", 0.5, f"2026-06-10T00:00:{i:02d}", 10, 5))
    for i in range(19):
        evals.append(_eval(f"only143-{i}.md", "v0.3.143", "Only143.pdf", 0.5, f"2026-06-11T00:00:{i:02d}", 10, 5))

    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.143")
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: evals)
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data()
    kt = data["kpi_trend"]
    assert kt["versions"] == ["v0.3.140", "v0.3.143"]
    assert kt["n"] == [25, 22]  # n>=20 in beiden -> der alte Guard allein hätte reliable=True ergeben
    vd = kt["deltas"]["hall"]
    assert vd["reliable"] is False
    assert vd["reason"] == "pdf_mix"
    assert vd["pdf_overlap"] == pytest.approx(3 / 22, abs=0.001)
