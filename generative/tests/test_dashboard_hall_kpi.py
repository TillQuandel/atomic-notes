"""Test für den Halluzinations-Headline-KPI (Bug 2026-06-27).

Befund: `_calc_kpis` aggregierte `hallucination_rate` per Median. Die Metrik
ist zero-inflated (>50 % der Notes haben 0 halluzinierte Anker), deshalb
kollabierte der Median systematisch auf 0,0 % → Dashboard zeigte „0 %
Halluzination". Fix: gepoolte Rate (Σ halluzinierte Anker / Σ Anker,
ankergewichtet), mit Mittelwert-Fallback wenn keine Roh-Counts vorliegen.
"""

from __future__ import annotations

from generative.eval_common import wilson_ci
from generative.eval_dashboard import (
    _calc_kpis,
    _dedupe_pdf_options,
    _pdf_filter_key,
    _pooled_hall_pct,
    _pooled_hall_stats,
    _top_versions,
)


def _qrow(ver: str, hall: float, total=None, hallucinated=None) -> dict:
    r = {"version": ver, "hallucination_rate": hall, "coverage_factual": 0.75}
    if total is not None:
        r["anchors_total"] = total
        r["anchors_hallucinated"] = hallucinated
    return r


# ── Helper: gepoolte Rate ───────────────────────────────────────────────────


def test_pooled_is_anchor_weighted_not_note_weighted():
    # Note A: 2 Anker, 1 falsch (50 %); Note B: 100 Anker, 5 falsch (5 %).
    # Mean-of-rates wäre 27,5 % — die kleine Note kippt ihn. Gepoolt: 6/102.
    rows = [_qrow("v1", 0.5, 2, 1), _qrow("v1", 0.05, 100, 5)]
    assert _pooled_hall_pct(rows) == 5.9


def test_pooled_falls_back_to_mean_without_counts():
    rows = [_qrow("v1", x) for x in (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.083)]
    # keine anchors_* → Mittelwert der Raten = 1,2 % (Median wäre 0,0 %)
    assert _pooled_hall_pct(rows) == 1.2


def test_pooled_empty_is_none():
    assert _pooled_hall_pct([]) is None


def test_pooled_mixed_rows_fall_back_to_mean():
    # Eine Row mit Counts, eine ohne → KEIN Teilmengen-Pooling (Codex+QWEN-Befund).
    # Teilmengen-Pool wäre 1/2 = 50 %; korrekt: mean([0.5, 0.0]) = 25 %.
    rows = [_qrow("v1", 0.5, 2, 1), _qrow("v1", 0.0)]
    assert _pooled_hall_pct(rows) == 25.0


# ── Wilson-CI der gepoolten Rate (#196 P4) ──────────────────────────────────


def test_pooled_stats_has_wilson_ci_with_counts():
    rows = [_qrow("v1", 0.5, 2, 1), _qrow("v1", 0.05, 100, 5)]
    st = _pooled_hall_stats(rows)
    assert st["pct"] == 5.9
    assert st["anchors_total"] == 102
    assert st["rows_n"] == 2
    lo, hi = wilson_ci(6, 102)
    assert st["ci_low"] == round(lo * 100, 1)
    assert st["ci_high"] == round(hi * 100, 1)
    assert st["ci_low"] < st["ci_high"]


def test_pooled_stats_no_ci_on_mean_fallback():
    # gemischte Rows → Mean-Fallback → kein CI (Roh-Counts fehlen)
    rows = [_qrow("v1", 0.5, 2, 1), _qrow("v1", 0.0)]
    st = _pooled_hall_stats(rows)
    assert st["pct"] == 25.0
    assert st["ci_low"] is None
    assert st["ci_high"] is None
    assert st["anchors_total"] is None


def test_pooled_stats_empty_is_none():
    assert _pooled_hall_stats([]) is None


def test_calc_kpis_exposes_ci_fields_with_counts():
    rows = [_qrow("v0.3.141", 0.0, 17, 0), _qrow("v0.3.141", 0.083, 12, 1)]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.141")
    assert kpis["hall_anchors_total"] == 29
    assert kpis["hall_ci_low"] is not None
    assert kpis["hall_ci_high"] is not None
    assert kpis["hall_ci_low"] < kpis["hall_ci_high"]
    assert kpis["hall_notes_n"] == 2


def test_calc_kpis_ci_null_on_mean_fallback():
    rows = [_qrow("v0.3.141", x) for x in (0.0, 0.0, 0.083)]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.141")
    assert kpis["avg_hall"] == 2.8  # Mean-Fallback (kein CI ableitbar)
    assert kpis["hall_ci_low"] is None
    assert kpis["hall_ci_high"] is None
    assert kpis["hall_anchors_total"] is None


# ── KPI-Integration ─────────────────────────────────────────────────────────


def test_avg_hall_pooled_on_latest_version():
    # current_version explizit: seit #191 ist die KPI-Version config-verankert,
    # nicht mehr höchste Nummer — der Test prüft weiterhin nur das Pooling.
    rows = [
        _qrow("v0.3.140", 0.5, 2, 1),  # andere Version, darf nicht einfließen
        _qrow("v0.3.141", 0.0, 17, 0),
        _qrow("v0.3.141", 0.083, 12, 1),
    ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.141")
    # nur v0.3.141: gepoolt = 1/29 = 3.4 %
    assert kpis["avg_hall"] == 3.4


def test_avg_hall_not_zero_on_zero_inflated_fallback():
    # Regressions-Wächter für den Original-Bug: Median wäre 0, darf es nicht sein.
    rows = [_qrow("v0.3.141", x) for x in (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.083)]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.141")
    assert kpis["avg_hall"] == 1.2


def test_avg_hall_ignores_sentinel_negative_in_fallback():
    rows = [
        _qrow("v0.3.141", -1.0),  # Sentinel "keine Daten"
        _qrow("v0.3.141", 0.10),
        _qrow("v0.3.141", 0.0),
    ]
    kpis = _calc_kpis({}, [], rows, [], current_version="v0.3.141")
    assert kpis["avg_hall"] == 5.0


# ── PDF-Dropdown-Normalisierung (#4) ────────────────────────────────────────


def test_pdf_filter_key_author_year():
    assert _pdf_filter_key("Bates - 2017 - Information Behavior.pdf") == "Bates - 2017"


def test_pdf_filter_key_no_year_keeps_name():
    assert _pdf_filter_key("Knowles - From Pedagogy to Andragogy.pdf") == "Knowles - From Pedagogy to Andragogy"


def test_dedupe_keeps_full_titles_and_distinct_years():
    # Volltitel bleibt erhalten; gleiche Autorin / verschiedene Jahre getrennt.
    opts = _dedupe_pdf_options(
        [
            "Beutelspacher - 2014 - Erfassung von Informationskompetenz.pdf",
            "Beutelspacher - 2022 - Information Literacy as a Fundamental Skill.pdf",
        ]
    )
    assert opts == [
        "Beutelspacher - 2014 - Erfassung von Informationskompetenz",
        "Beutelspacher - 2022 - Information Literacy as a Fundamental Skill",
    ]


def test_dedupe_drops_bare_author_keeps_full_title():
    # Verirrtes "Bates" neben "Bates - 2017 - ..." → nur der Volltitel-Eintrag.
    opts = _dedupe_pdf_options(["Bates", "Bates - 2017 - Information Behavior.pdf"])
    assert opts == ["Bates - 2017 - Information Behavior"]


def test_dedupe_collapses_title_variants_to_fullest():
    # Mehrere Längen derselben Quelle → der vollständigste Titel gewinnt.
    opts = _dedupe_pdf_options(
        [
            "Bates - 2017.pdf",
            "Bates - 2017 - Information Behavior.pdf",
        ]
    )
    assert opts == ["Bates - 2017 - Information Behavior"]


# ── Versions-Filter Recency-Cap (#1) ────────────────────────────────────────


def test_top_versions_keeps_newest_15_with_min_n():
    # höchste Version (v0.3.40) ist gerade → n=5, also greift die n≥3-Regel sauber
    counts = {f"v0.3.{i}": (5 if i % 2 == 0 else 1) for i in range(2, 41)}
    opts = _top_versions(counts, limit=15, min_n=3)
    assert len(opts) == 15
    assert all(counts[v] >= 3 for v in opts)  # nur robuste
    assert opts == sorted(opts, key=lambda v: [int(x) for x in __import__("re").findall(r"\d+", v)], reverse=True)


def test_top_versions_always_includes_newest_even_if_thin():
    # Neueste Version dünn (n=1) → trotzdem an erster Stelle.
    counts = {"v0.3.141": 1, "v0.3.140": 42, "v0.3.131": 8, "v0.3.130": 10}
    opts = _top_versions(counts, limit=15, min_n=3)
    assert opts[0] == "v0.3.141"
    assert "v0.3.140" in opts
