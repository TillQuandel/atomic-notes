"""Tests für #202: PDF-Filter bricht Panels bei pdf_key/pdf_label-Drift.

Dieselbe Quelle liegt je nach Pipeline-Version in vier Namensräumen vor:
Volltitel ("Bates - 2017 - Information Behavior.pdf", note_evals.pdf),
Kurzlabel ("Bates", pipeline_runs.pdf_label v0.3.46–74), Kebab-Key
("bates-2017", v0.3.79+) und Triple-Dash-Key
("bates---2017---information-behavior", Log-Dateien/v0.1.0).

Der Dropdown-Wert ist der Volltitel; rohes ``startswith`` matcht damit nur
Namensräume, die mit dem Volltitel beginnen — Token-/Laufzeit-/Agenten-Panels
und die Vergleichstabelle wurden leer oder matchten die falsche Version
(Bates → v0.1.0-Kosten). Fix: Slug-Kanonisierung + grenzbewusstes
Präfix-Matching in beide Richtungen (``_pdf_matches``).
"""

from __future__ import annotations

from generative.eval_dashboard import _dedupe_pdf_options, _pdf_group_key, _pdf_matches, _pdf_slug

# ── _pdf_slug: Kanonisierung über Namensräume ───────────────────────────────


def test_slug_strips_pdf_suffix_and_lowercases():
    assert _pdf_slug("Porst-2014-Auszug-S1-40.pdf") == "porst-2014-auszug-s1-40"


def test_slug_collapses_nonalnum_runs():
    # Volltitel und Triple-Dash-Log-Key derselben Quelle → identischer Slug.
    assert _pdf_slug("Bates - 2017 - Information Behavior.pdf") == _pdf_slug("bates---2017---information-behavior")


def test_slug_empty_and_none():
    assert _pdf_slug(None) == ""
    assert _pdf_slug("") == ""
    assert _pdf_slug(".pdf") == ""


# ── _pdf_matches: Filter-Wert (Dropdown) gegen Kandidaten-Felder ────────────


def test_matches_full_title_filter_short_label():
    # Hrastinski-Fall: pdf_label ist nur der Autor, Filter der Volltitel.
    assert _pdf_matches("Hrastinski - 2008 - Asynchronous and Synchronous E-Learning", "Hrastinski")


def test_matches_full_title_filter_kebab_key():
    # Bates-Fall: neuere Runs schreiben Kebab-Key als Label.
    assert _pdf_matches("Bates - 2017 - Information Behavior", "bates-2017")


def test_matches_full_title_filter_triple_dash_key():
    assert _pdf_matches("Bates - 2017 - Information Behavior", "bates---2017---information-behavior")


def test_matches_identity_porst():
    assert _pdf_matches("Porst-2014-Auszug-S1-40", "Porst-2014-Auszug-S1-40.pdf")


def test_matches_kebab_filter_full_title_candidate():
    # Richtung umgekehrt: Kebab-Dropdown-Eintrag, Kandidat Volltitel.
    assert _pdf_matches("bates-2017", "Bates - 2017 - Information Behavior.pdf")


def test_no_match_different_source_same_author():
    assert not _pdf_matches(
        "Beutelspacher - 2014 - Erfassung von Informationskompetenz",
        "Beutelspacher - 2022 - Information Literacy as a Fundamental Skill",
    )


def test_no_match_word_boundary():
    # Präfix nur an Segment-Grenzen: "bates" darf "batesworth-2020" nicht matchen.
    assert not _pdf_matches("batesworth - 2020 - Something", "Bates")
    assert not _pdf_matches("Bates - 2017 - Information Behavior", "batesworth-2020")


def test_no_match_empty_candidates():
    assert not _pdf_matches("Bates - 2017 - Information Behavior", None)
    assert not _pdf_matches("Bates - 2017 - Information Behavior", "")


def test_matches_any_of_multiple_candidates():
    # all_log_runs prüft label UND key.
    assert _pdf_matches("Bates - 2017 - Information Behavior", "", "bates-2017")


# ── Dropdown-Dedupe: Kebab-Variante fällt mit Volltitel zusammen ────────────


def test_dedupe_collapses_kebab_variant_into_full_title():
    opts = _dedupe_pdf_options(
        [
            "bates-2017.pdf",
            "Bates - 2017 - Information Behavior.pdf",
            "Bates",
        ]
    )
    assert opts == ["Bates - 2017 - Information Behavior"]


def test_dedupe_collapses_all_three_bates_drift_variants():
    # Realer Bestand (U6-Befund UX-Review 2026-07-15, PR pdf_key-Kanonisierung):
    # bare "Bates" (eval_version 1.3, Testlauf test-20260519-120000), Kebab-Key
    # und Volltitel waren schon vor diesem Fix eine Gruppe -- NEU dazu kommt die
    # "2017"-Missing-Year-Drift-Variante (note_evals.pdf ab v0.3.143/eval 4.3).
    # Vorher bis zu 3 "Bates"-Eintraege nebeneinander im Dropdown, jetzt einer.
    opts = _dedupe_pdf_options(
        [
            "Bates",
            "bates-2017.pdf",
            "Bates - 2017 - Information Behavior.pdf",
            "Bates - Information Behavior.pdf",
        ]
    )
    assert opts == ["Bates - 2017 - Information Behavior"]


def test_dedupe_keeps_distinct_years_separate():
    # Regressions-Wächter: Kanonisierung darf verschiedene Quellen nicht mischen.
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


# ── _pdf_group_key: belegte Drift-Aliase (PR pdf_key-Kanonisierung) ─────────
# Autor-Jahr-Slug (_pdf_filter_key) kollabiert nur Strukturvarianten mit
# gleicher Segmentzahl -- eine Quelle, deren Rohstring das Jahr verliert
# ("Bates - Information Behavior.pdf" statt "... - 2017 - ..."), bekommt sonst
# einen eigenen Slug. Jeder Alias unten ist mit einem DB-Beleg dokumentiert
# (run_id-Join bzw. exakte Note-Titel-Überschneidung, s. PR-Body).


def test_group_key_collapses_missing_year_drift_variant():
    # "Bates - Information Behavior.pdf" (note_evals.pdf ab v0.3.143, bestätigt
    # in eval_version 4.1 UND 4.3) hat "2017" verloren -> eigener Slug
    # "bates-information-behavior" statt "bates-2017". B3-Befund (Opus-
    # Statistiker-Abnahme PR #305): dadurch verlor _version_pair_compare Bates
    # aus common_pdfs für jedes v0.3.143-vs-alt-Paar.
    assert _pdf_group_key("Bates - Information Behavior.pdf") == _pdf_group_key(
        "Bates - 2017 - Information Behavior.pdf"
    )
    assert _pdf_group_key("Bates - Information Behavior.pdf") == "bates-2017"


def test_group_key_collapses_bare_author_drift_variant():
    # Bare "Bates" (note_evals.pdf, eval_version 1.3, Testlauf
    # test-20260519-120000) -- derselbe run_id trägt in pipeline_runs
    # pdf_source="Bates - 2017 - Information Behavior.pdf" (DB-Join-Beleg,
    # kein Konjekt).
    assert _pdf_group_key("Bates") == "bates-2017"


def test_group_key_keeps_unrelated_similar_author_separate():
    # Regressions-Wächter: der Alias darf keine andere Quelle einziehen --
    # "Batesworth" ist kein Präfix-/Alias-Treffer für "Bates".
    assert _pdf_group_key("Batesworth - 2020 - Something.pdf") != "bates-2017"


def test_group_key_collapses_rl_information_seeking_quantum_alias():
    # "rl-information-seeking-quantum.pdf" (v0.3.34) ist ein früher interner
    # Dateiname derselben Quelle wie "Jaiswal - 2020 - Reinforcement
    # Learning-driven Information Seeking A Quantum Probabilistic....pdf"
    # (v0.3.39): identischer Note-Titel "Information Seeking als
    # Reinforcement-Learning-Task.md" in beiden Runs (DB-Beleg, s. PR-Body).
    assert _pdf_group_key("rl-information-seeking-quantum.pdf") == "jaiswal-2020"


def test_group_key_collapses_cobaltite_paper_alias():
    # "cobaltite-paper.pdf" (v0.3.51/54) vs. "Dhanasekhar - 2025 - Coexistence
    # of magnetic and dielectric glassy states in alternating kagome and.pdf"
    # (v0.3.56/58): identischer Note-Titel "AC magnetization studies.md" in
    # beiden Runs (DB-Beleg, s. PR-Body).
    assert _pdf_group_key("cobaltite-paper.pdf") == "dhanasekhar-2025"


# ── build_data-Integration: Filter-Kaskade überbrückt Label-Drift ───────────


def _pipeline_run(run_id, ver, label, key, source, cost=1.0):
    return {
        "run_id": run_id,
        "timestamp": "2026-07-01T00:00:00",
        "pipeline_version": ver,
        "pdf_source": source,
        "pdf_key": key,
        "pdf_label": label,
        "model": "test-model-x",
        "cost_usd": cost,
        "n_generated": 4,
        "n_vault": 3,
        "n_inbox": 1,
        "n_merge": 0,
        "n_words": 5000,
        "n_dropped": 0,
        "duration_s": 60.0,
    }


def _eval_row(run_id, ver, pdf):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.05,
        "coverage_factual": 0.8,
        "pipeline_version": ver,
        "version": ver,
        "pdf": pdf,
        "language": "DE→DE",
        "eval_version": "4.1",
        "anchors_total": 10,
        "anchors_hallucinated": 1,
    }


def test_build_data_pdf_filter_bridges_label_drift(monkeypatch):
    """Repro #202 (Bates): Filter=Volltitel, pipeline_runs-Labels driften.

    Vorher: token_runs matchte nur das v0.1.0-Volltitel-Label → Kosten-Panel
    zeigte die falsche Version, Agenten-/Token-Panels leer. Nachher: alle
    Namensraum-Varianten derselben Quelle bleiben in der Kaskade.
    """
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    runs = [
        _pipeline_run(
            "r-old", "v0.1.0", "Bates - 2017 - Information Behavior", None, "Bates - 2017 - Information Behavior.pdf"
        ),
        _pipeline_run("r-mid", "v0.3.74", "Bates", "bates", "Bates - 2017 - Information Behavior.pdf"),
        _pipeline_run("r-new", "v0.3.134", "bates-2017", "bates-2017", "bates-2017.pdf"),
        _pipeline_run(
            "r-other", "v0.3.134", "Porst-2014-Auszug-S1-40", "porst-2014-auszug-s1-40", "Porst-2014-Auszug-S1-40.pdf"
        ),
    ]
    evals = [
        _eval_row("r-old", "v0.1.0", "Bates - 2017 - Information Behavior.pdf"),
        _eval_row("r-mid", "v0.3.74", "Bates - 2017 - Information Behavior.pdf"),
        _eval_row("r-new", "v0.3.134", "bates-2017.pdf"),
        _eval_row("r-other", "v0.3.134", "Porst-2014-Auszug-S1-40.pdf"),
    ]
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(r) for r in runs])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    # Dropdown-Wert wie vom Client gesendet (lowercase Volltitel)
    data = S.build_data(pdf="bates - 2017 - information behavior")

    # token_runs-Kaskade: alle drei Bates-Runs überleben → Kosten/Tokens je
    # Version enthalten die echten neuen Versionen, nicht nur v0.1.0.
    trend_vers = set(data["kpi_trend"]["versions"])
    assert "v0.3.74" in trend_vers and "v0.3.134" in trend_vers
    cost_by_ver = dict(zip(data["kpi_trend"]["versions"], data["kpi_trend"]["cost"]))
    assert cost_by_ver.get("v0.3.74") == 1.0
    assert cost_by_ver.get("v0.3.134") == 1.0

    # all_log_runs-Kaskade (DB-Fallback): Vergleichstabelle hat die Bates-Versionen
    assert "v0.3.74" in data["runs_by_version"]
    assert "v0.3.134" in data["runs_by_version"]

    # quality_rows: auch die Kebab-Eval-Zeile der neuesten Version bleibt
    assert "v0.3.134" in data["quality_by_version"]

    # Fremde Quelle bleibt draußen
    assert all(
        "porst" not in (v.get("pdfs") and str(v["pdfs"]) or "").lower() for v in data["runs_by_version"].values()
    )

    # Dropdown bietet Bates nur einmal an
    bates_opts = [o for o in data["all_pdfs"] if "bates" in o.lower()]
    assert bates_opts == ["Bates - 2017 - Information Behavior"]


def test_build_data_pdf_filter_short_label_only(monkeypatch):
    """Repro #202 (Hrastinski): pdf_label ist nur der Autor — vorher 0 Treffer."""
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    runs = [
        _pipeline_run(
            "r-h",
            "v0.3.140",
            "Hrastinski",
            "hrastinski",
            "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf",
        )
    ]
    evals = [_eval_row("r-h", "v0.3.140", "Hrastinski - 2008 - Asynchronous and Synchronous E-Learning.pdf")]
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [dict(r) for r in runs])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data(pdf="hrastinski - 2008 - asynchronous and synchronous e-learning")

    assert "v0.3.140" in data["runs_by_version"]
    assert "v0.3.140" in data["quality_by_version"]
    cost_by_ver = dict(zip(data["kpi_trend"]["versions"], data["kpi_trend"]["cost"]))
    assert cost_by_ver.get("v0.3.140") == 1.0
