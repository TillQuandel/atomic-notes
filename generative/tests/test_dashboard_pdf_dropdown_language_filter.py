"""Test für #D6 (Multi-Perspektiven-Review 2026-07-15): PDF-Dropdown ignoriert
aktiven Sprachfilter.

Befund: Das Versions-Dropdown (`_all_pvers_opts`, eval_dashboard_server.py)
schränkt sich bei aktivem Sprachfilter bereits auf die passenden Zeilen ein
(`if language: _pver_rows = [...]`). Das PDF-Dropdown direkt darüber
(`_all_pdfs_opts`) tat das nicht — bei z.B. Sprachfilter "DE→DE" erschienen
weiterhin PDFs, die nur in anderen Sprachen evaluiert wurden.
"""

from __future__ import annotations


def _eval_row(run_id, ver, pdf, language, eval_version="4.1"):
    return {
        "run_id": run_id,
        "note_path": f"notes/{run_id}.md",
        "acceptance_status": "vault",
        "hallucination_rate": 0.05,
        "coverage_factual": 0.8,
        "pipeline_version": ver,
        "version": ver,
        "pdf": pdf,
        "language": language,
        "eval_version": eval_version,
        "anchors_total": 10,
        "anchors_hallucinated": 1,
    }


def test_pdf_dropdown_respects_active_language_filter(monkeypatch):
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    evals = [
        _eval_row("r-de", "v0.3.140", "Deutsches-Paper.pdf", "DE→DE"),
        _eval_row("r-en", "v0.3.140", "English-Paper.pdf", "EN→DE"),
    ]
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data(language="DE→DE")

    assert any("Deutsches-Paper" in o for o in data["all_pdfs"])
    assert not any("English-Paper" in o for o in data["all_pdfs"])


def test_pdf_dropdown_shows_all_pdfs_without_language_filter(monkeypatch):
    from generative import db as _gdb
    from generative import eval_dashboard as D
    from generative import eval_dashboard_server as S

    evals = [
        _eval_row("r-de", "v0.3.140", "Deutsches-Paper.pdf", "DE→DE"),
        _eval_row("r-en", "v0.3.140", "English-Paper.pdf", "EN→DE"),
    ]
    monkeypatch.setattr(_gdb, "query_pipeline_runs", lambda *a, **k: [])
    monkeypatch.setattr(_gdb, "query_note_evals", lambda *a, **k: [dict(r) for r in evals])
    monkeypatch.setattr(D, "_read_all_log_runs", lambda: [])
    monkeypatch.setattr(D, "_read_token_runs", lambda: [])

    data = S.build_data()

    assert any("Deutsches-Paper" in o for o in data["all_pdfs"])
    assert any("English-Paper" in o for o in data["all_pdfs"])
