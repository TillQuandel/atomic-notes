"""Tests für den Stage-6-Crash-Report-Writer (Issue #17).

Bei einer Stage-6-Exception wird der unverifizierte Draft NICHT geschrieben,
stattdessen ein strukturierter JSON-Crash-Report nach .cache/failed/<run-id>/<slug>.json,
damit der Crash diagnostizierbar ist (Schritt, Exception, Prompt, roher Output, Draft, Phase).
"""
import json

from pipeline.crash_report import write_crash_report


def _payload():
    return {
        "title": "Information Foraging Theory",
        "step": "verifier",
        "exception": "RuntimeError: backend crashed",
        "traceback": "Traceback (most recent call last):\n  ...\nRuntimeError: backend crashed",
        "prompt": "Verifiziere die folgenden Anker ...",
        "raw_output": "<!--ANCHOR-->\nmalformed",
        "draft_body": "# Information Foraging Theory\n\nText ...",
        "phase": "initial",
        "run_meta": {"run_id": "abc123", "pdf": "pirolli.pdf", "backend": "subscription"},
    }


def test_write_crash_report_creates_json_file(tmp_path):
    failed_dir = tmp_path / "failed" / "abc123"
    payload = _payload()

    path = write_crash_report(failed_dir, payload)

    assert path.exists()
    assert path.parent == failed_dir
    assert path.suffix == ".json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in payload:
        assert data[key] == payload[key], f"Feld {key} fehlt oder weicht ab"


def test_write_crash_report_creates_missing_dir(tmp_path):
    failed_dir = tmp_path / "failed" / "neuer-run"
    assert not failed_dir.exists()

    path = write_crash_report(failed_dir, _payload())

    assert failed_dir.is_dir()
    assert path.exists()


def test_write_crash_report_slug_derived_from_title(tmp_path):
    failed_dir = tmp_path / "failed" / "run"
    payload = _payload()
    payload["title"] = "Café: Aboutness & Relevanz/Recall"

    path = write_crash_report(failed_dir, payload)

    # Dateiname dateisystem-sicher, kein Slash/Doppelpunkt
    assert "/" not in path.name and "\\" not in path.name and ":" not in path.name
    assert path.name.endswith(".json")


def test_write_crash_report_unicode_roundtrip(tmp_path):
    """Umlaute im Draft-Body müssen UTF-8-clean persistiert werden (kein Mojibake)."""
    failed_dir = tmp_path / "failed" / "run"
    payload = _payload()
    payload["draft_body"] = "Glättung über Schwellenwerte — größere Stichprobe nötig."

    path = write_crash_report(failed_dir, payload)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["draft_body"] == payload["draft_body"]
