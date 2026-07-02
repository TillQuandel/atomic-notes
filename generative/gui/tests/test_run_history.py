"""Tests fuer die Run-Historie (P4): reine Datei-/Record-Logik, kein FastAPI.

Record-Schema (P4-Auftrag): {run_id, started_at, finished_at, source_pdf,
dry_run, options, rc, notes}. `notes` ist die P3-Item-Struktur (aus
`app._output_items_from_events` — hier bereits fertig aggregiert uebergeben,
kein Duplikat der Aggregations-Logik).
"""

import logging
from pathlib import Path

from generative.gui import run_history


# --- make_run_id -----------------------------------------------------------


def test_make_run_id_matches_pattern():
    run_id = run_history.make_run_id(1_800_000_000.0)
    assert run_history.RUN_ID_RE.fullmatch(run_id)


def test_make_run_id_encodes_timestamp_prefix():
    # 2024-01-01T00:00:00Z == 1704067200
    run_id = run_history.make_run_id(1_704_067_200.0, suffix="abc123")
    assert run_id == "20240101000000-abc123"


def test_make_run_id_differs_for_same_timestamp_without_explicit_suffix():
    a = run_history.make_run_id(1_704_067_200.0)
    b = run_history.make_run_id(1_704_067_200.0)
    assert a != b  # zufaelliges Suffix vermeidet Kollisionen


def test_is_valid_run_id_accepts_generated_ids():
    assert run_history.is_valid_run_id(run_history.make_run_id(1_704_067_200.0))


def test_is_valid_run_id_rejects_traversal_and_specials():
    for bad in ("../x", "..", "a/b", "a\\b", "UPPER", "with space", "a.b", "", "C:x"):
        assert not run_history.is_valid_run_id(bad), bad


# --- build_run_record -------------------------------------------------------


def test_build_run_record_shape():
    record = run_history.build_run_record(
        run_id="20240101000000-abc123",
        started_at=100.0,
        finished_at=142.5,
        source_pdf="C:/pdfs/foo.pdf",
        dry_run=True,
        options={"backend": "litellm"},
        rc=0,
        notes=[{"title": "a.md", "routing": "vault"}],
    )
    assert record == {
        "run_id": "20240101000000-abc123",
        "started_at": 100.0,
        "finished_at": 142.5,
        "source_pdf": "C:/pdfs/foo.pdf",
        "dry_run": True,
        "options": {"backend": "litellm"},
        "rc": 0,
        "notes": [{"title": "a.md", "routing": "vault"}],
    }


def test_build_run_record_includes_duration_and_tokens_when_given():
    # P5: run_summary-Daten (Zeit/Tokens aus dem Final-Report) landen optional
    # im Record — Kurzzeile im Historie-Panel braucht sie.
    record = run_history.build_run_record(
        run_id="20240101000000-abc123",
        started_at=100.0,
        finished_at=142.5,
        source_pdf="C:/pdfs/foo.pdf",
        dry_run=True,
        options={},
        rc=0,
        notes=[],
        duration_s=12.4,
        tokens={"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0},
    )
    assert record["duration_s"] == 12.4
    assert record["tokens"] == {"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0}


def test_build_run_record_omits_duration_and_tokens_when_absent():
    # Rueckwaertskompatibilitaet: kein run_summary-Event -> keine erfundenen
    # Felder im Record (L5) statt None-Platzhalter.
    record = run_history.build_run_record(
        run_id="x",
        started_at=1.0,
        finished_at=2.0,
        source_pdf=None,
        dry_run=None,
        options=None,
        rc=None,
        notes=[],
    )
    assert "duration_s" not in record
    assert "tokens" not in record


def test_build_run_record_defaults_options_to_empty_dict():
    record = run_history.build_run_record(
        run_id="x",
        started_at=1.0,
        finished_at=2.0,
        source_pdf=None,
        dry_run=None,
        options=None,
        rc=None,
        notes=[],
    )
    assert record["options"] == {}


# --- write_run_record / read_run_record roundtrip ---------------------------


def test_write_and_read_run_record_roundtrip(tmp_path):
    record = run_history.build_run_record(
        run_id="20240101000000-abc123",
        started_at=1.0,
        finished_at=2.0,
        source_pdf="x.pdf",
        dry_run=True,
        options={},
        rc=0,
        notes=[],
    )
    path = run_history.write_run_record(record, tmp_path)
    assert path == tmp_path / "20240101000000-abc123.json"
    assert run_history.read_run_record("20240101000000-abc123", tmp_path) == record


def test_write_run_record_is_atomic_no_tempfile_left_behind(tmp_path):
    record = run_history.build_run_record(
        run_id="20240101000000-abc123",
        started_at=1.0,
        finished_at=2.0,
        source_pdf="x.pdf",
        dry_run=False,
        options={},
        rc=0,
        notes=[],
    )
    run_history.write_run_record(record, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["20240101000000-abc123.json"]


def test_read_run_record_rejects_invalid_run_id(tmp_path):
    for bad in ("../secret", "..", "a/b", "C:x", "UPPER", ""):
        assert run_history.read_run_record(bad, tmp_path) is None


def test_read_run_record_missing_file_returns_none(tmp_path):
    assert run_history.read_run_record("20240101000000-abc123", tmp_path) is None


def test_read_run_record_corrupt_json_returns_none_and_logs(tmp_path, caplog):
    (tmp_path / "20240101000000-abc123.json").write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert run_history.read_run_record("20240101000000-abc123", tmp_path) is None
    assert "20240101000000-abc123" in caplog.text


def test_read_run_record_non_dict_json_returns_none(tmp_path):
    (tmp_path / "20240101000000-abc123.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert run_history.read_run_record("20240101000000-abc123", tmp_path) is None


# --- list_run_records --------------------------------------------------------


def _write(tmp_path: Path, run_id: str, finished_at: float) -> None:
    record = run_history.build_run_record(
        run_id=run_id,
        started_at=finished_at - 1,
        finished_at=finished_at,
        source_pdf="x.pdf",
        dry_run=True,
        options={},
        rc=0,
        notes=[],
    )
    run_history.write_run_record(record, tmp_path)


def test_list_run_records_orders_newest_first(tmp_path):
    _write(tmp_path, "20240101000000-a", 100.0)
    _write(tmp_path, "20240102000000-b", 300.0)
    _write(tmp_path, "20240103000000-c", 200.0)
    records = run_history.list_run_records(tmp_path)
    assert [r["run_id"] for r in records] == ["20240102000000-b", "20240103000000-c", "20240101000000-a"]


def test_list_run_records_limits_to_given_limit(tmp_path):
    for i in range(5):
        _write(tmp_path, f"2024010{i}000000-x", float(i))
    records = run_history.list_run_records(tmp_path, limit=3)
    assert len(records) == 3
    assert [r["finished_at"] for r in records] == [4.0, 3.0, 2.0]


def test_list_run_records_skips_corrupt_without_crashing(tmp_path):
    _write(tmp_path, "20240101000000-a", 1.0)
    (tmp_path / "20240102000000-broken.json").write_text("{broken", encoding="utf-8")
    records = run_history.list_run_records(tmp_path)
    assert len(records) == 1
    assert records[0]["run_id"] == "20240101000000-a"


def test_list_run_records_empty_dir_returns_empty_list(tmp_path):
    assert run_history.list_run_records(tmp_path) == []


def test_list_run_records_missing_dir_returns_empty_list(tmp_path):
    assert run_history.list_run_records(tmp_path / "does-not-exist") == []


# --- prune_old_records --------------------------------------------------------


def test_prune_old_records_keeps_newest_n(tmp_path):
    for i in range(55):
        _write(tmp_path, f"run-{i:03d}", float(i))
    run_history.prune_old_records(tmp_path, keep=50)
    remaining = run_history.list_run_records(tmp_path, limit=100)
    assert len(remaining) == 50
    assert sorted(r["finished_at"] for r in remaining) == [float(i) for i in range(5, 55)]


def test_prune_old_records_noop_when_under_limit(tmp_path):
    for i in range(3):
        _write(tmp_path, f"run-{i:03d}", float(i))
    run_history.prune_old_records(tmp_path, keep=50)
    assert len(run_history.list_run_records(tmp_path, limit=100)) == 3


def test_prune_old_records_missing_dir_no_crash(tmp_path):
    run_history.prune_old_records(tmp_path / "does-not-exist", keep=50)  # darf nicht werfen
