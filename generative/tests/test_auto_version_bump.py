"""Tests für den kollisionssicheren Auto-Version-Bump (#191).

Zwei Befunde:

1. Kollisionsrisiko: `_auto_version_bump` zählte den Patch der config-Version
   +1 und kannte bereits „verbrannte" Versionsnummern nicht. WIP-Branch-Läufe
   hatten .141/.142 in quality_history/DB gestempelt, master stand auf .140 —
   der nächste Bump hätte .141 wiederverwendet und künftige Läufe mit den
   WIP-Zeilen vermischt (History-Pools, Eval-Cache-Match auf pipeline_version).
   Fix: Patch+1 der MAXIMAL bekannten Version (config, State, DB, JSONL, Logs).

2. Stale-Binding (Codex-Review-Fund): Der Bump setzte nur
   `generative.config.AGENT_VERSION`, nicht das from-Import-Global des
   Orchestrators — der laufende Run stempelte weiter die alte Version.
"""

from __future__ import annotations

import json

import pytest

from generative import config as _cfg
from generative import orchestrator


@pytest.fixture()
def bump_env(monkeypatch, tmp_path):
    """Isolierte Bump-Umgebung: config-Stub in tmp, Module-Globals auto-restauriert."""
    (tmp_path / "config.py").write_text('AGENT_VERSION = "v0.3.140"  # Kommentar bleibt\n', encoding="utf-8")
    monkeypatch.setattr(orchestrator, "AGENT_VERSION", "v0.3.140")
    monkeypatch.setattr(_cfg, "AGENT_VERSION", "v0.3.140")
    monkeypatch.setattr(orchestrator, "_known_pipeline_versions", lambda: set())
    return tmp_path


def _cfg_version(tmp_path) -> str:
    return (tmp_path / "config.py").read_text(encoding="utf-8")


def test_bump_skips_versions_known_from_history(bump_env, monkeypatch):
    # config .140, aber .142 wurde von einem WIP-Branch bereits gestempelt → .143
    monkeypatch.setattr(orchestrator, "_known_pipeline_versions", lambda: {"v0.3.142", "v0.3.130"})
    orchestrator._auto_version_bump(base_dir=bump_env)
    assert 'AGENT_VERSION = "v0.3.143"' in _cfg_version(bump_env)


def test_bump_without_foreign_sources_increments_config(bump_env):
    orchestrator._auto_version_bump(base_dir=bump_env)
    assert 'AGENT_VERSION = "v0.3.141"' in _cfg_version(bump_env)


def test_bump_updates_running_module_globals(bump_env, monkeypatch):
    # F1 aus dem Review: Der laufende Run muss die NEUE Version stempeln —
    # sowohl config-Attribut als auch das from-Import-Global im Orchestrator.
    monkeypatch.setattr(orchestrator, "_known_pipeline_versions", lambda: {"v0.3.142"})
    orchestrator._auto_version_bump(base_dir=bump_env)
    assert orchestrator.AGENT_VERSION == "v0.3.143"
    assert _cfg.AGENT_VERSION == "v0.3.143"


def test_bump_noop_when_code_hash_unchanged(bump_env):
    orchestrator._auto_version_bump(base_dir=bump_env)
    after_first = _cfg_version(bump_env)
    orchestrator._auto_version_bump(base_dir=bump_env)
    # Zweiter Lauf ohne Code-Änderung: kein weiterer Bump. NB: der erste Bump
    # ändert config.py und damit den Hash — der State muss den Post-Bump-Hash
    # speichern, sonst bumpt jeder Lauf.
    assert _cfg_version(bump_env) == after_first


def test_bump_survives_corrupt_state(bump_env):
    state = bump_env / ".cache" / "pipeline_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{kaputt", encoding="utf-8")
    orchestrator._auto_version_bump(base_dir=bump_env)
    assert 'AGENT_VERSION = "v0.3.141"' in _cfg_version(bump_env)
    # State wurde repariert und trägt die neue Version
    assert json.loads(state.read_text(encoding="utf-8"))["last_version"] == "v0.3.141"


def test_known_versions_filters_junk(monkeypatch):
    # Der Sammler liefert nur plausible generative Versions-Strings.
    monkeypatch.setattr(
        orchestrator,
        "_iter_raw_version_strings",
        lambda: iter(["v0.3.142", "extractive-v0.2.0", "", "unfug", "v0.3.140"]),
    )
    assert orchestrator._known_pipeline_versions() == {"v0.3.142", "v0.3.140"}


def test_raw_version_collector_reads_real_artifacts(monkeypatch, tmp_path):
    # Integrationstest (Codex-Fund PR-192-Review): der Sammler liest DB,
    # quality_history*.jsonl (inkl. Archiv) und Baseline-Log-Namen wirklich.
    from generative import config as _cfg
    from generative import db as _db

    history = tmp_path / "quality_history.jsonl"
    history.write_text('{"version": "v0.3.140", "note": "a"}\n', encoding="utf-8")
    (tmp_path / "quality_history_archive.jsonl").write_text(
        '{"pipeline_version": "v0.3.142", "note": "b"}\n', encoding="utf-8"
    )
    log_dir = tmp_path / "eval" / "baseline"
    log_dir.mkdir(parents=True)
    (log_dir / "bates_v0.3.99_run2.log").write_text("", encoding="utf-8")

    monkeypatch.setattr(_cfg, "QUALITY_HISTORY", history)
    monkeypatch.setattr(_db, "query_pipeline_runs", lambda: [{"pipeline_version": "v0.3.130"}])

    assert orchestrator._known_pipeline_versions() == {"v0.3.140", "v0.3.142", "v0.3.99", "v0.3.130"}
