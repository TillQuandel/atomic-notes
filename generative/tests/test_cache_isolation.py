"""#198 Nachbesserung: LLM-Cache- und Rotations-Senke bleiben tmp-isoliert.

Beweist, dass die autouse-Fixture `isolate_pipeline_side_effects` (generative/conftest.py)
zwei zuvor ungedeckte produktive Senken abfängt:

  Fix 1 — `base._cache_put` (LLM-Response-Cache): ein `call_claude_full(use_cache=True)`
          landet in tmp, nie in `<repo>/generative/.cache/llm/`.
  Fix 2 — `cache_rotation.rotate_run_caches()` (Lösch-Senke): rotiert das tmp-cache_root,
          nie das produktive `.cache/llm` bzw. `.cache/runs`.

Beide Tests snapshotten das REALE `.cache/`-Verzeichnis rein lesend (glob) und stellen
sicher, dass die Datei-Menge davor == danach ist — es werden nie echte Cache-Daten
angefasst. Die "Redirect greift"-Assertion steht bewusst VOR jedem Schreib-/Rotations-
Schritt: fällt die Isolation aus, bricht der Test ab, bevor irgendetwas passiert.
"""

from __future__ import annotations

import os

from generative import cache_rotation
from generative.agents import base
from generative.agents.base import CallResult
from generative.config import CACHE_DIR


def test_llm_cache_write_goes_to_tmp_not_production(isolate_pipeline_side_effects, monkeypatch):
    real_llm = CACHE_DIR / "llm"
    before = set(real_llm.glob("*.json")) if real_llm.exists() else set()

    # Redirect-Nachweis ZUERST — bei ausgefallener Isolation Abbruch vor dem Schreiben.
    assert base._LLM_CACHE_DIR == isolate_pipeline_side_effects.llm_cache_dir
    assert base._LLM_CACHE_DIR != real_llm

    # Gestubbtes Backend → deterministischer, netz-freier Call über den echten
    # _cache_put-Schreibpfad (use_cache=True ist Default).
    monkeypatch.setattr(
        base,
        "_backend_call_full",
        lambda prompt, **k: CallResult(text="antwort", input_tokens=3, output_tokens=1),
    )
    base.call_claude_full("ein prompt", agent="test", use_cache=True)

    # (a) Cache-File liegt in tmp.
    tmp_files = list(isolate_pipeline_side_effects.llm_cache_dir.glob("*.json"))
    assert tmp_files, f"kein Cache-File in tmp: {isolate_pipeline_side_effects.llm_cache_dir}"

    # (b) Produktives .cache/llm bekam keine neue Datei.
    after = set(real_llm.glob("*.json")) if real_llm.exists() else set()
    assert after == before, f"Produktives .cache/llm verschmutzt: {after - before}"


def test_rotation_targets_tmp_not_production(isolate_pipeline_side_effects, monkeypatch):
    real_llm = CACHE_DIR / "llm"
    real_runs = CACHE_DIR / "runs"
    llm_before = set(real_llm.iterdir()) if real_llm.exists() else set()
    runs_before = set(real_runs.iterdir()) if real_runs.exists() else set()

    # Caps runter, damit Rotation ohne 2001 Dateien greift. rotate_run_caches liest die
    # Caps als Modul-Globals zur Laufzeit → Monkeypatch wirkt.
    monkeypatch.setattr(cache_rotation, "CACHE_RUNS_MAX_FILES", 1)
    monkeypatch.setattr(cache_rotation, "CACHE_LLM_MAX_FILES", 1)

    cache_root = isolate_pipeline_side_effects.cache_root
    tmp_runs = cache_root / "runs"
    tmp_runs.mkdir(parents=True, exist_ok=True)
    old = tmp_runs / "old.jsonl"
    old.write_text("x", encoding="utf-8")
    new = tmp_runs / "new.jsonl"
    new.write_text("y", encoding="utf-8")
    os.utime(old, (1, 1))  # alt → wird bei keep=1 gelöscht

    # rotate_run_caches() ist von der Fixture auf cache_root umgebogen (kein Argument →
    # tmp-Default, wie orchestrator.main() es aufruft).
    _n_llm, n_runs = cache_rotation.rotate_run_caches()

    # (a) Rotation traf das tmp-cache_root: ältestes weg, neuestes bleibt.
    assert n_runs == 1, f"tmp-runs nicht rotiert (n_runs={n_runs})"
    assert new.exists() and not old.exists()

    # (b) Produktive .cache/llm und .cache/runs unberührt.
    llm_after = set(real_llm.iterdir()) if real_llm.exists() else set()
    runs_after = set(real_runs.iterdir()) if real_runs.exists() else set()
    assert llm_after == llm_before, f"Produktives .cache/llm verändert: {llm_after ^ llm_before}"
    assert runs_after == runs_before, f"Produktives .cache/runs verändert: {runs_after ^ runs_before}"
