"""Test-Isolation der Pipeline-Seiteneffekte für ALLE generative-Tests (#198 P2).

Vier produktive Senken wurden bisher aus Tests heraus beschrieben bzw. mutiert:

1. **Trace-Verzeichnis** — jeder LLM-Call (`agents.base.call_claude_full`) und jedes
   strukturierte Event routet über `agents.tracing._backend` und schreibt eine Zeile
   nach `generative/.cache/runs/<run-id>.jsonl`. Tests, die `call_claude_full` mit
   gestubbtem Backend aufrufen (z. B. test_call_record, test_eval_cache_namespace),
   erzeugten so echte Trace-Dateien mit agent="test"/"verifier"/... im produktiven
   `runs/`-Verzeichnis.

2. **LLM-Response-Cache** — `call_claude_full(..., use_cache=True)` (Default) legt die
   Backend-Antwort unter `base._cache_put` in `generative/.cache/llm/<key>.json` ab. Der
   in-process-main()-Smoke-E2E-Test persistierte damit Fake-Backend-Antworten unter
   ECHTEN Cache-Keys im produktiven LLM-Cache (#198 Nachbesserung Fix 1).

3. **Cache-Rotation** — `orchestrator.main()` ruft am Lauf-Ende
   `cache_rotation.rotate_run_caches()` (Default-Verzeichnis `config.CACHE_DIR`, Caps
   llm=2000 / runs=600, ÄLTESTE Dateien werden GELÖSCHT). Ein in-process-main()-Test auf
   einer gut gefüllten Maintainer-Maschine löscht so reale Cache-Dateien
   (#198 Nachbesserung Fix 2).

4. **Analytics-DB** — seit #198 P1 persistiert `orchestrator.main()` den pipeline_run
   UNBEDINGT (auch ohne Inline-Eval). Damit würde der In-Process-Smoke-E2E-Test
   (test_ci_smoke_e2e) in die produktive `.cache/atomic_analytics.db` schreiben.

Diese autouse-Fixture biegt alle vier Senken pro Test in ein isoliertes tmp-Verzeichnis
um — analog zum bestehenden Sicherheitsnetz in `generative/gui/tests/conftest.py`. Bewusst
über `tmp_path_factory` (eigener Ordner), NICHT über das test-eigene `tmp_path`: viele
Tests listen ihr `tmp_path` und würden sonst über die eingestreute `analytics.db` /
`trace-runs/` stolpern. Tests, die selbst einen Trace-Backend oder DB-Pfad setzen,
überschreiben die Fixture einfach danach (monkeypatch, last-wins) und bleiben unberührt.

Abgedeckt (in-process): tracing._backend, base._RUN_DIR, base._LLM_CACHE_DIR, db.DB_PATH
(Modul-Attribut) sowie die Lösch-Senke von cache_rotation.rotate_run_caches().

NICHT abgedeckt — bewusst dokumentiert:
  * **Late-Binding-Falle** — die db-Query-Helfer (`db.query_pipeline_runs`,
    `query_note_evals`, `query_kpi_trend`, `available_eval_versions`, `init_db`, `get_db`)
    binden `DB_PATH` als DEFAULT-ARGUMENT zur DEF-Zeit. Ein no-arg-Aufruf (`query_*()`)
    liest damit die REALE DB und umgeht den `db.DB_PATH`-Monkeypatch dieser Fixture. Nur
    Aufrufer, die den Pfad explizit durchreichen — wie orchestrator.main() via
    `get_db(db.DB_PATH)` — sind isoliert. Reine Lese-Query-Aufrufer sind harmlos (kein
    Write), aber nicht deterministisch gegen die reale DB.
  * **Subprozess-Tests** — `tests/test_e2e_baseline.py` startet die Pipeline als eigenen
    Prozess; diese in-process-Monkeypatches greifen dort prinzipiell nicht. Dieser Test
    isoliert die DB-Senke selbst per `ATOMIC_DB_PATH`-ENV (siehe dortiger Kommentar).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolate_pipeline_side_effects(tmp_path_factory, monkeypatch):
    """Trace-, LLM-Cache-, Rotations- und DB-Senken landen in tmp, nie im produktiven .cache/.

    Gibt ein SimpleNamespace mit `runs_dir`, `llm_cache_dir`, `cache_root` und `db_path`
    zurück, damit Tests die Zielpfade abfragen können (per-Namen anfordern, obwohl autouse).
    """
    from generative import cache_rotation, db
    from generative.agents import base, tracing

    base_dir = tmp_path_factory.mktemp("pipeline-side-effects")
    runs_dir = base_dir / "trace-runs"
    llm_cache_dir = base_dir / "llm-cache"
    cache_root = base_dir / "cache-root"
    db_path = base_dir / "analytics.db"

    # Trace-Backend + Lese-Pfad (base._RUN_DIR, von orchestrator zum Token-Aggregat
    # genutzt) auf denselben tmp-Ordner zeigen — Schreiben und Lesen konsistent.
    monkeypatch.setattr(tracing, "_backend", tracing.JsonlBackend(run_dir=runs_dir, run_id=tracing._RUN_ID))
    monkeypatch.setattr(base, "_RUN_DIR", runs_dir)

    # LLM-Response-Cache (_cache_get/_cache_put) auf tmp umbiegen — Fake-Backend-
    # Antworten aus Tests dürfen nie unter echten Cache-Keys im produktiven .cache/llm/
    # landen (Fix 1).
    monkeypatch.setattr(base, "_LLM_CACHE_DIR", llm_cache_dir)

    # Cache-Rotation (Lösch-Senke): orchestrator.main() ruft rotate_run_caches() ohne
    # Argument → Default = produktives config.CACHE_DIR, ältestes wird GELÖSCHT. Wrapper
    # bindet ein tmp-cache_root als Default, damit auf einer vollen Maintainer-Maschine
    # kein reales .cache/llm bzw. .cache/runs beschnitten wird (Fix 2). Die echte
    # Rotations-Logik (inkl. Caps) bleibt exerziert — nur das Zielverzeichnis ist tmp.
    _real_rotate = cache_rotation.rotate_run_caches
    monkeypatch.setattr(
        cache_rotation,
        "rotate_run_caches",
        lambda cache_dir=cache_root: _real_rotate(cache_dir),
    )

    # get_db(DB_PATH)/init_db lesen db.DB_PATH zur Laufzeit — Umbiegen genügt.
    # Schema gleich anlegen: ein Insert (orchestrator #198 P1) findet die Tabelle,
    # und Queries auf die leere DB liefern [] statt OperationalError.
    db.init_db(db_path)
    monkeypatch.setattr(db, "DB_PATH", db_path)

    return SimpleNamespace(
        runs_dir=runs_dir,
        llm_cache_dir=llm_cache_dir,
        cache_root=cache_root,
        db_path=db_path,
    )
