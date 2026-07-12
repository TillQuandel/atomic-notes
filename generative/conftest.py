"""Test-Isolation der Pipeline-Seiteneffekte für ALLE generative-Tests (#198 P2).

Zwei produktive Senken wurden bisher aus Tests heraus beschrieben:

1. **Trace-Verzeichnis** — jeder LLM-Call (`agents.base.call_claude_full`) und jedes
   strukturierte Event routet über `agents.tracing._backend` und schreibt eine Zeile
   nach `generative/.cache/runs/<run-id>.jsonl`. Tests, die `call_claude_full` mit
   gestubbtem Backend aufrufen (z. B. test_call_record, test_eval_cache_namespace),
   erzeugten so echte Trace-Dateien mit agent="test"/"verifier"/... im produktiven
   `runs/`-Verzeichnis.

2. **Analytics-DB** — seit #198 P1 persistiert `orchestrator.main()` den pipeline_run
   UNBEDINGT (auch ohne Inline-Eval). Damit würde der In-Process-Smoke-E2E-Test
   (test_ci_smoke_e2e) in die produktive `.cache/atomic_analytics.db` schreiben.

Diese autouse-Fixture biegt beide Senken pro Test in ein isoliertes tmp-Verzeichnis um
— analog zum bestehenden Sicherheitsnetz in `generative/gui/tests/conftest.py`. Bewusst
über `tmp_path_factory` (eigener Ordner), NICHT über das test-eigene `tmp_path`: viele
Tests listen ihr `tmp_path` und würden sonst über die eingestreute `analytics.db` /
`trace-runs/` stolpern. Tests, die selbst einen Trace-Backend oder DB-Pfad setzen,
überschreiben die Fixture einfach danach (monkeypatch, last-wins) und bleiben unberührt.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolate_pipeline_side_effects(tmp_path_factory, monkeypatch):
    """Trace-Writes und pipeline_runs-Insert landen in tmp, nie im produktiven .cache/.

    Gibt ein SimpleNamespace mit `runs_dir` und `db_path` zurück, damit Tests die
    Zielpfade abfragen können (per-Namen anfordern, obwohl autouse).
    """
    from generative import db
    from generative.agents import base, tracing

    base_dir = tmp_path_factory.mktemp("pipeline-side-effects")
    runs_dir = base_dir / "trace-runs"
    db_path = base_dir / "analytics.db"

    # Trace-Backend + Lese-Pfad (base._RUN_DIR, von orchestrator zum Token-Aggregat
    # genutzt) auf denselben tmp-Ordner zeigen — Schreiben und Lesen konsistent.
    monkeypatch.setattr(tracing, "_backend", tracing.JsonlBackend(run_dir=runs_dir, run_id=tracing._RUN_ID))
    monkeypatch.setattr(base, "_RUN_DIR", runs_dir)

    # get_db(DB_PATH)/init_db lesen db.DB_PATH zur Laufzeit — Umbiegen genügt.
    # Schema gleich anlegen: ein Insert (orchestrator #198 P1) findet die Tabelle,
    # und Queries auf die leere DB liefern [] statt OperationalError.
    db.init_db(db_path)
    monkeypatch.setattr(db, "DB_PATH", db_path)

    return SimpleNamespace(runs_dir=runs_dir, db_path=db_path)
