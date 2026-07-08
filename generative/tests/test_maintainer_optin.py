"""Tests für das Maintainer-Opt-in der schreibenden Erst-Lauf-Nebeneffekte (#156).

`_auto_version_bump` mutiert die getrackte `generative/config.py` (AGENT_VERSION),
`_auto_start_dashboard` spawnt einen Server auf :8051. Beides überrascht Fremd-
Nutzer beim ersten (Dry-)Run und bricht das Doku-Versprechen "nothing written
until you say so". Ab #156 laufen beide Aktionen nur noch bei
`ATOMIC_AGENT_MAINTAINER=1`; die bestehende `ATOMIC_AGENT_GUI=1`-Unterdrückung
bleibt zusätzlich intakt.

Kein voller Pipeline-Lauf: das Gate sitzt direkt zwischen `_setup_phoenix_tracing`
und `load_runtime_config` in orchestrator.main — wir monkeypatchen beide Enden als
Seam und brechen unmittelbar nach dem Gate kontrolliert ab (Muster wie
test_orchestrator_export.py).
"""

from __future__ import annotations

import pytest

from generative import orchestrator


def _arm_gate_seam(monkeypatch):
    """Spies auf beide Auto-Aktionen + kontrollierter Abbruch direkt nach dem Gate.

    Gibt (calls, run_main) zurück: `calls` sammelt die Namen der aufgerufenen
    Auto-Aktionen; `run_main()` ruft orchestrator.main mit einer neutralen --source
    auf und erwartet den stop-here-Abbruch (load_runtime_config läuft unmittelbar
    nach dem Gate).
    """
    calls: list[str] = []
    monkeypatch.setattr(orchestrator, "_setup_phoenix_tracing", lambda: None)
    monkeypatch.setattr(orchestrator, "_auto_start_dashboard", lambda: calls.append("dashboard"))
    monkeypatch.setattr(orchestrator, "_auto_version_bump", lambda: calls.append("version_bump"))
    monkeypatch.setattr(
        orchestrator,
        "load_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeError("stop-here")),
    )

    def run_main():
        with pytest.raises(RuntimeError, match="stop-here"):
            orchestrator.main(["--source", "irrelevant.pdf"])

    return calls, run_main


def test_no_maintainer_flag_suppresses_both_side_effects(monkeypatch):
    # Fremd-Nutzer-Default: Flag nicht gesetzt (auch nicht via geladenem .env) →
    # weder Version-Bump noch Dashboard-Autostart.
    monkeypatch.delenv("ATOMIC_AGENT_MAINTAINER", raising=False)
    monkeypatch.delenv("ATOMIC_AGENT_GUI", raising=False)
    calls, run_main = _arm_gate_seam(monkeypatch)
    run_main()
    assert calls == [], f"Auto-Aktionen ohne Maintainer-Flag ausgelöst: {calls}"


def test_maintainer_flag_enables_both_side_effects(monkeypatch):
    # Maintainer-Opt-in: beide Aktionen laufen.
    monkeypatch.setenv("ATOMIC_AGENT_MAINTAINER", "1")
    monkeypatch.delenv("ATOMIC_AGENT_GUI", raising=False)
    calls, run_main = _arm_gate_seam(monkeypatch)
    run_main()
    assert set(calls) == {"dashboard", "version_bump"}, f"Nicht beide Auto-Aktionen liefen: {calls}"


def test_gui_flag_suppresses_even_for_maintainer(monkeypatch):
    # GUI-Kontext unterdrückt weiterhin, unabhängig vom Maintainer-Flag: das
    # GUI-Subprocess spawnt sonst ein zweites Dashboard und mutiert config.py.
    monkeypatch.setenv("ATOMIC_AGENT_MAINTAINER", "1")
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    calls, run_main = _arm_gate_seam(monkeypatch)
    run_main()
    assert calls == [], f"GUI=1 unterdrückt die Auto-Aktionen nicht: {calls}"
