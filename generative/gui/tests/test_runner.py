"""Tests fuer den Subprocess-Runner der Live-GUI.

Echte Subprocesses (kein Mock): ein kleines `python -c`-Skript druckt
orchestrator-typische Marker-Zeilen, der Runner muss daraus die geparsten
Events streamen.
"""
import sys

from generative.gui.runner import build_argv, iter_run_events


def test_build_argv_dry_run_default():
    argv = build_argv("C:/x/foo.pdf", dry_run=True)
    assert argv[0] == sys.executable
    assert "-m" in argv and "generative.orchestrator" in argv
    assert "--source" in argv
    assert argv[argv.index("--source") + 1] == "C:/x/foo.pdf"
    assert "--dry-run" in argv


def test_build_argv_real_run_omits_dry_run():
    argv = build_argv("foo.pdf", dry_run=False)
    assert "--dry-run" not in argv


def test_iter_run_events_streams_parsed_events():
    script = (
        "print('[1/7] PDF extrahieren…'); "
        "print('=== Fertig: 1 Notes (dry-run) ==='); "
    )
    evs = list(iter_run_events([sys.executable, "-c", script]))
    types = [e["type"] for e in evs]
    assert "stage" in types
    assert "done" in types
    # Terminal-Event ist IMMER `exited` (nicht `done`) — der Orchestrator druckt
    # nach `=== Fertig ===` noch Routing-Report + Stage-8-Eval.
    assert evs[-1]["type"] == "exited"
    assert evs[-1]["returncode"] == 0


def test_iter_run_events_nonzero_exit_is_exited_with_returncode():
    script = "import sys; print('[1/7] x'); sys.exit(3)"
    evs = list(iter_run_events([sys.executable, "-c", script]))
    assert evs[-1]["type"] == "exited"
    assert evs[-1]["returncode"] == 3


def test_iter_run_events_emits_started_first():
    evs = list(iter_run_events([sys.executable, "-c", "pass"]))
    assert evs[0]["type"] == "started"
