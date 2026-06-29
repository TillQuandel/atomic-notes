"""Phoenix-Server-Auto-Start: die Pipeline startet den Server selbst, wenn
Tracing aktiv ist und der Server noch nicht läuft.

Lifecycle-Tests (Server-Start), NICHT Span-Erzeugung — die deckt
test_phoenix_span.py ab. Subprocess.Popen wird durchgängig gemockt, damit
kein echter Phoenix-Server startet.
"""

from pathlib import Path

from generative import orchestrator


def test_server_already_running_no_spawn(monkeypatch):
    monkeypatch.setattr(orchestrator, "_phoenix_server_running", lambda _p: True)
    spawned = {"n": 0}
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1))

    assert orchestrator._ensure_phoenix_server(port=6006, venv=Path("/nonexistent")) is True
    assert spawned["n"] == 0  # kein zweiter Server


def test_venv_missing_returns_false_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator, "_phoenix_server_running", lambda _p: False)
    spawned = {"n": 0}
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: spawned.__setitem__("n", spawned["n"] + 1))

    # tmp_path enthält kein phoenix-Binary → graceful False, kein Spawn, kein Crash.
    assert orchestrator._ensure_phoenix_server(port=6006, venv=tmp_path, timeout=1.0) is False
    assert spawned["n"] == 0


def test_spawns_and_waits_until_port_open(monkeypatch, tmp_path):
    # Fake-Binary am plattform-korrekten Pfad anlegen, damit _phoenix_exe es findet.
    exe = orchestrator._phoenix_exe(tmp_path)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")

    calls = {"n": 0}

    def fake_running(_port):
        calls["n"] += 1
        return calls["n"] > 1  # 1. Check (vor Spawn) tot, danach offen

    monkeypatch.setattr(orchestrator, "_phoenix_server_running", fake_running)
    spawned = {"args": None}
    monkeypatch.setattr("subprocess.Popen", lambda a, **k: spawned.__setitem__("args", a))

    assert orchestrator._ensure_phoenix_server(port=6006, venv=tmp_path, timeout=5.0) is True
    assert spawned["args"][0] == str(exe)
    assert spawned["args"][1] == "serve"


def test_popen_oserror_is_graceful(monkeypatch, tmp_path):
    exe = orchestrator._phoenix_exe(tmp_path)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    monkeypatch.setattr(orchestrator, "_phoenix_server_running", lambda _p: False)

    def boom(*a, **k):
        raise OSError("WinError 193")

    monkeypatch.setattr("subprocess.Popen", boom)

    # Beschädigtes Binary → kein Crash, graceful False (Docstring-Versprechen).
    assert orchestrator._ensure_phoenix_server(port=6006, venv=tmp_path, timeout=1.0) is False


def test_setup_skips_wiring_when_server_unavailable(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENT_TRACING", "phoenix")
    monkeypatch.setattr(orchestrator, "_TRACER", None)
    monkeypatch.setattr(orchestrator, "_ensure_phoenix_server", lambda *a, **k: False)

    orchestrator._setup_phoenix_tracing()
    # Server nicht verfügbar → kein Tracer verdrahtet (sonst Tot-Port-Span-Spam).
    assert orchestrator._TRACER is None


def test_setup_noop_when_tracing_disabled(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENT_TRACING", raising=False)
    monkeypatch.setattr(orchestrator, "_TRACER", None)
    called = {"ensure": 0}
    monkeypatch.setattr(
        orchestrator,
        "_ensure_phoenix_server",
        lambda *a, **k: called.__setitem__("ensure", called["ensure"] + 1) or True,
    )

    orchestrator._setup_phoenix_tracing()
    assert called["ensure"] == 0  # ohne ENV-Flag wird der Server nie gestartet
    assert orchestrator._TRACER is None
