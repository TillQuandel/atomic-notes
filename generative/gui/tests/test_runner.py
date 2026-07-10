"""Tests fuer den Subprocess-Runner der Live-GUI.

Echte Subprocesses (kein Mock): ein kleines `python -c`-Skript druckt
orchestrator-typische Marker-Zeilen, der Runner muss daraus die geparsten
Events streamen.
"""

import ctypes
import os
import queue
import signal
import subprocess
import sys
import threading
import time

import pytest

from generative.gui.runner import build_argv, build_run_spec, iter_run_events, terminate_process_tree


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
    script = "print('[1/7] PDF extrahieren…'); print('=== Fertig: 1 Notes (dry-run) ==='); "
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


def test_iter_run_events_sets_gui_env_flag():
    # Der GUI-Subprocess markiert sich via ATOMIC_AGENT_GUI=1, damit der
    # Orchestrator seine schreibenden Auto-Aktionen (Version-Bump, Dashboard-
    # Spawn) unterdrückt — ein Vorschau-Lauf darf nichts mutieren.
    script = "import os; print('[1/7] flag=' + os.environ.get('ATOMIC_AGENT_GUI', 'UNSET'))"
    evs = list(iter_run_events([sys.executable, "-c", script]))
    logs = [e.get("label", "") + e.get("text", "") for e in evs]
    assert any("flag=1" in s for s in logs)


# --- build_run_spec (P1: Lauf-Einstellungen) ------------------------------


def test_build_run_spec_no_options_matches_plain_build_argv():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options=None)
    assert argv == build_argv("foo.pdf", dry_run=True)
    assert env == {}


def test_build_run_spec_empty_options_dict_matches_plain_build_argv():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options={})
    assert argv == build_argv("foo.pdf", dry_run=True)
    assert env == {}


def test_build_run_spec_backend_option_sets_env_var():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={"backend": "litellm"})
    assert env == {"ATOMIC_AGENT_BACKEND": "litellm"}
    assert "--no-llm" not in argv


def test_build_run_spec_profile_option_sets_env_var():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={"profile": "fast"})
    assert env == {"ATOMIC_AGENT_PROFILE": "fast"}


def test_build_run_spec_no_llm_option_appends_flag():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={"no_llm": True})
    assert "--no-llm" in argv
    assert env == {}


def test_build_run_spec_no_llm_false_omits_flag():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={"no_llm": False})
    assert "--no-llm" not in argv


def test_build_run_spec_combined_options():
    argv, env = build_run_spec(
        "foo.pdf",
        dry_run=True,
        options={"backend": "litellm", "profile": "quality", "no_llm": True},
    )
    assert env == {"ATOMIC_AGENT_BACKEND": "litellm", "ATOMIC_AGENT_PROFILE": "quality"}
    assert "--no-llm" in argv
    assert "--dry-run" in argv


# --- build_run_spec vault_path (B2: Subprocess-Override) -------------------


def test_build_run_spec_vault_path_sets_env_var():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options=None, vault_path="C:/Users/x/Vault")
    assert env == {"ATOMIC_AGENT_VAULT_PATH": "C:/Users/x/Vault"}


def test_build_run_spec_no_vault_path_omits_env_var():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options=None, vault_path=None)
    assert "ATOMIC_AGENT_VAULT_PATH" not in env


def test_build_run_spec_vault_path_combines_with_other_options():
    argv, env = build_run_spec(
        "foo.pdf",
        dry_run=False,
        options={"backend": "litellm"},
        vault_path="/some/vault",
    )
    assert env == {"ATOMIC_AGENT_BACKEND": "litellm", "ATOMIC_AGENT_VAULT_PATH": "/some/vault"}


# --- build_run_spec inbox_dir (B3: Output-Ziel waehlbar) -------------------


def test_build_run_spec_inbox_dir_write_mode_appends_flag():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={"inbox_dir": "C:/export"})
    assert "--inbox-dir" in argv
    assert argv[argv.index("--inbox-dir") + 1] == "C:/export"


def test_build_run_spec_inbox_dir_dry_run_omits_flag():
    # Dry-Run ignoriert inbox_dir -- vault_writer schreibt im Dry-Run ohnehin
    # nur eval-Kopien nach .cache/eval/baseline, unabhaengig von --inbox-dir.
    argv, env = build_run_spec("foo.pdf", dry_run=True, options={"inbox_dir": "C:/export"})
    assert "--inbox-dir" not in argv


def test_build_run_spec_no_inbox_dir_omits_flag():
    argv, env = build_run_spec("foo.pdf", dry_run=False, options={})
    assert "--inbox-dir" not in argv


def test_build_run_spec_inbox_dir_combines_with_other_options():
    argv, env = build_run_spec(
        "foo.pdf",
        dry_run=False,
        options={"backend": "litellm", "no_llm": True, "inbox_dir": "C:/export"},
    )
    assert "--no-llm" in argv
    assert "--inbox-dir" in argv
    assert argv[argv.index("--inbox-dir") + 1] == "C:/export"
    assert env == {"ATOMIC_AGENT_BACKEND": "litellm"}


# --- build_run_spec export_formats (F4: Export-Formatwahl als Lauf-Option) --


def test_build_run_spec_export_formats_appends_flags():
    argv, env = build_run_spec(
        "foo.pdf",
        dry_run=True,
        options={"export_formats": ["docx", "pdf"], "export_formats_dir": "C:/exports/run-1"},
    )
    assert "--export-format" in argv
    assert argv[argv.index("--export-format") + 1] == "docx,pdf"
    assert "--export-dir" in argv
    assert argv[argv.index("--export-dir") + 1] == "C:/exports/run-1"


def test_build_run_spec_export_formats_empty_list_omits_flags():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options={"export_formats": []})
    assert "--export-format" not in argv
    assert "--export-dir" not in argv


def test_build_run_spec_no_export_formats_omits_flags():
    argv, env = build_run_spec("foo.pdf", dry_run=True, options={})
    assert "--export-format" not in argv
    assert "--export-dir" not in argv


def test_build_run_spec_export_formats_without_dir_omits_export_dir_flag():
    # Sollte serverseitig nicht vorkommen (start_run setzt export_formats_dir
    # immer mit), aber die Funktion selbst darf nicht abstuerzen.
    argv, env = build_run_spec("foo.pdf", dry_run=True, options={"export_formats": ["docx"]})
    assert "--export-format" in argv
    assert "--export-dir" not in argv


def test_build_run_spec_export_formats_works_in_dry_run():
    # Anders als inbox_dir gilt export_formats AUCH im Dry-Run (json/portable-md/
    # docx/... brauchen keinen Vault-Schreib-Lauf, nur die Drafts).
    argv, env = build_run_spec(
        "foo.pdf",
        dry_run=True,
        options={"export_formats": ["json"], "export_formats_dir": "C:/exports/run-1"},
    )
    assert "--dry-run" in argv
    assert "--export-format" in argv
    assert argv[argv.index("--export-format") + 1] == "json"


# --- terminate_process_tree (#61: Cancel muss den ganzen Prozessbaum killen) --

# Windows-API-Konstanten fuer den Alive-Check unten (kein `os.kill(pid, 0)` --
# das ist unter Windows KEIN Existenz-Check, sondern killt die PID aktiv).
_WIN_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN_STILL_ACTIVE = 259

# Parent-Skript: spawnt selbst ein Kind, druckt dessen PID, schlaeft dann lang
# genug, um von `terminate_process_tree` waehrend des Schlafs erwischt zu werden.
_PARENT_SCRIPT = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(120)\n"
)


def _pid_alive(pid: int) -> bool:
    """Test-Helper: existiert der Prozess mit `pid` noch?

    POSIX: `os.kill(pid, 0)` sendet kein Signal, wirft aber `ProcessLookupError`,
    wenn die PID nicht mehr existiert -- das IST dort ein reiner Existenz-Check.
    Windows: `os.kill(pid, 0)` ist dagegen KEIN Existenz-Check, sondern ein
    duenner Wrapper um `TerminateProcess` (killt aktiv!) -- deshalb dort ueber
    die WinAPI (`OpenProcess` + `GetExitCodeProcess`).
    """
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(_WIN_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == _WIN_STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_line_with_timeout(stream, timeout: float) -> str | None:
    """Liest eine Zeile aus `stream` mit Timeout -- verhindert einen CI-Haenger,
    falls das Parent-Skript aus irgendeinem Grund nie druckt. `select`
    funktioniert unter Windows nicht auf Pipes, deshalb der Read in einem
    Hilfsthread, Rueckgabe ueber eine Queue."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def _reader() -> None:
        try:
            result.put(stream.readline())
        except Exception:
            result.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty:
        return None


def test_terminate_process_tree_kills_parent_and_child():
    """#61: Ein GUI-Cancel muss den KOMPLETTEN Prozessbaum beenden, nicht nur
    den direkten Orchestrator-Subprocess -- sonst laufen vom Orchestrator
    gespawnte Kinder (z.B. `claude -p`) als Waisen weiter. Echter Prozessbaum
    (kein Mock): ein Parent-Subprocess spawnt selbst ein Kind; nach
    `terminate_process_tree` muessen beide tot sein."""
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        # Eigene Prozessgruppe, analog `iter_run_events` -- sonst wuerde
        # `killpg()` weiter unten die GESAMTE Pytest-Prozessgruppe treffen.
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, "-c", _PARENT_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    child_pid: int | None = None
    try:
        line = _read_line_with_timeout(proc.stdout, timeout=10.0)
        assert line, "Parent-Skript hat keine Kind-PID gedruckt (Timeout)"
        child_pid = int(line.strip())
        assert _pid_alive(child_pid)

        terminate_process_tree(proc)

        assert proc.poll() is not None  # Parent beendet

        # Das Kind kann wenige ms nach dem Kill-Kommando noch leben -- pollen
        # statt sofort zu assertieren.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _pid_alive(child_pid):
            time.sleep(0.2)
        assert not _pid_alive(child_pid)  # Kind ebenfalls beendet, kein Waise
    finally:
        # Kein Test-Leak: falls Parent/Kind noch leben, hart aufraeumen.
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child_pid is not None and _pid_alive(child_pid):
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(child_pid), "/F"], capture_output=True)
            else:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_terminate_process_tree_noop_when_already_dead():
    """Bereits beendeter Prozess: darf nicht werfen. (Auf POSIX geht dabei ein
    ProcessLookupError-tolerantes `killpg` an die nicht mehr existente Gruppe --
    Absicherung der Leader-tot-Kinder-leben-Race, siehe Test unten.)"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    terminate_process_tree(proc)  # darf nicht werfen


# Parent-Skript fuer die Leader-tot-Race: spawnt ein Kind, druckt dessen PID
# und exitet SOFORT -- das Kind laeuft (in derselben Prozessgruppe) weiter.
_EXITING_PARENT_SCRIPT = (
    "import subprocess, sys\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "print(child.pid, flush=True)\n"
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="dokumentierte taskkill-/T-Grenze: tote Parent-Kette ist nicht mehr traversierbar",
)
def test_terminate_process_tree_kills_orphans_after_leader_exit():
    """#61-Race (Codex-Review-Fund): Orchestrator crasht/exitet, seine Kinder
    leben noch. Auf POSIX ueberlebt die Prozessgruppe ihren Leader --
    `terminate_process_tree` darf deshalb NICHT early-returnen, sondern muss
    die Gruppe (pgid == pid dank start_new_session) trotzdem signalisieren."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _EXITING_PARENT_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        line = _read_line_with_timeout(proc.stdout, timeout=10.0)
        assert line, "Parent-Skript hat keine Kind-PID gedruckt (Timeout)"
        child_pid = int(line.strip())
        proc.wait(timeout=10.0)  # Leader ist jetzt sicher tot, Kind lebt
        assert _pid_alive(child_pid)

        terminate_process_tree(proc)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _pid_alive(child_pid):
            time.sleep(0.2)
        assert not _pid_alive(child_pid)  # Waise trotz toten Leaders erwischt
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child_pid is not None and _pid_alive(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
