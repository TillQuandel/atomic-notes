"""Subprocess-Runner fuer die Live-GUI.

Faehrt `python -m generative.orchestrator …` als Subprocess und streamt dessen
stdout zeilenweise durch den RunParser zu strukturierten Events. Ein Subprocess
(statt In-Process-Aufruf) isoliert den Lauf sauber: der Orchestrator liest
VAULT/BACKEND beim Import aus ENV (config.py:7-12) und ruft `sys.exit()` —
beides unkritisch in einem eigenen Prozess.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator

from generative.gui.run_parser import RunParser


def build_argv(pdf_path: str, *, dry_run: bool, extra: list[str] | None = None) -> list[str]:
    argv = [sys.executable, "-m", "generative.orchestrator", "--source", pdf_path]
    if dry_run:
        argv.append("--dry-run")
    if extra:
        argv.extend(extra)
    return argv


def build_run_spec(
    pdf_path: str,
    *,
    dry_run: bool,
    options: dict | None = None,
    vault_path: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Uebersetzt bereits validierte Lauf-Einstellungen (P1) in argv + Env-Overrides.

    Erwartet normalisierte `options` (Server-seitig gegen die Whitelist geprueft,
    z.B. in `app.py:_validate_run_options`) — diese Funktion validiert selbst
    nicht, sie ist reine Uebersetzung. `options=None`/`{}` verhaelt sich exakt
    wie ein `build_argv`-Aufruf ohne Extras (Rueckwaertskompatibilitaet).

    `vault_path` (B2, Punkt 3): der zur Laufzeit in der GUI gewaehlte Vault --
    reine Uebersetzung in `ATOMIC_AGENT_VAULT_PATH`, keine Validierung hier
    (die passiert serverseitig in `app.py` vor dem Lauf-Start bzw. in
    `PUT /api/vault`). `None`/leer laesst den Subprocess das Standard-ENV
    (bzw. `.env`) erben -- unveraendertes Verhalten vor B2.

    `options["inbox_dir"]` (B3): nur im SCHREIB-Modus (`dry_run=False`) an
    `--inbox-dir` durchgereicht -- im Dry-Run schreibt vault_writer ohnehin nur
    eval-Kopien nach `.cache/eval/baseline`, unabhaengig von `--inbox-dir`
    (Orchestrator-Flag greift dort nicht). Reine Uebersetzung, keine Existenz-
    Validierung hier (die passiert serverseitig in `app.py:start_run`).

    `options["export_formats"]` (F4, Output-Projekt): anders als `inbox_dir`
    AUCH im Dry-Run wirksam -- json/portable-md/docx/... brauchen keinen
    Vault-Schreib-Lauf, nur die bereits erzeugten Drafts (s.
    `pipeline.export_runner.run_export`). `options["export_formats_dir"]` ist
    der von `app.py:start_run` server-seitig berechnete Session-Export-Ordner
    (Snapshot, analog `session.vault_path`) -- ohne ihn wird `--export-format`
    zwar gesetzt, aber ohne `--export-dir` (Orchestrator faellt dann auf
    seinen eigenen Default zurueck).
    """
    options = options or {}
    extra: list[str] = []
    if options.get("no_llm"):
        extra.append("--no-llm")
    if options.get("inbox_dir") and not dry_run:
        extra.extend(["--inbox-dir", str(options["inbox_dir"])])
    if options.get("export_formats"):
        extra.extend(["--export-format", ",".join(options["export_formats"])])
        if options.get("export_formats_dir"):
            extra.extend(["--export-dir", str(options["export_formats_dir"])])
    argv = build_argv(pdf_path, dry_run=dry_run, extra=extra or None)
    env_overrides: dict[str, str] = {}
    if options.get("backend"):
        env_overrides["ATOMIC_AGENT_BACKEND"] = options["backend"]
    if options.get("profile"):
        env_overrides["ATOMIC_AGENT_PROFILE"] = options["profile"]
    if vault_path:
        env_overrides["ATOMIC_AGENT_VAULT_PATH"] = str(vault_path)
    return argv, env_overrides


def iter_run_events(
    argv: list[str],
    *,
    env: dict | None = None,
    cwd: str | None = None,
    on_proc=None,
) -> Iterator[dict]:
    """Startet den Subprocess und yieldet geparste Events (inkl. started/error).

    on_proc: optionaler Callback, der mit dem Popen-Handle aufgerufen wird, sobald
    der Subprocess läuft — erlaubt dem Aufrufer (RunSession), den Lauf zu canceln.
    """
    yield {"type": "started", "argv": argv}
    run_env = {**os.environ, **(env or {})}
    # Unbuffered Python-Subprocess, damit stdout live ankommt; UTF-8 erzwingen
    # (Umlaute/⚠️ in den Notes-Titeln und Dry-Run-Flags).
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("PYTHONIOENCODING", "utf-8")
    # Markiert den Lauf als GUI-getrieben → der Orchestrator unterdrückt seine
    # schreibenden Auto-Aktionen (Version-Bump in config.py, Eval-Dashboard-Spawn
    # auf :8051). Ein Vorschau-Lauf darf weder Quellcode mutieren noch Prozesse leaken.
    run_env["ATOMIC_AGENT_GUI"] = "1"
    popen_kwargs: dict = {}
    if sys.platform != "win32":
        # Eigene Prozessgruppe (#61): `terminate_process_tree` killt auf POSIX
        # per `killpg` -- ohne eigene Gruppe wuerde das den GUI-Server-Prozess
        # (Elternteil dieses Popen) mittreffen, nicht nur diesen einen Lauf.
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=run_env,
        cwd=cwd,
        **popen_kwargs,
    )
    if on_proc is not None:
        on_proc(proc)
    parser = RunParser()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            for ev in parser.feed(line):
                yield ev
        for ev in parser.flush():
            yield ev
        rc = proc.wait()
        # Terminal-Event: IMMER `exited` (auch bei rc==0). `done` aus
        # `=== Fertig ===` ist NICHT das Ende — der Orchestrator druckt danach
        # noch Routing-Report + Stage-8-Eval. Erst `exited` schliesst den Stream.
        yield {"type": "exited", "returncode": rc}
    finally:
        # Generator vorzeitig geschlossen (SSE-Client trennt, Lauf abgebrochen):
        # Child (UND dessen eigene Kind-Prozesse, #61) nicht verwaisen lassen.
        terminate_process_tree(proc)


def terminate_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    """Beendet `proc` UND dessen Kind-Prozesse (best-effort).

    #61: Ein simples `proc.terminate()` beendet auf Windows nur den direkten
    Subprocess (`TerminateProcess` kennt keinen Prozessbaum) -- vom
    Orchestrator per asyncio gespawnte Kinder (`claude -p`) liefen als Waisen
    weiter. Windows: `taskkill /T /F` killt den kompletten PID-Baum. POSIX:
    `iter_run_events` startet `proc` mit `start_new_session=True` (eigene
    Prozessgruppe), `killpg` trifft daher nur diesen Lauf, nie den
    GUI-Server-Prozess selbst.

    Hinweis: `taskkill /T` findet nur die noch lebende Parent-Kette -- bereits
    verwaiste Enkel (deren direkter Parent schon tot ist) erwischt es nicht.
    Auf POSIX faengt der `killpg`-Pfad auch diesen Fall (die Gruppe ueberlebt
    ihren Leader). Bewusst kein Job-Object-Ansatz (keine neue Dependency).

    Voraussetzung: NUR fuer Popen-Handles aus `iter_run_events` gedacht --
    der POSIX-Pfad verlaesst sich auf `start_new_session=True` (pgid == pid);
    ein fremdes Popen ohne eigene Gruppe wuerde die Gruppe des Aufrufers
    (GUI-Server, pytest) mitsignalisieren.
    """
    if sys.platform == "win32":
        if proc.poll() is not None:
            return  # bereits beendet; verwaiste Enkel sind hier nicht auffindbar
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            pass  # haengendes taskkill darf den Cancel-Thread nicht blocken
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return
    # POSIX: KEIN Early-Return bei totem Leader -- die Prozessgruppe kann ihn
    # ueberleben (Orchestrator crasht, `claude -p`-Kinder laufen weiter).
    # `start_new_session=True` garantiert pgid == proc.pid, also direkt die
    # pid als Gruppen-Id nutzen statt getpgid() (wirft nach Reap des Leaders).
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # Gruppe komplett weg
    if proc.poll() is not None:
        return  # Leader war schon gereapt; SIGTERM an Restgruppe war best-effort
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
