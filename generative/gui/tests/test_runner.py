"""Tests fuer den Subprocess-Runner der Live-GUI.

Echte Subprocesses (kein Mock): ein kleines `python -c`-Skript druckt
orchestrator-typische Marker-Zeilen, der Runner muss daraus die geparsten
Events streamen.
"""

import sys

from generative.gui.runner import build_argv, build_run_spec, iter_run_events


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
