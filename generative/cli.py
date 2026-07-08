"""atomic-notes — Konsolen-Entry-Point.

Subkommandos:
    run     generative Pipeline: PDF → geprüfte Atomic Notes (delegiert an
            generative.orchestrator; alle Orchestrator-Flags werden durchgereicht)
    doctor  Preflight-Checks: poppler, LLM-Backend, Vault-Pfad, optionale Deps
    gui     lokale Web-GUI: PDF wählen, Lauf live verfolgen, Notes-Preview
            (FastAPI; benötigt das Extra `[gui]`)

Der Orchestrator-Import passiert lazy im run-Zweig — `atomic-notes --help`
bleibt dadurch schnell und funktioniert auch ohne schwere Dependencies.
"""

from __future__ import annotations

import sys

from generative.ui_strings import msg


def _parse_gui_args(rest: list[str]) -> tuple[int, bool]:
    """`gui`-Argumente robust parsen. ValueError bei ungültigem `--port`."""
    open_browser = "--no-browser" not in rest
    port = 8052
    if "--port" in rest:
        idx = rest.index("--port")
        if idx + 1 >= len(rest):
            raise ValueError(msg("cli.port_expects_portnumber"))
        raw = rest[idx + 1]
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(msg("cli.port_expects_number", raw=raw))
        if not (1 <= port <= 65535):
            raise ValueError(msg("cli.port_out_of_range", port=port))
    return port, open_browser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        print(msg("cli.usage"))
        return 2
    if args[0] in ("-h", "--help"):
        print(msg("cli.usage"))
        return 0

    cmd, rest = args[0], args[1:]
    if cmd == "run":
        from generative import orchestrator

        return orchestrator.main(rest) or 0
    if cmd == "doctor":
        from generative import doctor

        return doctor.main()
    if cmd == "gui":
        try:
            port, open_browser = _parse_gui_args(rest)
        except ValueError as exc:
            print(msg("cli.invalid_argument", exc=exc), file=sys.stderr)
            return 2
        try:
            from generative.gui.app import serve
        except ImportError:
            print(msg("cli.gui_deps_missing"), file=sys.stderr)
            return 1
        serve(port=port, open_browser=open_browser)
        return 0

    print(msg("cli.usage"))
    print(msg("cli.unknown_command", cmd=cmd), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
