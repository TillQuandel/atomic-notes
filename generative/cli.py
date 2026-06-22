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

_USAGE = """\
atomic-notes — PDF → geprüfte Atomic Notes für Obsidian

Verwendung:
  atomic-notes run --source <pdf> [Orchestrator-Flags]   Pipeline starten
  atomic-notes run --help                                alle Pipeline-Flags
  atomic-notes doctor                                    Preflight-Checks
  atomic-notes gui [--port N] [--no-browser]             lokale Web-GUI

Umgebung:
  ATOMIC_AGENT_BACKEND   LLM-Backend: 'subscription' (Default, Claude-Code-CLI,
                         kein API-Key) oder 'litellm' (Anthropic/OpenAI per
                         API-Key; lokales Ollama via OLLAMA_API_BASE).
                         Setup prüfen mit: atomic-notes doctor
"""


def _parse_gui_args(rest: list[str]) -> tuple[int, bool]:
    """`gui`-Argumente robust parsen. ValueError bei ungültigem `--port`."""
    open_browser = "--no-browser" not in rest
    port = 8052
    if "--port" in rest:
        idx = rest.index("--port")
        if idx + 1 >= len(rest):
            raise ValueError("--port erwartet eine Portnummer")
        raw = rest[idx + 1]
        try:
            port = int(raw)
        except ValueError:
            raise ValueError(f"--port erwartet eine Zahl, nicht '{raw}'")
        if not (1 <= port <= 65535):
            raise ValueError(f"--port außerhalb 1–65535: {port}")
    return port, open_browser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        print(_USAGE)
        return 2
    if args[0] in ("-h", "--help"):
        print(_USAGE)
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
            print(f"Ungültiges Argument: {exc}", file=sys.stderr)
            return 2
        try:
            from generative.gui.app import serve
        except ImportError:
            print("GUI-Dependencies fehlen. Installation: pip install -e '.[gui]'",
                  file=sys.stderr)
            return 1
        serve(port=port, open_browser=open_browser)
        return 0

    print(_USAGE)
    print(f"Unbekanntes Kommando: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
