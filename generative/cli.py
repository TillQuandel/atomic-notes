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
        port = 8052
        open_browser = True
        if "--no-browser" in rest:
            open_browser = False
        if "--port" in rest:
            port = int(rest[rest.index("--port") + 1])
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
