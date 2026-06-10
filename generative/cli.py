"""atomic-notes — Konsolen-Entry-Point.

Subkommandos:
    run     generative Pipeline: PDF → geprüfte Atomic Notes (delegiert an
            generative.orchestrator; alle Orchestrator-Flags werden durchgereicht)
    doctor  Preflight-Checks: poppler, LLM-Backend, Vault-Pfad, optionale Deps

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

    print(_USAGE)
    print(f"Unbekanntes Kommando: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
