#!/usr/bin/env python3
"""End-to-end demo on the bundled example PDF.

Runs the generative pipeline in --dry-run (no files written) on
examples/zettelkasten-primer.pdf so a fresh clone can see the pipeline work in
one command:  python scripts/demo.py

Works with the default dev setup (`uv sync --extra dev`) — no extractive/GLiNER
deps needed. It DOES call the configured LLM backend (uses Claude-subscription
quota by default); see README "Configure backend" if it reports a backend error.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "examples" / "zettelkasten-primer.pdf"


def main() -> None:
    if not PDF.is_file():
        sys.exit(f"FEHLER: Beispiel-PDF fehlt: {PDF}")

    print(f"Demo: generative Pipeline (--dry-run) auf {PDF.name}\nEs wird NICHTS geschrieben.\n")

    cmd = ["uv", "run", "atomic-notes", "run", "--source", str(PDF), "--dry-run"]
    try:
        # Backend-Precheck: doctor faengt fehlendes CLI/Backend ab, bevor die halbe
        # Pipeline laeuft und tief im Stacktrace stirbt (frischer Clone ohne Login).
        if subprocess.run(["uv", "run", "atomic-notes", "doctor"], cwd=ROOT).returncode != 0:
            sys.exit(
                "\nBackend nicht bereit (siehe doctor-Ausgabe oben). "
                "Setup: README -> 'Configure backend'. Demo abgebrochen."
            )
        raise SystemExit(subprocess.run(cmd, cwd=ROOT).returncode)
    except FileNotFoundError:
        sys.exit("FEHLER: 'uv' nicht gefunden — Setup siehe README/CONTRIBUTING.")


if __name__ == "__main__":
    main()
