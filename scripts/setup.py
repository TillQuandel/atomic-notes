#!/usr/bin/env python3
"""Lokales Dev-Setup: uv-Umgebung aus dem Lockfile synchronisieren + doctor.

Cross-platform (Windows + Linux). Loescht das .venv NICHT (uv synct in-place),
bricht aber ab, wenn man versehentlich aus dem .venv heraus startet.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"


def main() -> None:
    exe = Path(sys.executable).resolve()
    if VENV == exe.parent.parent or VENV in exe.parents:
        sys.exit(
            "FEHLER: aus dem .venv heraus gestartet. Mit System-Python ausfuehren "
            "(vorher 'deactivate') — sonst Datei-Locks beim Synchronisieren (Windows)."
        )
    if not shutil.which("uv"):
        sys.exit("FEHLER: uv nicht gefunden. Installation: https://docs.astral.sh/uv/")

    print("-> uv sync --extra dev")
    subprocess.run(["uv", "sync", "--extra", "dev"], cwd=ROOT, check=True)

    print("-> doctor (Preflight)")
    subprocess.run(["uv", "run", "python", "-m", "generative.cli", "doctor"], cwd=ROOT, check=True)

    print("\nSetup fertig. Tools laufen via 'uv run <cmd>'.")


if __name__ == "__main__":
    main()
