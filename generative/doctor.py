"""atomic-notes doctor — Preflight-Checks mit konkretem Installationshinweis pro Fehlschlag.

Checks: poppler-Tools (pdftotext/pdfinfo), LLM-Backend (subscription: claude-CLI
vorhanden + eingeloggt; litellm: API-Key gesetzt), Vault-Pfad, optionale Imports.
Alle Checks sind pure Funktionen mit injizierbaren Abhängigkeiten (testbar ohne System).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

_POPPLER_HINT = (
    "poppler installieren: Ubuntu/Debian `sudo apt install poppler-utils`, "
    "Windows `choco install poppler` (oder scoop), macOS `brew install poppler` — "
    "danach neue Shell öffnen, damit PATH greift."
)

# litellm liest die üblichen Provider-Keys aus der Umgebung
_LITELLM_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_API_BASE")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def check_tool(tool: str, which: Callable[[str], str | None] = shutil.which) -> CheckResult:
    """Poppler-Werkzeug (pdftotext/pdfinfo) im PATH?"""
    path = which(tool)
    if path:
        return CheckResult(name=tool, ok=True, detail=f"{tool}: {path}")
    return CheckResult(name=tool, ok=False, detail=f"{tool} nicht im PATH", hint=_POPPLER_HINT)


def check_backend(
    backend: str,
    which: Callable[[str], str | None] = shutil.which,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> CheckResult:
    """LLM-Backend erreichbar? subscription: CLI + Login-Heuristik; litellm: Key gesetzt."""
    home = Path.home() if home is None else home
    env = os.environ if env is None else env

    if backend == "subscription":
        from generative.config import CLAUDE_BIN

        cli = which(CLAUDE_BIN)
        if not cli:
            return CheckResult(
                name="backend (subscription)", ok=False,
                detail=f"claude-CLI '{CLAUDE_BIN}' nicht im PATH",
                hint=(
                    "Claude-Code-CLI installieren: `npm install -g @anthropic-ai/claude-code`, "
                    "danach einmal `claude` starten und einloggen (Pro/Max-Abo). "
                    "Alternative ohne CLI: ATOMIC_AGENT_BACKEND=litellm + API-Key."
                ),
            )
        credentials = home / ".claude" / ".credentials.json"
        if not credentials.exists():
            return CheckResult(
                name="backend (subscription)", ok=False,
                detail=f"claude-CLI gefunden ({cli}), aber {credentials} fehlt",
                hint=(
                    "Vermutlich nicht eingeloggt: einmal `claude` starten und den "
                    "Login durchlaufen. (Heuristik — falls anders authentifiziert, "
                    "Check mit einem echten Lauf gegenprüfen.)"
                ),
            )
        return CheckResult(
            name="backend (subscription)", ok=True,
            detail=f"claude-CLI: {cli}, Credentials vorhanden",
        )

    if backend == "litellm":
        set_vars = [v for v in _LITELLM_KEY_VARS if env.get(v)]
        if not set_vars:
            return CheckResult(
                name="backend (litellm)", ok=False,
                detail="kein Provider-Key in der Umgebung",
                hint=(
                    "API-Key setzen, z. B. ANTHROPIC_API_KEY oder OPENAI_API_KEY "
                    "(in .env oder Umgebung). Geprüft: " + ", ".join(_LITELLM_KEY_VARS)
                ),
            )
        return CheckResult(
            name="backend (litellm)", ok=True, detail="gesetzt: " + ", ".join(set_vars)
        )

    return CheckResult(
        name=f"backend ({backend})", ok=False,
        detail="unbekannter Backend-Wert",
        hint="ATOMIC_AGENT_BACKEND auf 'subscription' (Default) oder 'litellm' setzen.",
    )


def check_vault(vault: Path) -> CheckResult:
    """Vault-Pfad vorhanden und beschreibbar? (Schreibprobe statt os.access — Windows.)"""
    if not vault.is_dir():
        return CheckResult(
            name="vault", ok=False,
            detail=f"Vault-Pfad existiert nicht: {vault}",
            hint=(
                "ATOMIC_AGENT_VAULT_PATH auf den Obsidian-Vault (oder einen "
                "beliebigen Zielordner) setzen — in .env oder Umgebung."
            ),
        )
    probe = vault / ".atomic-notes-doctor-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return CheckResult(
            name="vault", ok=False,
            detail=f"Vault nicht beschreibbar: {vault} ({e})",
            hint="Schreibrechte des Ordners prüfen.",
        )
    return CheckResult(name="vault", ok=True, detail=str(vault))


def check_import(module: str, hint: str) -> CheckResult:
    """Modul installiert? (find_spec — importiert nicht, bleibt schnell.)"""
    if importlib.util.find_spec(module) is not None:
        return CheckResult(name=module, ok=True, detail=f"{module} installiert")
    return CheckResult(name=module, ok=False, detail=f"{module} fehlt", hint=hint)


def run_all() -> list[CheckResult]:
    from generative.config import BACKEND, VAULT

    results = [
        check_tool("pdftotext"),
        check_tool("pdfinfo"),
        check_backend(BACKEND),
        check_vault(VAULT),
        check_import("pypdf", "pip install pypdf (PDF-Metadaten-Enrichment)"),
        check_import("sentence_transformers",
                     "pip install sentence-transformers (Embeddings/Entity-Resolution)"),
    ]
    if BACKEND == "litellm":
        results.append(check_import("litellm", "pip install litellm"))
    return results


def main() -> int:
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        mark = "OK " if r.ok else "FEHLT"
        print(f"[{mark:5}] {r.name:<{width}}  {r.detail}")
        if not r.ok and r.hint:
            print(f"        -> {r.hint}")
    failed = [r for r in results if not r.ok]
    print()
    if failed:
        print(f"doctor: {len(failed)} von {len(results)} Checks fehlgeschlagen.")
        return 1
    print(f"doctor: alle {len(results)} Checks ok.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
