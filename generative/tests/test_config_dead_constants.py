"""Dead-Constant-Guard (#103): jede Modul-Konstante in generative/config.py muss
entweder mindestens einen Konsumenten außerhalb von config.py haben ODER im
zugehörigen Quelltext-Kommentar das Wort „Backlog" tragen (ehrliche Markierung
für bewusst-nicht-verdrahtete Stellschrauben).

Fängt die #103-Klasse: tote Konfiguration täuscht Stellschrauben vor und falsche
Kommentare kosten Verifikationszeit. Der Guard erzwingt eine Entscheidung pro
Konstante — einlösen (Consumer bauen), löschen oder als Backlog markieren.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[1] / "config.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Quell-Pakete, in denen echte Konsumenten leben (schnell + deterministisch;
# .venv/.cache/__pycache__ bleiben außen vor).
_SOURCE_DIRS = ("generative", "extractive", "lib", "shared", "decision_engine")


def _module_constants() -> list[tuple[str, int]]:
    """Modul-Level GROSS_KONSTANTEN aus config.py (Name, Zeile). `_`-Präfix (privat)
    ausgeschlossen."""
    tree = ast.parse(_CONFIG.read_text(encoding="utf-8"))
    out: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                    out.append((target.id, node.lineno))
    return out


def _consumer_files() -> list[Path]:
    files: list[Path] = []
    for d in _SOURCE_DIRS:
        root = _REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p.resolve() == _CONFIG:
                continue
            files.append(p)
    return files


def _has_external_use(name: str, sources: list[str]) -> bool:
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    return any(pat.search(src) for src in sources)


def _has_backlog_marker(name: str, lineno: int, lines: list[str]) -> bool:
    """True, wenn die Definitionszeile ODER der direkt darüberstehende
    zusammenhängende Kommentarblock das Wort „Backlog" enthält."""
    # Definitionszeile selbst (Trailing-Kommentar).
    idx = lineno - 1
    if "backlog" in lines[idx].lower():
        return True
    # Kontinuierlicher Kommentarblock direkt darüber.
    j = idx - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        if "backlog" in lines[j].lower():
            return True
        j -= 1
    return False


def test_no_dead_config_constants() -> None:
    lines = _CONFIG.read_text(encoding="utf-8").splitlines()
    sources = [p.read_text(encoding="utf-8", errors="ignore") for p in _consumer_files()]

    offenders: list[str] = []
    for name, lineno in _module_constants():
        if _has_external_use(name, sources):
            continue
        if _has_backlog_marker(name, lineno, lines):
            continue
        offenders.append(f"{name} (config.py:{lineno})")

    assert not offenders, (
        "Tote config.py-Konstante(n) ohne Konsument und ohne Backlog-Markierung "
        "(#103): einlösen, löschen oder als Backlog markieren:\n  " + "\n  ".join(offenders)
    )
