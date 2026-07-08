"""Schichtungs-Test: gerichtete agents→pipeline-Kopplung erzwingen (#153).

Verhindert die Rückkehr der bidirektionalen Kopplung, die vor #153 bestand:

- agents-Module dürfen `generative.pipeline` NICHT auf Modul-Ebene importieren.
  Lazy Imports in Funktions-/Methodenkörpern sind erlaubt — sie brechen keinen
  Import-Zyklus und werden bewusst so gehalten (Backend-/Pipeline-Aufrufe erst
  zur Laufzeit).
- Über die agents↔pipeline-Grenze dürfen keine privaten Namen (führender `_`)
  importiert werden (Privat-Import bricht bei interner Umbenennung ohne
  Deprecation-Signal — Auslöser: `_parse_filename_fallback`,
  `_has_overview_marker`).

AST-basiert, damit lazy Imports (im Funktionskörper) sauber von Modul-Level-
Imports unterschieden werden.
"""

from __future__ import annotations

import ast
from pathlib import Path

_GENERATIVE = Path(__file__).resolve().parent.parent
_AGENTS_DIR = _GENERATIVE / "agents"
_PIPELINE_DIR = _GENERATIVE / "pipeline"


def _py_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _targets_pkg(module: str | None, pkg: str) -> bool:
    module = module or ""
    return module == pkg or module.startswith(pkg + ".")


def _module_level_imports(tree: ast.Module):
    """Nur Import-Nodes direkt im Modulkörper — nicht in Funktionen/Klassen."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def test_agents_no_module_level_pipeline_import():
    """agents darf pipeline nicht top-level importieren (lazy im Funktionskörper ok)."""
    offenders: list[str] = []
    for f in _py_files(_AGENTS_DIR):
        for node in _module_level_imports(_parse(f)):
            if isinstance(node, ast.ImportFrom) and _targets_pkg(node.module, "generative.pipeline"):
                offenders.append(f"{f.name}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(_targets_pkg(a.name, "generative.pipeline") for a in node.names):
                offenders.append(f"{f.name}:{node.lineno}")
    assert not offenders, "Modul-Level agents→pipeline-Importe gefunden (lazy im Funktionskörper nutzen): " + ", ".join(
        offenders
    )


def test_no_private_imports_across_agents_pipeline_boundary():
    """Kein `from … import _x` über die agents↔pipeline-Grenze (beide Richtungen)."""
    offenders: list[str] = []
    for f in _py_files(_AGENTS_DIR):
        for node in ast.walk(_parse(f)):
            if isinstance(node, ast.ImportFrom) and _targets_pkg(node.module, "generative.pipeline"):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        offenders.append(f"agents/{f.name}:{node.lineno} → pipeline.{alias.name}")
    for f in _py_files(_PIPELINE_DIR):
        for node in ast.walk(_parse(f)):
            if isinstance(node, ast.ImportFrom) and _targets_pkg(node.module, "generative.agents"):
                for alias in node.names:
                    if alias.name.startswith("_"):
                        offenders.append(f"pipeline/{f.name}:{node.lineno} → agents.{alias.name}")
    assert not offenders, "Privat-Importe über die agents↔pipeline-Grenze gefunden: " + ", ".join(offenders)
