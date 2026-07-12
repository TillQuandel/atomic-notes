"""E2E-Baseline-Tests: prüft Vault-Quote auf den 3 Baseline-PDFs gegen Mindest-Schwellen.

Läuft NICHT im normalen pytest — nur mit:
    pytest -m slow tests/test_e2e_baseline.py

Voraussetzungen:
  - Baseline-PDFs in ~/OneDrive/Dokumente/Literatur/ vorhanden
  - Pipeline ohne --dry-run würde schreiben; hier immer --dry-run

Schwellen (absoluter Vault-Floor je PDF, kein Totalcheck — expected_total=None;
gelockert gegenüber v35, weil der Vault gewachsen ist → der Planner skippt mehr
Notes als „existing", die Vault-Quote sinkt entsprechend; siehe THRESHOLDS):
  - Schlebbe 2022: ≥ 5
  - Kuhlthau ISP:  ≥ 6
  - Bates 2017:    ≥ 2

Warnung: Jeder Run dauert ~15–30 min. Nur für Release-Checks und
nach größeren Refactors ausführen, nicht im Entwicklungs-Loop.
"""

from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent
from generative.config import LITERATURE_DIR as _LIT  # noqa: E402

BASELINE_PDFS = {
    "schlebbe": _LIT / "Schlebbe und Greifeneder - 2022 - Information Need, Informationsbedarf und -bedürfnis.pdf",
    "kuhlthau": _LIT / "Kuhlthau - INFORMATION SEARCH PROCESS.pdf",
    "bates": _LIT / "Bates - 2017 - Information Behavior.pdf",
}

# Mindest-Vault-Quote: (min_vault, expected_total).
# expected_total=None → nur min_vault als absoluter Floor, kein Totalcheck.
# Konservativ: Bestwert - 2 als Puffer für Sampling-Drift.
THRESHOLDS: dict[str, tuple[int, int | None]] = {
    "schlebbe": (5, None),  # Vault gewachsen seit v35 → Planner skippt mehr als existing; None = kein total-Check
    "kuhlthau": (6, None),  # analog
    "bates": (2, None),  # analog
}

_DRY_RUN_RE = re.compile(r"^\s*\[DRY-RUN\]")
_VAULT_SUMMARY_RE = re.compile(r"->\s+Vault:\s+(\d+)")
_TOTAL_SUMMARY_RE = re.compile(r"=== Fertig:\s+(\d+)\s+Notes")


def run_pipeline(pdf: Path, env_overrides: dict | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["ENABLE_NLI_VALIDATION"] = "1"
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "orchestrator.py", "--source", str(pdf), "--dry-run"],
        cwd=SCRIPTS_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,  # 60 min Hard-Timeout (Schlebbe ~17 Notes braucht bis 40 min)
    )
    return result.stdout + result.stderr


def parse_output(output: str) -> tuple[int, int]:
    """Returns (vault_count, total_count).

    In dry-run alle Notes → Inbox-Pfad; Vault-Count aus '=== Fertig ===' Zeile.
    """
    vault = 0
    total = 0
    for line in output.splitlines():
        m = _VAULT_SUMMARY_RE.search(line)
        if m:
            vault = int(m.group(1))
        m2 = _TOTAL_SUMMARY_RE.search(line)
        if m2:
            total = int(m2.group(1))
    # Fallback: DRY-RUN-Zeilen zählen wenn Fertig-Zeile fehlt
    if total == 0:
        total = sum(1 for line in output.splitlines() if _DRY_RUN_RE.match(line))
    return vault, total


@pytest.mark.slow
@pytest.mark.parametrize("key", ["schlebbe", "kuhlthau", "bates"])
def test_vault_quote_baseline(key: str, tmp_path):
    """Vault-Quote darf nicht unter Mindest-Schwelle fallen."""
    pdf = BASELINE_PDFS[key]
    if not pdf.exists():
        pytest.skip(f"Baseline-PDF nicht gefunden: {pdf}")

    # DB-Senke isolieren: dieser Test startet die Pipeline als SUBPROZESS, die
    # in-process-autouse-Fixture (generative/conftest.py) greift dort NICHT. db.DB_PATH
    # wird zur db.py-Import-Zeit aus ATOMIC_DB_PATH gelesen — im frischen Prozess also
    # bindbar. Ohne Override würde jeder Baseline-Lauf eine echte pipeline_runs-Zeile in
    # die produktive .cache/atomic_analytics.db schreiben (#198 Nachbesserung Fix 3).
    # Residuum (bewusst NICHT isoliert): Trace (.cache/runs), LLM-Cache (.cache/llm) und
    # Cache-Rotation kennen keine ENV-Steuerung; sie in tmp umzubiegen bräuchte einen
    # ATOMIC_CACHE_DIR-Umbau in config.py mit breitem Import-Rippel. Da dies ein ECHTER
    # Pipeline-Lauf ist (kein Fake-Backend), sind diese Trace-/Cache-Schreibvorgänge
    # legitime Laufdaten (rotations-gedeckelt), keine schädliche Test-Verschmutzung.
    min_vault, expected_total = THRESHOLDS[key]
    output = run_pipeline(pdf, env_overrides={"ATOMIC_DB_PATH": str(tmp_path / "e2e_analytics.db")})
    vault, total = parse_output(output)

    assert vault > 0, f"[{key}] Keine Notes extrahiert — Pipeline-Fehler?\n{output[-500:]}"
    assert vault >= min_vault, (
        f"[{key}] Vault-Quote {vault}/{total} unter Schwelle {min_vault}. "
        f"Mögliche Regression! Output-Tail:\n" + "\n".join(output.splitlines()[-30:])
    )
    if expected_total is not None:
        assert total == expected_total, (
            f"[{key}] Unerwartete Note-Anzahl: {total} statt {expected_total}. "
            f"Planner-Drift oder neue Halluzinations-Filter?"
        )
    print(f"[{key}] {vault}/{total} Vault ✓ (Schwelle: ≥{min_vault})")
