# tests/extractive/test_e2e.py
"""E2E-Akzeptanztests auf Bates 2017 (EN, 12 Seiten)."""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

BATES = Path(__file__).parent.parent / "fixtures" / "bates-2017.pdf"
pytestmark = pytest.mark.skipif(not BATES.exists(), reason="Bates PDF fehlt in tests/fixtures/")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def out_dir():
    """Eigenes tmpdir — umgeht pytest-asyncio STRICT-mode Bug mit tmp_path."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _run_pipeline(tmp_path, extra_args=None):
    cmd = [
        "python", str(REPO / "extractive" / "orchestrator.py"),
        "--source", str(BATES),
        "--out-dir", str(tmp_path),
        "--eval-jsonl", str(tmp_path / "eval.jsonl"),
    ] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


def test_pipeline_exits_zero(out_dir):
    r = _run_pipeline(out_dir)
    assert r.returncode == 0, f"Pipeline fehlgeschlagen:\n{r.stderr}"


def test_produces_min_3_notes(out_dir):
    r = _run_pipeline(out_dir)
    assert r.returncode == 0, r.stderr
    notes = list(out_dir.glob("*.md"))
    assert len(notes) >= 3, f"Nur {len(notes)} Notes erzeugt — erwartet >= 3"


def test_runs_within_600s(out_dir):
    t0 = time.time()
    r = _run_pipeline(out_dir)
    elapsed = time.time() - t0
    assert r.returncode == 0, r.stderr
    assert elapsed < 600, f"Pipeline zu langsam: {elapsed:.0f}s (max 600s)"


def test_zero_hallucination(out_dir):
    r = _run_pipeline(out_dir)
    assert r.returncode == 0, r.stderr
    jl = out_dir / "eval.jsonl"
    assert jl.exists(), "eval.jsonl nicht erzeugt"
    rates = [json.loads(l)["hallucination_rate"] for l in jl.read_text().splitlines() if l.strip()]
    assert rates, "Keine Eval-Ergebnisse"
    assert all(r == 0.0 for r in rates), f"Halluzinationen gefunden: {rates}"


def test_anchor_rate_above_80(out_dir):
    r = _run_pipeline(out_dir)
    assert r.returncode == 0, r.stderr
    jl = out_dir / "eval.jsonl"
    rates = [json.loads(l)["anchor_rate"] for l in jl.read_text().splitlines() if l.strip()]
    if not rates:
        pytest.skip("Keine Eval-Ergebnisse")
    avg = sum(rates) / len(rates)
    assert avg >= 0.8, f"Anchor-Rate zu niedrig: {avg:.2f} (min 0.8)"


def test_output_is_valid_obsidian_markdown(out_dir):
    _run_pipeline(out_dir)
    notes = list(out_dir.glob("*.md"))
    if not notes:
        pytest.skip("Keine Notes erzeugt")
    for note in notes:
        content = note.read_text(encoding="utf-8")
        assert content.startswith("---"), f"{note.name}: fehlt Frontmatter"
        assert "title:" in content, f"{note.name}: fehlt title im Frontmatter"
        assert "## Quellen" in content, f"{note.name}: fehlt ## Quellen"
