# Contributing

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/). Dependencies are pinned in
`uv.lock` (resolved for Windows, Linux, and macOS). On Windows/Linux `torch` is
mapped to the CPU wheel index; on macOS it comes from PyPI (CPU-only there anyway).

```bash
git clone https://github.com/TillQuandel/atomic-notes.git
cd atomic-notes
uv sync --extra dev        # core + test deps, from the lockfile
# or:  python scripts/setup.py   (uv sync + preflight doctor)
```

Run everything through the environment with `uv run <cmd>` — no manual venv
activation. The `dev` extra installs pytest, pytest-asyncio, the OTel SDK, and the
FastAPI test stack. The heavy GLiNER/torch stack for the extractive pipeline lives
in the separate `extractive` extra (`uv sync --extra extractive`) and is **not**
installed by default; the canonical test suite guards against its absence.

## Running Tests

```bash
uv run python -m pytest generative lib/decision_engine/tests -q
```

This is the canonical, LLM-free suite (mirrors CI on ubuntu + windows + macOS). It
runs in a couple of minutes locally. Format check:

```bash
uv run ruff format --check .       # CI gate; `uv run ruff format .` to apply
```

Only formatting is enforced for now; the `ruff` linter (`ruff check`) is
intentionally deferred and not part of CI yet.

## ML notes (model cache & slow tests)

The pipelines load sentence-transformer / HuggingFace models. To avoid
re-downloading several GB per checkout or per branch, point HuggingFace at one
shared cache:

```bash
export HF_HOME="$HOME/.cache/huggingface"
# Windows (PowerShell):  setx HF_HOME "$env:USERPROFILE\.cache\huggingface"
```

Tests that exercise the full pipeline or load heavy models are marked `slow` and
excluded from the default dev loop. Run only the fast suite, or include slow tests
explicitly:

```bash
uv run python -m pytest -m "not slow" generative -q   # fast (default in dev)
uv run python -m pytest generative -q                 # everything
```

GPU override (optional, local only): the lockfile pins CPU `torch` (Win/Linux). To
use a CUDA build, install it outside the locked sync:
`uv pip install torch --index https://download.pytorch.org/whl/cu121 --reinstall`.
This **dirties the venv** relative to `uv.lock` — run `uv sync` to revert to the
locked CPU build. (Not applicable on macOS, which has no CUDA.)

## TDD is the Project Norm

Write the test before writing the implementation. A failing test (RED) that
documents the intended behaviour is required before any new feature or bug fix
lands. Make it pass (GREEN), then refactor. PRs that skip the failing-test step
will be asked to add it.

## Branch Naming

- `feat/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `chore/<short-description>` / `build/<short-description>` — tooling, deps, CI

## Pull Requests

- Target the `master` branch.
- Keep each PR focused on a single change.
- Include a brief description of *why*, not just *what*.
- CI (tests + `ruff format --check`) must be green on ubuntu, windows **and** macOS.

## Issues

Use GitHub Issues for bug reports and feature requests. Include a minimal
reproduction for bugs.
