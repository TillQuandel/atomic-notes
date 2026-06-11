# Contributing

## Development Setup

```bash
git clone https://github.com/TillQuandel/atomic-notes.git
cd atomic-notes
pip install -e .[dev]
```

The `dev` extra installs pytest and pytest-asyncio. If you only need the core
package without test dependencies, use `pip install -e .` instead.

## Running Tests

```bash
python -m pytest generative -q
```

The generative test suite (~430 tests) is LLM-free and runs in roughly 30 s.
CI runs on ubuntu-latest and windows-latest with poppler installed.

## TDD is the Project Norm

Write the test before writing the implementation. A failing test (RED) that
documents the intended behaviour is required before any new feature or bug fix
lands. Make it pass (GREEN), then refactor. PRs that skip the failing-test step
will be asked to add it.

## Branch Naming

- `feat/<short-description>` — new features
- `fix/<short-description>` — bug fixes

## Pull Requests

- Target the `master` branch.
- Keep each PR focused on a single change.
- Include a brief description of *why*, not just *what*.

## Issues

Use GitHub Issues for bug reports and feature requests. Include a minimal
reproduction for bugs.
