# Documentation backlog

Deferred documentation work surfaced by a multi-agent doc review (2026-06-30:
fact-audit, fresh-clone onboarding test, editorial, adversarial/completeness).

These are **OSS-maturity** items — valuable once external contributors actually
arrive, but out of scope for the current "show it to a colleague + work better"
goal. The user-facing docs (README, ARCHITECTURE, CONTRIBUTING) are factually
accurate and onboarding-tested; this list is the next tier, not a correctness gap.

## Backlog (rough effort estimates)

- **TROUBLESHOOTING.md** — error catalogue: backend 401/429 (rate window + reset),
  `doctor` exit codes, poppler PATH, PDF encoding issues. (~2h, highest adoption value)
- **SECURITY.md / data-flow** — a short table of what each backend sends where
  (subscription = your Claude account; litellm = the provider API; extractive =
  local only) + a note that PDF text is not anonymized. Privacy is a selling point,
  so a brief version is worth more than the others. (~1.5h)
- **CONTRIBUTING: "developing generative stages"** — iterate without burning quota:
  use `--dry-run`, the LLM-free test suite + fixtures, `doctor` for window status.
  (~1.5h)
- **CONTRIBUTING: "writing tests"** — fixture patterns for LLM stages, the `slow`
  marker, where unit-test patterns live (`lib/decision_engine/tests`). (~1.5h)
- **CHANGELOG.md + release process** — note: packaging `version` (0.1.0) and the
  pipeline version constants (`AGENT_VERSION` v0.3.x / `EXTRACTIVE_VERSION` v0.2.0)
  are intentionally separate; document the scheme. (~2h)
- **CODE_STYLE.md** — function-size guidance, `_draft`/`_verified` naming, error
  handling (`sys.exit` only in CLI `main`). (~1h)
- **.github/ISSUE_TEMPLATE/ + pull_request_template.md** — backend type, repro with
  `doctor` output, test-coverage checklist. (~0.5h)
- **CALIBRATION.md** — how to add a label in `generative/calibration/labels-active/`
  and tune thresholds (links to `faithfulness-gate-plan.md`). (~1h)
- **ADR collection (`docs/adr/`)** — beyond the dual-pipeline ADR already in
  ARCHITECTURE.md: pdftotext-vs-pdfplumber, verifier+critic design. (~2h)

## Not doing (verified non-issues)

- "version 0.1.0 vs v0.3.x" is **not** a contradiction — packaging version vs.
  pipeline version constants; both match the code. (fact-audit, 2026-06-30)
