# decision_engine

Domain-agnostic rule pipeline for claim decisions. The module knows only abstract labels,
retrieval confidence, evidence verification state, parse status, and optional audit labels.
It does not import `atomic-agent` code and does not know about PDFs, prompts, vault notes, or
JSONL persistence.

## Public API

```python
from decision_engine import ClaimInput, Label, determine_decision

inp = ClaimInput(
    primary_label=Label.SUPPORTED_PARAPHRASE,
    audit_label=Label.CONTRADICTED,
    cosine=0.82,
    evidence_verified=True,
    parse_failed=False,
)

decision = determine_decision(inp)
```

`decision.label` is the final label, `decision.flags` explains rule side effects, and
`decision.source` is one of `primary`, `audit_override`, `system`, or `downgrade`.

## Rule Order

1. Parse failures become `parse_error`.
2. Low cosine becomes `retrieval_or_parse_uncertain`.
3. Unverified exact/paraphrase evidence is downgraded.
4. Stricter normal audit labels override primary labels.
5. Otherwise the primary label is kept.

System labels sit outside the strictness order and are never audit-overridden.

## Tests

Run from this directory:

```powershell
python -m pytest tests/ -v
```

Tests use Hypothesis when installed. In minimal environments they fall back to deterministic
ADT combination coverage.
