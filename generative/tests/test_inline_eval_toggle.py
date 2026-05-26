from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import inline_eval_enabled


def test_inline_eval_enabled_by_default():
    assert inline_eval_enabled({}) is True


def test_inline_eval_disabled_only_by_zero():
    assert inline_eval_enabled({"ATOMIC_AGENT_INLINE_EVAL": "0"}) is False
    assert inline_eval_enabled({"ATOMIC_AGENT_INLINE_EVAL": "1"}) is True
    assert inline_eval_enabled({"ATOMIC_AGENT_INLINE_EVAL": "false"}) is True
