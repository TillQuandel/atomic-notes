"""Tests für claim-level Agreement in collect.py (#24).

Der alte Code rechnete `agreement_rate` auf Aggregat-Halluzinationsraten pro Note
(`1 - |human_hall - llm_hall|`) — zwei Notes mit gleicher Rate zeigten 100%
"Agreement", auch wenn sie bei keinem einzelnen Claim übereinstimmten.
`note_claim_agreement` paart stattdessen Mensch- und Pipeline-Label pro claim_idx.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.collect import note_claim_agreement


def test_full_agreement():
    human = {0: "s", 1: "h", 2: "s"}
    pipeline = {("n", 0): "s", ("n", 1): "h", ("n", 2): "s"}
    assert note_claim_agreement(human, pipeline, "n") == 1.0


def test_regression_same_rate_but_zero_claim_agreement():
    # Beide 50% Halluzinationsrate → alte Aggregat-Logik: 100% "Agreement".
    # Claim-level: bei KEINEM Claim einig → 0.0. Das ist der Bug-Beweis.
    human = {0: "s", 1: "h"}
    pipeline = {("n", 0): "h", ("n", 1): "s"}
    assert note_claim_agreement(human, pipeline, "n") == 0.0


def test_partial_agreement_rounded():
    human = {0: "s", 1: "h", 2: "s"}
    pipeline = {("n", 0): "s", ("n", 1): "s", ("n", 2): "s"}  # 2/3 einig
    assert note_claim_agreement(human, pipeline, "n") == 0.6667


def test_no_overlap_returns_none():
    human = {0: "s", 1: "h"}
    pipeline = {("other", 0): "s"}
    assert note_claim_agreement(human, pipeline, "n") is None


def test_only_shared_claims_counted():
    # Pipeline hat claim 2 nicht → nur 0 und 1 paaren.
    human = {0: "s", 1: "h", 2: "s"}
    pipeline = {("n", 0): "s", ("n", 1): "h"}
    assert note_claim_agreement(human, pipeline, "n") == 1.0
