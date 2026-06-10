"""Tests für ER-Stage-1-Blocking-Predikat (asymmetric subset + hub-generic).

Siehe `orchestrator.er_stage1_decision` und [[Atomic-Agent-Pipeline]].
Das Predikat entscheidet ob ein Title-Token-Paar in die teure Body-Cosine-Stage
darf. Zwei Filter gegen Hub-Sub-False-Positives:
    (a) asymmetrischer Subset mit |longer\\shorter| ≤ ER_MAX_TOKEN_DIFF
    (b) Hub-Generic-Blocklist: shorter darf nicht ⊆ ER_HUB_GENERIC_TOKENS sein
"""
from __future__ import annotations
import sys
from pathlib import Path


from generative.orchestrator import er_stage1_decision
from generative.agents.cross_reference import _tokens
from generative.config import ER_MAX_TOKEN_DIFF, ER_HUB_GENERIC_TOKENS


def _decide(title_a: str, title_b: str) -> tuple[str, int]:
    return er_stage1_decision(_tokens(title_a), _tokens(title_b))


# ---- skip-mono: zu wenig Signal -----------------------------------------

def test_mono_token_left_skipped():
    verdict, _ = _decide("HIB", "Human Information Behavior")
    assert verdict == "skip-mono"


def test_mono_token_right_skipped():
    verdict, _ = _decide("Information Search Process", "ISP")
    assert verdict == "skip-mono"


def test_both_mono_skipped():
    verdict, _ = _decide("HIB", "ISP")
    assert verdict == "skip-mono"


# ---- skip-no-subset: disjunkte Konzepte ---------------------------------

def test_disjoint_concepts_skipped():
    verdict, _ = _decide("Five Laws Library Science",
                         "Information Search Process")
    assert verdict == "skip-no-subset"


def test_partial_overlap_no_subset_skipped():
    # Schnittmenge aber keine Subset-Relation.
    verdict, _ = _decide("Information Search Process",
                         "Information Need Wilson")
    assert verdict == "skip-no-subset"


# ---- accept: Author-Suffix-Pattern (diff=1) -----------------------------

def test_author_suffix_accepted():
    # „Five Laws Library Science" ⊂ „Five Laws Library Science Bates" → diff=1
    verdict, diff = _decide("Five Laws Library Science",
                            "Five Laws Library Science Bates")
    assert verdict == "accept"
    assert diff == 1


def test_author_suffix_year_accepted():
    verdict, diff = _decide("ISP Modell Kuhlthau",
                            "ISP Modell Kuhlthau 1991")
    assert verdict == "accept"
    assert diff == 1


def test_specific_concept_with_author_accepted():
    # Spezifisches Sub-Konzept mit Author-Suffix — diff=1, kein Hub-Generic-Match.
    verdict, diff = _decide("Berrypicking Foraging Pattern",
                            "Berrypicking Foraging Pattern Bates")
    assert verdict == "accept"
    assert diff == 1


def test_hub_generic_shorter_blocked_even_with_author_suffix():
    # Author-Suffix-diff=1 reicht NICHT zum Akzeptieren wenn shorter komplett
    # hub-generic ist — Hub-Generic-Filter überstimmt den diff-Filter.
    verdict, diff = _decide("Information Behavior Concept",
                            "Information Behavior Concept Bates")
    assert verdict == "skip-hub-generic"
    assert diff == 1


# ---- skip-token-diff: Hub-Sub mit diff > 1 ------------------------------

def test_hub_sub_two_token_diff_skipped():
    # „Taylor Information Need" ⊂ „Taylor Vier Stufen Typologie Information Need"
    # → diff=2 (vier, stufen, typologie zusätzlich, aber „taylor information need"
    # hat 3 Tokens; longer hat 5+ → diff ≥ 2)
    verdict, diff = _decide("Taylor Information Need Model",
                            "Taylor Vier Stufen Typologie Information Need Model")
    assert verdict == "skip-token-diff"
    assert diff > ER_MAX_TOKEN_DIFF


def test_specialization_three_token_diff_skipped():
    verdict, diff = _decide("Information Behavior Bates",
                            "Information Behavior Bates Berrypicking Foraging Theory")
    assert verdict == "skip-token-diff"
    assert diff >= 3


# ---- skip-hub-generic: shorter ist reines Hub-Konzept -------------------

def test_information_need_shorter_blocked_as_hub_generic():
    # „Information Need" (Hub) ⊂ „Information Need Wilson" — diff=1, würde
    # ohne Hub-Generic-Filter durchrutschen. „information"+„need" sind beide
    # in ER_HUB_GENERIC_TOKENS.
    verdict, diff = _decide("Information Need", "Information Need Wilson")
    assert verdict == "skip-hub-generic"
    assert diff == 1


def test_information_behavior_shorter_blocked_as_hub_generic():
    verdict, _ = _decide("Information Behavior", "Information Behavior Bates")
    assert verdict == "skip-hub-generic"


def test_search_process_blocked_as_hub_generic():
    verdict, _ = _decide("Search Process", "Search Process Kuhlthau")
    assert verdict == "skip-hub-generic"


# ---- Negativ-Probe: nicht-generische shorter werden NICHT geblockt ------

def test_specific_shorter_not_blocked():
    # „Berrypicking Bates" ⊂ „Berrypicking Bates 1989" — „berrypicking" ist
    # NICHT in der Hub-Generic-Liste → accept.
    verdict, diff = _decide("Berrypicking Bates", "Berrypicking Bates 1989")
    assert verdict == "accept"
    assert diff == 1


def test_kuhlthau_isp_specific_not_blocked():
    # „ISP Kuhlthau Modell" ⊂ „ISP Kuhlthau Modell 1991" — keiner der Tokens
    # ist Hub-Generic.
    verdict, diff = _decide("Kuhlthau ISP Phasen", "Kuhlthau ISP Phasen Empirie")
    assert verdict == "accept"


# ---- Symmetrie: Reihenfolge egal ----------------------------------------

def test_decision_symmetric():
    # Reihenfolge der Argumente darf das verdict nicht ändern (außer bei
    # asymmetrischen Token-Counts welcher als shorter gilt — Filter aber
    # symmetrisch).
    v1, _ = _decide("Five Laws Library Science", "Five Laws Library Science Bates")
    v2, _ = _decide("Five Laws Library Science Bates", "Five Laws Library Science")
    assert v1 == v2


def test_hub_generic_symmetric():
    v1, _ = _decide("Information Need", "Information Need Wilson")
    v2, _ = _decide("Information Need Wilson", "Information Need")
    assert v1 == v2 == "skip-hub-generic"


# ---- Sanity: Config-Konstanten konsistent -------------------------------

def test_max_token_diff_default_is_one():
    # Tests sind gegen ER_MAX_TOKEN_DIFF=1 kalibriert (Author-Suffix erlaubt,
    # 2-Token-Spezialisierung blockiert). Wenn der Default geändert wird,
    # müssen die Tests neu kalibriert werden.
    assert ER_MAX_TOKEN_DIFF == 1


def test_hub_generic_contains_core_tokens():
    # Pflichtmitglieder der Hub-Generic-Liste — Rückversicherung gegen
    # versehentliches Entfernen.
    for t in ("information", "need", "behavior", "search", "process", "model"):
        assert t in ER_HUB_GENERIC_TOKENS, f"{t!r} fehlt in ER_HUB_GENERIC_TOKENS"
