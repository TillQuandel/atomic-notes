from extractive.eval.extractive_eval import compute_anchor_rate, compute_hallucination_rate

def test_anchor_rate_all_anchored():
    assert compute_anchor_rate(["IB is broad. (S. 1)", "Bates defines it. (S. 2)"]) == 1.0

def test_anchor_rate_partial():
    assert compute_anchor_rate(["IB is broad. (S. 1)", "No anchor."]) == 0.5

def test_anchor_rate_empty():
    assert compute_anchor_rate([]) == 0.0

def test_hallucination_zero_for_verbatim():
    sents = ["Information behavior is the total experience."]
    fulltext = "Information behavior is the total experience. It includes seeking and use."
    assert compute_hallucination_rate(sents, fulltext) == 0.0

def test_hallucination_one_for_invented():
    sents = ["Quantum teleportation invented by Bates in 1992."]
    fulltext = "Information behavior is the total experience."
    assert compute_hallucination_rate(sents, fulltext) == 1.0

def test_hallucination_skips_short_sentences():
    sents = ["IB.", "Very short."]
    fulltext = "completely different text here"
    # Kurze Saetze (< 10 Zeichen nach anchor-strip) werden uebersprungen
    rate = compute_hallucination_rate(sents, fulltext)
    assert rate == 0.0
