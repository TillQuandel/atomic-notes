from extractive.pipeline.sentence_extractor import find_concept_sentences, add_page_anchors, strip_anchors

TEXT = (
    "Information behavior is the total experience with information. "
    "Bates defines information behavior broadly including passive acquisition. "
    "Active seeking is also part of information behavior research. "
    "Many models exist for studying information behavior patterns."
)


def test_strip_anchors_removes_marker():
    assert strip_anchors("IB is broad. (S. 3)") == "IB is broad."


def test_strip_anchors_no_change():
    assert strip_anchors("No anchor here.") == "No anchor here."


def test_find_sentences_finds_matches():
    sents = find_concept_sentences("information behavior", TEXT)
    assert len(sents) >= 2
    assert any("information behavior" in s.lower() for s in sents)


def test_add_page_anchors_appends():
    result = add_page_anchors(["IB is broad.", "Bates defines it."], [1, 2])
    assert "(S. 1)" in result[0]
    assert "(S. 2)" in result[1]


def test_add_page_anchors_strips_existing():
    result = add_page_anchors(["IB is broad. (S. 99)"], [1])
    assert "(S. 99)" not in result[0]
    assert "(S. 1)" in result[0]


def test_add_page_anchors_single_page():
    result = add_page_anchors(["Sent 1.", "Sent 2.", "Sent 3."], [5])
    assert all("(S. 5)" in s for s in result)


def test_extract_body_returns_empty_when_no_concept_sentences():
    """Kein Fallback auf ganzen Text wenn Konzept nicht vorkommt."""
    from extractive.pipeline.sentence_extractor import extract_body_for_concept

    text = "This sentence is about cats. Another sentence about dogs. A third about fish."
    result = extract_body_for_concept("information literacy", text)
    assert result == [], f"Expected [], got {result}"
