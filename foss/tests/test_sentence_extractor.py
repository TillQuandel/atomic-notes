def test_extract_body_returns_empty_when_no_concept_sentences():
    """Kein Fallback auf ganzen Text wenn Konzept nicht vorkommt."""
    from foss.pipeline.sentence_extractor import extract_body_for_concept
    text = "This sentence is about cats. Another sentence about dogs. A third about fish."
    result = extract_body_for_concept("information literacy", text)
    assert result == [], f"Expected [], got {result}"
