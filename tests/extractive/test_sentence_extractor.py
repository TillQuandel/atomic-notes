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


def test_sumy_language_maps_detected_code():
    """Fix #167b: langdetect-ISO-Code -> sumy-Tokenizer-Sprachname, Fallback english."""
    from extractive.pipeline.sentence_extractor import sumy_language

    assert sumy_language("en") == "english"
    assert sumy_language("de") == "german"
    assert sumy_language("unknown") == "english"
    assert sumy_language("") == "english"


def test_map_sentences_to_pages_uses_physical_page():
    """Fix #167c: ein Satz von physischer Seite 2 bekommt Seite 2 (nicht die
    Chunk-Startseite), aufgeloest ueber den vorhandenen Seitenindex."""
    from extractive.pipeline.sentence_extractor import map_sentences_to_pages

    page_texts = [
        "Alpha concept is introduced on the very first page of the document here.",
        "Beta discussion continues about the alpha concept on the second page exclusively.",
    ]
    pages = map_sentences_to_pages(
        ["Beta discussion continues about the alpha concept on the second page exclusively."],
        page_texts,
        fallback_page=1,
    )
    assert pages == [2]


def test_map_sentences_to_pages_first_page_and_fallback():
    from extractive.pipeline.sentence_extractor import map_sentences_to_pages

    page_texts = [
        "Alpha concept is introduced on the very first page of the document here.",
        "Beta discussion continues about the alpha concept on the second page exclusively.",
    ]
    # Satz von Seite 1 -> Seite 1
    assert map_sentences_to_pages(
        ["Alpha concept is introduced on the very first page of the document here."],
        page_texts,
        fallback_page=2,
    ) == [1]
    # Satz kommt in keiner Seite vor -> Fallback
    assert map_sentences_to_pages(
        ["A completely unrelated statement about quantum chromodynamics and gluons."],
        page_texts,
        fallback_page=7,
    ) == [7]
