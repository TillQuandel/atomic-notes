from foss.pipeline.gliner_planner import deduplicate_concepts


def test_dedup_removes_case_duplicates():
    concepts = [
        {"name": "Information Behavior", "type": "Concept", "page": 1, "score": 0.9},
        {"name": "information behavior", "type": "Concept", "page": 2, "score": 0.8},
        {"name": "Information Search Process", "type": "Theory", "page": 1, "score": 0.85},
    ]
    result = deduplicate_concepts(concepts)
    assert len(result) == 2
    assert any(c["name"] == "Information Search Process" for c in result)


def test_dedup_keeps_higher_score():
    concepts = [
        {"name": "Information Behavior", "type": "Concept", "page": 1, "score": 0.9},
        {"name": "information behavior", "type": "Concept", "page": 2, "score": 0.8},
    ]
    result = deduplicate_concepts(concepts)
    assert result[0]["score"] == 0.9


def test_dedup_different_concepts_kept():
    concepts = [
        {"name": "LexRank", "type": "Method", "page": 1, "score": 0.8},
        {"name": "BERT", "type": "Model", "page": 2, "score": 0.85},
    ]
    result = deduplicate_concepts(concepts)
    assert len(result) == 2
