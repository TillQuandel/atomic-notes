import pytest
from unittest.mock import patch, MagicMock


def test_extract_concepts_default_threshold_is_higher():
    """Threshold muss 0.75 sein."""
    import inspect
    from foss.pipeline.gliner_planner import extract_concepts
    sig = inspect.signature(extract_concepts)
    assert sig.parameters["threshold"].default == 0.75


def test_plan_concepts_respects_max_cap():
    """plan_concepts darf max_concepts nicht ueberschreiten."""
    from foss.pipeline.gliner_planner import plan_concepts
    chunks = [MagicMock(text=f"Information literacy framework concept model {i}", page=1) for i in range(5)]
    with patch("foss.pipeline.gliner_planner.extract_concepts") as mock_ec:
        mock_ec.return_value = [
            {"name": f"concept_{i}", "type": "Concept", "page": 1, "score": 0.9}
            for i in range(20)
        ]
        result = plan_concepts(chunks, max_concepts=10)
    assert len(result) <= 10


def test_generic_concepts_filtered():
    from foss.pipeline.gliner_planner import _is_specific_concept
    assert _is_specific_concept("mean") is False
    assert _is_specific_concept("management") is False
    assert _is_specific_concept("advantage") is False
    assert _is_specific_concept("information") is False
    assert _is_specific_concept("methods") is False
    assert _is_specific_concept("surveys") is False


def test_specific_concepts_pass():
    from foss.pipeline.gliner_planner import _is_specific_concept
    assert _is_specific_concept("information literacy framework") is True
    assert _is_specific_concept("ADKAR model") is True
    assert _is_specific_concept("LexRank") is True
    assert _is_specific_concept("constructivism") is True
