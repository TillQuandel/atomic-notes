from unittest.mock import MagicMock, patch

from extractive.pipeline.gliner_planner import deduplicate_concepts


def test_extract_concepts_default_threshold_is_higher():
    """Threshold muss 0.75 sein."""
    import inspect
    from extractive.pipeline.gliner_planner import extract_concepts

    sig = inspect.signature(extract_concepts)
    assert sig.parameters["threshold"].default == 0.75


def test_plan_concepts_respects_max_cap():
    """plan_concepts darf max_concepts nicht ueberschreiten."""
    from extractive.pipeline.gliner_planner import plan_concepts

    chunks = [MagicMock(text=f"Information literacy framework concept model {i}", page=1) for i in range(5)]
    with patch("extractive.pipeline.gliner_planner.extract_concepts") as mock_ec:
        mock_ec.return_value = [{"name": f"concept_{i}", "type": "Concept", "page": 1, "score": 0.9} for i in range(20)]
        result = plan_concepts(chunks, max_concepts=10)
    assert len(result) <= 10


def test_prominence_gate_counts_distinct_concepts():
    """Fix #167a-Kalibrierung: das Prominenz-Gate darf INSTANZEN nicht mit
    distinkten Konzepten verwechseln. Zwei prominente Konzepte (je in >= 2 Chunks)
    duerfen den 'zu wenige prominent'-Fallback auf all_concepts nicht blockieren,
    sonst gehen legitime 1-Chunk-Mehrwortkonzepte verloren (bates: 6 -> 2 Notes)."""
    from extractive.pipeline.gliner_planner import plan_concepts

    # 'alpha beta' + 'gamma delta' je 2x (prominent), 'epsilon zeta' 1x (nur wenn Fallback)
    per_chunk = [
        [
            {"name": "alpha beta", "type": "Concept", "page": 1, "score": 0.9},
            {"name": "gamma delta", "type": "Concept", "page": 1, "score": 0.85},
            {"name": "epsilon zeta", "type": "Theory", "page": 1, "score": 0.8},
        ],
        [
            {"name": "alpha beta", "type": "Concept", "page": 2, "score": 0.9},
            {"name": "gamma delta", "type": "Concept", "page": 2, "score": 0.85},
        ],
    ]
    chunks = [MagicMock(text="x", page=i + 1) for i in range(2)]
    with (
        patch("extractive.pipeline.gliner_planner.extract_concepts", side_effect=per_chunk),
        patch("extractive.pipeline.gliner_planner._keybert_fallback", side_effect=lambda c, existing, **k: existing),
    ):
        result = plan_concepts(chunks, min_concepts=3, min_chunk_count=2)
    names = {c["name"] for c in result}
    assert "epsilon zeta" in names, names


def test_generic_concepts_filtered():
    from extractive.pipeline.gliner_planner import _is_specific_concept

    assert _is_specific_concept("mean") is False
    assert _is_specific_concept("management") is False
    assert _is_specific_concept("advantage") is False
    assert _is_specific_concept("information") is False
    assert _is_specific_concept("methods") is False
    assert _is_specific_concept("surveys") is False


def test_specific_concepts_pass():
    from extractive.pipeline.gliner_planner import _is_specific_concept

    assert _is_specific_concept("information literacy framework") is True  # Mehrwort
    assert _is_specific_concept("ADKAR model") is True  # Mehrwort
    assert _is_specific_concept("LexRank") is True  # Einwort mit Grossbuchstabe (Eigenname)
    assert _is_specific_concept("Zettelkasten") is True  # Einwort-Eigenname (Grossbuchstabe)


def test_lowercase_single_word_concepts_rejected():
    """Fix #167a: rein-lowercase Einzelwoerter sind GLiNER-Artefakte und keine
    Atomic-Note-Titel (die reale Note 'chemistry' handelte von Atomicity).
    Statt des ehemaligen >=8-Zeichen-Proxys wird die ganze Klasse verworfen;
    Eigennamen ueberleben ueber den Grossbuchstaben (siehe test oben)."""
    from extractive.pipeline.gliner_planner import _is_specific_concept

    for junk in ("chemistry", "knowledge", "retrieval", "conceptual"):
        assert _is_specific_concept(junk) is False, junk


def test_language_drift_filtered():
    from extractive.pipeline.gliner_planner import _matches_language

    assert _matches_language("bekanntheitsgrad", "en") is False
    assert _matches_language("datenschutz", "en") is False
    assert _matches_language("information literacy", "en") is True
    assert _matches_language("informationskompetenz", "de") is True


def test_extract_concepts_filters_language():
    from extractive.pipeline.gliner_planner import extract_concepts

    with patch("extractive.pipeline.gliner_planner._get_model") as mock_model:
        mock_model.return_value.predict_entities.return_value = [
            {"text": "bekanntheitsgrad", "label": "Concept", "score": 0.9}
        ]
        result = extract_concepts("some english text", main_language="en")
    assert len(result) == 0


def test_keybert_fallback_filtered():
    from extractive.pipeline.gliner_planner import _keybert_fallback

    chunks = [MagicMock(text="change management process", page=1)]
    with patch("keybert.KeyBERT") as mock_kb:
        mock_kb.return_value.extract_keywords.return_value = [
            ("management", 0.9),
            ("information literacy framework", 0.8),
        ]
        result = _keybert_fallback(chunks, [], main_language="en")
    names = [r["name"] for r in result]
    assert "management" not in names
    assert "information literacy framework" in names


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


def test_get_model_passes_device_as_map_location():
    """Fix #167d: --device wird eingeloest — GLiNER.from_pretrained bekommt
    map_location=<device> (gliner 0.2.27 unterstuetzt das)."""
    from extractive.pipeline import gliner_planner

    gliner_planner._get_model.cache_clear()
    with patch("gliner.GLiNER") as mock_gliner:
        gliner_planner._get_model("cuda")
    mock_gliner.from_pretrained.assert_called_once()
    _, kwargs = mock_gliner.from_pretrained.call_args
    assert kwargs.get("map_location") == "cuda"
    gliner_planner._get_model.cache_clear()


def test_plan_concepts_threads_device_to_extract():
    """device fliesst von plan_concepts bis extract_concepts durch."""
    from extractive.pipeline.gliner_planner import plan_concepts

    chunks = [MagicMock(text="Information literacy framework", page=1)]
    with patch("extractive.pipeline.gliner_planner.extract_concepts") as mock_ec:
        mock_ec.return_value = [{"name": "information literacy framework", "type": "Concept", "page": 1, "score": 0.9}]
        plan_concepts(chunks, min_concepts=1, min_chunk_count=1, device="cuda")
    _, kwargs = mock_ec.call_args
    assert kwargs.get("device") == "cuda"
