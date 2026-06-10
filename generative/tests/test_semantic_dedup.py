"""Tests für Semantic Title Deduplication (v35).
Prüft ob Title-Varianten ohne Token-Overlap via Cosine-Similarity gemergt werden.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


from generative.orchestrator import entity_resolution
from generative.schemas.atomic_note import AtomicNoteDraft

@pytest.mark.anyio
async def test_er_v35_semantic_title_accept():
    # Zwei Drafts ohne Token-Subset (Red Thread vs Roter Faden)
    # Tokens: {red, thread, information} vs {roter, faden, information}
    d1 = AtomicNoteDraft(
        title="Red Thread of Information", 
        body="Body Content A", 
        action="create",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high"
    )
    d2 = AtomicNoteDraft(
        title="Roter Faden der Information", 
        body="Body Content A", 
        action="create",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high"
    )
    
    # Mocking
    with patch("generative.orchestrator.embeddings") as mock_emb:
        # Stage 1: Blocking
        # embed_title für d1 und d2
        mock_emb.embed_title.side_effect = ["emb1", "emb2"]
        # cosine für Title-Vergleich (0.95 > 0.88 Threshold)
        # cosine für Body-Vergleich (0.99 > 0.85 Threshold)
        mock_emb.cosine.side_effect = [0.95, 0.99]
        
        # embed_body für d1 und d2 (Stage 2)
        mock_emb.embed_body.side_effect = ["body_emb1", "body_emb2"]
        
        with patch("generative.orchestrator.canonicalizer.merge_cluster") as mock_merge:
            # Simuliere Merge-Erfolg
            mock_merge.return_value = d1
            
            results = await entity_resolution([d1, d2])
            
            # Verifikation
            assert len(results) == 1
            assert results[0].title == "Red Thread of Information"
            
            # Sicherstellen dass semantic accept geloggt wurde (via mock calls)
            assert mock_emb.embed_title.call_count == 2
            assert mock_emb.cosine.call_count == 2 # 1x Title, 1x Body

@pytest.mark.anyio
async def test_er_v35_semantic_title_reject():
    # Zwei Drafts die semantisch verschieden sind (ADKAR Awareness vs Desire)
    d1 = AtomicNoteDraft(
        title="ADKAR Awareness", 
        body="Content A", 
        action="create",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high"
    )
    d2 = AtomicNoteDraft(
        title="ADKAR Desire", 
        body="Content B", 
        action="create",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high"
    )
    
    with patch("generative.orchestrator.embeddings") as mock_emb:
        mock_emb.embed_title.side_effect = ["emb1", "emb2"]
        # Cosine 0.5 (verschieden)
        mock_emb.cosine.return_value = 0.5
        
        results = await entity_resolution([d1, d2])
        
        # Kein Merge
        assert len(results) == 2
        assert results[0].title == "ADKAR Awareness"
        assert results[1].title == "ADKAR Desire"
        
        # Nur Title-Cosine wurde geprüft (Stage 1 Blocking)
        assert mock_emb.cosine.call_count == 1
        assert mock_emb.embed_body.call_count == 0
