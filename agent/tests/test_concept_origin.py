"""Tests für das origin-Feld in ConceptItem (Secondary Citation Handling)."""
from schemas.atomic_note import ConceptItem


def test_concept_item_origin_defaults():
    c = ConceptItem(title="Test", priority="medium", chapter="ch1", action="create")
    assert c.origin == "primary"
    assert c.cited_authors == []


def test_concept_item_origin_secondary():
    c = ConceptItem(
        title="Information Foraging Theory",
        priority="medium",
        chapter="ch1",
        action="create",
        origin="secondary_mention",
        cited_authors=["Pirolli", "Card"],
    )
    assert c.origin == "secondary_mention"
    assert "Pirolli" in c.cited_authors


def test_concept_item_origin_extension():
    c = ConceptItem(
        title="Jaiswal's Critique of IFT",
        priority="high",
        chapter="ch2",
        action="create",
        origin="extension",
        cited_authors=["Pirolli"],
    )
    assert c.origin == "extension"
    assert c.cited_authors == ["Pirolli"]


def test_secondary_mention_routing_logic():
    """Konzepte mit origin=secondary_mention werden als Related Mentions gesammelt."""
    from schemas.atomic_note import ConceptPlan

    plan = ConceptPlan(
        source_title="Test",
        source_summary="Test.",
        concepts=[
            ConceptItem("Primary Concept", "high", "ch1", "create", origin="primary"),
            ConceptItem("IFT", "medium", "bg", "create", origin="secondary_mention",
                        cited_authors=["Pirolli", "Card"]),
            ConceptItem("Extension Note", "medium", "ch2", "create", origin="extension"),
        ],
    )
    secondary = [c for c in plan.concepts if c.origin == "secondary_mention"]
    actionable = [c for c in plan.concepts
                  if c.action != "skip" and c.origin != "secondary_mention"]
    related_mentions = [c.title for c in secondary]

    assert len(actionable) == 2
    assert "IFT" not in [c.title for c in actionable]
    assert "IFT" in related_mentions
