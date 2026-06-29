"""Tests für MoC-Auto-Vorschlag bei marker-losen Clustern (#4).

`cross_draft_hub.resolve()` erkennt Hubs nur bei Übersichts-Marker im Title
(Modell/Framework/MoC). Thematische Cluster OHNE Marker (z.B. 8 Agent-Notes aus
einem Guide-Run) bekommen keinen MoC. `suggest_unmarked_clusters()` findet sie:
≥5 marker-lose Nicht-Hub-Drafts mit gemeinsamem Title-Token → MoC-Vorschlag.
Seiteneffektfrei — schlägt nur vor, erzeugt keine Note.
"""

from __future__ import annotations


from generative.schemas.atomic_note import AtomicNoteDraft
from generative.pipeline.cross_draft_hub import suggest_unmarked_clusters


def _draft(title: str, action: str = "create") -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title=title,
        body="",
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="medium",
        action=action,
    )


def test_five_shared_token_drafts_suggested():
    drafts = [
        _draft(t)
        for t in [
            "Agent Architektur",
            "Agent Memory",
            "Agent Loop",
            "Agent Tooling",
            "Agent Planung",
        ]
    ]
    result = suggest_unmarked_clusters(drafts)
    assert len(result) == 1
    token, members = result[0]
    assert token == "agent"
    assert len(members) == 5


def test_four_below_threshold():
    drafts = [
        _draft(t)
        for t in [
            "Agent Architektur",
            "Agent Memory",
            "Agent Loop",
            "Agent Tooling",
        ]
    ]
    assert suggest_unmarked_clusters(drafts) == []


def test_marker_draft_excluded_from_cluster():
    # 4 marker-lose + 1 Marker-Draft ("Framework") → nur 4 Kandidaten → unter Schwelle.
    drafts = [
        _draft(t)
        for t in [
            "Agent Architektur",
            "Agent Memory",
            "Agent Loop",
            "Agent Tooling",
        ]
    ] + [_draft("Agent Framework")]
    assert suggest_unmarked_clusters(drafts) == []


def test_hub_draft_excluded_from_cluster():
    drafts = [
        _draft(t)
        for t in [
            "Agent Architektur",
            "Agent Memory",
            "Agent Loop",
            "Agent Tooling",
        ]
    ] + [_draft("Agent Overview", action="hub")]
    assert suggest_unmarked_clusters(drafts) == []


def test_stopword_token_not_suggested():
    # Alle teilen nur das Stoppwort "oder" — kein echter thematischer Cluster.
    drafts = [
        _draft(t)
        for t in [
            "Recht oder Gesetz",
            "Steuer oder Abgabe",
            "Kauf oder Miete",
            "Lohn oder Gehalt",
            "Zins oder Rendite",
        ]
    ]
    assert suggest_unmarked_clusters(drafts) == []


def test_short_token_not_suggested():
    # Gemeinsamer Token "is" (<4 Zeichen) wird ignoriert.
    drafts = [
        _draft(t)
        for t in [
            "AI is Cool",
            "ML is Hard",
            "RL is Slow",
            "NLP is Big",
            "CV is Old",
        ]
    ]
    assert suggest_unmarked_clusters(drafts) == []


def test_strongest_cluster_first():
    # "agent" 5×, "memory" 2× → agent zuerst (nach Häufigkeit sortiert).
    drafts = [
        _draft(t)
        for t in [
            "Agent Architektur",
            "Agent Memory Store",
            "Agent Memory Cache",
            "Agent Loop",
            "Agent Tooling",
        ]
    ]
    result = suggest_unmarked_clusters(drafts)
    assert result[0][0] == "agent"
    assert len(result[0][1]) == 5


def test_resolved_hub_members_excluded():
    # resolve() hat 'ADKAR-Modell' als Hub erkannt; die 5 Stage-Member stehen in
    # hub_subconcepts und teilen Token 'adkar' — der Cluster ist abgedeckt und
    # darf NICHT erneut vorgeschlagen werden.
    member_titles = [
        "ADKAR Awareness",
        "ADKAR Desire",
        "ADKAR Knowledge",
        "ADKAR Ability",
        "ADKAR Reinforcement",
    ]
    hub = _draft("ADKAR-Modell", action="hub")
    hub.hub_subconcepts = list(member_titles)
    drafts = [hub] + [_draft(t) for t in member_titles]
    assert suggest_unmarked_clusters(drafts) == []


def test_identical_member_sets_deduped():
    # Alle 5 Titel teilen 'agent' UND 'memory' → ohne Dedup zwei Vorschläge für
    # denselben Cluster; es darf nur einer bleiben (stärkstes/alphabetisch erstes Token).
    drafts = [
        _draft(t)
        for t in [
            "Agent Memory Store",
            "Agent Memory Cache",
            "Agent Memory Index",
            "Agent Memory Eviction",
            "Agent Memory TTL",
        ]
    ]
    result = suggest_unmarked_clusters(drafts)
    assert len(result) == 1
    assert result[0][0] == "agent"
    assert len(result[0][1]) == 5
