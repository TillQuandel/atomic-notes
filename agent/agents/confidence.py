"""Confidence-Agent: setzt synthesis-confidence basierend auf CERQual-Logik.

Quelle der Regeln: [[Synthesis-Confidence]] (98-system/claude-code/).

Vier CERQual-Komponenten — bei Bedenken in einer Komponente eine Stufe runter.

1. **Methodische Limits** — Tier der Quellen (A/B/C). Nur Tier-C → mind. eine Stufe runter.
2. **Coherence** — widersprechen sich Quellen an Kernaussagen? Wenn ja und Konflikt nicht
   in der Note benannt: runter.
3. **Adequacy** — sind Kernaussagen ≥2-fach belegt? Monoquellig: runter.
4. **Relevance** — passen die Quellen zum Konzept der Note (oder sind sie Pointer zu
   Nachbarthemen)? Nur Pointer: runter.

Stufen-Mapping:
- Alle 4 OK → high
- Bedenken in 1 Komponente → medium
- Bedenken in ≥2 Komponenten → low

Für Pipeline-erzeugte Notes: Default low, weil ein PDF = monoquellig (Adequacy fail).
medium nur wenn Cross-Reference echte Vault-Belege gefunden hat (zweite Quelle).
high nur bei ≥2 unabhängigen Tier-A/B-Quellen (heuristisch via QualityReport-Flags).
"""
from __future__ import annotations

from schemas.atomic_note import AtomicNoteDraft


def run(draft: AtomicNoteDraft, has_vault_corroboration: bool = False,
        peer_reviewed: bool = False, citation_count: int | None = None) -> AtomicNoteDraft:
    """CERQual-Stufung. Pipeline-Default low, hochstufen nur bei klarer Evidenz.

    Args:
        draft: Note nach Cross-Reference (related-Links bereits gesetzt)
        has_vault_corroboration: True wenn Cross-Reference ≥1 verwandte Vault-Note
            gefunden hat (Hinweis auf zweite Quelle, Adequacy-Komponente bessert sich)
        peer_reviewed: aus QualityReport
        citation_count: aus QualityReport (None wenn unbekannt)
    """
    components_pass = 4
    reasons: list[str] = []

    # 1. Methodische Limits — Tier-Heuristik via peer_reviewed + Citations
    if peer_reviewed is False:
        components_pass -= 1
        reasons.append("nicht peer-reviewed (Methodische Limits)")
    elif citation_count is not None and citation_count < 5:
        # peer-reviewed aber wenig zitiert — Tier-B/C-Indikator
        components_pass -= 1
        reasons.append(f"niedrige Zitationsanzahl n={citation_count} (Methodische Limits)")

    # 2. Coherence — Widersprüche aus Cross-Reference?
    # Nur harte ⚠️-Flags (Haiku+NLI bestätigt) zählen als Coherence-Penalty.
    # Soft-Warnings (ℹ️ Möglicher Widerspruch, nur Haiku) sind kein Confidence-Signal.
    has_contradiction = any(
        f.startswith("⚠️") and "Widerspruch" in f
        for f in draft.quality_flags
    )
    if has_contradiction:
        components_pass -= 1
        reasons.append("Widerspruch zu Vault-Note erkannt (Coherence)")

    # 3. Adequacy — monoquellig wenn keine Vault-Bestätigung
    if not has_vault_corroboration:
        components_pass -= 1
        reasons.append("monoquellig — keine zweite Quelle im Vault (Adequacy)")

    # 4. Relevance — heuristisch: <2 source_anchors → schwache Quellenbindung
    if len(draft.source_anchors) < 2:
        components_pass -= 1
        reasons.append(f"nur {len(draft.source_anchors)} Anker (Relevance)")

    # Stufen-Mapping
    if components_pass >= 4:
        level = "high"
    elif components_pass == 3:
        level = "medium"
    else:
        level = "low"

    draft.synthesis_confidence = level
    if reasons:
        draft.confidence_reasoning = "; ".join(reasons)
    # Stabilisierungs-Refactor: Confidence-Notiz NICHT mehr in den Body schreiben.
    # confidence_reasoning bleibt als Draft-Feld; Renderer fügt es beim finalen Schreiben
    # ein (Body-Sektion oder Frontmatter, je nach F3-Entscheidung). Damit bleibt der
    # Critic-Input cache-stabil bei Renderer-Tweaks.
    return draft
