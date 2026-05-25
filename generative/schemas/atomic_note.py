from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextAnchor:
    quote: str           # wörtliches Zitat oder Paraphrase
    page: Optional[str]  # "S. 42" — vom LLM-Verifier exakt bestätigt, oder None
    fuzzy_page: Optional[str] = None  # F8: Fuzzy-Match-Fallback wenn LLM-exact-match scheitert.
                                       # Renderer nutzt `page or fuzzy_page` für Quellen-Block.
                                       # Critic-Input bleibt nur `page` → cache-stabil.


@dataclass
class AtomicNoteDraft:
    title: str
    body: str                          # Markdown-Body ohne Frontmatter (Anker inline mit Seitenzahl)
    source_anchors: list[TextAnchor]   # vom Verifier bestätigt (interne Liste, nicht im Frontmatter)
    related: list[str]                 # Wikilinks zu existierenden Notes
    tags: list[str]
    synthesis_confidence: str          # "high" | "medium" | "low"
    aliases: list[str] = field(default_factory=list)  # DE/EN-Schreibvarianten für Wikilink-Auflösung
    quality_flags: list[str] = field(default_factory=list)  # ⚠️-Marker
    action: str = "create"             # "create" | "extend" | "hub"
    extend_path: Optional[str] = None  # Pfad wenn action == "extend"
    hub_subconcepts: list[str] = field(default_factory=list)  # bei action=="hub": gefundene Sub-Konzept-Titel
    hub_subconcept_descriptions: dict[str, str] = field(default_factory=dict)  # title → Kerncharakteristik aus Sub-Note-H1
    critic_score: int = 0              # 0–5 (5 Tests: Title, Glance, Future-Self, Quellen, Deletion)
    hard_gates_pass: bool = False      # Glance + Future-Self + Quellen alle bestanden
    revision_hint: Optional[str] = None  # für Self-Refine-Loop (Milestone 3.6)
    confidence_reasoning: Optional[str] = None  # CERQual-Begründung bei low/medium
    auto_vault_recommended: Optional[bool] = None  # v23: vault-vs-inbox-Routing ist
                                                    # jetzt Tag-basiert (Auto-Note-Mover);
                                                    # dieses Feld wird Frontmatter-Marker
                                                    # für Inbox-Reviewer
    proposed_tags: list[str] = field(default_factory=list)  # Bootstrap-Pfad: Tag-Vorschläge
                                                            # für neue Domains. KEIN Routing,
                                                            # User-Review beim Inbox-Triage.
                                                            # Nach Bestätigung wandert Tag in
                                                            # tag_registry.yml und wird beim
                                                            # nächsten Run regulär nutzbar.
    tag_review_status: Optional[str] = None  # "needs-review" wenn proposed_tags nicht leer
    refine_key: Optional[str] = None         # concept plan title für concept_map-Lookup nach ER (Bug #5)


@dataclass
class ConceptItem:
    title: str
    priority: str   # "high" | "medium" | "low"
    chapter: str    # Kapitel/Abschnitt wo das Konzept erwartet wird
    action: str     # "create" | "extend" | "skip"
    extend_path: Optional[str] = None
    category: str = "conceptual"   # "architectural" | "operational" | "conceptual"
    # Pass 1 (Prompt) → architectural/conceptual, Pass 2 → operational.
    # Default conceptual für Backward-Compat mit alten Caches/Parsern.
    origin: str = "primary"        # "primary" | "extension" | "secondary_mention"
    cited_authors: list[str] = field(default_factory=list)


@dataclass
class ConceptPlan:
    source_title: str
    source_summary: str   # 2 Sätze: worum geht es insgesamt
    concepts: list[ConceptItem]


@dataclass
class QualityReport:
    peer_reviewed: Optional[bool]
    citation_count: Optional[int]
    retracted: bool
    flags: list[str]   # fertige ⚠️-Strings für Frontmatter
    # F2: CrossRef-Metadata durchreichen, damit Renderer den Quellen-Block mit
    # autoritativen Werten überschreiben kann (überschreibt pdf_metadata)
    crossref_title: Optional[str] = None
    crossref_author: Optional[str] = None
    crossref_year: Optional[str] = None
