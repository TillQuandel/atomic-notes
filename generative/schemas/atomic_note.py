"""AtomicNoteDraft ist das Pipeline-Blackboard: eine einzelne Instanz wird von
extractor bis vault_writer durchgereicht, jede Stage liest und mutiert
Teilmengen der Felder weiter (keine State-Machine, kein Owner-Objekt).
Konvention (#99): jedes Feld trägt einen `# ownership: writer=<modul>[,...]
reader=<modul>[,...]`-Kommentar; eine neue/geänderte Ownership-Zeile ist
Pflichtteil jedes Diffs, der ein Feld hinzufügt oder seine Schreiber/Leser
ändert — `tests/test_schema_ownership.py` erzwingt das per Drift-Test.
Sonderfall Stage 7: `vault_writer.write_note` mutiert als nomineller
„Reader" trotzdem `auto_vault_recommended`, hängt an `quality_flags` an und
kann `proposed_tags`/`tag_review_status` aus der Inbox zurückschreiben —
`note_json.py` exportiert deshalb bewusst zwei Varianten (Draft-Snapshot
`note.auto_vault_recommended`, ggf. `None`, plus frisch berechnetes
`routing.auto_vault_recommended`, siehe note_json.py:22-27). Der generische
Checkpoint-Roundtrip in `orchestrator._save_draft_state`/`_load_draft_state`
(`dataclasses.asdict` / `AtomicNoteDraft(**d)`) fasst jedes Feld an, ist aber
reine Resume-Serialisierung ohne Fachlogik und taucht in den Ownership-Zeilen
unten bewusst nicht auf.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ownership (Klassen-Block, #99): quote/page werden von extractor (initiale
# LLM-Extraktion) und verifier (alleiniger Lifecycle-Owner danach: Fuzzy-/
# Semantic-/LLM-Resolution, inkl. In-Place-Fill von fuzzy_page) geschrieben.
# page zusätzlich gelesen von critic (Prompt-Anzeige) und figure_alt (exakter
# Seiten-Match — NIE fuzzy_page); quote+page gelesen von canonicalizer als
# Dedup-Key beim Anker-Merge. fuzzy_page bewusst nicht an critic durchgereicht
# (Cache-Stabilität, siehe Feld-Kommentar unten). vault_writer.collect_anchor_pages
# liest page/fuzzy_page, hat aktuell aber keinen Produktions-Aufrufer (nur
# test_portable_md.py) — de-facto tot.
@dataclass
class TextAnchor:
    quote: str  # wörtliches Zitat oder Paraphrase
    page: Optional[str]  # "S. 42" — vom LLM-Verifier exakt bestätigt, oder None
    fuzzy_page: Optional[str] = None  # F8: Fuzzy-Match-Fallback wenn LLM-exact-match scheitert.
    # Renderer nutzt `page or fuzzy_page` für Quellen-Block.
    # Critic-Input bleibt nur `page` → cache-stabil.


@dataclass
class AtomicNoteDraft:
    # ownership: writer=extractor,canonicalizer reader=verifier,critic,cross_reference,canonicalizer,cross_draft_hub,boilerplate_dedup,vault_writer,orchestrator,export_runner,note_json
    title: str
    # ownership: writer=extractor,canonicalizer,orchestrator,boilerplate_dedup,cross_draft_hub,figure_alt reader=verifier,critic,cross_reference,canonicalizer,boilerplate_dedup,cross_draft_hub,figure_alt,vault_writer,orchestrator,note_json
    body: str  # Markdown-Body ohne Frontmatter (Anker inline mit Seitenzahl)
    # ownership: writer=extractor,verifier,canonicalizer,orchestrator reader=verifier,critic,confidence,figure_alt,canonicalizer,orchestrator,note_json
    source_anchors: list[TextAnchor]  # vom Verifier bestätigt (interne Liste, nicht im Frontmatter)
    # ownership: writer=extractor,cross_reference,canonicalizer,orchestrator,vault_writer reader=cross_reference,orchestrator,vault_writer,boilerplate_dedup,note_json
    related: list[str]  # Wikilinks zu existierenden Notes
    # ownership: writer=extractor,canonicalizer,orchestrator reader=canonicalizer,orchestrator,vault_writer,note_json
    tags: list[str]
    # ownership: writer=extractor,verifier,confidence,critic,canonicalizer reader=critic,canonicalizer,vault_writer,routing_report,note_json
    synthesis_confidence: str  # "high" | "medium" | "low"
    # ownership: writer=extractor,canonicalizer,orchestrator reader=canonicalizer,orchestrator,critic,cross_reference,boilerplate_dedup,cross_draft_hub,vault_writer,note_json
    aliases: list[str] = field(default_factory=list)  # DE/EN-Schreibvarianten für Wikilink-Auflösung
    # ownership: writer=verifier,critic,cross_reference,canonicalizer,boilerplate_dedup,cross_draft_hub,citation_check,vault_writer,orchestrator reader=canonicalizer,citation_check,vault_writer,orchestrator,confidence,routing_report,note_json
    quality_flags: list[str] = field(default_factory=list)  # ⚠️-Marker
    # ownership: writer=extractor,critic,cross_reference,cross_draft_hub,orchestrator,canonicalizer reader=critic,cross_draft_hub,orchestrator,canonicalizer,vault_writer,boilerplate_dedup,figure_alt,routing_report,note_json
    action: str = "create"  # "create" | "extend" | "hub"
    # ownership: writer=extractor,cross_reference,canonicalizer,orchestrator reader=canonicalizer,orchestrator,vault_writer
    extend_path: Optional[str] = None  # Pfad wenn action == "extend"
    # ownership: writer=critic,cross_draft_hub reader=cross_draft_hub,vault_writer,boilerplate_dedup,note_json
    hub_subconcepts: list[str] = field(default_factory=list)  # bei action=="hub": gefundene Sub-Konzept-Titel
    # ownership: writer=cross_draft_hub reader=vault_writer
    hub_subconcept_descriptions: dict[str, str] = field(
        default_factory=dict
    )  # title → Kerncharakteristik aus Sub-Note-H1
    # ownership: writer=critic reader=vault_writer,orchestrator,note_json
    critic_score: int = 0  # 0–5 (5 Tests: Title, Glance, Future-Self, Quellen, Deletion)
    # ownership: writer=critic reader=vault_writer,orchestrator,routing_report,note_json
    hard_gates_pass: bool = False  # Glance + Future-Self + Quellen alle bestanden
    # ownership: writer=orchestrator reader=vault_writer,note_json
    faithfulness_fail: bool = False  # Faithfulness-Gate (E6): >=1 High-Risk-Claim failed
    # — Routing-Veto in auto_write_decision
    # ownership: writer=critic reader=orchestrator,note_json
    revision_hint: Optional[str] = None  # für Self-Refine-Loop (Milestone 3.6)
    # ownership: writer=confidence reader=vault_writer,note_json
    confidence_reasoning: Optional[str] = None  # CERQual-Begründung bei low/medium
    # ownership: writer=vault_writer reader=vault_writer,note_json
    auto_vault_recommended: Optional[bool] = None  # v23: vault-vs-inbox-Routing ist
    # jetzt Tag-basiert (Auto-Note-Mover);
    # dieses Feld wird Frontmatter-Marker
    # für Inbox-Reviewer
    # ownership: writer=extractor,vault_writer reader=vault_writer,note_json
    proposed_tags: list[str] = field(default_factory=list)  # Bootstrap-Pfad: Tag-Vorschläge
    # für neue Domains. KEIN Routing,
    # User-Review beim Inbox-Triage.
    # Nach Bestätigung wandert Tag in
    # tag_registry.yml und wird beim
    # nächsten Run regulär nutzbar.
    # ownership: writer=extractor,vault_writer reader=vault_writer,note_json
    tag_review_status: Optional[str] = None  # "needs-review" wenn proposed_tags nicht leer
    # ownership: writer=orchestrator reader=orchestrator
    refine_key: Optional[str] = None  # concept plan title für concept_map-Lookup nach ER (Bug #5)
    # ownership: writer=orchestrator reader=orchestrator,vault_writer,routing_report,note_json
    source_status: Optional[str] = None  # #45: "unresolved" wenn die Quelle (Autor/Jahr/DOI)
    # nicht zuverlässig aufgelöst werden konnte (Enrichment
    # leer ODER CrossRef-Override fail-closed verworfen).
    # Schmales Frontmatter-Flag, kein Erklär-Absatz.


# ownership (Klassen-Block, #99): alle 8 Felder werden ausschließlich von
# planner geschrieben (einziger Konstruktor, aus geparstem LLM-Output).
# Gelesen von extractor (title, priority, action, chapter, extend_path — steuert
# Extraktions-Prompt sowie action/extend_path-Vererbung an AtomicNoteDraft) und
# orchestrator (title, action, origin, priority — Chunk-Matching, Halluzinations-
# Filter-Report, Planungs-Logging). category hat außer planners eigenem
# Kategorien-Verteilungs-Log keinen Leser. cited_authors hat aktuell KEINEN
# Produktions-Leser (Grep-verifiziert: nur geschrieben, nirgends gelesen).
@dataclass
class ConceptItem:
    title: str
    priority: str  # "high" | "medium" | "low"
    chapter: str  # Kapitel/Abschnitt wo das Konzept erwartet wird
    action: str  # "create" | "extend" | "skip"
    extend_path: Optional[str] = None
    category: str = "conceptual"  # "architectural" | "operational" | "conceptual"
    # Pass 1 (Prompt) → architectural/conceptual, Pass 2 → operational.
    # Default conceptual für Backward-Compat mit alten Caches/Parsern.
    origin: str = "primary"  # "primary" | "extension" | "secondary_mention"
    cited_authors: list[str] = field(default_factory=list)


@dataclass
class ConceptPlan:
    source_title: str
    source_summary: str  # 2 Sätze: worum geht es insgesamt
    concepts: list[ConceptItem]


@dataclass
class QualityReport:
    peer_reviewed: Optional[bool]
    citation_count: Optional[int]
    retracted: bool
    flags: list[str]  # fertige ⚠️-Strings für Frontmatter
    # F2: CrossRef-Metadata durchreichen, damit Renderer den Quellen-Block mit
    # autoritativen Werten überschreiben kann (überschreibt pdf_metadata)
    crossref_title: Optional[str] = None
    crossref_author: Optional[str] = None
    crossref_year: Optional[str] = None
    # True wenn die DOI per Title-RATEN (CrossRef-Title-Match) statt per harter ID
    # gefunden wurde → crossref_*-Override darf die Quelle nur bei Titel-Match setzen.
    doi_from_title_match: bool = False
