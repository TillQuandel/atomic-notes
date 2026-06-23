"""Tests für Intra-Run-Sibling-Dedup (Befund D).

cross_reference erkennt zwei Near-Dup-Drafts EINES Laufs (dup_risk=high) und setzt
action=extend + extend_path=<Sibling-Titel>. Da der Sibling keine Vault-Datei ist,
verpufft das beim Writer und BEIDE Notes werden geschrieben. resolve_sibling_dups()
wertet genau dieses vorhandene Signal aus und mergt/skippt die Siblings VOR dem
Schreiben — ohne Eingriff ins Title-Blocking (kein False-Positive-Risiko).
"""
from generative.orchestrator import resolve_sibling_dups
from generative.agents.cross_reference import MAX_RELATED
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor


def _draft(title, *, body="", action="create", extend_path=None,
           related=None, source_anchors=None, critic_score=0, aliases=None):
    return AtomicNoteDraft(
        title=title,
        body=body or f"Body von {title}",
        source_anchors=source_anchors or [],
        related=related or [],
        tags=[],
        synthesis_confidence="high",
        aliases=aliases or [],
        action=action,
        extend_path=extend_path,
        critic_score=critic_score,
    )


def test_pair_extend_to_sibling_merges_to_one():
    # d_b ist ein Near-Dup von d_a und zeigt per extend_path auf dessen Titel.
    d_a = _draft("Affective Access", body="Langer verifizierter Body " * 5,
                 critic_score=4, related=["[[Information Behavior]]"])
    d_b = _draft("Affektiver Zugang", action="extend",
                 extend_path="Affective Access", critic_score=2,
                 related=["[[Kuhlthau ISP]]"])

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 1
    assert len(kept) == 1
    survivor = kept[0]
    assert survivor.title == "Affective Access"        # höherer critic_score + längerer Body
    assert survivor.action == "create"                  # kein dangling extend
    # related des gedroppten Drafts wandern verlustarm in den Survivor
    assert "[[Kuhlthau ISP]]" in survivor.related
    # gedroppter Titel lebt als Alias weiter → [[Affektiver Zugang]] löst auf den Survivor auf
    assert "Affektiver Zugang" in survivor.aliases


def test_cycle_a_to_b_and_b_to_a_resolves_to_one():
    # Beide Drafts flaggen sich gegenseitig als Dup (paralleler per-Draft-Call).
    d_a = _draft("A", action="extend", extend_path="B", critic_score=3)
    d_b = _draft("B", action="extend", extend_path="A", critic_score=1)

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 1
    assert len(kept) == 1
    assert kept[0].title == "A"                          # höherer critic_score gewinnt
    assert kept[0].action == "create"                   # Zyklus aufgelöst, kein dangling extend


def test_chain_a_b_c_collapses_to_one():
    # A→B, B→C: Union-Find muss die ganze Kette zu einem Cluster verbinden.
    d_a = _draft("A", action="extend", extend_path="B", critic_score=1)
    d_b = _draft("B", action="extend", extend_path="C", critic_score=5)
    d_c = _draft("C", critic_score=2)

    kept, dropped = resolve_sibling_dups([d_a, d_b, d_c])

    assert dropped == 2
    assert len(kept) == 1
    assert kept[0].title == "B"                          # höchster critic_score


def test_vault_extend_is_not_touched():
    # extend_path zeigt auf eine echte Vault-Note (Stem matcht KEINEN Sibling-Titel)
    # → legitimer Vault-Extend, darf NICHT angefasst werden.
    d_a = _draft("Concept A", critic_score=3)
    d_b = _draft("Concept B", action="extend",
                 extend_path="04-wissen/Some Vault Note.md", critic_score=2)

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 0
    assert len(kept) == 2
    survivor_b = next(d for d in kept if d.title == "Concept B")
    assert survivor_b.action == "extend"
    assert survivor_b.extend_path == "04-wissen/Some Vault Note.md"


def test_related_union_capped_and_no_dangling_self_link():
    # Survivor-related enthält bereits einen Link auf den gedroppten Titel
    # (cross_reference fügt [[dup-stem]] ein). Nach dem Drop darf kein Link auf
    # den gedroppten Titel als Dead-Link überleben (Alias-Auflösung übernimmt),
    # und die Gesamtzahl ist auf MAX_RELATED gedeckelt.
    d_a = _draft("Survivor", critic_score=5,
                 related=["[[Affektiver Zugang]]", "[[L1]]", "[[L2]]"])
    d_b = _draft("Affektiver Zugang", action="extend", extend_path="Survivor",
                 critic_score=1, related=["[[L3]]", "[[L4]]", "[[L5]]"])

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 1
    survivor = kept[0]
    assert survivor.title == "Survivor"
    assert len(survivor.related) <= MAX_RELATED
    # kein Self-Link auf den absorbierten Titel
    assert "[[Affektiver Zugang]]" not in survivor.related


def test_no_extend_drafts_is_noop():
    d_a = _draft("A", critic_score=1)
    d_b = _draft("B", critic_score=2)
    kept, dropped = resolve_sibling_dups([d_a, d_b])
    assert dropped == 0
    assert len(kept) == 2


# --- Cross-Model-Review-Befunde (Codex 2026-06-23) ---

def test_vault_extend_propagates_to_survivor():
    # HIGH#2: A (bester Body) ist Near-Dup von B, B ist zugleich Dup einer EXISTIERENDEN
    # Vault-Note V. Survivor A behält seinen besseren Body, MUSS aber B's Vault-Bezug erben
    # — sonst wird eine Dublette der Vault-Note geschrieben. Vault-Stem wird Alias, damit der
    # title-/alias-basierte Writer die Vault-Note findet.
    d_a = _draft("A", body="Langer Body " * 10, action="extend",
                 extend_path="B", critic_score=5)
    d_b = _draft("B", action="extend", extend_path="Vault Concept", critic_score=1)
    existing = {"vault concept": "01-studium/Vault Concept.md"}

    kept, dropped = resolve_sibling_dups([d_a, d_b], existing)

    assert dropped == 1
    survivor = kept[0]
    assert survivor.title == "A"                         # besserer Body bleibt Survivor
    assert survivor.action == "extend"                   # Vault-Bezug NICHT verloren
    assert survivor.extend_path == "Vault Concept"
    assert any("vault concept" == a.lower() for a in survivor.aliases)  # Writer findet Vault-Note


def test_bare_title_with_slash_matches():
    # MED#4: Ein Sibling-Titel mit "/" darf nicht über Path().stem zerlegt werden.
    d_a = _draft("TCP/IP", critic_score=4)
    d_b = _draft("TCP IP Stack", action="extend", extend_path="TCP/IP", critic_score=1)

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 1
    assert kept[0].title == "TCP/IP"


def test_survivor_tiebreak_is_order_independent():
    # LOW#5: bei gleichem critic_score UND gleicher Body-Länge muss der Survivor
    # deterministisch sein, unabhängig von der Eingabereihenfolge.
    def make_pair():
        a = _draft("Alpha", body="x" * 50, action="extend", extend_path="Beta", critic_score=3)
        b = _draft("Beta", body="y" * 50, action="extend", extend_path="Alpha", critic_score=3)
        return a, b

    a1, b1 = make_pair()
    kept1, _ = resolve_sibling_dups([a1, b1])
    a2, b2 = make_pair()
    kept2, _ = resolve_sibling_dups([b2, a2])  # umgekehrte Reihenfolge

    assert kept1[0].title == kept2[0].title


def test_heading_self_link_removed():
    # LOW#6: [[Titel#Abschnitt]]-Self-Link auf den absorbierten Titel muss entfernt werden.
    d_a = _draft("Survivor", critic_score=5,
                 related=["[[Affektiver Zugang#Definition]]", "[[Keep]]"])
    d_b = _draft("Affektiver Zugang", action="extend", extend_path="Survivor", critic_score=1)

    kept, dropped = resolve_sibling_dups([d_a, d_b])

    assert dropped == 1
    assert not any("affektiver zugang" in l.lower() for l in kept[0].related)
    assert "[[Keep]]" in kept[0].related
