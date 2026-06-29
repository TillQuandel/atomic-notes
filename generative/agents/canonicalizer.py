"""Canonicalizer-Agent: Cluster aus Title-Varianten desselben Konzepts → eine
konsolidierte Note via LLM-Merge.

Stage 4 der Entity-Resolution-Pipeline (Christen 2012, GraphRAG-Pattern):
mehrere Bodies aus demselben PDF die laut Embedding-Cosine ein Konzept-Cluster
bilden, werden hier zu einem Body gemergt — alle Aspekte/Anker konsolidiert,
keine Inhaltsfakten verloren, keine erfundenen Aussagen.

Anker-Listen werden konkateniert (deterministisch, kein LLM-Schreiben), nur
der Body-Text geht durch den LLM-Merge.
"""

from __future__ import annotations
from generative.agents.base import call_claude_async
from generative.agents.structured_output import parse_canonicalizer_output
from generative.config import MODEL_CANONICALIZER
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor

_PROMPT = """Du erhältst {n} Konzept-Note-Varianten zum SELBEN Konzept aus DERSELBEN Quelle.
Sie sind durch Embedding-Cluster als inhaltlich äquivalent erkannt worden.

Aufgabe: erzeuge EINE konsolidierte Note die alle inhaltlichen Aspekte aus allen Quell-Varianten enthält.

## Harte Regeln
- Erfinde NICHTS. Jede Aussage muss in mindestens einer der Quell-Varianten unten stehen.
- Behalte ALLE Seiten-Anker `(S. X)` exakt wie in den Quellen — kein Anker darf verschwinden, keine neuen Seitenzahlen erfinden.
- Behalte ALLE Direktzitate exakt wörtlich — keine Umformulierung.
- Bei widersprüchlichen Aussagen zwischen Varianten: beide aufnehmen mit Markierung „je nach Lesart" oder ähnlich, nicht eine Variante unterdrücken.
- Body-Länge: 25–40 Zeilen, Hardlimit 30 Zeilen. Nutze die 3-Phasen-Struktur Definition→Substanz→Empirie wie bei normalen Konzept-Notes.
- Deutsch. Stilkonvention: Direktzitate mit deutschen Guillemets `„..."`. ASCII-`"` ist im Output-Format erlaubt, Guillemets sind Vault-Konvention.

## Was du synthetisierst (NICHT erfindest)
- Aliases: vereinige die `aliases`-Listen aller Varianten + füge die Title-Varianten als Aliases hinzu
- Tags: Schnittmenge der Tag-Listen (konservativ — nur was in mindestens 2 Varianten steht; bei <2 Varianten: Schnittmenge = Vereinigung)

## Quell-Varianten

{variants_block}

## Output — NUR dieses Format, kein erklärender Text, KEIN JSON:

<!--NOTE-->
title: konsolidierter Konzept-Titel (atomar, eine Idee)
aliases: alle Title-Varianten, alle bestehenden Aliases (comma-separated)
tags: konservative Tag-Schnittmenge (comma-separated)
<!--BODY-->
Markdown-Body 25–40 Zeilen mit Ankern inline, alle Fakten aus allen Varianten.
Darf beliebige Zeichen enthalten — auch ASCII-Quotes, Markdown-HR, Backticks.
<!--END-->

**Format-Regeln**: Sentinels exakt ALL_CAPS wie oben. Header-Lines `key: value` einzeilig. Lists comma-separated. Body als Heredoc nach `<!--BODY-->`. Genau **ein** `<!--NOTE-->` Block.
"""


def _format_variant(idx: int, draft: AtomicNoteDraft) -> str:
    return f"### Variante {idx}: {draft.title}\nAliases: {draft.aliases}\nTags: {draft.tags}\nBody:\n{draft.body}\n"


async def merge_cluster(cluster: list[AtomicNoteDraft]) -> AtomicNoteDraft:
    """Merget Cluster-Mitglieder zu einer konsolidierten Note.

    Anker-Listen werden deterministisch konkateniert (kein LLM-Schreiben),
    nur Body+Title+Aliases+Tags gehen durch den Merge-Call. Bei LLM-Fail
    Fallback auf den ersten Draft (Repräsentant) mit gemergten Aliases —
    keine Datenverluste bei Pipeline-Fehlern.
    """
    if len(cluster) <= 1:
        return cluster[0]

    variants_block = "\n".join(_format_variant(i + 1, d) for i, d in enumerate(cluster))
    prompt = _PROMPT.format(n=len(cluster), variants_block=variants_block)

    try:
        raw = await call_claude_async(prompt, model=MODEL_CANONICALIZER, agent="canonicalizer")
        data, parse_warnings = parse_canonicalizer_output(raw)
        if parse_warnings:
            import sys

            for w in parse_warnings:
                print(f"      [canonicalizer-warn] {w}", file=sys.stderr)
        if not data:
            data = None
    except RuntimeError:
        data = None

    # Anker deterministisch zusammenführen — kein LLM-Schreiben.
    # Dedup über (quote, page) — gleicher Anker aus mehreren Varianten nur einmal.
    seen_anchors: set[tuple] = set()
    merged_anchors: list[TextAnchor] = []
    for d in cluster:
        for a in d.source_anchors:
            key = (a.quote.strip()[:80], a.page or "")
            if key not in seen_anchors:
                seen_anchors.add(key)
                merged_anchors.append(a)

    # Aliases-Vereinigung deterministisch (Backup falls LLM-Output unvollständig)
    all_aliases: set[str] = set()
    for d in cluster:
        all_aliases.update(d.aliases or [])
        # Title der nicht-Repräsentanten als Alias aufnehmen
    rep = cluster[0]
    for d in cluster[1:]:
        all_aliases.add(d.title)
    # Repräsentant-Title NICHT in seine eigenen Aliases
    all_aliases.discard(rep.title)

    if data is None:
        # Fallback: Repräsentant behalten, nur Aliases+Anker mergen
        rep.aliases = sorted(all_aliases)
        rep.source_anchors = merged_anchors
        return rep

    # Tags-Schnittmenge: nur was in mindestens 2 Varianten steht (konservativ).
    # Bei nur 2 Varianten: Schnittmenge = Vereinigung (sonst leer wenn nicht identisch).
    from collections import Counter

    tag_counts = Counter()
    for d in cluster:
        tag_counts.update(set(d.tags))
    min_count = 2 if len(cluster) >= 3 else 1
    safe_tags = sorted(t for t, c in tag_counts.items() if c >= min_count)

    # LLM-Output übernehmen, deterministische Felder vorrangig
    return AtomicNoteDraft(
        title=data.get("title") or rep.title,
        body=data.get("body") or rep.body,
        source_anchors=merged_anchors,
        related=[],  # Cross-Reference läuft NACH Canonicalization
        tags=safe_tags or data.get("tags") or rep.tags,
        aliases=sorted(set(data.get("aliases") or []) | all_aliases),
        synthesis_confidence=rep.synthesis_confidence,
        action=rep.action,
        extend_path=rep.extend_path,
        quality_flags=list({f for d in cluster for f in d.quality_flags}),
    )
