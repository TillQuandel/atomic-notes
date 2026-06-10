"""Boilerplate-Dedup zwischen Hub-Drafts und ihren Sub-Konzept-Drafts (Hebel #5).

Konzeptzentrierter Extractor produziert pro Sub-Konzept eigene Empirie-/Methodik-
Blöcke, oft wortgleich (z.B. 6 Kuhlthau-ISP-Phasen-Notes mit identischem 3-Satz-
Empirie-Block). Diese Wiederholung gehört in die Hub-Note; Sub-Konzepte sollten
nur phase-spezifischen Inhalt tragen.

Strategie: Sätze die in ≥SHARED_THRESHOLD Sub-Drafts identisch vorkommen werden
als Boilerplate erkannt, aus den Sub-Drafts entfernt und (falls fehlend) dem
Hub-Draft am Ende des Bodys angehängt. Deterministisch, keine LLM-Calls.
"""
from __future__ import annotations
import re

from generative.schemas.atomic_note import AtomicNoteDraft

# Anzahl Sub-Drafts in denen ein Satz vorkommen muss, um als Boilerplate zu gelten.
# 2 = ein einziges Duplikat reicht. Konservativ: 3 vermeidet False-Positives bei
# zufälligen Phrasen-Ähnlichkeiten zwischen 2 Notes.
SHARED_THRESHOLD = 3
# Mindest-Wörter pro Boilerplate-Satz. Verhindert dass kurze Verbindungssätze
# („Vgl. dazu Kuhlthau.") als Boilerplate gestrippt werden.
MIN_SENTENCE_WORDS = 6
# Mindest-Wörter die ein Sub-Draft nach Stripping behalten muss. Sonst wird die
# Note inhaltsleer und das Stripping wird verworfen.
MIN_SUB_BODY_WORDS = 40

# Satz-Splitter: . ! ? gefolgt von Whitespace + Großbuchstabe oder Zeilenende.
# Behält Trennzeichen am Satzende.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])|\n\n+")

# Normalisierung für Match-Vergleich: Anker `(S. N)` und Whitespace raus.
_PAGE_REF = re.compile(r"\s*\(S\.?\s*\d+(?:[–\-]\d+)?\)")
_WS = re.compile(r"\s+")
_DISAMBIG = re.compile(r"\s*\([^)]*\)\s*")


def _strip_disambig_local(s: str) -> str:
    return _DISAMBIG.sub(" ", s).strip()


def _normalize(sentence: str) -> str:
    s = _PAGE_REF.sub("", sentence)
    s = _WS.sub(" ", s).strip().lower()
    return s


def _split_sentences(body: str) -> list[str]:
    """Body in Sätze splitten. Markdown-Listen-Items und Callouts bleiben atomar
    (werden nicht weiter zerlegt — Boilerplate sind Prosa-Sätze)."""
    parts = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # Listen, Callouts, Headings: 1 Block = 1 Einheit, nicht weiter splitten
        if block.startswith(("- ", "* ", "> ", "#", "|")):
            parts.append(block)
            continue
        for sent in _SENTENCE_RE.split(block):
            sent = sent.strip()
            if sent:
                parts.append(sent)
    return parts


def _strip_sentences(body: str, to_remove_normalized: set[str]) -> str:
    """Entfernt alle Sätze aus body deren Normalform in to_remove_normalized ist."""
    sentences = _split_sentences(body)
    kept = [s for s in sentences if _normalize(s) not in to_remove_normalized]
    # Reassemble: Absatzbrüche wiederherstellen — heuristisch je nach Original-Format.
    # Einfach: Doppelnewline zwischen Sätzen ist semantisch konservativ (mehr Whitespace
    # ist besser als verklebte Sätze). Markdown-Renderer kollabiert sowieso.
    return "\n\n".join(kept).strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _build_subconcept_index(drafts: list[AtomicNoteDraft]) -> dict[str, AtomicNoteDraft]:
    """Map lowercase-key (title + aliases) → draft. Für Hub→Sub-Auflösung."""
    idx: dict[str, AtomicNoteDraft] = {}
    for d in drafts:
        idx[d.title.lower()] = d
        for a in d.aliases:
            idx[a.lower()] = d
    return idx


def dedup_hub_subconcepts(drafts: list[AtomicNoteDraft]) -> tuple[list[AtomicNoteDraft], int]:
    """Strippt Boilerplate-Sätze aus Sub-Drafts pro Hub. Mutiert Drafts in-place.
    Returns (drafts, total_stripped_sentences)."""
    if not drafts:
        return drafts, 0

    sub_index = _build_subconcept_index(drafts)
    total_stripped = 0

    for hub in drafts:
        if hub.action != "hub":
            continue

        # Sub-Drafts via zwei Kanälen:
        # (a) hub.hub_subconcepts (vom Hub-Detector — Body-Mentions im Hub-Note)
        # (b) reverse: Siblings die den Hub als [[Wikilink]] in ihrer related-Liste führen.
        #     Beispiel ISP: ISP-Modell-Hub erwähnt Phasen nicht im Body, aber jede Phase-Note
        #     hat [[ISP Modell (Kuhlthau)]] in related → Phasen sind Sub-Konzepte des Modells.
        sub_drafts: list[AtomicNoteDraft] = []
        seen_ids: set[int] = set()

        # Hub-Token-Sets aus Title und Aliases (Disambig-Klammern entfernt). ≥2 Tokens
        # Mindestlänge, sonst matchen generische Single-Words.
        hub_token_sets: list[frozenset[str]] = []
        for label in [hub.title, *hub.aliases]:
            toks = frozenset(_strip_disambig_local(label).lower().split())
            if len(toks) >= 2:
                hub_token_sets.append(toks)

        for sc in hub.hub_subconcepts:
            sub = sub_index.get(sc.lower())
            if sub is not None and id(sub) != id(hub) and id(sub) not in seen_ids:
                sub_drafts.append(sub)
                seen_ids.add(id(sub))

        # Reverse-Mapping: Sibling der Hub-Title (Token-Subset) in related-Wikilink hat.
        for d in drafts:
            if id(d) == id(hub) or id(d) in seen_ids:
                continue
            for r in d.related:
                # `[[Foo Bar]]` → stripped Tokens
                stripped = re.sub(r"^\[\[|\]\]$", "", r.strip()).strip()
                stripped = _strip_disambig_local(stripped)
                rel_toks = frozenset(stripped.lower().split())
                if len(rel_toks) < 2:
                    continue
                if any(rel_toks <= h or h <= rel_toks for h in hub_token_sets):
                    sub_drafts.append(d)
                    seen_ids.add(id(d))
                    break

        if len(sub_drafts) < SHARED_THRESHOLD:
            continue

        # Satz-Häufigkeit über alle Sub-Drafts (1× pro Draft, Mehrfach-Vorkommen
        # innerhalb eines Drafts zählt 1×).
        sentence_to_drafts: dict[str, set[int]] = {}
        sentence_canonical: dict[str, str] = {}  # Normalform → Original-Satz (1. Vorkommen)
        for sub in sub_drafts:
            seen_norms_local: set[str] = set()
            for sent in _split_sentences(sub.body):
                if _word_count(sent) < MIN_SENTENCE_WORDS:
                    continue
                norm = _normalize(sent)
                if not norm or norm in seen_norms_local:
                    continue
                seen_norms_local.add(norm)
                sentence_to_drafts.setdefault(norm, set()).add(id(sub))
                sentence_canonical.setdefault(norm, sent)

        boilerplate_norms = {n for n, ids in sentence_to_drafts.items()
                             if len(ids) >= SHARED_THRESHOLD}
        if not boilerplate_norms:
            continue

        # Stripping: pro Sub-Draft prüfen ob nach Stripping noch genug Body übrig.
        # Wenn nicht: Sub-Draft unverändert lassen (Schutz vor inhaltsleeren Notes).
        hub_existing_norms = {_normalize(s) for s in _split_sentences(hub.body)}
        stripped_in_hub_run = 0
        for sub in sub_drafts:
            stripped_body = _strip_sentences(sub.body, boilerplate_norms)
            if _word_count(stripped_body) < MIN_SUB_BODY_WORDS:
                continue
            removed = len(_split_sentences(sub.body)) - len(_split_sentences(stripped_body))
            if removed <= 0:
                continue
            sub.body = stripped_body
            sub.quality_flags.append(
                f"Boilerplate-Dedup: {removed} Satz/Sätze aus Hub [[{hub.title}]] zentralisiert"
            )
            stripped_in_hub_run += removed

        if stripped_in_hub_run == 0:
            continue
        total_stripped += stripped_in_hub_run

        # Boilerplate-Sätze die noch nicht im Hub sind, am Hub-Body anhängen unter
        # eigenem Abschnitt — semantisch klar abgegrenzt von Hub-eigenem Prosa.
        missing = [sentence_canonical[n] for n in boilerplate_norms
                   if n not in hub_existing_norms]
        if missing:
            block = "\n\n".join(missing).strip()
            hub.body = hub.body.rstrip() + "\n\n## Geteilte Empirie\n\n" + block + "\n"
            hub.quality_flags.append(
                f"Boilerplate-Dedup: {len(missing)} geteilten Satz/Sätze aus Sub-Notes übernommen"
            )

    return drafts, total_stripped
