"""Cross-Draft-Hub-Resolution: erkennt MoC-/Hub-Notes anhand parallel erzeugter
Stage-Drafts. Wird nach Entity-Resolution und vor Critic aufgerufen.

Problem: Hub-Detector in `critic.run` prüft `draft.body` gegen `existing_concepts`
(Vault-Index VOR Pipeline-Run). Stage-Notes die parallel erzeugt werden, sind dort
nicht. Plus: Bodies enthalten Plain-Text-Mentions („Awareness", „Desire") statt
Wikilinks, weil zum Extraktions-Zeitpunkt die Stage-Notes noch nicht existieren.
Folge: ein Modell-Übersichts-Draft (z.B. ADKAR-Modell) bleibt fälschlich als
`atomic`, obwohl er die 5 Stage-Notes referenziert.

Fix: aggregiere die parallel erzeugten Drafts als pending_concepts. Common-Prefix
unter Drafts (z.B. „ADKAR " bei 6 Drafts) wird detektiert und der Suffix als
impliziter Single-Token-Match-Key registriert (Body sagt „Awareness" allein, aber
Title ist „ADKAR Awareness" → impliziter Key „awareness"). Hub-Klassifikation nur
wenn Title `_has_overview_marker()` triggert (Modell, Framework, MoC etc.) —
verhindert False-Positives bei Stage-Notes die sich gegenseitig erwähnen.
"""

from __future__ import annotations
import re

from generative.schemas.atomic_note import AtomicNoteDraft
from generative.agents.critic import _has_overview_marker

HUB_MIN_CROSS_MENTIONS = 3
SUGGEST_MIN_CLUSTER = 5  # ab wievielen marker-losen Drafts mit gemeinsamem Token ein MoC vorgeschlagen wird (#4)
# Token-Stoppwörter, die nicht als impliziter Single-Token-Key zählen (zu generisch)
_STOPWORD_TOKENS = {
    "modell",
    "model",
    "framework",
    "moc",
    "übersicht",
    "atlas",
    "taxonomie",
    "der",
    "die",
    "das",
    "und",
    "oder",
    "von",
    "the",
    "of",
    "and",
    "or",
}


_TITLE_SPLIT_RE = re.compile(r"[\s\-—]+")


def _detect_common_prefix(drafts: list[AtomicNoteDraft]) -> str:
    """Findet einen gemeinsamen Wort-Prefix in den Titles (z.B. „ADKAR" bei 6
    ADKAR-Notes mit Mix `ADKAR-Modell` und `ADKAR Awareness`). Tokenisiert über
    Whitespace UND Bindestrich/Em-Dash (sonst zählt „ADKAR-Modell" als 1 Token,
    Match scheitert). Returns lower-cased prefix oder "".
    """
    if len(drafts) < 3:
        return ""
    title_tokens = [_TITLE_SPLIT_RE.split(d.title.strip()) for d in drafts]
    if not all(title_tokens):
        return ""
    first_token = title_tokens[0][0].lower()
    if all(t and t[0].lower() == first_token for t in title_tokens):
        return first_token
    return ""


def _build_pending_index(drafts: list[AtomicNoteDraft]) -> dict[str, str]:
    """Title + Aliases jedes Drafts → Title des Drafts. Lower-keyed.
    Plus: bei Common-Prefix unter Drafts wird der Suffix-Token zusätzlich als
    impliziter Match-Key registriert (z.B. „awareness" bei „ADKAR Awareness").
    """
    idx: dict[str, str] = {}
    common_prefix = _detect_common_prefix(drafts)
    for d in drafts:
        idx[d.title.lower()] = d.title
        for alias in d.aliases:
            if alias and alias.lower() not in idx:
                idx[alias.lower()] = d.title
        # Implizit: Suffix-Tokens nach Common-Prefix (z.B. "ADKAR Awareness" → "awareness",
        # "ADKAR-Modell" → "modell"). Tokenisiert wie _detect_common_prefix.
        if common_prefix:
            tokens = _TITLE_SPLIT_RE.split(d.title.strip())
            if tokens and tokens[0].lower() == common_prefix:
                suffix = " ".join(tokens[1:]).lower().strip()
                if suffix and suffix not in _STOPWORD_TOKENS and suffix not in idx and len(suffix) >= 4:
                    idx[suffix] = d.title
    return idx


def _find_cross_mentions(draft: AtomicNoteDraft, pending_idx: dict[str, str]) -> list[str]:
    """Liefert canonical Titles anderer Drafts, die im draft.body als Plain-Text
    oder Wikilink vorkommen. Self-Match wird gefiltert. Single-Token-Match
    erlaubt — pending_idx ist kuratierte Planner-Liste, kein Vault-weiter Index.

    Reihenfolge der Treffer = erstes Vorkommen im Body. Bei Sequenz-Modellen
    (ADKAR-Stages, ISP-Phasen) entspricht die Body-Reihenfolge der Modell-
    Reihenfolge — Renderer kann daraus eine nummerierte Liste in korrekter
    Sequenz erzeugen.
    """
    self_keys = {draft.title.lower()} | {a.lower() for a in draft.aliases if a}
    body = draft.body
    body_lower = body.lower()
    found: dict[str, int] = {}  # canonical → first-match-position
    for key, canonical in pending_idx.items():
        if key in self_keys or canonical == draft.title or canonical in found:
            continue
        if key not in body_lower:
            continue
        m = re.search(rf"\b{re.escape(key)}\b", body, re.IGNORECASE)
        if m:
            found[canonical] = m.start()
    return [c for c, _ in sorted(found.items(), key=lambda x: x[1])]


def _embed_wikilinks(body: str, mentions: list[str], pending_idx: dict[str, str]) -> str:
    """Ersetzt Plain-Text-Mentions durch [[Wikilinks]]. Pro mentioned Title nur
    erste Occurrence ersetzen (Über-Verlinkung vermeiden). Bestehende Wikilinks
    werden nicht doppelt gewrappt.
    """
    canonical_to_keys: dict[str, list[str]] = {}
    for key, canonical in pending_idx.items():
        canonical_to_keys.setdefault(canonical, []).append(key)

    for canonical in mentions:
        # Längsten Key zuerst (spezifischste Match-Form)
        keys = sorted(canonical_to_keys.get(canonical, [canonical.lower()]), key=len, reverse=True)
        replaced = False
        for key in keys:
            if replaced:
                break
            # Negative-lookbehind/-ahead: bestehende [[..]] nicht doppelt wrappen
            pattern = re.compile(
                rf"(?<!\[\[)(?<!\[)\b{re.escape(key)}\b(?!\]\])(?!\])",
                re.IGNORECASE,
            )
            m = pattern.search(body)
            if m:
                body = body[: m.start()] + f"[[{canonical}]]" + body[m.end() :]
                replaced = True
    return body


def _extract_description_from_h1(body: str) -> str:
    """Extrahiert {Kerncharakteristik} aus '# {Title}: {Kerncharakteristik}'.
    Liefert leeren String wenn H1 oder Doppelpunkt-Trenner fehlt.
    """
    first_line = body.lstrip().split("\n", 1)[0].strip()
    if first_line.startswith("# ") and ": " in first_line:
        return first_line.split(": ", 1)[1].strip()
    return ""


def suggest_unmarked_clusters(drafts: list[AtomicNoteDraft]) -> list[tuple[str, list[str]]]:
    """Findet thematische Cluster OHNE Übersichts-Marker (#4): ≥SUGGEST_MIN_CLUSTER
    Drafts, deren Title einen gemeinsamen nicht-generischen Token teilen, und die
    `resolve()` mangels Marker nie als Hub erkennt (z.B. 8 Agent-Notes aus einem
    Guide-Run). Liefert (token, member_titles), nach Cluster-Größe absteigend.

    Bewusst seiteneffektfrei: schlägt nur einen MoC-Titel vor (`MoC-{Token}`),
    erzeugt KEINE Note — Auto-Anlage wäre Synthese-/Fabrikations-Risiko und ist
    eine separate, vom User zu treffende Entscheidung.

    Hub- und Marker-Drafts werden ausgeschlossen (die deckt `resolve()` ab),
    ebenso Member bereits aufgelöster Hubs (`hub_subconcepts`) — sonst würde
    derselbe Cluster direkt nach der Hub-Resolution erneut vorgeschlagen.
    Token < 4 Zeichen und Stoppwörter zählen nicht; Token mit identischer
    Member-Menge werden dedupliziert (das stärkste/alphabetisch erste gewinnt).
    """
    hub_member_titles = {title for d in drafts if d.action == "hub" for title in d.hub_subconcepts}
    candidates = [
        d
        for d in drafts
        if d.action != "hub" and not _has_overview_marker(d.title) and d.title not in hub_member_titles
    ]
    token_to_titles: dict[str, list[str]] = {}
    for d in candidates:
        seen: set[str] = set()
        for tok in _TITLE_SPLIT_RE.split(d.title.strip()):
            t = tok.lower()
            if len(t) >= 4 and t not in _STOPWORD_TOKENS and t not in seen:
                seen.add(t)
                token_to_titles.setdefault(t, []).append(d.title)
    out = [(tok, titles) for tok, titles in token_to_titles.items() if len(titles) >= SUGGEST_MIN_CLUSTER]
    out.sort(key=lambda x: (-len(x[1]), x[0]))
    seen_member_sets: set[frozenset[str]] = set()
    deduped = []
    for tok, titles in out:
        key = frozenset(titles)
        if key not in seen_member_sets:
            seen_member_sets.add(key)
            deduped.append((tok, titles))
    return deduped


def resolve(drafts: list[AtomicNoteDraft]) -> int:
    """Erkennt Hub-Drafts in der Liste, modifiziert sie in-place.
    Returns: Anzahl als Hub umklassifizierte Drafts.
    """
    if len(drafts) < HUB_MIN_CROSS_MENTIONS + 1:
        return 0
    pending_idx = _build_pending_index(drafts)
    title_to_draft = {d.title: d for d in drafts}
    hub_count = 0
    for draft in drafts:
        if draft.action == "hub":
            continue
        # Marker-Override: Hub-Klassifikation nur wenn Title-Marker (Modell/Framework/
        # MoC etc.) gesetzt ist. Verhindert dass Stage-Notes die sich gegenseitig
        # erwähnen fälschlich als Hub markiert werden.
        if not _has_overview_marker(draft.title):
            continue
        mentions = _find_cross_mentions(draft, pending_idx)
        if len(mentions) < HUB_MIN_CROSS_MENTIONS:
            continue
        draft.body = _embed_wikilinks(draft.body, mentions, pending_idx)
        draft.action = "hub"
        draft.hub_subconcepts = mentions
        # Beschreibungen aus zugehörigen Stage-Draft-H1s ziehen (Format
        # `# {Title}: {Kerncharakteristik}` — die Kerncharakteristik ist der
        # destillierte Glance-Satz und passt als Listen-Beschreibung).
        for sub_title in mentions:
            sub = title_to_draft.get(sub_title)
            if sub:
                desc = _extract_description_from_h1(sub.body)
                if desc:
                    draft.hub_subconcept_descriptions[sub_title] = desc
        draft.quality_flags.append(
            f"Hub-Auto-Detect: {len(mentions)} Cross-Mentions zu parallel erzeugten "
            f"Drafts ({', '.join(mentions[:4])}{'...' if len(mentions) > 4 else ''})"
        )
        hub_count += 1
    return hub_count
