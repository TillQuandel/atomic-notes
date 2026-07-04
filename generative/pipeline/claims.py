"""High-Risk-Claim-Dekomposition (Faithfulness-Gate E2, #69).

Pure, deterministische Zerlegung eines Draft-Bodys (Markdown ohne Frontmatter,
Seitenanker inline als `(S. N)`/`(zit. n. Autor, S. N)`, Blockquotes mit
`> `-Prefix) in risiko-markierte Claims. Kein LLM, keine Embeddings, keine
Pipeline-Verdrahtung — nur Regex/String-Logik.

Die Attributions-Regexes (`attribution_risk` + Konstanten) sind bewusst
modulöffentlich und einzeln importierbar, weil #96 (CitationMeta-Validierung)
dieselben Muster wiederverwenden wird.

**Satz-Split-Abweichung (dokumentiert, empirisch begründet):** Das Sentence-
Splitting nutzt `_SENT_SPLIT` aus `generative/pipeline/embeddings.py` (bereits
cross-modul präzedenzhaft genutzt, siehe `agents/planner.py`). `_SENT_SPLIT`
matched aber jedes `.` gefolgt von Whitespace, unabhängig vom Folgezeichen —
für den dortigen Embedding-Use-Case unkritisch ("keine perfekte Tokenisierung
nötig"), hier aber fatal: JEDER Seitenanker `(S. N)` und die Attribution
`zit. n.` enthalten genau dieses Muster (`S.` + Leerzeichen + Ziffer) und
würden sonst mitten im Anker aufgespalten (verifiziert: `_SENT_SPLIT.split(
"... (S. 3)")` → `["... (S.", "3)"]`). Deshalb werden diese beiden bekannten
Abkürzungs-Muster (`S. <Zahl>`, `zit. n.`, `et al.`) vor dem Split maskiert
(Punkt → Sentinel) und danach wiederhergestellt. Das ist keine Neuerfindung
des Sentence-Splittings — nur ein Schutz der zwei Token-Klassen, die dieses
Modul selbst als Risk-Pattern führt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from generative.pipeline.embeddings import _SENT_SPLIT


@dataclass
class Claim:
    text: str
    anchor_page: int | None
    anchor_span: tuple[int, int]
    risk_types: list[str]
    is_quote: bool


# ---- Attribution -----------------------------------------------------------

ZIT_N_RE = re.compile(r"zit\.\s*n\.")
ET_AL_RE = re.compile(r"et al\.")
AUTHOR_YEAR_RE = re.compile(r"[A-ZÄÖÜ][\wäöüß\-]+(?:\s+(?:&|und)\s+[A-ZÄÖÜ][\wäöüß\-]+)?\s*\(?(?:19|20)\d{2}\)?")
LAUT_RE = re.compile(r"\b(?i:laut)\s+[A-ZÄÖÜ]\w+")
ZUFOLGE_RE = re.compile(r"\b[A-ZÄÖÜ]\w+\s+(?i:zufolge)\b")
NACH_ANSICHT_VON_RE = re.compile(r"(?i:nach Ansicht von)")

ATTRIBUTION_PATTERNS = (
    ZIT_N_RE,
    ET_AL_RE,
    AUTHOR_YEAR_RE,
    LAUT_RE,
    ZUFOLGE_RE,
    NACH_ANSICHT_VON_RE,
)


def attribution_risk(text: str) -> bool:
    return any(p.search(text) for p in ATTRIBUTION_PATTERNS)


# ---- Number -----------------------------------------------------------------

# Seitenverweis-Kontext (auch Spannen `S. 5-8`, `S. 5, 8`) — wird vor der
# Ziffern-Suche ausmaskiert, da Seitenzahlen selbst kein number-Risk sind.
PAGE_REF_RE = re.compile(r"S\.\s*\d+(?:\s*[\-–,]\s*(?:S\.\s*)?\d+)*")
# Reine Jahreszahl (optional in Klammern) — ebenfalls kein number-Risk.
YEAR_TOKEN_RE = re.compile(r"\(?(?:19|20)\d{2}\)?")


def number_risk(text: str) -> bool:
    masked = PAGE_REF_RE.sub("", text)
    masked = YEAR_TOKEN_RE.sub("", masked)
    return bool(re.search(r"\d", masked))


# ---- Comparison ---------------------------------------------------------------

COMPARISON_TERMS = (
    "mehr als",
    "weniger als",
    "höher",
    "niedriger",
    "stärker",
    "schwächer",
    "größer",
    "kleiner",
    "effektiver",
    "besser als",
    "schlechter als",
    "im Vergleich",
    "gegenüber",
    "überlegen",
    "unterlegen",
)
COMPARISON_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in COMPARISON_TERMS) + r")\b",
    re.IGNORECASE,
)


def comparison_risk(text: str) -> bool:
    return bool(COMPARISON_RE.search(text))


# ---- Causal -------------------------------------------------------------------

CAUSAL_TERMS = (
    "führt zu",
    "führen zu",
    "bewirkt",
    "verursacht",
    "weil",
    "daher",
    "deshalb",
    "infolge",
    "aufgrund",
    "bedingt durch",
    "zur Folge",
)
CAUSAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in CAUSAL_TERMS) + r")\b",
    re.IGNORECASE,
)


def causal_risk(text: str) -> bool:
    return bool(CAUSAL_RE.search(text))


# ---- Zeilen-Skip + Satz-Split -----------------------------------------------

_QUELLEN_HEADING_RE = re.compile(r"^#+\s*Quellen\s*$", re.IGNORECASE)
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^[^\]]+\]:")
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^>\s?")

# Bekannte Abkürzungs-Muster, deren Punkt vor dem Sentence-Split geschützt
# werden muss (siehe Modul-Docstring).
_ABBREV_GUARD_RE = re.compile(r"S\.\s*\d+|zit\.\s*n\.|et al\.")
_MASK_CHAR = "\x00"

# Seitenzahl-Extraktion: erste Übereinstimmung `S. <Zahl>` im Satz.
_PAGE_NUM_RE = re.compile(r"S\.\s*(\d+)")


def _split_sentences(content: str) -> list[str]:
    protected = _ABBREV_GUARD_RE.sub(lambda m: m.group(0).replace(".", _MASK_CHAR), content)
    fragments = _SENT_SPLIT.split(protected)
    return [f.replace(_MASK_CHAR, ".").strip() for f in fragments if f.strip()]


def _risk_types(text: str) -> list[str]:
    types: list[str] = []
    if attribution_risk(text):
        types.append("attribution")
    if number_risk(text):
        types.append("number")
    if comparison_risk(text):
        types.append("comparison")
    if causal_risk(text):
        types.append("causal")
    return types


def _anchor_page(text: str) -> int | None:
    match = _PAGE_NUM_RE.search(text)
    return int(match.group(1)) if match else None


def decompose_claims(body: str) -> list[Claim]:
    """Zerlegt `body` satzweise, liefert nur Sätze mit >=1 Risk-Match.

    Übersprungen (nicht gesplittet): Überschriften (`#`-Prefix), Footnote-
    Definitionen (`[^i]:`-Prefix), alles ab einer `## Quellen`-Überschrift
    bis Body-Ende, Leerzeilen. Blockquote-Sätze (`> `-Prefix) werden MIT
    `is_quote=True` zurückgegeben — Filterung ist Aufgabe des späteren Gates.
    """
    if not body:
        return []

    claims: list[Claim] = []
    offset = 0
    in_sources = False

    for line in body.split("\n"):
        line_start = offset
        offset += len(line) + 1  # +1 für den beim Split entfernten Zeilenumbruch

        stripped = line.strip()
        if in_sources or not stripped:
            continue

        if stripped.startswith("#"):
            if _QUELLEN_HEADING_RE.match(stripped):
                in_sources = True
            continue

        if _FOOTNOTE_DEF_RE.match(stripped):
            continue

        is_quote = stripped.startswith(">")
        content = _BLOCKQUOTE_PREFIX_RE.sub("", stripped) if is_quote else stripped

        search_from = line_start
        for sentence in _split_sentences(content):
            idx = body.find(sentence, search_from)
            if idx == -1:
                idx = body.find(sentence)

            risk_types = _risk_types(sentence)
            if not risk_types:
                if idx != -1:
                    search_from = idx + len(sentence)
                continue

            if idx == -1:
                start, end = line_start, line_start + len(line)
            else:
                start, end = idx, idx + len(sentence)
                search_from = end

            claims.append(
                Claim(
                    text=sentence,
                    anchor_page=_anchor_page(sentence),
                    anchor_span=(start, end),
                    risk_types=risk_types,
                    is_quote=is_quote,
                )
            )

    return claims
