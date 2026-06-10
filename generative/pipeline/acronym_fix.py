"""Sprachagnostische Akronym-Auflösung — Hybrid Schwartz-Hearst + LLM-Fallback.

Architektur (siehe [[Akronym-Erkennung]] im Wissenspool):

1. **Schwartz-Hearst-Stage** — scannt das Quell-PDF nach „Long Form (Short Form)"
   und „Short Form (Long Form)" Pattern. Letter-Matching prüft, dass jeder Char
   der Short Form in der Long Form in Reihenfolge vorkommt. Sprachagnostisch
   (Pattern funktioniert in DE+EN gleich), 96% Precision laut Originalpaper.
2. **Body-Insertion** — fügt beim ersten Vorkommen jedes erkannten Akronyms
   `(<Auflösung>)` ein. Idempotent.
3. **LLM-Fallback (opt-in via ENABLE_ACRONYM_LLM_FALLBACK)** — für Akronyme
   die im Body auftauchen aber nicht im Schwartz-Hearst-Dict (globale Akronyme
   ohne Auflösung im PDF). LLM erhält Kontext-Fenster + Akronym, schlägt
   Auflösung vor. Halluzinations-Risiko via Quality-Flag transparent gehalten.

Keine private Whitelist — alle Auflösungen kommen aus dem Quell-PDF oder via
LLM-Inferenz. Publish-tauglich, multilingual, multi-domain.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass


# ---------- Schwartz-Hearst Pattern Detection -----------------------------

# Short-Form-Validierung: 2-10 chars, mindestens ein Buchstabe, erster Char alphanumerisch.
# Schwartz & Hearst (2003) Originalspec.
_SHORT_FORM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-&]{1,9}$")

# Pattern (i): "Long Form (Short Form)" — Klammer enthält Kandidaten für Short Form
# Klammer-Inhalt-Limit großzügig (200 chars) damit auch lange Long Forms in
# Pattern (ii) ("Short Form (Long Form)") matchen.
_PAT_LONG_PAREN_SHORT = re.compile(
    r"([^()]{1,200}?)\s*\(\s*([A-Za-z0-9][A-Za-z0-9\-&,.\s]{1,200}?)\s*\)"
)

# Pattern (ii): "Short Form (Long Form)" — Klammer enthält Long Form (>2 Wörter)
# wird unten anhand Wort-Anzahl in der Klammer entschieden


@dataclass
class AcronymPair:
    short: str
    long: str
    pattern: str  # "i" or "ii"


def _short_is_valid(s: str) -> bool:
    """Short Form: 2-10 chars, erster Char Großbuchstabe, mindestens 50%
    Großbuchstaben (Akronym-typisch). Schwartz-Hearst-Original ist hier zu lax —
    diese strengere Variante eliminiert normale Wörter wie „Technology" oder
    „least", die durch Pattern-Match in lange Klammer-Inhalte rutschen würden.
    """
    if not _SHORT_FORM_RE.match(s):
        return False
    if len(s) < 2 or len(s) > 10:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_count = sum(1 for c in letters if c.isupper())
    # Mindestens 2 Großbuchstaben (eliminiert „Technology", „least") UND ≥50%
    # Großbuchstaben-Ratio. Lässt Mixed-Case-Akronyme durch (fMRT, mRNA, ASIS&T).
    if upper_count < 2:
        return False
    if upper_count / len(letters) < 0.5:
        return False
    return True


def _letter_match(short: str, long: str) -> bool:
    """Schwartz-Hearst Letter-Matching: jeder Char (case-insensitive) der Short Form
    muss in der Long Form in Reihenfolge vorkommen. Erster Char muss am Wort-Anfang
    matchen.

    Beispiel: "CSCW" matched "Computer-Supported Cooperative Work" weil C,S,C,W
    in dieser Reihenfolge auftauchen, jedes am Wortanfang.
    """
    short = short.lower()
    long_lower = long.lower()
    s_idx = len(short) - 1
    l_idx = len(long_lower) - 1

    # Iteriere von hinten — erster Short-Char muss am Wortanfang in Long matchen
    while s_idx >= 0:
        char = short[s_idx]
        # Nicht-Buchstaben in Short überspringen (z.B. & in „R&D")
        if not char.isalnum():
            s_idx -= 1
            continue
        found = False
        while l_idx >= 0:
            if long_lower[l_idx] == char:
                # Erster Short-Char (s_idx==0) muss am Wortanfang stehen
                if s_idx == 0 and l_idx > 0 and long_lower[l_idx - 1].isalnum():
                    l_idx -= 1
                    continue
                found = True
                l_idx -= 1
                break
            l_idx -= 1
        if not found:
            return False
        s_idx -= 1
    return True


def _trim_long_form(long_candidate: str, short: str) -> str | None:
    """Long Form trimmen: maximal min(|A|+5, |A|·2) Wörter, die letzten n Wörter vor
    der Klammer (Schwartz-Hearst 2003).

    Iteriert von 1 Wort bis max_len, gibt den kürzesten letter-matchenden
    Kandidaten zurück. Wichtig für Compound-Words mit Bindestrich:
    „Mountain-Bike (MTB)" zählt als 1 Wort beim split(), letter-match greift
    aber durch alle Buchstaben.
    """
    words = long_candidate.split()
    if not words:
        return None
    max_len = min(len(words), max(len(short) + 5, len(short) * 2))
    for n in range(1, max_len + 1):
        candidate = " ".join(words[-n:])
        if _letter_match(short, candidate):
            return candidate
    return None


def extract_acronym_pairs(text: str) -> dict[str, str]:
    """Schwartz-Hearst-Scanner über kompletten Quell-Text.

    Returns: {short_form: long_form} dict, dynamisch aus dem Text extrahiert.
    Keine globalen Whitelists, keine privaten Daten.

    Beide Patterns abgedeckt:
    - „Long Form (Short Form)" — typisch in wissenschaftlichen PDFs
    - „Short Form (Long Form)" — typisch in Lehrbüchern, Glossaren
    """
    pairs: dict[str, str] = {}
    seen_orders: dict[str, str] = {}  # Prävention von späteren falschen Re-Definitionen

    for match in _PAT_LONG_PAREN_SHORT.finditer(text):
        before = match.group(1).strip()
        in_paren = match.group(2).strip()

        # Pattern (ii): wenn Klammer-Inhalt > 2 Wörter ist, dann ist short=before, long=in_paren
        in_paren_words = in_paren.split()
        if len(in_paren_words) >= 3:
            # Short Form ist letztes Wort vor Klammer
            before_words = before.split()
            if not before_words:
                continue
            short = before_words[-1].rstrip(",.;:")
            if not _short_is_valid(short):
                continue
            if _letter_match(short, in_paren) and short not in pairs:
                pairs[short] = in_paren.strip()
            continue

        # Pattern (i): Klammer-Inhalt ist Short Form, before enthält Long Form
        short = in_paren
        if not _short_is_valid(short):
            continue
        if short in pairs:
            continue
        long_form = _trim_long_form(before, short)
        if long_form:
            pairs[short] = long_form

    return pairs


# ---------- Body-Insertion (idempotent) -----------------------------------

def _already_resolved(body: str, acronym: str, expansion: str) -> bool:
    """Akronym gilt als aufgelöst, wenn die Langform irgendwo im Body steht
    (egal ob als `(...)` direkt nach dem Akronym oder als eigenständige Phrase)."""
    return expansion.lower() in body.lower()


def expand_acronyms(body: str,
                    whitelist: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Fügt beim ersten Vorkommen jedes Akronyms aus der Whitelist `(<Auflösung>)` ein.

    Returns: (modifizierter Body, Liste der eingefügten Akronyme).

    Regeln:
    - Word-boundary-Match um Substrings wie `Sub-CERQual` auszuschließen.
    - Pre-Check `_already_resolved`: wenn Langform irgendwo im Body steht, skip.
    - Wenn das Akronym bereits direkt von `(...)` gefolgt wird, skip
      (idempotent gegenüber LLM-Output der die Klammer schon gesetzt hat).
    - Nur das ERSTE Vorkommen wird modifiziert.
    """
    if not whitelist:
        return body, []
    expanded: list[str] = []
    for acronym, expansion in whitelist.items():
        if _already_resolved(body, acronym, expansion):
            continue
        pattern = re.compile(rf"\b{re.escape(acronym)}\b(?!\s*\()")
        m = pattern.search(body)
        if not m:
            continue
        insert_at = m.end()
        body = body[:insert_at] + f" ({expansion})" + body[insert_at:]
        expanded.append(acronym)
    return body, expanded


# ---------- LLM-Fallback für globale Akronyme -----------------------------

# Heuristik: Akronym-Kandidaten im Body, die NICHT im Schwartz-Hearst-Dict stehen.
# Pattern: ≥2 chars, ≥2 Großbuchstaben (mind. 60%), 2-10 chars total.
_BODY_ACRONYM_RE = re.compile(r"\b([A-ZÄÖÜ][A-ZÄÖÜ0-9\-&]{1,9})\b")
_COMMON_NON_ACRONYMS = frozenset({
    "DER", "DIE", "DAS", "UND", "ODER", "ICH", "WIR", "SIE", "ER", "ES",
    "THE", "AND", "OR", "FOR", "NOT", "BUT", "YES", "NO", "TO", "OF",
    "GMBH", "KG", "EG", "EU", "USA", "UK",  # Common but rarely need expansion
})


def _candidate_acronyms_in_body(body: str, resolved: set[str]) -> list[tuple[str, str]]:
    """Returns [(acronym, context_snippet)] für Akronym-Kandidaten im Body, die
    NICHT bereits aufgelöst sind. Context-Snippet = ±100 chars um erstes Vorkommen.
    """
    seen = set()
    candidates: list[tuple[str, str]] = []
    for m in _BODY_ACRONYM_RE.finditer(body):
        token = m.group(1)
        if token in seen:
            continue
        if token in resolved:
            continue
        if token in _COMMON_NON_ACRONYMS:
            continue
        # Skip wenn sofort von `(` gefolgt — das ist eine bereits-gegebene Auflösung
        if body[m.end():m.end() + 2].lstrip().startswith("("):
            continue
        seen.add(token)
        start = max(0, m.start() - 100)
        end = min(len(body), m.end() + 100)
        ctx = body[start:end]
        candidates.append((token, ctx))
    return candidates


_LLM_FALLBACK_PROMPT = """Du bekommst ein Akronym + Kontext-Fenster aus einem Dokument.

Aufgabe: Falls das Akronym im Kontext eindeutig identifizierbar ist, gib die wahrscheinlichste Langform zurück. Bei Unsicherheit oder Mehrdeutigkeit antworte „UNKNOWN".

## Regeln
- Nur antworten wenn die Langform aus Kontext + Allgemeinwissen mit hoher Sicherheit ableitbar ist.
- Bei mehreren plausiblen Interpretationen → UNKNOWN.
- Keine Phantasie, keine Spekulation. Lieber UNKNOWN als falsch.

## Akronym
{acronym}

## Kontext
{context}

## Output (genau eine Zeile)
LONG_FORM: <Langform>  ODER  LONG_FORM: UNKNOWN
"""


def llm_resolve_unknown(acronym: str, context: str) -> str | None:
    """LLM-Fallback für ein einzelnes Akronym mit Kontext. Returns Langform oder None.

    Synchron, blockierend — pro PDF typischerweise ≤5 Calls (nur globale
    Akronyme ohne lokale Auflösung). Token-Cost vernachlässigbar.
    """
    from generative.agents.base import call_claude_sync
    from generative.config import MODEL_HAIKU

    prompt = _LLM_FALLBACK_PROMPT.format(acronym=acronym, context=context)
    try:
        raw = call_claude_sync(prompt, model=MODEL_HAIKU, agent="acronym_llm")
    except Exception:
        return None
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("LONG_FORM:"):
            value = line[len("LONG_FORM:"):].strip()
            if value and value != "UNKNOWN" and len(value) > len(acronym):
                return value
    return None


def llm_fallback_resolve(body: str, schwartz_hearst_dict: dict[str, str]) -> dict[str, str]:
    """Sucht im Body nach Akronymen die NICHT im SH-Dict sind, fragt LLM nach Auflösung.

    Returns: dict mit zusätzlichen {acronym: long_form}. Leeres Dict wenn LLM-Fallback
    deaktiviert ist (ENABLE_ACRONYM_LLM_FALLBACK=0).
    """
    if os.getenv("ENABLE_ACRONYM_LLM_FALLBACK", "0") not in ("1", "true", "True"):
        return {}
    resolved_set = set(schwartz_hearst_dict.keys())
    candidates = _candidate_acronyms_in_body(body, resolved_set)
    extra: dict[str, str] = {}
    for acronym, ctx in candidates[:10]:  # Cap auf 10 Kandidaten pro Body
        long_form = llm_resolve_unknown(acronym, ctx)
        if long_form:
            extra[acronym] = long_form
    return extra
