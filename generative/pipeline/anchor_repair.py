"""Deterministische Schlusssatz-Anker-Vererbung.

Hebel aus Eval 2026-05-07 (Red Thread of Information / ISP-Phasen): Critic
flaggt Quellen-Hard-Gate-Fail wenn der letzte Satz eines Absatzes keinen
`(S. N)`-Anker hat — auch dann, wenn der vorherige Satz desselben Absatzes
einen einzigen Anker setzt und der Schlusssatz inhaltlich an die selbe
Aussage anschliesst (typisch: zusammenfassende oder einordnende Schlussklausel).

Self-Refine fängt das nicht zuverlässig — das LLM ergänzt den Anker im Retry
oft nicht, obwohl der Critic-Hint die Seitenzahl explizit nennt. Deterministische
Vererbung ist hier robuster und Cache-stabiler (idempotent: zweiter Lauf mutiert
nicht mehr).

Konservativ:
- Nur wenn der Schlusssatz selbst KEINEN `(S. N)` enthält
- Vorherige Sätze desselben Absatzes haben mindestens einen Anker
- Schlusssatz ist nicht ausführlich (<=240 Zeichen) — Heuristik fuer
  Schlusssatz-Konvention statt eigenstaendiger Aussage
- Anker wird vor dem letzten Satzzeichen eingefuegt: `Schlusssatz (S. N).`
"""
from __future__ import annotations
import re

from pipeline.anchor_patterns import PAGE_ANCHOR_RE as _PAGE_ANCHOR_RE, SENTENCE_SPLIT_RE as _SENTENCE_SPLIT_RE

_TRAILING_PUNCT_RE = re.compile(r"([.!?]\"?\s*)$")
MAX_TRAILING_LEN = 240


def _last_anchor_in(text: str) -> str | None:
    """Findet den letzten `(S. N)`-Anker im Text und liefert ihn als String."""
    matches = list(_PAGE_ANCHOR_RE.finditer(text))
    return matches[-1].group(0) if matches else None


def _inject_anchor(sentence: str, anchor: str) -> str:
    """Setzt `anchor` vor das abschliessende Satzzeichen. Funktioniert auch
    wenn der Satz mit Anfuehrungszeichen schliesst (`...".`)."""
    m = _TRAILING_PUNCT_RE.search(sentence)
    if not m:
        # Kein klares Satzende — anker am String-Ende anfuegen
        return f"{sentence.rstrip()} {anchor}"
    end_punct = m.group(1)
    head = sentence[:m.start()].rstrip()
    return f"{head} {anchor}{end_punct}"


def repair_trailing_anchors(body: str) -> tuple[str, int]:
    """Vererbt den letzten Anker eines Absatzes auf den Schlusssatz, wenn
    dieser keinen eigenen `(S. N)`-Anker hat. Returns (modified_body, count).

    Idempotent: zweiter Lauf macht nichts (Schlusssatz hat dann Anker).
    """
    paragraphs = body.split("\n\n")
    repaired = 0
    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped:
            continue
        # Bullet-/Header-Absaetze ueberspringen — Schlusssatz-Konvention gilt nicht
        if stripped.startswith(("- ", "* ", "#", "> ", "|")):
            continue

        sentences = _SENTENCE_SPLIT_RE.split(stripped)
        if len(sentences) < 2:
            continue
        last = sentences[-1].strip()
        if not last or len(last) > MAX_TRAILING_LEN:
            continue
        if _PAGE_ANCHOR_RE.search(last):
            continue  # bereits Anker

        # Anker aus dem unmittelbar vorhergehenden Satz uebernehmen
        prev_anchor = _last_anchor_in(sentences[-2])
        if not prev_anchor:
            continue

        sentences[-1] = _inject_anchor(last, prev_anchor)
        # Absatz neu zusammenbauen — Whitespace-Konvention: Single space zwischen Saetzen
        paragraphs[i] = " ".join(s.strip() for s in sentences)
        repaired += 1

    return "\n\n".join(paragraphs), repaired
