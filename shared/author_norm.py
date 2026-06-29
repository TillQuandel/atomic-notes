"""Autor-Normalisierung — geteilt zwischen Dateiname-Parsern (pdf_enrich +
vault_writer). Reine Funktion, keine internen Deps (unterste Schicht).

Bug-Klasse (Mahmood-Lauf 2026-06-25): Zotero (oft deutsche Locale) hängt die
Affiliation als zweiten "Autor" an — `Mahmood und University of the Punjab`.
Die Affiliation ist kein Koautor; sie als solchen zu behandeln verfälscht jede
Inline-Zitation ("Mahmood & Punjab"), die Planner-origin-Klassifikation und das
Footnote-Label ("et al.").
"""
from __future__ import annotations

import re

# Klar institutionelle Marker (Wortgrenze, case-insensitiv). Bewusst KEINE
# Akronyme (MIT, ETH) und kein "school"/"college" allein — diese sind als
# Nachnamen mehrdeutig. Die ≥1-Person-bleibt-Garantie unten schützt zusätzlich:
# eine reine Personenliste wird nie angefasst, ein reiner Korporativ-Autor bleibt.
_INSTITUTION_RE = re.compile(
    r"\b("
    r"universi(?:ty|t[äa]t|dad|t[ée]|t[àa])"
    r"|institut(?:e|o|ion)?"
    r"|department|fakult[äa]t|faculty"
    r"|hochschule|polytechnic"
    r"|academy|akademie"
    r"|laborator(?:y|ies)|laboratoire"
    r"|hospital|klinik|clinic"
    r"|minist(?:ry|erium|ère)"
    r"|foundation|stiftung"
    r"|society|gesellschaft|associat(?:ion|ed)|verband"
    r"|council|committee|kommission|commission"
    r"|corporation|incorporated|gmbh|inc|ltd|llc|plc"
    r"|centre|center|zentrum"
    r"|bureau|agency|agentur"
    r"|organi[sz]ation"
    r")\b",
    re.IGNORECASE,
)

# Autor-Trenner: ';', ' und ', ' and ', ' & '. (Komma NICHT — würde
# 'Lastname, Firstname' fälschlich splitten.) IGNORECASE: Zotero/manuelle
# Renames liefern auch 'UND'/'AND' (Qwen-Review HIGH 1).
_AUTHOR_SEP_RE = re.compile(r"\s*;\s*|\s+und\s+|\s+and\s+|\s*&\s*", re.IGNORECASE)


def _looks_institutional(segment: str) -> bool:
    """Ein Segment gilt nur als institutionell, wenn es ≥2 Tokens hat UND einen
    Institutions-Marker trägt. Der Token-Guard schützt 1-Wort-Nachnamen, die
    zufällig ein Marker-Wort sind (Hospital, Bureau, Center, Foundation als
    Personenname) vor falschem Strippen (Qwen/Codex-Review)."""
    if len(segment.split()) < 2:
        return False
    return bool(_INSTITUTION_RE.search(segment))


def drop_institutional_coauthors(author: str) -> str:
    """Entfernt institutionelle Affiliations-Segmente aus einem Autor-String —
    aber nur, wenn mindestens ein Personen-Segment übrig bleibt.

    - ``"Mahmood und University of the Punjab"`` → ``"Mahmood"``
    - ``"Schlebbe und Greifeneder"`` → unverändert (beides Personen)
    - ``"World Health Organization"`` → unverändert (reiner Korporativ-Autor)

    Der Trenner zwischen verbleibenden Personen wird aus dem Original übernommen,
    damit Downstream-Parser (`_short_author`, `_short_label`) unverändert greifen.
    """
    if not author or not author.strip():
        return author
    sep_match = _AUTHOR_SEP_RE.search(author)
    if not sep_match:
        return author  # ein einziges Segment — nichts zu trennen
    parts = [p.strip() for p in _AUTHOR_SEP_RE.split(author) if p.strip()]
    if len(parts) < 2:
        return author
    persons = [p for p in parts if not _looks_institutional(p)]
    institutional = [p for p in parts if _looks_institutional(p)]
    # Nur eingreifen, wenn sich Personen UND Institutionen mischen.
    if not persons or not institutional:
        return author
    sep = sep_match.group()
    return sep.join(persons)
