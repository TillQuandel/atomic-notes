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
#
# Plural-Toleranz (#74 Bug A): jede Alternative endete zuvor mit `\b` → das
# Plural-`s` (ein Word-Char) verhinderte den Match ("National Institutes of
# Health" blieb ungestrippt). y→ies-Marker tragen ihre Pluralform inline
# (universi…ties, academ…ies …), alle übrigen deckt das nachgestellte
# `(?:e?s)?` vor `\b` ab (Institutes, Departments, Centers, Hospitals …).
_INSTITUTION_RE = re.compile(
    r"\b(?:"
    r"universi(?:t(?:y|ies)|t[äa]t|dad|t[ée]|t[àa])"
    r"|institut(?:e|o|ion)?"
    r"|department|fakult[äa]t|facult(?:y|ies)"
    r"|hochschule|polytechnic"
    r"|academ(?:y|ies)|akademie"
    r"|laborator(?:y|ies)|laboratoire"
    r"|hospital|klinik|clinic"
    r"|minist(?:r(?:y|ies)|erium|ère)"
    r"|foundation|stiftung"
    r"|societ(?:y|ies)|gesellschaft|associat(?:ion|ed)|verband"
    r"|council|committee|kommission|commission"
    r"|corporation|incorporated|gmbh|inc|ltd|llc|plc"
    r"|centre|center|zentrum"
    r"|bureau|agenc(?:y|ies)|agentur"
    r"|organi[sz]ation"
    r")(?:e?s)?\b",
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


def _strip_comma_affiliation(segment: str) -> str:
    """Entfernt eine institutionelle Komma-Affiliation aus EINEM Segment und behält
    die Person(en): ``"Mahmood, University of the Punjab"`` → ``"Mahmood"``.

    Komma ist bewusst kein Autor-Trenner (schützt ``"Lastname, Firstname"``).
    Hier wird nur eingegriffen, wenn eine Komma-Sub-Part institutionell ist UND
    mindestens eine Person-Sub-Part übrig bleibt (#74 Bug B) — sonst bleibt das
    Segment unangetastet:

    - ``"Schlebbe, Kirsten"`` → unverändert (keine Sub-Part institutionell)
    - ``"University of X, Department of Y"`` → unverändert (keine Person übrig;
      das ganze Segment wird oben als reine Institution behandelt)
    """
    if "," not in segment:
        return segment
    subs = [s.strip() for s in segment.split(",") if s.strip()]
    person_subs = [s for s in subs if not _looks_institutional(s)]
    inst_subs = [s for s in subs if _looks_institutional(s)]
    if inst_subs and person_subs:
        return ", ".join(person_subs)
    return segment


def drop_institutional_coauthors(author: str) -> str:
    """Entfernt institutionelle Affiliations-Segmente aus einem Autor-String —
    aber nur, wenn mindestens ein Personen-Segment übrig bleibt.

    - ``"Mahmood und University of the Punjab"`` → ``"Mahmood"``
    - ``"Mahmood und National Institutes of Health"`` → ``"Mahmood"`` (Plural, #74 A)
    - ``"Mahmood, University of the Punjab und Schlebbe"`` → ``"Mahmood und Schlebbe"``
      (Komma-Affiliation an echtem und-Trenner, #74 B — Autor bleibt erhalten)
    - ``"Schlebbe und Greifeneder"`` → unverändert (beides Personen)
    - ``"Schlebbe, Kirsten und Greifeneder"`` → unverändert (Lastname, Firstname)
    - ``"World Health Organization"`` → unverändert (reiner Korporativ-Autor)

    Der Trenner zwischen verbleibenden Personen wird aus dem Original übernommen,
    damit Downstream-Parser (`_short_author`, `_short_label`) unverändert greifen.
    """
    if not author or not author.strip():
        return author
    sep_match = _AUTHOR_SEP_RE.search(author)
    if sep_match:
        parts = [p.strip() for p in _AUTHOR_SEP_RE.split(author) if p.strip()]
    else:
        # Kein Autor-Trenner, aber evtl. eine einzelne "Person, Affiliation"-Form.
        parts = [author.strip()]
    # 1) Komma-Affiliationen innerhalb jedes Segments entfernen ("Person, Institut").
    cleaned = [_strip_comma_affiliation(p) for p in parts]
    # 2) Ganze institutionelle Segmente ausfiltern.
    persons = [p for p in cleaned if not _looks_institutional(p)]
    institutional = [p for p in cleaned if _looks_institutional(p)]
    # Reiner Korporativ-Autor (keine Person bleibt) → unangetastet lassen.
    if not persons:
        return author
    # Nur eingreifen, wenn tatsächlich etwas entfernt wurde: ein institutionelles
    # Segment ODER eine Komma-Affiliation. Sonst (reine Personenliste) Original zurück.
    if not institutional and cleaned == parts:
        return author
    sep = sep_match.group() if sep_match else " "
    return sep.join(persons)
