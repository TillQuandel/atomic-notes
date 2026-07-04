"""Deterministische Attribution-Heuristik (Faithfulness-Gate E4b, #69).

Reine Präsenz-Prüfung: prüft NUR ob die im Claim-Text NAMENTLICH genannten
(fremden) Autoren-Nachnamen im zugehörigen Seitenfenster (`source_window`)
überhaupt VORKOMMEN — keine semantische Aussage-Zuordnung. Die feinere Prüfung
("sagt das Fenster wirklich das, was dem Autor zugeschrieben wird") macht
später NLI (Etappe E5, `nli.py`). Das ist bewusst NUR ein Namens-Existenz-Check.

Fängt den dokumentierten Fehlerfall ab, dass ein LLM eine Aussage dem falschen
zitierten Autor zuordnet (Sekundärzitate mit vertauschtem Autor) — komplementär
zu `citation_check.validate_citation_attributions` (E3b), das Autor/Jahr GEGEN
die `CitationMeta` prüft, nicht ob der Name im Quelltext selbst auftaucht.

Namens-Extraktion aus dem Claim-Text (Regexes aus `claims.py`, importiert):

1. `AUTHOR_YEAR_RE`-Treffer ("Merrill (2006)", "Schlebbe & Greifeneder (2020)")
   — Nachnamen via `&`/`und`-Split wie in `citation_check._match_names_and_year`.
2. `zit. n.`-Konstruktion (`ZIT_N_RE`): NUR der Name NACH "zit. n." (das
   Sekundärzitat-Ziel, i.d.R. der Primärautor). Der berichtete Fremdautor VOR
   der Klammer ("…, führt Haythornthwaite aus (zit. n. Hrastinski)…") wird
   bewusst NICHT heuristisch extrahiert: "letzter großgeschriebener Token" ist
   im Deutschen regelmäßig ein Substantiv ("…soziale Unterstützung (zit. n. …)")
   und matcht gegen englische Quellfenster nie — am echten Hrastinski-PDF
   erzeugten 2 von 3 realitätsnahen zit.-n.-Sätzen author_missing-False-
   Positives (Review 2026-07-04). Die Aussage↔Autor-Zuordnung solcher Sätze
   prüft die NLI-Stufe (E5), nicht diese Präsenz-Heuristik.
3. `laut <Name>` / `<Name> zufolge` (`LAUT_RE`/`ZUFOLGE_RE`).

Bekannte Grenzen (bewusst, Qwen-Review E4): `\b`-Wortgrenzen matchen „Smith"
auch in „Smith-Jones" (Bindestrich ist Non-Word-Char); der Primärautoren-
Set-Vergleich ist case-sensitiv (Metadaten liefern konsistente Groß-
schreibung). Beides Randfälle ohne beobachtete Instanz — bei Bedarf in der
E5-Kalibrierung nachschärfen.

Primärautoren (`primary_surnames`) sind von der Präsenz-Prüfung ausgenommen —
die Quelle IST der Primärautor, er muss sich nicht selbst im eigenen Fenster
nennen. Sind NACH Abzug der Primärautoren keine Fremd-Namen mehr übrig
("Primärautor-Attribution ohne Fenster-Präsenz"), gilt das als `supported`
(Ausnahme greift), unabhängig davon ob überhaupt ein Fenster vorliegt.
"""

from __future__ import annotations

import re

from generative.pipeline.claims import AUTHOR_YEAR_RE, LAUT_RE, ZIT_N_RE, ZUFOLGE_RE, Claim

_NAME_TOKEN_RE = re.compile(r"[A-ZÄÖÜ][\wäöüß\-]+")
_MULTI_AUTHOR_SPLIT_RE = re.compile(r"\s+(?:&|und)\s+")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _surnames_from_author_year(text: str) -> set[str]:
    surnames: set[str] = set()
    for match in AUTHOR_YEAR_RE.finditer(text):
        name_part = _YEAR_RE.sub("", match.group(0)).strip(" ()")
        surnames.update(n.strip() for n in _MULTI_AUTHOR_SPLIT_RE.split(name_part) if n.strip())
    return surnames


def _surnames_from_zit_n(text: str) -> set[str]:
    surnames: set[str] = set()
    for match in ZIT_N_RE.finditer(text):
        after = text[match.end() :].lstrip()
        after_match = _NAME_TOKEN_RE.match(after)
        if after_match:
            surnames.add(after_match.group(0))
    return surnames


def _surnames_from_laut_zufolge(text: str) -> set[str]:
    surnames: set[str] = set()
    for match in LAUT_RE.finditer(text):
        surnames.add(match.group(0).split()[-1])
    for match in ZUFOLGE_RE.finditer(text):
        surnames.add(match.group(0).split()[0])
    return surnames


def _attributed_surnames(text: str) -> set[str]:
    return _surnames_from_author_year(text) | _surnames_from_zit_n(text) | _surnames_from_laut_zufolge(text)


def check_attribution(claim: Claim, source_window: str | None, primary_surnames: list[str]) -> str:
    """Präsenz-Check: kommt jeder attribuierte Fremd-Nachname im Fenster vor?

    Rückgabe: `"supported"` | `"author_missing"` | `"no_window"` | `"not_applicable"`.
    Siehe Modul-Docstring für Namens-Extraktion und Primärautoren-Ausnahme.
    """
    if "attribution" not in claim.risk_types:
        return "not_applicable"

    all_surnames = _attributed_surnames(claim.text)
    if not all_surnames:
        return "not_applicable"

    foreign_surnames = all_surnames - set(primary_surnames)
    if not foreign_surnames:
        return "supported"  # nur Primärautor genannt — Ausnahme greift

    if not source_window:
        return "no_window"

    for surname in foreign_surnames:
        if not re.search(r"\b" + re.escape(surname) + r"\b", source_window):
            return "author_missing"

    return "supported"
