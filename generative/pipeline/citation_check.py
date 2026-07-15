"""Validierungsnetz für Zitations-Attributionen gegen CitationMeta (#96, Etappe E3b).

Pure, deterministische Prüfung: LLM-generierte Autor-/Jahr-Attributionen im
Note-Body (`"Landry 2019"`, `"Merrill (2006)"`, …) werden gegen die kanonische
`CitationMeta` (E3a) abgeglichen. Mismatch → `quality_flags`-Eintrag — kein
Body-Edit, kein Routing-Eingriff (analog zu `flag_redundant_siblings`, #8:
seiteneffekt-freier Review-Hinweis statt Eingriff).

Regressionsfall (Issue #96): `Andragogical Design Process.md` zitierte
durchgängig „Landry 2019" statt des tatsächlichen Autors Knowles (Dateiname
ohne Jahr) — das LLM konfabulierte Autor UND Jahr, der Critic-Flag hat es
durchgewunken. Dieser Check fängt genau diesen Fall deterministisch ab.

Wiederverwendet `AUTHOR_YEAR_RE`/`ZIT_N_RE`/`_split_sentences` aus
`generative.pipeline.claims` (dort bewusst modulöffentlich für genau diesen
Zweck gebaut, siehe dortiger Docstring) statt sie zu duplizieren.
"""

from __future__ import annotations

import re
from pathlib import Path

from generative.pipeline.claims import (
    AUTHOR_YEAR_RE,
    ZIT_N_RE,
    _FOOTNOTE_DEF_RE,
    _QUELLEN_HEADING_RE,
    _split_sentences,
)
from generative.schemas.atomic_note import AtomicNoteDraft
from generative.schemas.citation import CitationMeta

# Multi-Autor-Trenner in CitationMeta.author — identisch zu CitationMeta.short_label
# (";"/"und"/"and"), damit beide Stellen dieselben Nachnamen als Primärautor sehen.
_AUTHOR_SPLIT_RE = re.compile(r"\s*;\s*|\s+(?:und|and)\s+")
# Obsidian-Callout-Header (`[!quote]- Landry 2019, S. 24` nach Blockquote-Strip):
# der Header-Text ist LLM-generiert (Prompt-Template `[!quote]- {author_short}
# {year}, S. N`) und damit prüfpflichtig — anders als Quote-INHALT (wörtliche
# Zitate dürfen fremde Namen nennen) und Footnote-Defs (deterministisch, E3a).
_CALLOUT_HEADER_TEXT_RE = re.compile(r"^\[![\w-]+\][+-]?\s*(?P<text>.*)$")
# Co-Autor-Trenner INNERHALB einer Body-Attribution ("Schlebbe & Greifeneder (2020)").
_MATCH_NAME_SPLIT_RE = re.compile(r"\s+(?:&|und)\s+")
_MATCH_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# #289: Stopword-artige Nicht-Nachnamen, die AUTHOR_YEAR_RE faelschlich als Autor-
# Jahr-Attribution matcht, weil deutsche Grossschreibung auch normale Substantive
# und Ortsnamen erfasst. Beide Faelle belegt (Opus-Breitenreview, Laeufe 2-5):
# "Jahr 1989" aus einer "aus dem Jahr 1989"-Konstruktion, "Potsdam 2019" als
# Ort+Jahr am Ende eines Veranstaltungsnamens ("8. Potsdamer IScience Tag").
# Bewusst NUR die zwei belegten Tokens -- keine spekulative Erweiterung auf
# weitere Stopwoerter/Ortsnamen (im Zweifel bleibt das Flag stehen, siehe
# _is_non_surname_match).
_NON_SURNAME_TOKENS = {"Jahr", "Potsdam"}


def _is_non_surname_match(names: list[str]) -> bool:
    """True, wenn ALLE genannten Namen eines Treffers bekannte Nicht-Nachnamen
    sind (#289) — konservativ per `all()`: ein Treffer mit einem echten (nicht
    gelisteten) Namen neben einem Stopword wird weiterhin geflaggt."""
    return bool(names) and all(n in _NON_SURNAME_TOKENS for n in names)


def _primary_surnames(author: str | None) -> list[str]:
    """Nachnamen aller Primärautoren aus `CitationMeta.author` — Multi-Autor via
    ';'/'und'/'and' (identische Split-Logik wie `CitationMeta.short_label`)."""
    if not author or not author.strip():
        return []
    parts = [p.strip() for p in _AUTHOR_SPLIT_RE.split(author) if p.strip()]
    return [p.split(",", 1)[0].strip() if "," in p else p.split()[-1] for p in parts if p]


def _match_names_and_year(match_text: str) -> tuple[list[str], str]:
    """Zerlegt einen `AUTHOR_YEAR_RE`-Treffer ('Merrill (2006)', 'Schlebbe & Greifeneder
    (2020)') in die genannten Nachnamen und das Jahr."""
    # Defensive: AUTHOR_YEAR_RE erzwingt ein Jahr, year_m ist praktisch nie None;
    # year="" wuerde als abweichendes Jahr flaggen (fail-safe).
    year_m = _MATCH_YEAR_RE.search(match_text)
    year = year_m.group(0) if year_m else ""
    name_part = match_text[: year_m.start()] if year_m else match_text
    name_part = name_part.strip(" (")
    names = [n.strip() for n in _MATCH_NAME_SPLIT_RE.split(name_part) if n.strip()]
    return names, year


def _iter_checkable_sentences(body: str):
    """Liefert Sätze aus prüfbaren Body-Zeilen — Blockquotes (`>`-Prefix, inkl.
    Callout-Header), alles ab `## Quellen` und Footnote-Definitionen (`[^i]:`)
    werden übersprungen (deterministisch aus CitationMeta gerendert, kein
    LLM-Risiko dort, siehe #96 E3a)."""
    in_sources = False
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if in_sources:
            continue
        if stripped.startswith("#"):
            if _QUELLEN_HEADING_RE.match(stripped):
                in_sources = True
            continue
        if _FOOTNOTE_DEF_RE.match(stripped):
            continue
        if stripped.startswith(">"):
            # Quote-INHALT skippen, aber Callout-HEADER prüfen: der Header
            # (`> [!quote]- Landry 2019, S. 24`) ist LLM-generiert — genau dort
            # standen die Fehl-Attributionen der historischen #96-Notes.
            inner = stripped.lstrip("> ").strip()
            header = _CALLOUT_HEADER_TEXT_RE.match(inner)
            if header and header.group("text"):
                yield header.group("text")
            continue
        yield from _split_sentences(stripped)


def validate_citation_attributions(body: str, citation: CitationMeta) -> list[str]:
    """Prüft Autor-/Jahr-Attributionen im Fließtext gegen `citation`.

    Regeln (Issue #96, E3b):
    1. Satz enthält `zit. n.` → Sekundärzitat-Ausnahme: ein fremder Autor ist
       legitim, aber der Teil NACH `zit. n.` muss den Primärautor (Nachname)
       oder den Datei-Stem nennen — sonst Flag.
    2. Satz OHNE `zit. n.`: Attribution mit Primärautor-Nachname, aber
       abweichendem Jahr (oder `citation.year is None`) → Flag.
    3. Satz OHNE `zit. n.`: Attribution mit fremdem Nachname → Flag
       (Landry/Merrill-Fall #96).
    `[o. J.]`/`o. J.`/`o.J.` matchen `AUTHOR_YEAR_RE` nicht (kein Jahres-Digit)
    → nie ein Flag dafür. Rein pure Funktion, keine Seiteneffekte.
    """
    if not body or not citation or not citation.author:
        return []

    primary_surnames = _primary_surnames(citation.author)
    if not primary_surnames:
        return []

    primary_stem = Path(citation.source_file).stem if citation.source_file else ""
    flags: list[str] = []

    for sentence in _iter_checkable_sentences(body):
        zit_n_match = ZIT_N_RE.search(sentence)
        if zit_n_match:
            after = sentence[zit_n_match.end() :]
            # Wortgrenzen statt Substring (Qwen-Review): "Berg" darf nicht via
            # "Bergbau" als genannt gelten (und umgekehrt beim Datei-Stem).
            references_primary = any(re.search(r"\b" + re.escape(s) + r"\b", after) for s in primary_surnames) or (
                bool(primary_stem) and primary_stem in after
            )
            if not references_primary:
                flag = f"⚠️ Sekundärzitat verweist nicht auf die Primärquelle: {sentence}"
                if flag not in flags:
                    flags.append(flag)
            continue  # zit. n. macht jeden Fremdautor im Satz legitim (Rule 2 greift statt 3/4)

        for match in AUTHOR_YEAR_RE.finditer(sentence):
            match_text = match.group(0)
            names, year = _match_names_and_year(match_text)
            is_primary = any(n in primary_surnames for n in names)

            if is_primary:
                if citation.year is None or year != citation.year:
                    flag = f"⚠️ Attribution mit ungedecktem Jahr: '{match_text}' — CitationMeta: {citation.short_label}"
                    if flag not in flags:
                        flags.append(flag)
            elif _is_non_surname_match(names):
                continue  # #289: bekannter Nicht-Nachname (Jahr/Ort-Konstruktion), kein Flag
            else:
                flag = (
                    f"⚠️ Attribution ohne Quellendeckung: '{match_text}' — weder Primärautor "
                    f"({'/'.join(primary_surnames)}) noch als Sekundärzitat (zit. n.) gekennzeichnet"
                )
                if flag not in flags:
                    flags.append(flag)

    return flags


_PHYSICAL_PAGES_FLAG = "Seitenangaben = PDF-Position (Quelle ohne /PageLabels — gedruckte Seiten unbekannt)"


def apply_physical_pages_flag(drafts: list[AtomicNoteDraft], citation: CitationMeta) -> int:
    """Markiert alle Drafts einer Quelle ohne `/PageLabels` (Issue #95): die vom
    Renderer als `PDF-S. N` gekennzeichneten Seitenangaben sind die physische
    PDF-Position, nicht die gedruckte Seite. Seiteneffekt-freier Review-Hinweis,
    idempotent wie `apply_citation_check` — gibt die Zahl neu geflaggter Drafts
    zurück."""
    if not citation or not citation.physical_pages:
        return 0
    added = 0
    for draft in drafts:
        if _PHYSICAL_PAGES_FLAG not in draft.quality_flags:
            draft.quality_flags.append(_PHYSICAL_PAGES_FLAG)
            added += 1
    return added


def apply_citation_check(drafts: list[AtomicNoteDraft], citation: CitationMeta) -> int:
    """Wendet `validate_citation_attributions` auf jeden Draft an und hängt neue
    Flags an `draft.quality_flags` — idempotent (Duplikat-Check vor dem Anhängen,
    analog zum `_add_flag`-Muster in `orchestrator.flag_redundant_siblings`),
    damit ein zweifacher Aufruf keine doppelten Flags erzeugt. Gibt die Zahl neu
    angehängter Flags zurück."""
    added = 0
    for draft in drafts:
        for flag in validate_citation_attributions(draft.body, citation):
            if flag not in draft.quality_flags:
                draft.quality_flags.append(flag)
                added += 1
    return added
