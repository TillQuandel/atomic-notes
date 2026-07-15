#!/usr/bin/env python3
"""eval_common.py -- geteilte Eval-Utilities, extrahiert aus v1/v2 -- #98.

PDF-Text-Extraktion, Note-Parsing, Claim-Extraktion, Chunking und Sprach-Paar-
Erkennung, die v4 (und andere Produktionsmodule wie calibration/sample.py)
konsumieren. v1 (eval_quality.py) und v2 (eval_quality_v2.py) re-exportieren
dieselben Namen fuer ihre eigenen Runner-Teile und fuer bestehende Tests.

Reiner Schichten-Schnitt (#98) -- Code-Move ohne Logik-Aenderung. Kein Import
von eval_quality/eval_quality_v2/eval_dashboard hier (sonst Zirkularimport).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF -- nur fuer Typ-Hinweise (fitz.Document) auf den Parametern unten.

# ---------------------------------------------------------------------------
# Text-Extraktion + Normalisierung (aus eval_quality.py / v1)
# ---------------------------------------------------------------------------


def _extract_page_text(pdf_doc: fitz.Document, page_num: int) -> str:
    """Extrahiert Seiten-Text via blocks für korrekte Lesereihenfolge.

    Normalisierung (Gemini-Finding): Silbentrennung, Ligaturen, Whitespace.
    Gibt "" zurück wenn Seite nicht existiert oder leer (OCR-PDF).
    """
    try:
        page = pdf_doc[page_num - 1]  # 1-basiert → 0-basiert
    except IndexError:
        return ""
    # get_text("blocks") → [(x0,y0,x1,y1,text,block_no,block_type), ...]
    # Nur Text-Blöcke (block_type=0), nach Y-Position sortiert
    blocks = page.get_text("blocks")
    text_parts = [b[4] for b in sorted(blocks, key=lambda b: b[1]) if b[6] == 0]
    raw = " ".join(text_parts)
    return _normalize(raw)


def _normalize(text: str) -> str:
    """Normalisiert PDF-Text für robustes Matching.

    - Silbentrennung: 'Infor-\nmations-' → 'Informations'
    - Ligaturen: fi, fl → fi, fl (PyMuPDF normalisiert meist, aber sicher ist sicher)
    - Whitespace kollabieren
    - Sonderzeichen die Fuzzy stören
    - CID-Font-Artefakte (#278): siehe unten
    """
    # --- #278: CID-Font-Artefakt-Rueckmapping (VOR Silbentrennung/Whitespace) ---
    # Manche eingebetteten Schriftarten (verifiziert an "Jockisch - 2010 - Das
    # Technologieakzeptanzmodell.pdf" per PyMuPDF) mappen ihr Leerzeichen- bzw.
    # Trennstrich-Glyph auf U+0231 ("ȱ") bzw. U+022C ("Ȭ") statt auf ASCII-Space/
    # -Hyphen. PyMuPDF liest das buchstabengetreu -> der Text besteht praktisch nur
    # noch aus zusammengeschriebenen Woertern ("DieȱAkzeptanzȱistȱGegenstandȱ..."),
    # wodurch sowohl Cosine-Chunk-Matching als auch die Zitat-Fuzzy-Verifikation
    # (_normalize_for_evidence in eval_quality_v4.py, ruft _normalize auf) an einer
    # Quelle scheitern, die das generierende LLM korrekt gelesen hat -> False-
    # Positive-Halluzinationen + 0% Coverage bei quellentreuen Notes (Repro-Runs
    # 20260714-185639/-215345). Haeufigkeitsbeleg am Gesamt-PDF: U+0231 5173x,
    # U+022C 240x -- alle anderen Nicht-ASCII-Zeichen sind Groessenordnungen
    # seltener (<=8x) und bleiben bewusst ungefixt (keine belegte Wirkung auf das
    # Coverage-Symptom). Trennstrich zuerst zurueckmappen, damit die bestehende
    # Silbentrennungs-Regex direkt danach greift.
    text = text.replace("Ȭ", "-").replace("ȱ", " ")
    # Silbentrennung am Zeilenende
    text = re.sub(r"-\s*\n\s*", "", text)
    # Normales Newline → Leerzeichen
    text = re.sub(r"\s+", " ", text)
    # Anführungszeichen normalisieren
    text = text.replace("„", '"').replace("“", '"').replace("’", "'")
    return text.strip()


# ---------------------------------------------------------------------------
# Wilson-Konfidenzintervall (aus eval_quality.py / v1)
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-Score-Konfidenzintervall für Binomial-Rate. z=1.96 → 95% CI."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return max(0.0, round(center - margin, 3)), min(1.0, round(center + margin, 3))


# ---------------------------------------------------------------------------
# Chunk-Modell + Konstanten (aus eval_quality_v2.py / v2)
# ---------------------------------------------------------------------------

TOP_K = 5
CHUNK_MIN_TOKENS = 100
CHUNK_MAX_TOKENS = 180
EXPANSION_MAX_TOKENS = 450


@dataclass
class Chunk:
    idx: int
    text: str
    pages: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Note-Body lesen (aus eval_quality_v2.py / v2)
# ---------------------------------------------------------------------------


def _read_note_body(note_path: Path) -> str:
    text = note_path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        fm_end = text.find("\n---", 3)
        if fm_end != -1:
            text = text[text.find("\n", fm_end + 4) + 1 :]

    title = re.search(r"^#\s+.+$", text, flags=re.MULTILINE)
    if title:
        text = text[title.start() :]

    sources = re.search(r"^##\s+Quellen\b", text, flags=re.MULTILINE | re.I)
    if sources:
        text = text[: sources.start()]
    return text


# ---------------------------------------------------------------------------
# Claim-Extraktion (aus eval_quality_v2.py / v2)
# ---------------------------------------------------------------------------

_ABBREVIATIONS = [
    "z.B.",
    "d.h.",
    "bzw.",
    "vgl.",
    "bspw.",
    "usw.",
    "etc.",
    "ggf.",
    "Hrsg.",
    "Jg.",
    "Bd.",
    "S.",
    "Nr.",
    "Abb.",
    "Tab.",
    "Kap.",
    "ff.",
]

_FOOTNOTE_DEF_RE = re.compile(r"^\[\^\d+\]:.*$", re.MULTILINE)
_FOOTNOTE_MARKER_RE = re.compile(r"\[\^\d+\]")


def _drop_quote_callouts(markdown: str) -> str:
    lines = markdown.splitlines()
    kept: list[str] = []
    in_quote_callout = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("> [!quote]"):
            in_quote_callout = True
            continue
        if in_quote_callout:
            if stripped.startswith(">"):
                continue
            in_quote_callout = False
        kept.append(line)
    return "\n".join(kept)


def _protect_sentence_periods(text: str) -> str:
    sentinel = "__PERIOD__"
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", sentinel))

    text = re.sub(r"\b[A-Z]\.", lambda m: m.group(0).replace(".", sentinel), text)
    text = re.sub(r"\d+\.\d+", lambda m: m.group(0).replace(".", sentinel), text)
    text = re.sub(r"S\.\s*\d+(?:-\d+)?", lambda m: m.group(0).replace(".", sentinel), text)
    return text


def _split_sentences(text: str) -> list[str]:
    protected = _protect_sentence_periods(text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace("__PERIOD__", ".").strip() for p in parts if p.strip()]


# --- Fix A: Metatext-Vorfilter (Cross-Model-Fehleranalyse 2026-06-28) ---------
# Pipeline-/Vault-Metatext leakt in die Claim-Extraktion und wird fälschlich
# gegen das PDF geprüft (→ „not_in_context") — das trieb die hallucination_rate
# künstlich hoch (7 von 14 Fehl-Flags). Deterministischer Vorfilter, kein LLM.
# Wiki-Link mit Capture-Group; Alias [[Ziel|Anzeige]] → der Anzeigetext bleibt.
_WIKILINK_RE = re.compile(r"\[\[([^\]]*)\]\]")
# Reiner Wiki-Link-Pointer (optional triviales Suffix wie „ 2.") — kein Claim.
_POINTER_RE = re.compile(r"^\s*\[\[[^\]]*\]\]\s*\d*\.?\s*$")
_META_MARKER_RE = re.compile(
    r"merge stub|bestehende note existiert|pipeline hat das konzept",
    re.IGNORECASE,
)
# Reines Zitat-Fragment, z.B. „Schlebbe und Greifeneder 2022, S. 145." — eine
# Folge namens-artiger Token (Großbuchstaben-Wort/Initiale, Jahr, „et al.",
# Adelspartikel van/von/de/…), per Komma/&/und/Space verbunden, + Seitenangabe.
# Ein normales kleingeschriebenes Wort (Verb/Artikel) ist KEIN Namens-Token →
# bricht die Kette → echte Sätze wie „Das Konzept ist widerlegt, S. 12." bleiben.
# (Cross-Model-Review Qwen+Mistral, 2 Runden, 2026-06-28.)
_NAME_TOKEN = (
    r"(?:[A-ZÄÖÜ][\wäöüß.\-]*|\d{4}[a-z]?|et\s+al\.?|"
    r"van|von|der|den|de|del|della|du|da|la|le|di)"
)
_CITATION_FRAGMENT_RE = re.compile(
    rf"^{_NAME_TOKEN}(?:(?:\s*[,&]\s*|\s+und\s+|\s+){_NAME_TOKEN})*"
    rf"\s*,?\s*S\.\s*\d+(?:\s*[–-]\s*\d+)?\s*\)?\.?$"
)
_MIN_CLAIM_LEN = 30


def _replace_wikilink(m: "re.Match") -> str:
    """[[Ziel]] → Ziel, [[Ziel|Anzeige]] → Anzeige (der in Obsidian sichtbare Text)."""
    inner = m.group(1)
    return inner.split("|", 1)[1] if "|" in inner else inner


def filter_meta_claims(claims: list[str]) -> list[str]:
    """Verwirft Nicht-Claims (Pipeline-/Vault-Metatext) aus der Claim-Liste.

    Verwirft reine Wiki-Link-Pointer, Merge-Stub-Marker, Vault-Meta-Hinweise und
    reine Zitat-Fragmente. Echte Inhalts-Claims (auch mit Wiki-Link) bleiben erhalten
    — der Link wird durch seinen Anzeigetext ersetzt, das Subjekt also bewahrt.
    """
    out: list[str] = []
    for claim in claims:
        if _POINTER_RE.match(claim):
            continue
        stripped = _WIKILINK_RE.sub(_replace_wikilink, claim)
        stripped = re.sub(r"\s+", " ", stripped).strip(" \t\r\n:;,-")
        if len(stripped) < _MIN_CLAIM_LEN:
            continue
        if _META_MARKER_RE.search(stripped):
            continue
        if _CITATION_FRAGMENT_RE.match(stripped):
            continue
        out.append(stripped)
    return out


def extract_claims(note_path: Path) -> list[str]:
    """Extrahiert atomare Claims aus dem Note-Body."""
    body = _read_note_body(note_path)
    body = _drop_quote_callouts(body)
    body = _FOOTNOTE_DEF_RE.sub("", body)
    body = _FOOTNOTE_MARKER_RE.sub("", body)
    body = re.sub(r"^#+\s+", "", body, flags=re.MULTILINE)
    body = re.sub(r"[*_`>#-]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    claims: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(body):
        sentence = sentence.strip(" \t\r\n:;,-")
        if len(sentence) < 30:
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(sentence)
    return filter_meta_claims(claims)


# ---------------------------------------------------------------------------
# PDF-Satz-Extraktion + Boilerplate-Filter (aus eval_quality_v2.py / v2)
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(r"^\s*(Abb\.|Abbildung|Fig\.|Figure|Tab\.|Tabelle)\s*\d+[:.]", re.I)
_BIB_START_RE = re.compile(r"^\s*(literatur|bibliographie|references|bibliography)\s*$", re.I)
_BIB_ENTRY_RE = re.compile(r"^\s*[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'`-]+,\s+.*\b(19|20)\d{2}\b")


def _raw_page_lines(pdf_doc: fitz.Document, page_num: int) -> list[str]:
    try:
        page = pdf_doc[page_num - 1]
    except IndexError:
        return []
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if len(b) > 6 and b[6] == 0]
    text_blocks.sort(key=lambda b: (round(float(b[1]) / 8) * 8, float(b[0])))
    lines: list[str] = []
    for block in text_blocks:
        for line in str(block[4]).splitlines():
            line = _normalize(line)
            if line:
                lines.append(line)
    return lines


def _recurring_short_lines(pages: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update({line for line in lines if len(line) <= 80})
    threshold = max(3, int(len(pages) * 0.2))
    return {line for line, count in counts.items() if count >= threshold}


def _filter_boilerplate_pages(pages: list[list[str]]) -> list[list[str]]:
    recurring = _recurring_short_lines(pages)
    filtered_pages: list[list[str]] = []
    for lines in pages:
        bib_seen = False
        bib_hits = 0
        filtered: list[str] = []
        for line in lines:
            if line in recurring or _CAPTION_RE.match(line):
                continue
            if _BIB_START_RE.match(line):
                bib_seen = True
                continue
            if bib_seen:
                if _BIB_ENTRY_RE.match(line):
                    bib_hits += 1
                if bib_hits >= 3 or _BIB_ENTRY_RE.match(line):
                    continue
            filtered.append(line)
        filtered_pages.append(filtered)
    return filtered_pages


def _pdf_sentences(pdf_doc: fitz.Document, page_numbers: list[int] | None = None) -> list[tuple[str, int]]:
    """Extrahiert (Satz, Seitenzahl)-Paare. `page_numbers` (#80 Fund 1): Anker-
    Seitenzahl je physischem 0-basiertem Index (aus `pdf_chunker.anchor_page_numbers`)
    — Druckseiten-Label statt physischem Index, falls das PDF `/PageLabels` fuehrt.
    `None` (Default) erhaelt das alte i+1-Verhalten fuer rueckwaertskompatible
    Aufrufer."""
    raw_pages = [_raw_page_lines(pdf_doc, page) for page in range(1, len(pdf_doc) + 1)]
    if not any(raw_pages):
        raw_pages = [[_extract_page_text(pdf_doc, page)] for page in range(1, len(pdf_doc) + 1)]
    pages = _filter_boilerplate_pages(raw_pages)

    sentences: list[tuple[str, int]] = []
    for i, lines in enumerate(pages):
        page_num = page_numbers[i] if page_numbers is not None else i + 1
        text = _normalize(" ".join(lines))
        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if len(sentence) >= 20:
                sentences.append((sentence, page_num))
    return sentences


# ---------------------------------------------------------------------------
# Chunking (aus eval_quality_v2.py / v2)
# ---------------------------------------------------------------------------


def _chunks_from_sentences(sentences: list[tuple[str, int]]) -> list[Chunk]:
    """Reine Chunking-Logik (kein fitz-I/O) — aus build_chunks extrahiert, damit die
    Stage-8-PDF-Memoisierung (#151) dieselben Chunks OHNE erneutes fitz.open bauen
    kann. build_chunks bleibt die einzige öffentliche Chunk-Quelle (SSoT)."""
    chunks: list[Chunk] = []
    current: list[str] = []
    pages: set[int] = set()
    token_count = 0

    for sentence, page in sentences:
        sent_tokens = len(sentence.split())
        if current and token_count + sent_tokens > CHUNK_MAX_TOKENS and token_count >= CHUNK_MIN_TOKENS:
            chunks.append(Chunk(len(chunks), _normalize(" ".join(current)), tuple(sorted(pages))))
            overlap = current[-1:]
            current = overlap[:]
            pages = {page}
            token_count = sum(len(s.split()) for s in current)

        current.append(sentence)
        pages.add(page)
        token_count += sent_tokens

    if current:
        chunks.append(Chunk(len(chunks), _normalize(" ".join(current)), tuple(sorted(pages))))
    return chunks


def _expand_context(chunks: list[Chunk], idx: int) -> str:
    selected = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(chunks)]
    words = " ".join(chunks[i].text for i in selected).split()
    if len(words) > EXPANSION_MAX_TOKENS:
        return " ".join(words[:EXPANSION_MAX_TOKENS])
    return " ".join(words)


# ---------------------------------------------------------------------------
# Sprachpaar-Erkennung (aus eval_quality_v2.py / v2 -- NICHT die einfachere,
# nur-Umlaut-basierte Variante aus v1.eval_quality, die dort lokal bleibt)
# ---------------------------------------------------------------------------


def _detect_language_pair(note_text: str, pdf_text: str) -> str:
    umlaut_re = re.compile(r"[äöüßÄÖÜ]")
    de_words_re = re.compile(r"\b(der|die|das|und|ist|werden|nicht|eine|einer)\b", re.I)
    en_words_re = re.compile(r"\b(the|and|is|are|not|with|from|this|that)\b", re.I)

    def lang(sample: str) -> str:
        if umlaut_re.search(sample) or len(de_words_re.findall(sample[:2000])) >= len(
            en_words_re.findall(sample[:2000])
        ):
            return "DE"
        return "EN"

    return f"{lang(pdf_text)}→{lang(note_text)}"
