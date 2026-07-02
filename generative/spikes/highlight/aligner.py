# -*- coding: utf-8 -*-
"""Zitat -> Geometrie-Lokalisierung per Sequence-Alignment gegen den
bbox-tragenden get_text("words")-Token-Strom (Ansatz A der Feature-Recherche).

Reine Logik, kein PDF-I/O: Tokens werden als (text, (x0,y0,x1,y1))-Tupel
uebergeben, damit die Kernfunktionen ohne PyMuPDF testbar bleiben.
"""

import re
import unicodedata

from rapidfuzz import fuzz

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_WS_RE = re.compile(r"\s+")


def strip_editorial_brackets(quote: str) -> str:
    """`[T]his` -> `This`, `[asynchronous discussions]` -> `asynchronous discussions`.

    Akademische Zitier-Konvention (angepasste Grossschreibung / eingefuegte
    Referenten); steht so nicht woertlich in der Quelle -> vor dem Matching weg.
    """
    return _BRACKET_RE.sub(r"\1", quote)


def normalize_text(s: str) -> str:
    """NFKC (loest Ligaturen wie fi/fl auf), Whitespace kollabiert, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s)
    return s.strip().lower()


def build_page_string(tokens):
    """Baut einen normalisierten Seiten-String aus dem Token-Strom + eine
    parallele Liste char_to_word (Wort-Index pro Zeichen).

    Silbentrennung ueber Zeilenumbruch wird aufgeloest: endet ein Token auf
    "-" und folgt ein weiteres, wird der Bindestrich entfernt und die Tokens
    ohne Trenn-Space verklebt ("ar-" + "chitecture" -> "architecture").
    """
    chars = []
    char_to_word = []
    n = len(tokens)
    for wi, (text, _rect) in enumerate(tokens):
        norm = normalize_text(text)
        if not norm:
            continue
        dehyphenate = norm.endswith("-") and len(norm) > 1 and wi + 1 < n
        if dehyphenate:
            norm = norm[:-1]
        for ch in norm:
            chars.append(ch)
            char_to_word.append(wi)
        if not dehyphenate and wi + 1 < n:
            chars.append(" ")
            char_to_word.append(wi)
    return "".join(chars), char_to_word


def locate(quote: str, tokens, min_score: float = 82.0, min_len_ratio: float = 0.9):
    """Aligned `quote` gegen den Token-Strom EINER Seite (Page-Constraint liegt
    beim Aufrufer) und gibt die getroffenen Wort-Bboxes zurueck.

    Guardrails: Score-Schwellwert UND Laengen-Ratio (Match-Span / Quote-Laenge).
    Beides muss passieren, sonst None (Alignment matcht sonst immer, auch falsch).
    """
    q = normalize_text(strip_editorial_brackets(quote))
    if len(q) < 12:
        return None
    page_str, char_to_word = build_page_string(tokens)
    if not page_str:
        return None

    al = fuzz.partial_ratio_alignment(q, page_str)
    if al is None:
        return None

    span = al.dest_end - al.dest_start
    len_ratio = span / len(q)
    if al.score < min_score or len_ratio < min_len_ratio:
        return None

    word_indices = []
    for i in range(al.dest_start, min(al.dest_end, len(char_to_word))):
        w = char_to_word[i]
        if not word_indices or word_indices[-1] != w:
            word_indices.append(w)
    word_indices = sorted(set(word_indices))
    rects = [tokens[w][1] for w in word_indices]
    return {
        "score": float(al.score),
        "len_ratio": float(len_ratio),
        "word_indices": word_indices,
        "rects": rects,
    }
