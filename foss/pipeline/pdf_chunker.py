from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

try:
    from langdetect import detect as _detect
    def detect_language(text: str) -> str:
        try:
            return _detect(text[:500])
        except Exception:
            return "unknown"
except ImportError:
    def detect_language(text: str) -> str:
        return "unknown"

_ANCHOR_RE = re.compile(r"\s*\(S\.\s*\d+(?:-\d+)?\)")
_HEADER_RE = re.compile(r"^\d{0,2}\.?\s*[A-Z][A-Za-z\s]{3,60}$")


def strip_anchors(text: str) -> str:
    return _ANCHOR_RE.sub("", text).strip()


@dataclass
class Chunk:
    text: str
    page: int
    header: str = ""


def _is_header(text: str, size: float, avg_size: float) -> bool:
    return size > avg_size * 1.15 or bool(_HEADER_RE.match(text.strip()))


def extract_chunks(pdf_path, max_words: int = 3000) -> list[Chunk]:
    """Extrahiert Text-Chunks via pdfplumber (kein OS-Paket noetig)."""
    chunks, current, current_page, current_header, word_count = [], [], 1, "", 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        sizes = [float(w["size"]) for p in pdf.pages
                 for w in (p.extract_words(extra_attrs=["size"]) or []) if w.get("size")]
        avg = sum(sizes) / len(sizes) if sizes else 10.0
        for page in pdf.pages:
            for w in (page.extract_words(extra_attrs=["size"]) or []):
                t = w.get("text", "").strip()
                if not t:
                    continue
                sz = float(w.get("size", avg))
                if _is_header(t, sz, avg):
                    if current and word_count > 50:
                        chunks.append(Chunk(" ".join(current), current_page, current_header))
                        current, word_count = [], 0
                    current_header, current_page = t, page.page_number
                else:
                    current.append(t)
                    word_count += 1
                    if word_count >= max_words:
                        chunks.append(Chunk(" ".join(current), current_page, current_header))
                        current, word_count, current_page = [], 0, page.page_number
    if current:
        chunks.append(Chunk(" ".join(current), current_page, current_header))
    return chunks


def extract_fulltext(pdf_path) -> str:
    """Volltext fuer Ankerpruefung (rapidfuzz)."""
    with pdfplumber.open(str(pdf_path)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)
