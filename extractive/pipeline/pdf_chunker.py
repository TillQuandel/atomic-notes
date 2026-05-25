from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

try:
    import fitz as _fitz  # PyMuPDF — bessere Spacing-Erkennung als pdfplumber
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

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

        page_texts = _page_text_clean(pdf_path)
        for idx, page in enumerate(pdf.pages):
            page_text = page_texts[idx] if idx < len(page_texts) else ""
            # extract_words() nur fuer Font-Size-basierte Header-Erkennung
            header_words = {
                w["text"].strip()
                for w in (page.extract_words(extra_attrs=["size"]) or [])
                if w.get("size") and _is_header(w["text"].strip(), float(w["size"]), avg)
            }

            for line in page_text.splitlines():
                t = line.strip()
                if not t:
                    continue
                if t in header_words:
                    if current and word_count > 50:
                        chunks.append(Chunk(" ".join(current), current_page, current_header))
                        current, word_count = [], 0
                    current_header, current_page = t, page.page_number
                else:
                    words = t.split()
                    current.extend(words)
                    word_count += len(words)
                    if word_count >= max_words:
                        chunks.append(Chunk(" ".join(current), current_page, current_header))
                        current, word_count, current_page = [], 0, page.page_number
    if current:
        chunks.append(Chunk(" ".join(current), current_page, current_header))
    return chunks


def _page_text_clean(pdf_path) -> list[str]:
    """Pro-Seite sauberer Text. PyMuPDF bevorzugt (besseres Spacing), Fallback pdfplumber."""
    if _HAS_PYMUPDF:
        doc = _fitz.open(str(pdf_path))
        return [doc[i].get_text() for i in range(len(doc))]
    with pdfplumber.open(str(pdf_path)) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def extract_fulltext(pdf_path) -> str:
    """Volltext fuer Ankerpruefung (rapidfuzz)."""
    return "\n".join(_page_text_clean(pdf_path))
