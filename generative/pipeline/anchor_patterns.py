"""Geteilte Regex-Konstanten für Seitenanker und Sentence-Splitting.

Vor v14 lebten dieselben Patterns parallel in `agents/critic.py`, `pipeline/anchor_repair.py`
und `agents/verifier.py`. Ein Bug im Pattern (`(S. N, S. M)` wurde nicht erkannt) hat 3 Stages
gleichzeitig betroffen, weil der Regex per Copy-Paste dupliziert war. Lesson v14: gemeinsame
Pattern-Konstanten gehoeren in ein Utility-Modul.

Pattern-Konventionen:
- Body-Anker: `(S. N)`, `(S. N-M)`, `(S. N, M)`, `(S. N, S. M)` — alle Varianten
- Page-Marker: `[S. N]` — nur in PDF-extrahiertem Volltext, vom pdf_chunker injiziert
- Sentence-Split: nach `.!?` + Whitespace + Großbuchstabe oder Anführungszeichen
"""
from __future__ import annotations
import re

# Inline-Anker im Body: `(S. 5)`, `(S. 1-3)`, `(S. 1, 3)`, `(S. 1, S. 3)`,
# `(S. 1, S. 2, S. 3)`. Optionales `S.`-Präfix vor Folgezahlen erlaubt mehrfache
# Wiederholungen wie sie in deutschen Direktzitaten typisch sind.
PAGE_ANCHOR_RE = re.compile(r"\(S\.\s*\d+(?:\s*[\-–,]\s*(?:S\.\s*)?\d+)*\)")

# Page-Marker im PDF-Volltext, vom pdf_chunker an Seitenanfänge injiziert.
PAGE_MARKER_RE = re.compile(r"\[S\.\s*(\d+)\]")

# Variante mit Capture-Group für reine Zahlen-Extraktion (Verifier).
PAGE_ANCHOR_NUMS_RE = re.compile(r"\(S\.\s*(\d+(?:\s*,\s*(?:S\.\s*)?\d+)*)\)")

# Satz-Split: nach Satzzeichen + Whitespace, gefolgt von Großbuchstabe oder
# Öffnung-Anführungszeichen. Splittet nicht innerhalb von Klammern (z.B. `(S. 1)`).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ\"„])")
