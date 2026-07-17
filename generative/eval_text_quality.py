#!/usr/bin/env python3
"""Generische CID-Font-/Korruptions-Erkennung auf Eval-Seite (#306).

Hintergrund: #278 fixt exakt zwei Codepoints eines defekten eingebetteten
Fonts im Jockisch (2010)-PDF (U+0231 -> Leerzeichen, 5173x; U+022C ->
Trennstrich, 240x; per PyMuPDF am Original nachgemessen, siehe
`eval_common._normalize`). Jedes andere PDF mit einem anderen CID-Glyph-
Mapping erzeugt dasselbe Fehlerbild (Cosine-Matching + Zitat-Fuzzy-
Verifikation scheitern an korrumpiertem Quelltext -> quellentreue Notes
werden faelschlich als Halluzination markiert) und bleibt ungefixt, bis
jemand manuell eine neue Replace-Zeile ergaenzt.

Dieses Modul reklassifiziert NICHTS und mapt NICHTS zurueck (das bleibt
absichtlich Handarbeit pro PDF wie in #278) -- es liefert nur eine
Heuristik, die ein verdaechtiges Nicht-ASCII-Haeufigkeitsprofil erkennt und
als Warnung meldet: ein einzelner Codepoint ausserhalb des in deutschen/
englischen Fachtexten erwartbaren Bereichs (ASCII + Latin-1 Supplement +
Latin Extended-A + gaengige typografische Satzzeichen + etablierte
Nicht-Latein-Schriftbloecke wie Griechisch/Kyrillisch, s.
`_ESTABLISHED_SCRIPT_RANGES`), der einen Anteils-Schwellwert an der
Gesamtzeichenzahl ueberschreitet. Genau dieses Muster zeigte das
Jockisch-PDF (ein Glyph ersetzt praktisch jedes Leerzeichen).

Adversarialer Fund am #306-PR: ein statistik-/klassik-lastiger Absatz mit
haeufigem griechischem "α" (Koeffizienten-Notation) triggerte ohne die
Schriftblock-Ausnahme faelschlich (1,93 % > 0,5 %-Schwelle) -- Griechisch/
Kyrillisch sind in Fachtexten (Statistik, Linguistik, Klassik, Slawistik)
gehaeuft und legitim, waehrend CID-Korruptionsglyphen empirisch in Latin
Extended-B landen (siehe #278: U+0231/U+022C), nicht in etablierten
Nicht-Latein-Bloecken.

Kein Bezug zur Halluzinationsrate/den Judge-Labels -- reines Diagnostik-
Signal fuer den Eval-Betrieb (#306 harte Vorgabe: Flag/Warnung, kein
generisches Rueckmapping, keine Label-/Raten-Aenderung).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# Erwartbarer "gesunder" Codepoint-Bereich fuer per PyMuPDF extrahierten
# deutschen/englischen Fachtext:
#   - ASCII (0x00-0x7F)
#   - Latin-1 Supplement (0x80-0xFF): deckt Umlaute/ß (ä ö ü Ä Ö Ü ß),
#     Anfuehrungszeichen-Reste, Grad-/Paragraph-/Copyright-Zeichen ab.
#   - Latin Extended-A (0x100-0x17F): weitere lateinische Diakritika
#     (franz./poln./tschech. Namen in Literaturangaben etc.).
# Das Jockisch-Artefakt (U+0231, U+022C) liegt bewusst AUSSERHALB davon in
# Latin Extended-B (0x180-0x24F) -- der Bereich trifft die Erkennung exakt.
_EXPECTED_MAX_CODEPOINT = 0x17F

# Typografische Satzzeichen, die PyMuPDF-Extraktion regelmaessig produziert
# und die trotz Codepoint > 0x17F normaler, unverdaechtiger Text sind
# (Gedankenstriche, "smarte" Anfuehrungszeichen, Ellipse, Aufzaehlungspunkt,
# geschuetzter Bindestrich). eval_common._normalize wandelt nur eine
# Teilmenge der Anfuehrungszeichen nach ASCII um -- der Rest darf hier nicht
# als CID-Artefakt zaehlen, sonst waere die Heuristik bei dashreichem
# akademischem Text false-positive-anfaellig.
_TYPOGRAPHIC_PUNCTUATION_WHITELIST = frozenset(
    "‐‑‒–—―"  # Bindestriche/Gedankenstriche
    "‘’‚‛“”„‟"  # Anfuehrungszeichen-Varianten
    "•"  # Aufzaehlungspunkt
    "…"  # Ellipse
    "′″"  # Minute/Sekunde bzw. Prime (Formeln)
)

# Etablierte Nicht-Latein-Schriftbloecke, die in legitimen Fachtexten gehaeuft
# vorkommen -- Statistik-/Physik-Koeffizienten (α, β, χ², σ), Klassik-/
# Linguistik-Zitate (Griechisch), Slawistik/Osteuropa-Literaturangaben
# (Kyrillisch). CID-Font-Korruptionsglyphen wie U+0231/U+022C (#278) landen
# empirisch in Latin Extended-B, nicht in etablierten Nicht-Latein-Bloecken --
# ein griechischer Absatz mit vielen "α" ist ein plausibler Fachtext, kein
# CID-Artefakt (adversarialer Fund am #306-PR: 1,93 % α in einem Statistik-
# Absatz triggerte faelschlich). Bewusst nur die zwei haeufigsten Bloecke,
# nicht pauschal "alles ausserhalb Latein" -- das wuerde die Heuristik
# entwerten.
_ESTABLISHED_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x0400, 0x04FF),  # Cyrillic
)


def _in_established_script(codepoint: int) -> bool:
    return any(lo <= codepoint <= hi for lo, hi in _ESTABLISHED_SCRIPT_RANGES)


DEFAULT_RATIO_THRESHOLD = 0.005  # 0,5 % der Zeichen (Issue-Vorschlag #306)

QUALITY_FLAG_CID_SUSPECT = "cid_font_suspect"


@dataclass(frozen=True)
class CidSuspectResult:
    """Diagnostik-Ergebnis eines erkannten CID-Verdachts."""

    codepoint: str  # z.B. "U+0231"
    char: str
    count: int
    ratio: float
    total_chars: int

    def as_dict(self) -> dict:
        return {
            "codepoint": self.codepoint,
            "count": self.count,
            "ratio": self.ratio,
            "total_chars": self.total_chars,
        }


def detect_cid_suspect(
    text: str,
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> CidSuspectResult | None:
    """Prueft `text` auf ein CID-Font-Korruptions-Haeufigkeitsprofil.

    Zaehlt Codepoints ausserhalb von ASCII/Latin-1/Latin-Extended-A (abzueglich
    der typografischen Satzzeichen-Whitelist und etablierter Nicht-Latein-
    Schriftbloecke wie Griechisch/Kyrillisch, s. `_ESTABLISHED_SCRIPT_RANGES`).
    Wenn ein EINZELNER verbleibender Codepoint daraus `ratio_threshold` der
    Gesamtzeichenzahl ueberschreitet, gilt das als CID-Verdacht (dominantes
    Ersatz-Glyph fuer Leerzeichen/Trennstrich o.ae. -- das Muster, das #278 am
    Jockisch-PDF fuer U+0231/U+022C belegt hat).

    Reine Diagnostik: aendert `text` nicht, trifft keine Label-Entscheidung.
    Gibt `None` zurueck, wenn kein Codepoint den Schwellwert erreicht (oder
    `text` leer ist).
    """
    if not text:
        return None
    total = len(text)
    counts = Counter(
        ch
        for ch in text
        if ord(ch) > _EXPECTED_MAX_CODEPOINT
        and ch not in _TYPOGRAPHIC_PUNCTUATION_WHITELIST
        and not _in_established_script(ord(ch))
    )
    if not counts:
        return None
    char, count = counts.most_common(1)[0]
    ratio = count / total
    if ratio < ratio_threshold:
        return None
    return CidSuspectResult(
        codepoint=f"U+{ord(char):04X}",
        char=char,
        count=count,
        ratio=round(ratio, 5),
        total_chars=total,
    )
