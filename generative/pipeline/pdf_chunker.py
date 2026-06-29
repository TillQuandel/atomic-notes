"""PDF → Text-Chunks mit Seiten-Markern, aufgeteilt nach Kapitelstruktur.

Seiten-Marker `[S. N]` werden an Seitenanfängen eingefügt (basierend auf pdftotext's
\\f-Form-Feed-Markierung). Damit kann der Extractor Anker-Zitate mit korrekter Seitenzahl
versehen und der Verifier die Seitenzahl gegen den Originaltext prüfen.
"""
from __future__ import annotations
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from generative.config import CHUNK_WORDS, MIN_WORDS_PER_PAGE


@dataclass
class Chunk:
    title: str                   # Kapitelüberschrift oder "Abschnitt N"
    text: str                    # mit `[S. N]`-Markern an Seitenanfängen
    index: int
    page_start: int | None = None
    page_end: int | None = None


# Marker den Extractor/Verifier sehen: leere Zeile + [S. N] + leere Zeile
_PAGE_MARKER_RE = re.compile(r"\n*\[S\.\s*(\d+)\]\n*", re.MULTILINE)

# Zeilen-isolierte Variante NUR für die Seitenzählung (assess_text_quality):
# echte Pipeline-Marker stehen via `pages_to_marked_text` allein auf einer Zeile
# (`\n\n[S. N]\n\n`). Inline-Quellenverweise wie „vgl. [S. 12]" im Fließtext stehen
# NICHT allein und dürfen nicht als Seite zählen (sonst künstlich gedrückte
# words_per_page → falsch is_thin; Codex-Review G6/#27).
_PAGE_MARKER_LINE_RE = re.compile(r"^\s*\[S\.\s*\d+\]\s*$", re.MULTILINE)


def _pdf_page_labels(pdf_path: Path) -> list[str] | None:
    """Druckseiten-Bezeichner (`/PageLabels`) je PDF-Seite, oder None wenn das PDF
    keine führt. Fail-open: jeder pypdf-Fehler → None. **Nur** wenn `/PageLabels`
    real vorhanden ist, weicht das Ergebnis vom alten i+1-Verhalten ab — PDFs ohne
    Labels (die meisten Paper/Test-Fixtures) bleiben damit bit-identisch."""
    try:
        from pypdf import PdfReader
        from pypdf.generic import DictionaryObject
        reader = PdfReader(str(pdf_path))
        root = reader.trailer["/Root"].get_object()
        if not isinstance(root, DictionaryObject) or "/PageLabels" not in root:
            return None
        return _usable_page_labels(list(reader.page_labels))
    except Exception:
        return None


def _usable_page_labels(labels: list | None) -> list | None:
    """Gibt ``labels`` nur zurück, wenn ALLE numerisch UND eindeutig sind — sonst None.

    Verhindert, dass nicht-numerische (römische) Labels auf den i+1-Fallback fallen
    und mit echten numerischen Druckseiten im selben ``S. N``-Namespace kollidieren
    (→ False-Binds in figure_alt) bzw. dass doppelte Labels die Label→Index-Abbildung
    mehrdeutig machen. Gemischt/doppelt → einheitlicher i+1-Pfad für ALLE Konsumenten.
    (Codex-Review, 2. Durchgang.)"""
    if not labels:
        return None
    stripped = [str(label).strip() for label in labels]
    # isdecimal() statt isdigit(): isdigit() ist True für Unicode-Superscripts (²),
    # die int() dann nicht parsen kann (ValueError). isdecimal() == genau die von
    # int() akzeptierten Ziffern → kein Crash, sauberer Fallback. (Codex-Review.)
    if not all(s.isdecimal() for s in stripped):
        return None
    nums = [int(s) for s in stripped]
    # Eindeutigkeit auf der ZAHL prüfen, nicht dem String: "01" und "1" sind als
    # String verschieden, als Druckseite identisch → zwei Seiten "S. 1" (False-Bind).
    # Numerische Eindeutigkeit erzwingt zusammen mit der Monotonie echte strikte
    # Monotonie. (Qwen-Review, 2026-06-27.)
    if len(set(nums)) != len(nums):
        return None
    # Auch strikt monoton steigend verlangen: nicht-monotone (aber eindeutige)
    # Labels wie 100,1,2 würden in min/max-Chunk-Ranges (page_range_of_text,
    # split_by_chapters) falsche breite Spannen erzeugen. (Codex-Re-Review.)
    if nums != sorted(nums):
        return None
    return labels


def _resolve_page_numbers(
    pages_raw: list[str], labels: list | None
) -> list[tuple[int, str]]:
    """Ordnet jeder Seite ihre zitierfähige Seitenzahl zu: das numerische
    Druckseiten-Label, sonst die 1-basierte Form-Feed-Position.

    Nicht-numerische Labels (römisches Frontmatter) fallen bewusst auf den Index
    zurück — die Anker-Kette (`PAGE_MARKER_RE`, `_extract_page_span`) erwartet
    `\\d+`. Längen-Mismatch (pdftotext-Extraseite via finalem \\f) ist sicher."""
    out: list[tuple[int, str]] = []
    for i, page_text in enumerate(pages_raw):
        raw = labels[i] if labels and i < len(labels) else None
        # robust: pypdf-Labels können Whitespace (" 159 ") oder selten non-str
        # tragen → strippen/coercen statt aufs Form-Feed zurückzufallen/zu crashen.
        label = str(raw).strip() if raw is not None else ""
        num = int(label) if label.isdigit() else i + 1
        out.append((num, page_text))
    return out


def pdf_to_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Liefert [(page_num, page_text), ...] via pdftotext + \\f-Split.

    `page_num` ist die zitierfähige Druckseite aus den PDF-`/PageLabels`, falls das
    PDF welche führt (Buch: PDF-Seite 179 → Druckseite „159"); sonst die 1-basierte
    pdftotext-Position (Paper ohne Labels — unverändertes Verhalten)."""
    from generative.pipeline.error_hints import pdftotext_error_hint
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as e:
        # pdftotext-Binary fehlt/nicht ausführbar → genau der Setup-Fall, der den
        # handlungsanleitenden Hinweis (+ doctor) am meisten braucht.
        sys.exit(pdftotext_error_hint(f"{e} (pdftotext nicht gefunden?)"))
    if result.returncode != 0:
        sys.exit(pdftotext_error_hint(result.stderr))
    pages_raw = result.stdout.split("\f")
    labels = _pdf_page_labels(pdf_path)
    if labels is None:
        # Unverändertes Verhalten: leere Seiten verwerfen, lückenlos ab 1 zählen
        # (pdftotext hängt oft ein finales \\f → leere letzte Seite).
        pages_raw = [p for p in pages_raw if p.strip()]
        return [(i + 1, p) for i, p in enumerate(pages_raw)]
    # Mit Druckseiten-Labels: Label-Index = PDF-Seite, daher VOR dem Leerseiten-
    # Filter zuordnen (eine leere Seite mittendrin darf die Folgeseiten nicht
    # verschieben), dann leere Seiten verwerfen.
    numbered = _resolve_page_numbers(pages_raw, labels)
    return [(n, p) for n, p in numbered if p.strip()]


def pages_to_marked_text(pages: list[tuple[int, str]]) -> str:
    """Fügt `[S. N]`-Marker am Anfang jeder Seite ein."""
    return "".join(f"\n\n[S. {n}]\n\n{t}" for n, t in pages)


# Frontmatter-Indikatoren: Phrasen die typischerweise vor dem ersten Kapitel auftauchen.
# Englisch + Deutsch. Wortgrenzen-Match, case-insensitive.
_FRONTMATTER_PHRASES = (
    "advance praise", "praise for",
    "acknowledgments", "acknowledgements", "danksagung",
    "copyright", "all rights reserved", "alle rechte vorbehalten",
    "dedication", "widmung",
    "table of contents", "contents", "inhaltsverzeichnis", "inhalt",
    "foreword", "vorwort",
    "preface", "geleitwort",
    "about the author", "über den autor", "über die autorin",
    "isbn",
)
_FRONTMATTER_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _FRONTMATTER_PHRASES) + r")\b",
    re.IGNORECASE,
)
# Maximaler Anteil Seiten der als Frontmatter abgeschnitten werden darf — Schutz
# vor Misdetection bei Sammelbänden o.ä. die durchgehend Kapitel-Numbering haben.
_FRONTMATTER_MAX_DROP_RATIO = 0.5


def drop_frontmatter_pages(pages: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], int]:
    """Entfernt Frontmatter-Seiten (Advance Praise, Acknowledgments, Copyright,
    Preface, …) vor dem ersten Kapitel-Heading. Behält Original-Page-Numbers
    (kein Renumbering — Anker-Verifier braucht echte PDF-Seitenzahlen).

    Returns (pages_after_strip, dropped_count). dropped_count=0 wenn nichts
    abgeschnitten wurde (kein Chapter-Heading gefunden, erstes Chapter auf Page 1,
    oder Cap überschritten).

    Heuristik:
        1. Finde erste Page mit _CHAPTER_RE-Match → first_chapter_idx
        2. Falls keine Page davor: kein Crop
        3. Mindestens eine der Pre-Pages muss eine Frontmatter-Phrase enthalten
           (sonst sind die Pre-Pages wahrscheinlich Cover/Title ohne klare Frontmatter-Marker
           — konservativ behalten, könnten Inhalt sein)
        4. Cap: max _FRONTMATTER_MAX_DROP_RATIO der Pages dürfen weg
    """
    if len(pages) < 2:
        return pages, 0

    first_chapter_idx: int | None = None
    for i, (_, txt) in enumerate(pages):
        # Dual-Use mit split_by_chapters: TOC-Trail-Zeilen sind keine echten Headings,
        # sonst würde Frontmatter-Crop bei Inhaltsverzeichnis-Seiten zu früh greifen.
        # Zusätzlich: Matches deren Titel selbst eine Frontmatter-Phrase enthält
        # (z.B. „I. Vorwort", „Part I Preface") gelten NICHT als erstes Body-Kapitel —
        # Vorwort/Preface/Acknowledgments gehören zum Frontmatter.
        if any(
            _is_real_chapter_match(m) and not _FRONTMATTER_RE.search(m.group(2))
            for m in _CHAPTER_RE.finditer(txt)
        ):
            first_chapter_idx = i
            break

    if first_chapter_idx is None or first_chapter_idx == 0:
        return pages, 0

    pre_pages = pages[:first_chapter_idx]
    has_frontmatter_signal = any(_FRONTMATTER_RE.search(t) for _, t in pre_pages)
    if not has_frontmatter_signal:
        return pages, 0

    drop_ratio = first_chapter_idx / len(pages)
    if drop_ratio > _FRONTMATTER_MAX_DROP_RATIO:
        return pages, 0

    return pages[first_chapter_idx:], first_chapter_idx


def pdf_to_text(pdf_path: Path, strip_frontmatter: bool = True) -> str:
    """Backwards-compat: liefert Text mit eingebetteten Seiten-Markern.

    `strip_frontmatter=True` (Default) schneidet Frontmatter-Seiten (Advance Praise,
    Copyright, Acknowledgments, Preface, …) vor dem ersten Kapitel-Heading ab. Schützt
    `concept_text_window` vor Cluster-Treffern in der Pre-Chapter-Sektion (Sachbücher
    mit langem Frontmatter, in dem die Konzept-Begriffe en passant vorkommen).
    """
    pages = pdf_to_pages(pdf_path)
    if strip_frontmatter:
        pages, dropped = drop_frontmatter_pages(pages)
        if dropped:
            print(
                f"      [frontmatter-strip] {dropped} Seite(n) entfernt "
                f"(vor erstem Kapitel-Heading)",
                file=sys.stderr,
            )
    return pages_to_marked_text(pages)


def page_range_of_text(text: str) -> tuple[int | None, int | None]:
    """Extrahiert (page_start, page_end) aus eingebetteten `[S. N]`-Markern."""
    nums = [int(m.group(1)) for m in _PAGE_MARKER_RE.finditer(text)]
    if not nums:
        return None, None
    return min(nums), max(nums)


@dataclass
class TextQuality:
    """Ergebnis des Textqualitäts-Gates (G6/#27)."""
    total_words: int
    pages: int
    words_per_page: float
    is_empty: bool   # gar kein extrahierbarer Text
    is_thin: bool    # Text vorhanden, aber unter MIN_WORDS_PER_PAGE (gescannt/kaputt)


def assess_text_quality(text: str) -> TextQuality:
    """Bewertet die Dichte des extrahierten PDF-Texts.

    Zählt Wörter (ohne die `[S. N]`-Seiten-Marker) gegen die Seitenzahl (Anzahl
    Marker; ohne Marker, aber mit Text → mindestens 1 Seite). `is_thin` greift,
    wenn Text vorhanden ist, aber im Mittel weniger als ``MIN_WORDS_PER_PAGE``
    Wörter pro Seite liefert — typisch für gescannte/kaputte/copy-protected PDFs,
    bei denen sonst stiller dünner Text Coverage UND Halluzinationsrate verfälscht.

    Pure Funktion (fail-open): bewertet nur, löst nichts aus. Der Caller warnt.
    """
    page_count = len(_PAGE_MARKER_LINE_RE.findall(text))
    body = _PAGE_MARKER_RE.sub(" ", text)
    total_words = len(body.split())
    pages = max(page_count, 1) if total_words else page_count
    words_per_page = total_words / pages if pages else 0.0
    is_empty = total_words == 0
    is_thin = not is_empty and words_per_page < MIN_WORDS_PER_PAGE
    return TextQuality(
        total_words=total_words,
        pages=pages,
        words_per_page=words_per_page,
        is_empty=is_empty,
        is_thin=is_thin,
    )


def concept_text_window(full_text: str, search_terms: list[str],
                        window_words: int = 400, max_chars: int = 8000) -> str:
    """Sliding-Window Co-Occurrence Ranking — wählt die thematisch dichtesten
    Fenster aus dem Volltext (Option D, Gemini-Review 2026-05-17).

    Konvention: ``search_terms[0]`` ist der vollständige Konzept-Titel,
    ``search_terms[1:]`` sind Einzel-Tokens für den Co-Occurrence-Score.

    Scoring pro Fenster (window_words Wörter, 50%-Stride):
    - +100 pro Vorkommen des exakten Titels (case-insensitive Substring)
    - +1 pro **unterschiedlichem** Token aus search_terms[1:], das mindestens
      einmal im Fenster vorkommt (Wiederholung zählt nicht — verhindert dass
      ein einzelnes generisches Token wie ``agent`` den Score dominiert)

    Auswahl: Top-Fenster nach Score, gesammelt bis ``max_chars`` Chars erreicht;
    Overlaps werden in Dokumentenreihenfolge gemerged.

    Bei keinem Match leerer String (Halluzinations-Filter greift upstream).

    Vorgängerversion (vor 2026-05-17) hat um Treffer-Cluster ±window_words
    expandiert. Bei generischen Tokens (``agent``, ``system``) wuchs der
    Cluster über das ganze Dokument und der Extractor sah nur die ersten
    8000 chars (TOC+Intro), nie die Substanz-Kapitel.
    """
    if not search_terms:
        return full_text[:max_chars]

    words = full_text.split()
    if not words:
        return ""

    # Seite pro Wort-Index tracken: damit ein selektiertes Fenster, das mitten auf
    # einer Seite beginnt (der `[S. N]`-Marker stand am Seitenanfang, vor dem
    # Fenster), seinen korrekten Marker vorangestellt bekommt. Sonst erbt die
    # Downstream-Seitenableitung ("letzter [S. N]-Marker vor der Fundstelle":
    # Extractor-LLM, Verifier, Renderer) die Seite eines früheren Snippets →
    # falsche Fußnoten-Seite (#4 Anker-Clustering, Merrill-Run 2026-06-24).
    # NUR line-isolierte Pipeline-Marker (`\n\n[S. N]\n\n` aus pages_to_marked_text)
    # zählen als Seitenanfang — Inline-Quellenverweise wie „vgl. [S. 12]" im
    # Fließtext NICHT (sonst erbt Folgetext die zitierte statt der echten Seite;
    # Codex-Review 2026-06-24). re.finditer(r"\S+") liefert dieselbe Token-Folge
    # wie full_text.split() oben, plus Positionen fürs Marker-Mapping.
    _real_markers = [(m.start(), m.group(1)) for m in
                     re.finditer(r"(?m)^[ \t]*\[S\.\s*(\d+)\][ \t]*$", full_text)]
    page_at_word: list[str | None] = []
    _cur_page: str | None = None
    _mi = 0
    for _tok in re.finditer(r"\S+", full_text):
        while _mi < len(_real_markers) and _real_markers[_mi][0] <= _tok.start():
            _cur_page = _real_markers[_mi][1]
            _mi += 1
        page_at_word.append(_cur_page)

    # Title normalisieren auf gleiche Whitespace-Form wie `chunk` (single-space-join)
    # — sonst matcht z.B. "Multi-Agent\n\nSystem" nicht im normalisierten Chunk.
    title = " ".join((search_terms[0] or "").split())
    tokens = [t for t in search_terms[1:] if t]

    title_re = re.compile(re.escape(title), re.IGNORECASE) if title else None
    token_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in tokens]

    stride = max(1, window_words // 2)
    scored: list[tuple[int, int, int]] = []  # (score, start_word, end_word)
    start = 0
    while start < len(words):
        end = min(start + window_words, len(words))
        chunk = " ".join(words[start:end])
        score = 0
        if title_re:
            score += 100 * len(title_re.findall(chunk))
        for pat in token_res:
            if pat.search(chunk):
                score += 1
        if score > 0:
            scored.append((score, start, end))
        if end >= len(words):
            break
        start += stride

    if not scored:
        return ""

    scored.sort(key=lambda x: (-x[0], x[1]))

    picked: set[tuple[int, int]] = set()
    total_chars = 0
    sep_overhead = len("\n\n[...]\n\n")
    for score, s, e in scored:
        chunk_chars = sum(len(w) + 1 for w in words[s:e])
        added = chunk_chars + (sep_overhead if picked else 0)
        if total_chars + added > max_chars and picked:
            break
        picked.add((s, e))
        total_chars += added
        if total_chars >= max_chars:
            break
    # `total_chars` summiert Roh-Chunk-Längen vor dem Merge (Z. 78ff). Bei
    # 50%-Stride zählen überlappende Fenster ihren Overlap doppelt — Folge:
    # das tatsächlich ausgegebene `\n\n[...]\n\n`-Join ist kleiner als
    # `total_chars`, d.h. das Budget wird leicht unterausgenutzt, nie überschritten.
    # Bewusst akzeptiert; sauberes Tracking pro merged-Span wäre teurer als der Gewinn.

    spans = sorted(picked)
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            ps, pe = merged[-1]
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # Snippet-Bau: jedem markerlosen Snippet seinen gültigen Seitenmarker
    # voranstellen (s.o.). Das addiert ~"[S. N] " (≤ ~10 Zeichen) je injiziertem
    # Snippet — die max_chars-Aussage oben ist damit nicht mehr strikt, der
    # Overhead ist aber vernachlässigbar gegen das ohnehin unterausgenutzte Budget.
    snippets: list[str] = []
    for s, e in merged:
        snip = " ".join(words[s:e])
        page = page_at_word[s]
        if page is not None and not snip.lstrip().startswith("[S."):
            snip = f"[S. {page}] {snip}"
        snippets.append(snip)
    return "\n\n[...]\n\n".join(snippets)


def _parse_pdfinfo_output(stdout: str) -> dict[str, str]:
    """Parst pdfinfo-stdout zu Metadaten-dict (pure, testbar).

    Quellen-Treue (universell, nicht quellen-spezifisch): pdfinfo-`Author`
    (= Datei-Ersteller) und das Jahr aus `CreationDate` (= Speicher-/Abtipp-
    Zeitpunkt) sind NICHT zitierfähig — sie identifizieren weder Werk-Autor noch
    Publikationsjahr und führen bei abgetippten/gescannten/neu-gespeicherten PDFs
    zu systematischer Fehlattribution. Deshalb werden sie NICHT als `Author`/`Year`
    exportiert, sondern nur diagnostisch als `InfoDictAuthor`/`InfoDictCreationYear`
    (für Logging, nie für Zitate). Zitier-Autor/-Jahr kommen ausschließlich aus
    Dateiname, CrossRef/DOI oder validierter Titelseiten-Extraktion (Orchestrator).
    `Title`/`Pages`/`Subject` bleiben zitierfähig.
    """
    keep = {"Title", "Subject", "Pages"}
    meta: dict[str, str] = {}
    info_author = ""
    info_creationdate = ""
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if not val:
            continue
        if key in keep:
            meta[key] = val
        elif key == "Author":
            info_author = val
        elif key == "CreationDate":
            info_creationdate = val
    # Info-Dict-Autor + Year aus CreationDate nur diagnostisch ablegen (Format
    # z.B. "Mon Mar 15 14:23:01 2019 CET") — nie als zitierfähige Quelle.
    if info_author:
        meta["InfoDictAuthor"] = info_author
    if info_creationdate:
        m = re.search(r"\b(19|20)\d{2}\b", info_creationdate)
        if m:
            meta["InfoDictCreationYear"] = m.group(0)
    return meta


def pdf_metadata(pdf_path: Path) -> dict[str, str]:
    """Liest pdfinfo-Metadaten als dict (Title, Subject, Pages zitierfähig;
    Info-Dict-Autor/-CreationDate nur diagnostisch — siehe _parse_pdfinfo_output)."""
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        return {}
    return _parse_pdfinfo_output(result.stdout)


# Kapitel-Heading-Pattern. Erkennt:
#   - arabisch: "1 Titel", "Kapitel 2", "Chapter 3", "2.1 Untertitel"
#   - römisch:  "I. Einleitung", "Part II", "Kapitel III"
#   - ausgeschrieben: "Kapitel zwei Grundlagen", "Chapter two"
#   - erweiterte Prefixe: "Beitrag 3 Titel" (Sammelband-Beiträge), "Section"
# Titel-Länge bis 120 Zeichen (vorher 60) — viele Sachbücher haben längere Titel.
# Cross-Model-Konsens Codex/Gemini 2026-05-11. Dual-Use mit drop_frontmatter_pages()
# bedacht: TOC-Trail-Filter + Mindestlänge schützen vor False-Positives.
_CHAPTER_RE = re.compile(
    r"^\s*"
    r"(?:Kapitel|Chapter|Teil|Part|Beitrag|Section)?"
    r"\s*"
    r"("
        r"\d{1,2}(?:\.\d{1,2})*"            # arabisch (1, 2.3)
        r"|"
        r"[IVX]{1,4}"                        # römisch (Case-sensitive, max 4 Stellen)
        r"|"
        r"(?i:eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn"
            r"|one|two|three|four|five|six|seven|eight|nine|ten)"
    r")"
    r"[.:]?\s+"
    r"([A-ZÄÖÜ][^\n]{3,120})"
    r"\s*$",
    re.MULTILINE,
)

# TOC-Trail: Inhaltsverzeichnis-Zeilen wie "I. Einleitung .......... 12" oder
# mehrfaches Spacing + Seitenzahl. Solche Zeilen sind keine echten Kapitel-Headings.
_TOC_TRAIL_RE = re.compile(r"(?:\.{2,}|\s{3,}|\t)\s*\d{1,4}\s*$")


def _is_real_chapter_match(match: re.Match) -> bool:
    """Post-Filter: True wenn der Match ein echtes Kapitel-Heading ist, nicht TOC-Eintrag
    oder Aufzählungs-Bulletpoint. Greift auf die Match-Zeile als Ganzes zu."""
    line = match.group(0)
    if _TOC_TRAIL_RE.search(line):
        return False
    return True


def split_by_chapters(text: str) -> list[Chunk]:
    """Teilt Text (mit `[S. N]`-Markern) an Kapitel-Headings. Fallback: Word-Count."""
    matches = [m for m in _CHAPTER_RE.finditer(text) if _is_real_chapter_match(m)]
    if len(matches) < 2:
        return _split_by_words(text)

    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = f"{m.group(1)} {m.group(2).strip()}"
        chunk_text = text[start:end].strip()
        # Page-Range: letzter Marker VOR start (wenn Chunk auf bereits laufender Seite startet)
        # plus Marker IM Chunk
        prefix_pages = [int(mm.group(1)) for mm in _PAGE_MARKER_RE.finditer(text[:start])]
        chunk_pages = [int(mm.group(1)) for mm in _PAGE_MARKER_RE.finditer(chunk_text)]
        all_pages = ([prefix_pages[-1]] if prefix_pages else []) + chunk_pages
        page_start = min(all_pages) if all_pages else None
        page_end = max(all_pages) if all_pages else None
        chunks.append(Chunk(
            title=title, text=chunk_text, index=i,
            page_start=page_start, page_end=page_end,
        ))
    return chunks


def _split_by_words(text: str) -> list[Chunk]:
    """Word-basiertes Splitten — Seitenmarker bleiben im Text erhalten, page_start/end
    werden aus den Markern im jeweiligen Block abgeleitet."""
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, len(words), CHUNK_WORDS):
        block = " ".join(words[i:i + CHUNK_WORDS])
        page_start, page_end = page_range_of_text(block)
        chunks.append(Chunk(
            title=f"Abschnitt {i // CHUNK_WORDS + 1}",
            text=block,
            index=i // CHUNK_WORDS,
            page_start=page_start, page_end=page_end,
        ))
    return chunks


def extract_overview(text: str, max_words: int = 1500) -> str:
    """Repräsentativer Planner-Input über ALLE Kapitel, strikt innerhalb max_words.

    Alt: erste N + letzte K Wörter → mittlere Kapitel systematisch blind.
    Problem: operative Konzepte (Evals, Memory, Safety) in späteren Kapiteln
    wurden vom Planner nie gesehen → systematische Unterrepräsentation.

    Neu: Intro (min(600, max_words//3)) + Kapitel-Snippets (Budget-basiert,
    ohne Kapitel-1-Überlappung) + Fazit (min(300, max_words//5)).
    Alle Teile zusammen ≤ max_words. Fallback ohne Kapitel: Stichproben.
    """
    words = text.split()
    n = len(words)

    intro_budget = min(600, max_words // 3)
    outro_budget = min(300, max_words // 5) if n > intro_budget + 300 else 0
    snippet_budget = max(0, max_words - intro_budget - outro_budget)

    parts = [" ".join(words[:intro_budget])]

    chapters = split_by_chapters(text)
    # Kapitel-1-Überlappung vermeiden: erstes Kapitel hat oft denselben Inhalt
    # wie der Intro-Block → ab Index 1 beginnen (Gemini-Finding 2026-05-13).
    later_chapters = chapters[1:] if len(chapters) > 1 else []
    if later_chapters:
        per_chapter = max(50, snippet_budget // len(later_chapters))
        snippets = []
        for ch in later_chapters:
            ch_words = ch.text.split()
            snippet = " ".join(ch_words[:per_chapter])
            if not snippet.strip():
                continue  # leere Kapitel überspringen (Nemotron-Finding 2026-05-13)
            snippets.append(f"=== {ch.title} ===\n{snippet}")
        parts.append("[Kapitel-Überblick:]\n" + "\n\n".join(snippets))
    elif n > 3000:
        # Fallback: gleichmäßige Stichproben ohne Kapitel-Erkennung
        per_sample = max(100, snippet_budget // max(1, (n - intro_budget) // 1500))
        samples = []
        budget_used = 0
        for start in range(intro_budget, n - outro_budget, 1500):
            if budget_used >= snippet_budget:
                break
            take = min(per_sample, snippet_budget - budget_used)
            samples.append(" ".join(words[start:start + take]))
            budget_used += take
        if samples:
            parts.append("[Stichproben:]\n" + "\n\n[...]\n\n".join(samples))

    if outro_budget > 0:
        parts.append("[Ende:]\n" + " ".join(words[-outro_budget:]))

    return "\n\n".join(parts)
