"""PDF → Text-Chunks mit Seiten-Markern, aufgeteilt nach Kapitelstruktur.

Seiten-Marker `[S. N]` werden an Seitenanfängen eingefügt (basierend auf pdftotext's
\\f-Form-Feed-Markierung). Damit kann der Extractor Anker-Zitate mit korrekter Seitenzahl
versehen und der Verifier die Seitenzahl gegen den Originaltext prüfen.
"""

from __future__ import annotations
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from generative.config import (
    CHUNK_WORDS,
    MAX_SANE_OUTLINE_CHAPTERS,
    MIN_CHAPTER_SEGMENT_WORDS,
    MIN_WORDS_PER_PAGE,
)

# S5 (#150): Obergrenze fuer poppler-Subprozesse (pdftotext/pdfinfo). Ein
# defektes/boesartiges PDF darf die Pipeline nicht unbegrenzt haengen lassen —
# gleicher Wert wie figure_alt.py:67.
_PDF_SUBPROCESS_TIMEOUT_S = 120


@dataclass
class Chunk:
    title: str  # Kapitelüberschrift oder "Abschnitt N"
    text: str  # mit `[S. N]`-Markern an Seitenanfängen
    index: int
    page_start: int | None = None
    page_end: int | None = None
    # Herkunft des Splits (#345): "outline" = PDF-Lesezeichen, "heuristic" =
    # Text-Heading-Regex, "words" = Wort-Count-Fallback. None bei Alt-Konstruktion.
    source: str | None = None


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


def pdf_uses_physical_pages(pdf_path: Path) -> bool:
    """True wenn `pdf_to_pages` für dieses PDF auf den `i+1`-Fallback zurückfällt
    (keine nutzbaren `/PageLabels`) — die zurückgegebene Seitenzahl ist dann die
    physische PDF-Position, keine gedruckte Seite (Issue #95).

    Eigenständiger, günstiger Zweit-Check (dieselbe fail-open `_pdf_page_labels`-
    Logik, die `orchestrator.main()` bereits für die Edition-Verifikation nutzt)
    statt eines Rückgabe-Umbaus von `pdf_to_pages`/`pdf_to_text` — deren Signatur
    bleibt für die bestehenden Aufrufer (calibration-/eval-Skripte) unverändert.
    """
    return _pdf_page_labels(pdf_path) is None


def _resolve_page_numbers(pages_raw: list[str], labels: list | None) -> list[tuple[int, str]]:
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


def anchor_page_numbers(pdf_path: Path, n_pages: int) -> list[int]:
    """Anker-Seitenzahl je physischem 0-basiertem Index (0..n_pages-1) — derselbe
    Namespace wie `pdf_to_pages`/`source_anchors`: das numerische Druckseiten-
    Label aus `/PageLabels`, falls das PDF welche fuehrt, sonst i+1 (#80 Fund 1).

    Gemeinsames Mapping fuer eval_quality.py/_v2.py/_v4.py, die bis PR #79 den
    physischen PDF-Index lasen, obwohl source_anchors seither Druckseiten tragen.
    Wiederverwendet `_resolve_page_numbers` (keine zweite Label-Parsing-
    Implementierung, gleiche Semantik wie `figure_alt.pdf_index_to_anchor_page`
    fuer Einzelindizes)."""
    labels = _pdf_page_labels(pdf_path)
    numbered = _resolve_page_numbers([""] * n_pages, labels)
    return [n for n, _ in numbered]


def physical_pages_by_anchor(pdf_path: Path, n_pages: int) -> dict[int, int]:
    """Kehrt `anchor_page_numbers` um: {Anker-Seitenzahl: physischer 1-basierter
    Index}. Mit `/PageLabels` bijektiv (`_usable_page_labels` erzwingt Eindeutig-
    keit + Monotonie); ohne Labels die Identitaet (n -> n)."""
    return {num: i + 1 for i, num in enumerate(anchor_page_numbers(pdf_path, n_pages))}


def pdf_to_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Liefert [(page_num, page_text), ...] via pdftotext + \\f-Split.

    `page_num` ist die zitierfähige Druckseite aus den PDF-`/PageLabels`, falls das
    PDF welche führt (Buch: PDF-Seite 179 → Druckseite „159"); sonst die 1-basierte
    pdftotext-Position (Paper ohne Labels — unverändertes Verhalten)."""
    from generative.error_hints import pdftotext_error_hint

    try:
        result = subprocess.run(
            # S6 (#150): Pfad absolutieren -> ein relativer Name mit fuehrendem
            # "-" kann nicht als poppler-Option fehlinterpretiert werden.
            ["pdftotext", str(Path(pdf_path).resolve()), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PDF_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        # S5 (#150): PDF haengt pdftotext -> hart mit handlungsanleitender Meldung
        # abbrechen (analog OSError-Pfad) statt die Pipeline blockieren zu lassen.
        sys.exit(
            pdftotext_error_hint(
                f"pdftotext hat nach {_PDF_SUBPROCESS_TIMEOUT_S}s nicht geantwortet — "
                f"PDF evtl. defekt/boesartig: {pdf_path}"
            )
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
    "advance praise",
    "praise for",
    "acknowledgments",
    "acknowledgements",
    "danksagung",
    "copyright",
    "all rights reserved",
    "alle rechte vorbehalten",
    "dedication",
    "widmung",
    "table of contents",
    "contents",
    "inhaltsverzeichnis",
    "inhalt",
    "foreword",
    "vorwort",
    "preface",
    "geleitwort",
    "about the author",
    "über den autor",
    "über die autorin",
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
        if any(_is_real_chapter_match(m) and not _FRONTMATTER_RE.search(m.group(2)) for m in _CHAPTER_RE.finditer(txt)):
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
                f"      [frontmatter-strip] {dropped} Seite(n) entfernt (vor erstem Kapitel-Heading)",
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
    is_empty: bool  # gar kein extrahierbarer Text
    is_thin: bool  # Text vorhanden, aber unter MIN_WORDS_PER_PAGE (gescannt/kaputt)


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


def _iter_word_windows(n_words: int, window_words: int, stride: int | None = None):
    """Wort-Index-Fenster `(start, end)`, 50%-Stride per Default.

    Geteilte Sliding-Window-Geometrie zwischen `concept_text_window`s
    lexikalischem Scorer und dem #127-Fallback (`semantic_concept_window`) —
    beide sollen exakt dieselben Fenstergrenzen sehen, kein zweites
    Chunking-Schema für denselben Volltext.
    """
    stride = stride if stride is not None else max(1, window_words // 2)
    start = 0
    while start < n_words:
        end = min(start + window_words, n_words)
        yield start, end
        if end >= n_words:
            break
        start += stride


def _page_at_word_map(full_text: str) -> list[str | None]:
    """Seiten-Marker (`[S. N]`) pro Wort-Index — Vorarbeit für den Snippet-Bau
    in `concept_text_window` UND `semantic_concept_window` (#127). Nur
    line-isolierte Pipeline-Marker (`\\n\\n[S. N]\\n\\n` aus pages_to_marked_text)
    zählen als Seitenanfang — Inline-Quellenverweise im Fließtext nicht (siehe
    `concept_text_window`-Docstring für die volle Begründung)."""
    _real_markers = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^[ \t]*\[S\.\s*(\d+)\][ \t]*$", full_text)]
    page_at_word: list[str | None] = []
    _cur_page: str | None = None
    _mi = 0
    for _tok in re.finditer(r"\S+", full_text):
        while _mi < len(_real_markers) and _real_markers[_mi][0] <= _tok.start():
            _cur_page = _real_markers[_mi][1]
            _mi += 1
        page_at_word.append(_cur_page)
    return page_at_word


def _prefix_page_marker(snippet: str, page: str | None) -> str:
    """Stellt einem markerlosen Snippet seinen gültigen Seitenmarker voran
    (s. `_page_at_word_map`) — sonst erbt die Downstream-Seitenableitung
    (Extractor-LLM, Verifier, Renderer) die Seite eines früheren Snippets."""
    if page is not None and not snippet.lstrip().startswith("[S."):
        return f"[S. {page}] {snippet}"
    return snippet


# #127: Satz-Embeddings pro (Volltext, window_words) gecacht — mehrere in
# Stage 5 lexikalisch leer gebliebene Konzepte DESSELBEN Laufs (z.B. alle drei
# Top-Konzepte eines fast-Profils) teilen sich EIN Batch-Encode statt es pro
# Konzept zu wiederholen. Gleiches Muster wie planner._SENT_EMB_CACHE.
_WINDOW_SENT_CACHE: dict[tuple[int, int, int], tuple] = {}
_WINDOW_SENT_CACHE_MAX = 8


def _window_sentence_embeddings(full_text: str, words: list[str], window_words: int):
    """(spans, window_texts, sentences, sentence_embeddings) für die
    Sliding-Window-Fenster von `words`.

    Satz-Ebene statt Fenster-Mittelwert: Kalibrierung #127 zeigt, dass
    Mean-Pooling über ein ganzes 400-Wort-Fenster den Score eines einzelnen
    treffenden Satzes verwässert (Fenster enthält 10-15 Sätze, nur einer
    trägt den Konzeptbezug) — ein reales Rettungsfall-Beispiel („Andragogik“
    auf der Knowles-Quelle) lag im Fenster-Mittel bei 0.43, auf Satzebene aber
    deutlich über der Schwelle. Damit bleibt `TITLE_PRESENCE_COSINE_THRESHOLD`
    (kalibriert für genau diese Satz-MAX-Methodik in
    `planner._default_semantic_presence`) gültig wiederverwendbar.

    Sätze aus überlappenden Fenstern (50%-Stride) werden dedupliziert (Wert =
    Text), bevor EIN `model.encode()`-Call über alle eindeutigen Sätze läuft
    — sonst würde der Overlap-Bereich doppelt encodet.

    Wirft bei fehlendem/kaputtem Embedding-Modell — Aufrufer fängt ab
    (fail-closed, #127 ist ein reiner Rettungskanal, keine Pflichtstufe).
    """
    key = (len(full_text), hash(full_text), window_words)
    cached = _WINDOW_SENT_CACHE.get(key)
    if cached is not None:
        return cached

    from generative.embeddings import _sentences, _model
    import numpy as np

    spans = list(_iter_word_windows(len(words), window_words))
    window_texts = [" ".join(words[s:e]) for s, e in spans]

    uniq_sentences: list[str] = []
    seen: set[str] = set()
    for wtext in window_texts:
        for s2 in _sentences(wtext):
            if len(s2) > 15 and s2 not in seen:
                seen.add(s2)
                uniq_sentences.append(s2)

    if uniq_sentences:
        sent_embs = np.asarray(
            _model().encode(uniq_sentences, show_progress_bar=False, normalize_embeddings=True, batch_size=64)
        )
    else:
        sent_embs = np.zeros((0, _model().get_sentence_embedding_dimension()))

    result = (spans, window_texts, uniq_sentences, sent_embs)
    if len(_WINDOW_SENT_CACHE) >= _WINDOW_SENT_CACHE_MAX:
        _WINDOW_SENT_CACHE.pop(next(iter(_WINDOW_SENT_CACHE)))
    _WINDOW_SENT_CACHE[key] = result
    return result


def semantic_concept_window(
    full_text: str,
    title: str,
    threshold: float,
    window_words: int = 400,
    max_chars: int = 8000,
) -> tuple[str, float]:
    """#127: semantischer Fallback für `concept_text_window()` bei lexikalisch
    leerem Ergebnis.

    Der Stage-5-Skip (orchestrator.run_extractors_per_concept) ist rein
    lexikalisch und sprachblind: ein deutscher Planner-Titel auf einer
    englischen Quelle hat 0 Token-Overlap, obwohl dasselbe Konzept
    `planner.filter_hallucinated`s #66-Rettungsanker (semantische Präsenz,
    multilinguales MiniLM) bereits passiert haben kann — die beiden Checks
    laufen auf demselben Volltext, aber nur Stage 4 hat einen semantischen
    Kanal. Dieser Fallback bietet denselben Kanal auch hier an: MAX-Cosine
    zwischen Titel-Embedding und Satz-Embeddings (multilinguales MiniLM,
    `generative.embeddings`) — exakt dieselbe Methodik + Schwelle
    (`TITLE_PRESENCE_COSINE_THRESHOLD`) wie #66, damit der kalibrierte Wert
    gültig bleibt (siehe `_window_sentence_embeddings`-Docstring für den
    Kalibrierungsbefund, der Fenster-Mittelwert statt Satz-MAX verwarf).
    Das Ergebnisfenster für den Extraktor-Kontext ist trotzdem eines der
    SELBEN Sliding-Window-Fenster wie der lexikalische Scorer
    (`_iter_word_windows`) — kein zweites Chunking-Schema; es wird das erste
    Fenster (Dokumentreihenfolge) gewählt, das den Treffer-Satz enthält.

    Anders als `_default_semantic_presence` (reiner OR-Kanal in Stage 4,
    fail-OPEN) muss dieser Fallback FAIL-CLOSED sein: das Ergebnis wird
    direkt als Extraktor-Kontext weiterverwendet, nicht nur als binäres
    Keep/Reject-Signal — ein Encoding-Fehler darf keinen unkontrollierten
    Volltext-Ausschnitt durchreichen.

    Returns: (bestes Fenster mit Seitenmarker, höchste Cosine). Leerer String
    wenn kein Satz die Schwelle erreicht ODER das Embedding-Modell nicht
    verfügbar ist/das Encoding scheitert (Score dann bestmöglich, sonst 0.0)
    — der Aufrufer behält in diesem Fall den bisherigen `[skip]`.
    """
    if not title.strip() or not full_text.strip():
        return "", 0.0
    words = full_text.split()
    if not words:
        return "", 0.0

    try:
        from generative.embeddings import embed_title

        spans, window_texts, sentences, sent_embs = _window_sentence_embeddings(full_text, words, window_words)
        if not sentences:
            return "", 0.0
        te = embed_title(title)
        sims = sent_embs.dot(te)
        best_i = int(sims.argmax())
        best_score = float(sims[best_i])
    except Exception:
        return "", 0.0

    if best_score < threshold:
        return "", max(best_score, 0.0)

    best_sentence = sentences[best_i]
    page_at_word = _page_at_word_map(full_text)
    for (s, _e), wtext in zip(spans, window_texts):
        if best_sentence in wtext:
            return _prefix_page_marker(wtext, page_at_word[s])[:max_chars], best_score
    # Sollte nicht eintreten (Treffer-Satz stammt aus genau diesen Fenstern) —
    # fail-closed statt eines Fensters ohne nachvollziehbaren Bezug.
    return "", best_score


def concept_text_window(full_text: str, search_terms: list[str], window_words: int = 400, max_chars: int = 8000) -> str:
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

    # Seite pro Wort-Index tracken (s. `_page_at_word_map`): damit ein
    # selektiertes Fenster, das mitten auf einer Seite beginnt (der `[S. N]`-
    # Marker stand am Seitenanfang, vor dem Fenster), seinen korrekten Marker
    # vorangestellt bekommt. Sonst erbt die Downstream-Seitenableitung
    # ("letzter [S. N]-Marker vor der Fundstelle": Extractor-LLM, Verifier,
    # Renderer) die Seite eines früheren Snippets → falsche Fußnoten-Seite
    # (#4 Anker-Clustering, Merrill-Run 2026-06-24).
    page_at_word = _page_at_word_map(full_text)

    # Title normalisieren auf gleiche Whitespace-Form wie `chunk` (single-space-join)
    # — sonst matcht z.B. "Multi-Agent\n\nSystem" nicht im normalisierten Chunk.
    title = " ".join((search_terms[0] or "").split())
    tokens = [t for t in search_terms[1:] if t]

    title_re = re.compile(re.escape(title), re.IGNORECASE) if title else None
    token_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in tokens]

    scored: list[tuple[int, int, int]] = []  # (score, start_word, end_word)
    for start, end in _iter_word_windows(len(words), window_words):
        chunk = " ".join(words[start:end])
        score = 0
        if title_re:
            score += 100 * len(title_re.findall(chunk))
        for pat in token_res:
            if pat.search(chunk):
                score += 1
        if score > 0:
            scored.append((score, start, end))

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
        snippets.append(_prefix_page_marker(" ".join(words[s:e]), page_at_word[s]))
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
    Info-Dict-Autor/-CreationDate nur diagnostisch — siehe _parse_pdfinfo_output).

    Fail-soft wie der returncode-Pfad: fehlt das pdfinfo-Binary ganz (WinError 2
    — z. B. choco-Xpdf-Paketierung mit pdftotext, aber ohne pdfinfo), liefert
    die Funktion {} statt die Pipeline hart zu crashen. pdfinfo-Metadaten sind
    optional; die Nutzer-Meldung dafür macht der doctor-Check (check_tool).
    Fund: erster CI-Lauf des Smoke-E2E (#97) auf windows-latest.
    """
    try:
        result = subprocess.run(
            # S6 (#150): Pfad absolutieren (Argument-Injection-Schutz, s. pdf_to_pages).
            ["pdfinfo", str(Path(pdf_path).resolve())],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PDF_SUBPROCESS_TIMEOUT_S,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        # S5 (#150): Timeout wie fehlendes Binary behandeln — pdfinfo ist optional
        # (fail-soft: {}), ein haengendes PDF darf hier nichts blockieren.
        return {}
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
    r"\d{1,2}(?:\.\d{1,2})*"  # arabisch (1, 2.3)
    r"|"
    r"[IVX]{1,4}"  # römisch (Case-sensitive, max 4 Stellen)
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
# Erweiterung (#345): gespacte Dot-Leader (`. . . .` — PDF-Extraktionsartefakt) und
# römische Seitenzahlen (`... xii`, Frontmatter) fängt der Alt-Ausdruck nicht.
# NUR additive Alternativen — die bestehenden Fälle bleiben unverändert (test_chapter_regex).
# Römisch bewusst lowercase-only (Frontmatter-Konvention) und ohne IGNORECASE, damit
# ein Titel-Wort wie "civil" (∈ {i,v,x,l,c}) nach `\s{3,}` nicht als Seitenzahl gilt.
_TOC_TRAIL_RE = re.compile(r"(?:\.{2,}|(?:\.\s){2,}|\s{3,}|\t)\s*(?:\d{1,4}|[ivxl]{1,7})\s*$")


def _is_real_chapter_match(match: re.Match) -> bool:
    """Post-Filter: True wenn der Match ein echtes Kapitel-Heading ist, nicht TOC-Eintrag
    oder Aufzählungs-Bulletpoint. Greift auf die Match-Zeile als Ganzes zu."""
    line = match.group(0)
    if _TOC_TRAIL_RE.search(line):
        return False
    return True


# --- Outline-first Kapitel-Split (#345) -----------------------------------
# Kapitelgrenze = fitz-aufgelöste Outline-Zielseite, auf den `[S. N]`-Marker
# gemappt und per Titel-Wort-Overlap validiert — NICHT Titel-Matching im Volltext
# (P4-Befund: Heading-Regex liefert auf realen Büchern 124–1798 „Kapitel" statt
# 7–32). Ohne nutzbare Outline fällt `split_by_chapters` transparent auf den
# heuristischen Normalpfad zurück (ehrliche Grenze: Scans ohne Bookmarks).

_OUTLINE_VALIDATION_WINDOW_CHARS = 400  # Fenster ab gemapptem Marker (24/24-Empirie)
_OUTLINE_TITLE_OVERLAP_MIN = 0.5  # ≥50 % der Titel-Wörter im Fenster (R3/V4-5)
_OUTLINE_FUZZY_THRESHOLD = 80  # HiPS-Kaskade Zweitcheck (partial_ratio %)
_OUTLINE_GIANT_SEGMENT_RATIO = 0.70  # ein Segment >70 % Gesamtwörter → degeneriert
_OUTLINE_OFFSET_VOTING_SAMPLE = 20  # Stichprobe für den Map-Kreuzcheck
_MARKER_LINE_RE = re.compile(r"(?m)^[ \t]*\[S\.\s*(\d+)\][ \t]*$")

# Front-/Backmatter-Outline-Titel (kurze Bookmark-Titel, nicht Seiten-Body).
# `_FRONTMATTER_RE` bleibt SSoT der Body-Phrasen (drop_frontmatter_pages); hier
# NUR additiv für den Outline-Kanal (#345), Termliste exakt aus Plan §1:
#   Substring `verzeichnis` (fängt Literatur-/Abbildungs-/Abkürzungsverzeichnis…)
#   + Ganzwort Inhalt/Vorwort/Glossar/Register/Index/Anhang/Geleitwort/Impressum/…
#   + Präfix Danksag(ung)/Autor(en).
# Bewusst KEIN bare „literatur"/„abbildung": „Zweiter Teil. … Literatur, Bücher,
# Medien" (Gantert) ist ein echtes Kapitel — bare Substrings droppen reale Titel.
_OUTLINE_SKIP_EXACT = frozenset(
    {
        "inhalt",
        "vorwort",
        "glossar",
        "geleitwort",
        "grusswort",
        "register",
        "index",
        "anhang",
        "appendix",
        "widmung",
        "impressum",
        "stichwort",
    }
)
_OUTLINE_SKIP_PREFIXES = ("danksag", "autor")  # Danksagung; Autoren, Autorenverzeichnis


def _norm_text(s: str) -> str:
    """NFKD-entdiakritisiert, lowercase, nur alnum+space — für Titel-Overlap-Vergleich."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _clean_outline_title(title: str) -> str:
    """NUL-Padding (bei Gantert beobachtet) + Whitespace entfernen."""
    if not title:
        return ""
    return title.replace("\x00", "").strip()


def _is_outline_skip_title(title: str) -> bool:
    """Front-/Backmatter-Outline-Eintrag? `_FRONTMATTER_RE` (SSoT) + `verzeichnis`-
    Substring + Ganzwort-/Präfix-Termliste (Plan §1)."""
    if _FRONTMATTER_RE.search(title):
        return True
    if "verzeichnis" in title.lower():
        return True
    for w in _norm_text(title).split():
        if w in _OUTLINE_SKIP_EXACT or any(w.startswith(p) for p in _OUTLINE_SKIP_PREFIXES):
            return True
    return False


def _outline_raw_entries(pdf_path: Path) -> list[tuple[int, str, int]]:
    """[(level, title, phys0)] aus `get_toc(simple=False)`.

    Zielseite bevorzugt aus dem fitz-aufgelösten Tupel-Element (1-basiert, Index 2)
    — bei 2/5 Stichproben-Büchern ist `dest["page"]` eine unaufgelöste named-
    destination-String; das Tupel-Element ist dann der korrekte Wert (#345)."""
    import fitz

    doc = fitz.open(str(Path(pdf_path).resolve()))
    try:
        toc = doc.get_toc(simple=False)
    finally:
        doc.close()

    entries: list[tuple[int, str, int]] = []
    for item in toc:
        level = item[0]
        title = _clean_outline_title(item[1])
        page_1based = item[2]
        dest = item[3] if len(item) > 3 else None
        phys0: int | None = None
        if isinstance(page_1based, int) and page_1based > 0:
            phys0 = page_1based - 1
        elif isinstance(dest, dict):
            dp = dest.get("page", -1)
            if isinstance(dp, int) and dp >= 0:
                phys0 = dp
        if phys0 is not None and title:
            entries.append((level, title, phys0))
    return entries


def _select_main_level_entries(entries: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Hauptkapitel-Ebene nach Front-/Backmatter-Filter. Root-Descend (R3/V4-3):
    bleiben <2 Einträge UND existiert genau 1 Wurzel → eine Ebene tiefer (Moser-
    Muster: 55 Einträge unter einer Wurzel)."""
    if not entries:
        return []
    min_level = min(e[0] for e in entries)
    top = [e for e in entries if e[0] == min_level]
    kept = [e for e in top if not _is_outline_skip_title(e[1])]
    if len(kept) >= 2:
        return kept
    if len(top) == 1:
        deeper = [e for e in entries if e[0] == min_level + 1]
        kept_deeper = [e for e in deeper if not _is_outline_skip_title(e[1])]
        if len(kept_deeper) >= 2:
            return kept_deeper
    return kept


def _merge_duplicate_boundaries(entries: list[tuple[int, str, int]]) -> list[tuple[int, str, int]]:
    """Duplikate mit Seitenabstand ≤1 zusammenfassen — längerer Titel gewinnt, früheste
    Seite bleibt (DAMA-Doppel-Bookmarks). Voraussetzung: nach `phys0` sortiert."""
    merged: list[tuple[int, str, int]] = []
    for lvl, title, phys0 in sorted(entries, key=lambda e: e[2]):
        if merged and phys0 - merged[-1][2] <= 1:
            plvl, ptitle, pphys0 = merged[-1]
            if len(title) > len(ptitle):
                merged[-1] = (plvl, title, pphys0)
        else:
            merged.append((lvl, title, phys0))
    return merged


def _pdftotext_raw_pages(pdf_path: Path) -> list[str] | None:
    """Roh-Seiten (pdftotext, `\\f`-Split, OHNE Leerseiten-Filter) — physischer Index.
    Fail-open: jeder Fehler → None (der Outline-Pfad fällt dann auf den Normalpfad)."""
    try:
        result = subprocess.run(
            ["pdftotext", str(Path(pdf_path).resolve()), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PDF_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.split("\f")


def _physical_to_marker_map(pdf_path: Path) -> dict[int, int] | None:
    """{physischer 0-basierter Seitenindex: `[S. N]`-Markernummer}, passend zu den
    Markern aus `pdf_to_pages`/`pages_to_marked_text`. ZWEI Zweige (R3/V4-1):

    - Labels-nutzbar → Marker = numerisches Druckseiten-Label je physischer Seite
      (Issue-#95-Klasse „PDF-Seite 179 → Druckseite 159" ist real).
    - Labels-None → Marker = komprimierte Zählung nicht-leerer Seiten (Drift real
      3 [Klingenberg] bis 39 Seiten [Kuhlen]).

    `physical_pages_by_anchor` bleibt VERBOTEN (Identität ohne Labels, #342-Erbe)."""
    raw = _pdftotext_raw_pages(pdf_path)
    if raw is None:
        return None
    labels = _pdf_page_labels(pdf_path)
    mapping: dict[int, int] = {}
    if labels is not None:
        numbered = _resolve_page_numbers(raw, labels)
        for i, (num, page_text) in enumerate(numbered):
            if page_text.strip():
                mapping[i] = num
    else:
        n = 0
        for i, page_text in enumerate(raw):
            if page_text.strip():
                n += 1
                mapping[i] = n
    return mapping or None


def _nearest_marker(pmap: dict[int, int], phys0: int, max_ahead: int = 3) -> int | None:
    """Marker der Zielseite; ist sie leer/ungemappt, die nächste nicht-leere Seite."""
    for d in range(max_ahead + 1):
        if phys0 + d in pmap:
            return pmap[phys0 + d]
    return None


def _title_overlap(title: str, window_text: str) -> float:
    """Anteil der (inhaltlichen) Titel-Wörter, die im Fenster vorkommen (0..1)."""
    words = [w for w in _norm_text(title).split() if len(w) > 2]
    if not words:
        words = _norm_text(title).split()
    if not words:
        return 0.0
    haystack = set(_norm_text(window_text).split())
    hit = sum(1 for w in words if w in haystack)
    return hit / len(words)


def _validate_boundary(title: str, window_text: str) -> bool:
    """Trägt das Fenster ab gemapptem Marker den Kapitel-Opener? Primär Wort-Overlap
    (≥50 %); bei Fehlschlag HiPS-Kaskade (normalisiert Substring → Fuzzy 80 %)."""
    if _title_overlap(title, window_text) >= _OUTLINE_TITLE_OVERLAP_MIN:
        return True
    nt = _norm_text(title)
    nw = _norm_text(window_text)
    if nt and nt in nw:
        return True
    try:
        from rapidfuzz import fuzz

        if nt and fuzz.partial_ratio(nt, nw) >= _OUTLINE_FUZZY_THRESHOLD:
            return True
    except Exception:
        pass
    return False


def _find_marker_pos(text: str, marker: int) -> int | None:
    """Position der `[S. marker]`-Marker-Zeile im Text (Marker sind eindeutig)."""
    m = re.search(rf"(?m)^[ \t]*\[S\.\s*{marker}\][ \t]*$", text)
    return m.start() if m else None


def _outline_chapters(text: str, pdf_path: Path) -> list[Chunk] | None:
    """Outline-first Kapitel-Chunks oder None (→ heuristischer Normalpfad).

    Pipeline: Outline lesen → Hauptebene (+ Root-Descend) → Front-/Backmatter-Filter
    → Duplikat-Merge → Sanity (≥2, ≤MAX) → physisch→Marker-Map → je Grenze Titel-
    Overlap-Validierung (Fehlschlag: Grenze verwerfen/mergen; >50 % Fehlschläge:
    Outline verwerfen) → Degenerations-Guards (Median, Riesensegment).
    """
    try:
        entries = _outline_raw_entries(pdf_path)
    except Exception:
        return None
    if not entries:
        return None
    selected = _select_main_level_entries(entries)
    if len(selected) < 2:
        return None
    merged = _merge_duplicate_boundaries(selected)
    if not 2 <= len(merged) <= MAX_SANE_OUTLINE_CHAPTERS:
        return None

    pmap = _physical_to_marker_map(pdf_path)
    if not pmap:
        return None

    # Grenze = (Titel, Markernummer). Zielseite leer → nächste nicht-leere Seite.
    boundaries: list[tuple[str, int]] = []
    for _lvl, title, phys0 in merged:
        marker = _nearest_marker(pmap, phys0)
        if marker is not None:
            boundaries.append((title, marker))
    if len(boundaries) < 2:
        return None

    # Je Grenze am gemappten Marker validieren; Fehlschlag → Grenze verwerfen
    # (Segment mergt implizit in den Vorgänger, da kein Split dort entsteht).
    validated: list[tuple[str, int]] = []
    n_total = len(boundaries)
    for title, marker in boundaries:
        pos = _find_marker_pos(text, marker)
        if pos is None:
            continue
        window = text[pos + 1 : pos + 1 + _OUTLINE_VALIDATION_WINDOW_CHARS + 12]
        if _validate_boundary(title, window):
            validated.append((title, marker))
    n_dropped = n_total - len(validated)
    if len(validated) < 2 or n_dropped > n_total // 2:
        return None

    # Nach Marker-Position sortieren + Duplikat-Marker (längerer Titel gewinnt).
    positions: list[tuple[int, str, int]] = []
    seen_markers: set[int] = set()
    for title, marker in validated:
        if marker in seen_markers:
            continue
        seen_markers.add(marker)
        pos = _find_marker_pos(text, marker)
        if pos is not None:
            positions.append((pos, title, marker))
    positions.sort()
    if len(positions) < 2:
        return None

    chunks: list[Chunk] = []
    for i, (start, title, _marker) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk_text = text[start:end].strip()
        prefix_pages = [int(mm.group(1)) for mm in _PAGE_MARKER_RE.finditer(text[:start])]
        chunk_pages = [int(mm.group(1)) for mm in _PAGE_MARKER_RE.finditer(chunk_text)]
        all_pages = ([prefix_pages[-1]] if prefix_pages else []) + chunk_pages
        chunks.append(
            Chunk(
                title=title,
                text=chunk_text,
                index=i,
                page_start=min(all_pages) if all_pages else None,
                page_end=max(all_pages) if all_pages else None,
                source="outline",
            )
        )

    # Degenerations-Guards (Sicherheitsnetz gegen fehlerhafte Outlines): ein
    # Segment >70 % der Wörter ODER Median-Segment < MIN_CHAPTER_SEGMENT_WORDS.
    seg_words = [len(c.text.split()) for c in chunks]
    total_words = sum(seg_words) or 1
    if len(chunks) >= 3 and max(seg_words) > _OUTLINE_GIANT_SEGMENT_RATIO * total_words:
        return None
    if _median(seg_words) < MIN_CHAPTER_SEGMENT_WORDS:
        return None

    # Diagnose + Offset-Voting-Kreuzcheck (Confidence-Flag, verändert den Split nicht).
    agree, voted = _offset_vote(text, merged, pmap)
    conf = "" if voted == 0 or agree * 2 >= voted else f" [map-confidence niedrig: {agree}/{voted}]"
    print(
        f"      [chapter-split] {len(chunks)} Kapitel erkannt (Quelle: outline, "
        f"Validierung {len(validated)}/{n_total}, {n_dropped} Grenzen verworfen){conf}",
        file=sys.stderr,
    )
    return chunks


def _offset_vote(text: str, selected: list[tuple[int, str, int]], pmap: dict[int, int]) -> tuple[int, int]:
    """Kreuzvalidierung der Seiten-Map (Confidence-Flag, verändert den Split nicht).

    Für eine Stichprobe unabhängig ALLE Marker sammeln, deren Fenster den Titel trägt
    (Kolumnentitel/Running-Heads erzeugen ein Plateau mehrerer Marker), und prüfen, ob
    der Map-Marker DARIN liegt — Tie-Break Richtung Map/Shift 0 (Plan §2). Liegt er
    außerhalb, ist die Map systematisch verschoben → niedrige Confidence. (agree, total).
    """
    if len(selected) > _OUTLINE_OFFSET_VOTING_SAMPLE:
        step = len(selected) / _OUTLINE_OFFSET_VOTING_SAMPLE
        sample = [selected[int(i * step)] for i in range(_OUTLINE_OFFSET_VOTING_SAMPLE)]
    else:
        sample = selected
    # Marker-Fenster einmalig als normalisierte Wort-Mengen vorberechnen.
    marker_windows: list[tuple[int, set[str]]] = []
    for m in _MARKER_LINE_RE.finditer(text):
        window = text[m.end() : m.end() + _OUTLINE_VALIDATION_WINDOW_CHARS]
        marker_windows.append((int(m.group(1)), set(_norm_text(window).split())))
    agree = total = 0
    for _lvl, title, phys0 in sample:
        map_marker = _nearest_marker(pmap, phys0)
        if map_marker is None:
            continue
        words = [w for w in _norm_text(title).split() if len(w) > 2] or _norm_text(title).split()
        if not words:
            continue
        thresh = _OUTLINE_TITLE_OVERLAP_MIN * len(words)
        matches = {mk for mk, ws in marker_windows if sum(1 for w in words if w in ws) >= thresh}
        if not matches:
            continue
        total += 1
        if map_marker in matches:
            agree += 1
    return agree, total


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def split_by_chapters(text: str, pdf_path: Path | None = None) -> list[Chunk]:
    """Teilt Text (mit `[S. N]`-Markern) an Kapitelgrenzen.

    Mit `pdf_path` wird zuerst der Outline-first-Pfad (#345) versucht — Kapitelgrenze
    = validierte PDF-Lesezeichen-Zielseite. Ohne nutzbare Outline (oder ohne
    `pdf_path`) exakt das bisherige Verhalten: Heading-Heuristik, sonst Word-Count.
    """
    if pdf_path is not None:
        outline = _outline_chapters(text, pdf_path)
        if outline is not None:
            return outline

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
        chunks.append(
            Chunk(
                title=title,
                text=chunk_text,
                index=i,
                page_start=page_start,
                page_end=page_end,
                source="heuristic",
            )
        )
    return chunks


def _split_by_words(text: str) -> list[Chunk]:
    """Word-basiertes Splitten — Seitenmarker bleiben im Text erhalten, page_start/end
    werden aus den Markern im jeweiligen Block abgeleitet."""
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, len(words), CHUNK_WORDS):
        block = " ".join(words[i : i + CHUNK_WORDS])
        page_start, page_end = page_range_of_text(block)
        chunks.append(
            Chunk(
                title=f"Abschnitt {i // CHUNK_WORDS + 1}",
                text=block,
                index=i // CHUNK_WORDS,
                page_start=page_start,
                page_end=page_end,
                source="words",
            )
        )
    return chunks


def extract_overview(text: str, max_words: int = 1500, chapters: list[Chunk] | None = None) -> str:
    """Repräsentativer Planner-Input über ALLE Kapitel, strikt innerhalb max_words.

    Alt: erste N + letzte K Wörter → mittlere Kapitel systematisch blind.
    Problem: operative Konzepte (Evals, Memory, Safety) in späteren Kapiteln
    wurden vom Planner nie gesehen → systematische Unterrepräsentation.

    Neu: Intro (min(600, max_words//3)) + Kapitel-Snippets (Budget-basiert,
    ohne Kapitel-1-Überlappung) + Fazit (min(300, max_words//5)).
    Alle Teile zusammen ≤ max_words. Fallback ohne Kapitel: Stichproben.

    `chapters` (#345, M1): bereits berechnete Chunks injizieren — vermeidet einen
    zweiten Split pro Lauf und macht die Overview outline-basiert. Ohne `chapters`
    (externe Aufrufer ohne pdf_path) exakt unverändert (heuristischer Split).
    """
    words = text.split()
    n = len(words)

    intro_budget = min(600, max_words // 3)
    outro_budget = min(300, max_words // 5) if n > intro_budget + 300 else 0
    snippet_budget = max(0, max_words - intro_budget - outro_budget)

    parts = [" ".join(words[:intro_budget])]

    chapters = chapters if chapters is not None else split_by_chapters(text)
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
            samples.append(" ".join(words[start : start + take]))
            budget_used += take
        if samples:
            parts.append("[Stichproben:]\n" + "\n\n[...]\n\n".join(samples))

    if outro_budget > 0:
        parts.append("[Ende:]\n" + " ".join(words[-outro_budget:]))

    return "\n\n".join(parts)
