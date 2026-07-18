"""Outline-first Kapitel-Split (#345).

Kapitelgrenze = validierte PDF-Lesezeichen-Zielseite statt Titel-Matching im
Volltext. Deckt die Akzeptanzkriterien v4 PR 1 (a)–(h) ab:

- (a) 5-Bücher-Fixture: echte PDFs (skip wenn nicht vorhanden — Tills Privatbestand,
      nicht in CI). Kuhlen als härtester Drift-Fall (39 Leerseiten), alle Grenzen
      tragen den Kapitel-Opener (100 % Validierung).
- (b) Klingenberg 8–10 Kapitel, Quelle outline.
- (c) book-mode ohne Outline → Normalpfad-Fallback (synthetisch, läuft überall).
- (d) TOC-Trail-Negativtests aus echten Klingenberg-Zeilen (gespacte Dot-Leader).
- (e) Overview-Budget ≤ ~1500 Wörter.
- (g) synthetisches PageLabels-Fixture (Labels-Zweig der Seiten-Map).
- (h) Moser-artiges Single-Root-Fixture (Root-Descend).

Die synthetischen Fixtures (c/e/g/h) + TOC-Trail (d) laufen ohne Bestandsdateien
und sind die dauerhafte Abdeckung; die echten Bücher (a/b) grounden die Empirie.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from generative import config
from generative.pipeline import pdf_chunker as pc
from generative.pipeline.pdf_chunker import (
    _CHAPTER_RE,
    _is_real_chapter_match,
    _title_overlap,
    _validate_boundary,
    extract_overview,
    split_by_chapters,
)

fitz = pytest.importorskip("fitz")


# --------------------------------------------------------------------------- #
# Fixture-Bau: PDF mit Outline (+ optional PageLabels) und extrahierbarem Text
# --------------------------------------------------------------------------- #

_FILLER_LINE = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod"


def _make_pdf(
    tmp_path: Path,
    name: str,
    n_pages: int,
    openers: dict[int, str],
    toc: list | None,
    labels: list | None = None,
    filler_lines: int = 44,
) -> Path:
    """Baut ein PDF mit `n_pages` textreichen Seiten (mehrzeilig → pdftotext-lesbar),
    optionaler Outline (`set_toc`) und optionalen `/PageLabels`. `openers[i]` setzt
    den Seitenkopf physischer Seite i (Kapitel-Opener)."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=500, height=760)
        head = openers.get(i, f"Fuellseite {i} mit Inhalt")
        body = head + "\n" + "\n".join([_FILLER_LINE] * filler_lines)
        page.insert_text((36, 30), body, fontsize=7)
    if toc:
        doc.set_toc(toc)
    if labels:
        doc.set_page_labels(labels)
    out = tmp_path / name
    doc.save(str(out))
    doc.close()
    return out


def _opening_marker(chunk) -> int | None:
    m = re.search(r"\[S\.\s*(\d+)\]", chunk.text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# (g) Labels-Zweig der physisch→Marker-Seiten-Map
# --------------------------------------------------------------------------- #


def test_labels_branch_maps_to_printed_label(tmp_path):
    """PageLabels-Fixture: Druckseite = physischer Index + 100. Die Kapitelgrenze
    muss auf den LABEL-Marker ([S. 103]) mappen, nicht auf den physischen Index —
    beweist den Labels-Zweig (Issue-#95-Klasse „PDF-Seite 179 → Druckseite 159")."""
    pdf = _make_pdf(
        tmp_path,
        "labels.pdf",
        n_pages=14,
        openers={3: "Kapitel Eins Grundlagen der Theorie", 8: "Kapitel Zwei Methoden und Verfahren"},
        toc=[[1, "Kapitel Eins Grundlagen der Theorie", 4], [1, "Kapitel Zwei Methoden und Verfahren", 9]],
        labels=[{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 100}],
    )
    text = pc.pdf_to_text(pdf)
    chunks = split_by_chapters(text, pdf_path=pdf)

    assert [c.source for c in chunks] == ["outline", "outline"]
    # Physische Seite 3 → Label 103 (nicht der leerseiten-gezählte Kleinindex ~4).
    assert _opening_marker(chunks[0]) == 103
    assert _opening_marker(chunks[1]) == 108
    # Jede Grenze trägt ihren Opener (Validierung; synthetisch sauber → voller Overlap).
    assert _title_overlap(chunks[0].title, chunks[0].text[:500]) >= 0.5


# --------------------------------------------------------------------------- #
# (h) Root-Descend: eine Wurzel, Kapitel eine Ebene tiefer (Moser-Muster)
# --------------------------------------------------------------------------- #


def test_root_descend_single_root(tmp_path):
    """Genau 1 L1-Wurzel, echte Kapitel auf L2 → Root-Descend liefert die L2-Kinder
    (Moser-Muster: 55 Einträge unter einer Wurzel)."""
    pdf = _make_pdf(
        tmp_path,
        "moser.pdf",
        n_pages=14,
        openers={
            3: "Erstes Kapitel Bibliothek und Wissen",
            7: "Zweites Kapitel Erschliessung der Bestaende",
            10: "Drittes Kapitel Digitale Angebote",
        },
        toc=[
            [1, "Gesammelte Werke", 1],
            [2, "Erstes Kapitel Bibliothek und Wissen", 4],
            [2, "Zweites Kapitel Erschliessung der Bestaende", 8],
            [2, "Drittes Kapitel Digitale Angebote", 11],
        ],
    )
    text = pc.pdf_to_text(pdf)
    chunks = split_by_chapters(text, pdf_path=pdf)

    assert len(chunks) == 3
    assert all(c.source == "outline" for c in chunks)
    titles = [c.title for c in chunks]
    assert any("Erstes Kapitel" in t for t in titles)
    assert any("Drittes Kapitel" in t for t in titles)


# --------------------------------------------------------------------------- #
# (c) Kein Outline → transparenter Fallback auf den Normalpfad
# --------------------------------------------------------------------------- #


def test_no_outline_falls_back_to_normal_path(tmp_path):
    """PDF ohne Lesezeichen: `split_by_chapters(text, pdf_path)` fällt auf den
    heuristischen Normalpfad zurück (source != outline) — ehrliche Grenze."""
    pdf = _make_pdf(
        tmp_path,
        "no_outline.pdf",
        n_pages=10,
        openers={},  # nur Füllseiten, keine Kapitel-Opener
        toc=None,  # keine Outline
    )
    text = pc.pdf_to_text(pdf)
    chunks = split_by_chapters(text, pdf_path=pdf)

    assert chunks[0].source != "outline"
    assert chunks[0].source in ("words", "heuristic")


def test_no_pdf_path_is_backward_compatible(tmp_path):
    """Ohne `pdf_path` exakt das bisherige Verhalten: Heading-Heuristik greift,
    Outline wird nie konsultiert (Rückwärtskompatibilität aller Alt-Aufrufer)."""
    text = (
        "[S. 1]\n1 Einleitung\n" + "Fliesstext " * 50 + "\n"
        "[S. 5]\n2 Methoden\n" + "Mehr Text " * 50 + "\n"
        "[S. 9]\n3 Ergebnisse\n" + "Noch mehr " * 50
    )
    chunks = split_by_chapters(text)  # kein pdf_path
    assert len(chunks) == 3
    assert all(c.source == "heuristic" for c in chunks)


# --------------------------------------------------------------------------- #
# (d) TOC-Trail-Negativtests aus echten Klingenberg-Zeilen (gespacte Dot-Leader)
# --------------------------------------------------------------------------- #


def _real_chapter_titles(text: str) -> list[str]:
    return [m.group(0).strip() for m in _CHAPTER_RE.finditer(text) if _is_real_chapter_match(m)]


@pytest.mark.parametrize(
    "toc_line",
    [
        "1 Einleitung . . . . . . . . . . . . . . . . . . . . . . 1",
        "2 Data, Governance und Co. . . . . . . . . . . . . . . 9",
        "5 Rollen und Gremien fuer Data Governance . . . . . . . 101",
        "6 Datenqualitaet . . . . . . . . . . . . . . . . . . . . 115",
        "9 Zusammenfassung und Ausblick . . . . . . . . . . . . 257",
    ],
)
def test_spaced_dot_leader_toc_line_not_a_chapter(toc_line):
    """Echte Klingenberg-Inhaltsverzeichnis-Zeilen mit gespacten Dot-Leadern
    (`. . . . 9`) sind KEINE Kapitel-Headings — die alte `_TOC_TRAIL_RE` (nur
    `\\.{2,}`/`\\s{3,}`) ließ sie durch (#345-Ursache), die Erweiterung fängt sie."""
    assert _real_chapter_titles(toc_line) == []


def test_roman_page_number_toc_line_filtered():
    """Frontmatter-TOC-Zeile mit römischer Seitenzahl am Ende."""
    assert _real_chapter_titles("1 Vorbemerkung . . . . . . . . . . xii") == []


def test_extension_leaves_real_headings_intact():
    """Regressions-Schutz: echte Headings OHNE Trailer matchen weiter (keine
    Über-Filterung durch die Erweiterung)."""
    assert len(_real_chapter_titles("1 Einleitung\n2 Methoden\n3 Ergebnisse")) == 3


# --------------------------------------------------------------------------- #
# (e) Overview-Budget
# --------------------------------------------------------------------------- #


def test_overview_budget_from_chapters(tmp_path):
    """`extract_overview(text, chapters=chunks)` bleibt im Wort-Budget (~1500) und
    weit unter dem Volltext (N2-Baseline lief auf ~21k)."""
    openers = {i: f"Kapitel {i} Thema {i} Ueberschrift" for i in (3, 6, 9, 12)}
    pdf = _make_pdf(
        tmp_path,
        "overview.pdf",
        n_pages=15,
        openers=openers,
        toc=[[1, f"Kapitel {i} Thema {i} Ueberschrift", i + 1] for i in (3, 6, 9, 12)],
    )
    text = pc.pdf_to_text(pdf)
    chunks = split_by_chapters(text, pdf_path=pdf)
    overview = extract_overview(text, chapters=chunks)

    n_overview = len(overview.split())
    assert n_overview <= 1700  # ~1500-Budget + Sektions-Dekoration
    assert n_overview < len(text.split())


# --------------------------------------------------------------------------- #
# (a)/(b) Echte 5 Bücher — skip wenn Bestandsdateien fehlen (nicht in CI)
# --------------------------------------------------------------------------- #

_BOOK_DIR = Path(os.environ.get("ATOMIC_AGENT_TEST_BOOK_DIR", str(config.LITERATURE_DIR)))

_BOOKS = {
    "Klingenberg": (
        "Klingenberg und Weber - 2025 - Data Governance der Leitfaden für die Praxis.pdf",
        (8, 10),
    ),
    "DAMA-DMBOK": (
        "DAMA International - 2024 - DAMA-DMBOK Data management body of knowledge.pdf",
        (14, 20),
    ),
    "Kuhlen": (
        "Kuhlen et al. - 2022 - Grundlagen der Informationswissenschaft.pdf",
        (6, 6),
    ),
    "Gantert": (
        "Gantert - 2016 - Bibliothekarisches Grundwissen.pdf",
        (4, 9),
    ),
    "Hobohm": (
        "Hobohm - 2024 - Age of Access Grundfragen der Informationsgesellschaft.pdf",
        (10, 14),
    ),
}


def _resolve_book(name: str) -> Path:
    filename, _band = _BOOKS[name]
    path = _BOOK_DIR / filename
    if not path.exists():
        pytest.skip(f"Bestands-PDF nicht vorhanden ({name}); setze ATOMIC_AGENT_TEST_BOOK_DIR")
    return path


@pytest.mark.parametrize("name", list(_BOOKS))
def test_real_book_outline_split(name):
    """(a) Jedes Buch: Quelle outline, Kapitelzahl im erwarteten Band, und JEDE
    Grenze trägt den Kapitel-Opener im gemappten Fenster (100 % Validierung —
    kein stiller Falsch-Split)."""
    path = _resolve_book(name)
    lo, hi = _BOOKS[name][1]
    text = pc.pdf_to_text(path)
    chunks = split_by_chapters(text, pdf_path=path)

    assert chunks[0].source == "outline", f"{name}: erwartete outline-Quelle"
    assert lo <= len(chunks) <= hi, f"{name}: {len(chunks)} Kapitel außerhalb [{lo},{hi}]"
    # 100 % der emittierten Grenzen passieren das Validierungs-Gate (Wort-Overlap
    # ODER HiPS-Kaskade Substring/Fuzzy — dieselbe Funktion wie im Split, fängt
    # Encoding-Mojibake „Datenqualit�t" und Teil-Divider-Abweichungen ab).
    for c in chunks:
        assert _validate_boundary(c.title, c.text[:600]), f"{name}: Opener fehlt bei {c.title!r}"


def test_klingenberg_chapter_count_and_source():
    """(b) Klingenberg: 8–10 Kapitel, Quelle outline; und der Heuristik-Pfad
    (ohne pdf_path) degeneriert nachweislich (viele Mikro-Chunks) — der Grund für #345."""
    path = _resolve_book("Klingenberg")
    text = pc.pdf_to_text(path)
    outline_chunks = split_by_chapters(text, pdf_path=path)
    heuristic_chunks = split_by_chapters(text)  # Alt-Pfad = degeneriert

    assert outline_chunks[0].source == "outline"
    assert 8 <= len(outline_chunks) <= 10
    # Der Alt-Pfad produziert ein Vielfaches (TOC-Einträge als Mikro-Kapitel).
    assert len(heuristic_chunks) > 3 * len(outline_chunks)


def test_kuhlen_hardest_drift_case():
    """(a) Kuhlen ist der härteste Drift-Fall (39 Leerseiten, kein PageLabels):
    6 Teile A–F, alle über die Zähl-Map korrekt getroffen."""
    path = _resolve_book("Kuhlen")
    text = pc.pdf_to_text(path)
    chunks = split_by_chapters(text, pdf_path=path)

    assert chunks[0].source == "outline"
    assert len(chunks) == 6
    joined = " ".join(c.title for c in chunks)
    assert "Information Retrieval" in joined


def test_gantert_keeps_literatur_content_chapter():
    """Regressions-Schutz: „Zweiter Teil. … Literatur, Bücher, Medien" ist ein
    echtes Kapitel — der Backmatter-Filter darf es nicht wegen des Wortes
    „Literatur" droppen (bare-Substring-Falle)."""
    path = _resolve_book("Gantert")
    text = pc.pdf_to_text(path)
    chunks = split_by_chapters(text, pdf_path=path)

    assert chunks[0].source == "outline"
    assert any("Zweiter Teil" in c.title for c in chunks)
