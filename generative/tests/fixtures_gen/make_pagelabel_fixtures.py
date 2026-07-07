"""Erzeugt Mini-PDF-Fixtures mit echten `/PageLabels` (reproduzierbar, committed).

Deckt die #79-Wurzelfix-Kette ab (Druckseiten aus `/PageLabels` + das
`_usable_page_labels`-Gate gegen römisch↔arabisch-Namespace-Kollision), die bis
Issue #154 in KEINEM Test gegen eine real gelabelte PDF lief.

- pagelabels_arabic.pdf       : 4 Seiten, rein arabische Labels 159–162
                                (Buch-Auszug). `_pdf_page_labels` liefert die
                                echten Druckseiten → Happy-Path.
- pagelabels_roman_arabic.pdf : 6 Seiten, 2 röm. Frontmatter (i, ii) + 4 arab.
                                (159–162). Gemischt → `_usable_page_labels`
                                verwirft die Liste → i+1-Fallback (Gate-Fall).

Jede Seite trägt eindeutigen Text (`FRONT-i`, `CONTENT-159`, …), damit die
pdftotext→Druckseiten-Zuordnung end-to-end prüfbar ist.

Zwei-Stufen-Rezept (aus der #79-Historie: `pdfunite` verwirft Labels,
`pypdf.set_page_label` setzt sie): Roh-PDF mit extrahierbarem Text bauen, dann
via pypdf klonen und `/PageLabels` setzen. Kein externer Pfad, nur Stdlib + pypdf.
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def _build_text_pdf(pages_text: list[str]) -> bytes:
    """Roh-PDF mit je einer Textzeile pro Seite (pdftotext-extrahierbar).

    Korrekte xref-Offsets, damit pypdf die Seiten klonen kann. Objekt-Layout:
    1=Catalog, 2=Pages, dann je Seite (Page, Contents), zuletzt die geteilte Font.
    """
    header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
    out = bytearray(header)
    n = len(pages_text)
    font_num = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
    ]
    for i, txt in enumerate(pages_text):
        content = f"BT /F1 14 Tf 50 700 Td ({txt}) Tj ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {4 + 2 * i} 0 R /Resources << /Font << /F1 {font_num} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content.decode('latin-1')}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    offsets = [0]  # Objekt 0 ist frei
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    total = len(objects) + 1
    out += f"xref\n0 {total}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def _with_page_labels(raw: bytes, label_calls: list[dict]) -> bytes:
    """Klont das Roh-PDF via pypdf und setzt `/PageLabels` (set_page_label)."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(raw)))
    for call in label_calls:
        writer.set_page_label(**call)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_arabic() -> bytes:
    raw = _build_text_pdf(["CONTENT-159", "CONTENT-160", "CONTENT-161", "CONTENT-162"])
    return _with_page_labels(raw, [dict(page_index_from=0, page_index_to=3, style="/D", start=159)])


def make_roman_arabic() -> bytes:
    raw = _build_text_pdf(["FRONT-i", "FRONT-ii", "CONTENT-159", "CONTENT-160", "CONTENT-161", "CONTENT-162"])
    return _with_page_labels(
        raw,
        [
            dict(page_index_from=0, page_index_to=1, style="/r"),  # i, ii
            dict(page_index_from=2, page_index_to=5, style="/D", start=159),  # 159..162
        ],
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "pagelabels_arabic.pdf").write_bytes(make_arabic())
    (out_dir / "pagelabels_roman_arabic.pdf").write_bytes(make_roman_arabic())
    print(f"PageLabels-Fixtures geschrieben nach {out_dir}")


if __name__ == "__main__":
    main()
