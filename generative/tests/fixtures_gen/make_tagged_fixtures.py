"""Erzeugt minimale PDF-Fixtures fuer figure_alt-Tests (reproduzierbar, committed).

- tagged_one_figure.pdf  : 1 Seite Text + 1 /Figure-Strukturelement mit /Alt + /Pg
- untagged_plain.pdf     : 1 Seite Text, KEIN StructTreeRoot (Gate-Negativfall)

Roh-PDF mit korrekten xref-Offsets, damit PyMuPDF die Objekte ueber xref_object
parsen kann (gleicher Pfad wie der reale Felsmann-2025-Korpus). Kein externer
Pfad, keine Abhaengigkeit ausser der Stdlib.
"""
from __future__ import annotations

from pathlib import Path


def _build_pdf(objects: list[str], root_obj: int) -> bytes:
    """Setzt ein PDF aus 1-basiert nummerierten Objekt-Bodies zusammen."""
    header = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
    out = bytearray(header)
    offsets = [0]  # Objekt 0 ist frei
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {n} /Root {root_obj} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def _alt_hex(text: str) -> str:
    """UTF-16BE mit FEFF-BOM als Hex — exakt die Form, die PyMuPDF fuer /Alt liefert."""
    data = b"\xfe\xff" + text.encode("utf-16-be")
    return "<" + data.hex().upper() + ">"


def make_tagged() -> bytes:
    content = b"BT /F1 12 Tf 50 700 Td (Konzept A auf Seite eins.) Tj ET"
    objects = [
        # 1 Catalog
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 6 0 R >>",
        # 2 Pages
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        # 3 Page
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> /StructParents 0 >>",
        # 4 Contents
        f"<< /Length {len(content)} >>\nstream\n{content.decode('latin-1')}\nendstream",
        # 5 Font
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        # 6 StructTreeRoot
        "<< /Type /StructTreeRoot /K [7 0 R] >>",
        # 7 Document StructElem
        "<< /Type /StructElem /S /Document /P 6 0 R /K [8 0 R] >>",
        # 8 Figure StructElem
        f"<< /Type /StructElem /S /Figure /P 7 0 R /Pg 3 0 R /K 0 "
        f"/A 9 0 R /Alt {_alt_hex('Ein Saeulendiagramm der Suchhaeufigkeit.')} >>",
        # 9 Layout-Attribut mit BBox
        "<< /O /Layout /Placement /Block /BBox [50 600 545 760] >>",
    ]
    return _build_pdf(objects, root_obj=1)


def make_untagged() -> bytes:
    content = b"BT /F1 12 Tf 50 700 Td (Nur Text, keine Struktur.) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content.decode('latin-1')}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    return _build_pdf(objects, root_obj=1)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "tagged_one_figure.pdf").write_bytes(make_tagged())
    (out_dir / "untagged_plain.pdf").write_bytes(make_untagged())
    print(f"Fixtures geschrieben nach {out_dir}")


if __name__ == "__main__":
    main()
