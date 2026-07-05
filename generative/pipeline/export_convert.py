"""Konvertierungs-Schicht: portables Markdown (F2, `portable_md.py`) → docx/pdf/
html/odt/epub via pandoc+typst (Output-Projekt F3).

pip-only: `pypandoc_binary` bündelt ein pandoc.exe im Wheel (kein System-Pandoc
nötig), `typst` bettet den Typst-Compiler als Python-API ein (kein typst.exe im
PATH). Für PDF ist der Weg deshalb zweistufig: pandoc erzeugt aus dem Markdown
Typst-Quelltext (`-t typst`), `typst.compile()` kompiliert diesen zu PDF-Bytes.
Der naheliegende `--pdf-engine=typst`-Weg scheidet aus, weil er ein Typst-
Binary im PATH erwartet — das liefert `typst-py` nicht.

Pandoc-Input-Format: `gfm` (empirisch bestimmt, nicht `markdown`/`commonmark`).
Beleg (siehe `generative/tests/test_export_convert.py`):
- `gfm` unterstützt Footnotes ohne Extra-Extension (anders als `commonmark`,
  das `+footnotes` bräuchte).
- `gfm`s Auto-Identifiers stimmen exakt mit `portable_md.gfm_anchor_slug`
  überein — auch im Digit-Leading-Fall („3 Regeln für 2026" → `3-regeln-für-
  2026`), wo `markdown`s Auto-Identifiers die führende Zahl abschneiden
  (`regeln-fuer-2026`). Damit ist `gfm_anchor_slug` unverändert korrekt; keine
  F2-Anpassung nötig.
- `---` wird in beiden Kandidaten zu `<hr>`/thematic break (kein Unterscheidungskriterium).

Diese Datei verdrahtet nichts in Pipeline/CLI/GUI — reiner, injizierbarer
Konverter. F4 übernimmt die Verdrahtung.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

EXPORT_FORMATS = ("docx", "pdf", "html", "odt", "epub")

_PANDOC_INPUT_FORMAT = "gfm"


def export_available() -> tuple[bool, str]:
    """Importierbarkeit von pypandoc+typst prüfen (für doctor.py und F4)."""
    try:
        import pypandoc

        pandoc_version = pypandoc.get_pandoc_version()
    except Exception as e:  # pragma: no cover - Umgebungsfehler, nicht Logikpfad
        return False, f"pypandoc/pandoc nicht verfügbar: {e}"

    try:
        import typst

        typst_version = getattr(typst, "__version__", "?")
    except Exception as e:  # pragma: no cover - Umgebungsfehler, nicht Logikpfad
        return False, f"typst nicht verfügbar: {e}"

    return True, f"pandoc {pandoc_version}, typst {typst_version}"


def _metadata_args(*, title: str | None, author: str | None, date: str | None, lang: str) -> list[str]:
    """`--metadata`-Flags für die gesetzten Felder; `lang` immer (Regel: kein
    YAML-Block im Markdown, Dokument-Properties ausschließlich über pandoc-
    Metadaten)."""
    args = []
    if title:
        args.append(f"--metadata=title:{title}")
    if author:
        args.append(f"--metadata=author:{author}")
    if date:
        args.append(f"--metadata=date:{date}")
    args.append(f"--metadata=lang:{lang}")
    return args


def _build_typst_source(
    md_text: str,
    *,
    title: str | None = None,
    author: str | None = None,
    date: str | None = None,
    lang: str = "de",
) -> str:
    """`md_text` → Typst-Quelltext (pandoc `-t typst`, standalone-Template).

    Empirisch geprüft: pandocs Standalone-Typst-Template übersetzt das
    `--metadata=lang:<lang>`-Metadatum bereits in einen
    `#set text(lang: lang, region: region, size: fontsize)`-Aufruf innerhalb
    der generierten `conf()`-Funktion — `hyphenate` ist dort aber NICHT
    enthalten (Default in Typst ist `hyphenate: auto`, das mit `lang: "de"`
    allein noch keine deutsche Silbentrennung erzwingt). Deshalb wird hier ein
    minimaler Präfix `#set text(hyphenate: true)` vorangestellt: `set`-Regeln
    in Typst wirken auf alles danach im selben Scope und überschreiben nur die
    explizit genannten Felder — der spätere `set text(lang: ..., ...)`-Aufruf
    aus dem Template lässt `hyphenate` unangetastet, sodass unser Präfix-Wert
    erhalten bleibt.
    """
    import pypandoc

    typst_src = pypandoc.convert_text(
        md_text,
        "typst",
        format=_PANDOC_INPUT_FORMAT,
        extra_args=["--standalone", *_metadata_args(title=title, author=author, date=date, lang=lang)],
    )
    return "#set text(hyphenate: true)\n\n" + typst_src


def _convert_pdf(md_text: str, out_path: Path, *, title, author, date, lang) -> Path:
    import typst

    typst_src = _build_typst_source(md_text, title=title, author=author, date=date, lang=lang)
    with tempfile.TemporaryDirectory() as tmp_dir:
        typ_path = Path(tmp_dir) / "doc.typ"
        typ_path.write_text(typst_src, encoding="utf-8")
        pdf_bytes = typst.compile(str(typ_path), output=None)
    out_path.write_bytes(pdf_bytes)
    return out_path


def convert_portable_md(
    md_text: str,
    fmt: str,
    out_path: Path,
    *,
    title: str | None = None,
    author: str | None = None,
    date: str | None = None,
    lang: str = "de",
) -> Path:
    """Portables Markdown (F2-Output) → `fmt`-Datei unter `out_path`.

    `fmt` eines aus `EXPORT_FORMATS`. Dokument-Properties (`title`/`author`/
    `date`, nur gesetzte; `lang` immer) laufen über pandoc `--metadata` — kein
    YAML-Block im Markdown. PDF läuft zweistufig über Typst (siehe
    `_build_typst_source`), alle anderen Formate direkt über
    `pypandoc.convert_text(..., outputfile=...)`.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unbekanntes Export-Format {fmt!r}; gültig: {', '.join(EXPORT_FORMATS)}")

    out_path = Path(out_path)
    # Zielverzeichnis anlegen — sonst quittiert pandoc ein fehlendes Verzeichnis
    # mit einem kryptischen Writer-Fehler statt einem klaren OSError (Mistral-MED, PR #135).
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pdf":
        return _convert_pdf(md_text, out_path, title=title, author=author, date=date, lang=lang)

    import pypandoc

    extra_args = _metadata_args(title=title, author=author, date=date, lang=lang)
    if fmt == "html":
        # standalone: sonst fehlt das <html>/<head>-Geruest (title, meta charset).
        extra_args = ["--standalone", *extra_args]

    pypandoc.convert_text(md_text, fmt, format=_PANDOC_INPUT_FORMAT, outputfile=str(out_path), extra_args=extra_args)
    return out_path
