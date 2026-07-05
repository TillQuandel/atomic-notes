"""Format-Auswahl-Verdrahtung fürs Output-Projekt (F4): verbindet F1
(`note_json.py`, kanonisches JSON-Schema), F2 (`portable_md.py`, portabler
Markdown-Renderer) und F3 (`export_convert.py`, pandoc+typst-Konvertierung) zu
einer einzigen Funktion, die aus einem Pipeline-Lauf (Drafts + Citation) eine
frei wählbare Menge an Export-Dateien erzeugt. Konsumenten: CLI (`--export-
format`, `orchestrator.py`) und GUI (Lauf-Einstellungen, `generative/gui/`).

Format-Kern-Set (`EXPORT_FORMAT_CHOICES`): `json` (F1-Contract direkt),
`obsidian-md` (Kopie der bereits vom Vault-Writer geschriebenen `.md`-Dateien —
kein neuer Render-Pfad), `portable-md` (F2), `docx`/`pdf`/`html` (F3, immer
verfügbar wenn `pandoc+typst` installiert sind). `odt`/`epub` sind zusätzlich
über F3 verfügbar, aber nicht Teil des CLI-Hilfetexts als "Kern" (zuschaltbar).

`FUTURE_FORMATS` ist reine Dokumentation (Entscheid Till 2026-07-05): Formate,
die geplant, aber noch nicht angebunden sind. `parse_export_formats` erkennt
sie und weist sie mit einem eigenen Hinweis zurück (nicht einfach "unbekannt"),
damit ein Nutzer nicht rätselt, ob er sich vertippt hat.
"""

from __future__ import annotations

from pathlib import Path

from generative.pipeline import vault_writer
from generative.pipeline.note_json import dumps, note_to_json_dict, run_to_json_dict
from generative.pipeline.portable_md import render_portable_note, render_portable_run
from generative.schemas.atomic_note import AtomicNoteDraft
from generative.schemas.citation import CitationMeta

EXPORT_FORMAT_CHOICES = ("json", "obsidian-md", "portable-md", "docx", "pdf", "html", "odt", "epub")

# Geplante/auf Anfrage aktivierbare Formate — NUR Doku-Konstante, nicht wählbar
# (Entscheid Till 2026-07-05). `parse_export_formats` gibt bei einem Treffer
# hier einen expliziten "geplant, noch nicht aktiviert"-Hinweis statt eines
# generischen "unbekanntes Format"-Fehlers.
FUTURE_FORMATS = ("rtf", "latex", "typst", "mediawiki", "rst", "epub3")

# Formate, die über F3 (pandoc+typst) laufen — brauchen die optionale
# `[export]`-Dependency-Gruppe, alle anderen Formate (json/obsidian-md/
# portable-md) nicht.
_BINARY_FORMATS = ("docx", "pdf", "html", "odt", "epub")

# Datei-Endungen, die im Export-Ordner eines Laufs vorkommen können — genutzt
# von der GUI (`app.py`) für die Pfad-Whitelist des Session-Export-Ordners.
EXPORT_FILE_SUFFIXES = frozenset({".json", ".md", ".docx", ".pdf", ".html", ".odt", ".epub"})


def parse_export_formats(raw: str) -> tuple[str, ...]:
    """`"pdf, DOCX,pdf"` → `("pdf", "docx")`: kommasepariert, getrimmt,
    case-insensitiv, ordnungserhaltend dedupliziert. Leerer/`None`-Input →
    leeres Tupel (kein Export gewünscht).

    Unbekanntes oder als `FUTURE_FORMATS` erkanntes Format → `ValueError` mit
    der vollständigen gültigen Liste; bei einem `FUTURE_FORMATS`-Treffer
    zusätzlich ein "geplant, noch nicht aktiviert"-Hinweis (Unterscheidung zu
    einem echten Tippfehler).
    """
    if not raw or not raw.strip():
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        fmt = part.strip().lower()
        if not fmt:
            continue
        if fmt not in EXPORT_FORMAT_CHOICES:
            future_hint = ""
            if fmt in FUTURE_FORMATS:
                future_hint = (
                    f" {fmt!r} ist als geplantes Format vorgesehen, aber noch nicht aktiviert "
                    f"(geplante Formate: {', '.join(FUTURE_FORMATS)})."
                )
            raise ValueError(
                f"Unbekanntes Export-Format {fmt!r}; gültig: {', '.join(EXPORT_FORMAT_CHOICES)}.{future_hint}"
            )
        if fmt not in seen:
            seen.append(fmt)
    return tuple(seen)


def _unique_stems(titles: list[str]) -> list[str]:
    """Dateiname-Stem je Note aus `vault_writer.slugify(title)`; Kollisionen
    innerhalb des Laufs (zwei Notes mit identischem Titel) bekommen einen
    `-2`/`-3`/…-Suffix statt sich gegenseitig zu überschreiben."""
    used: dict[str, int] = {}
    stems: list[str] = []
    for title in titles:
        base = vault_writer.slugify(title)
        count = used.get(base, 0)
        used[base] = count + 1
        stems.append(base if count == 0 else f"{base}-{count + 1}")
    return stems


def _build_exported_titles(note_dicts: list[dict], stems: list[str]) -> dict[str, str]:
    """Map normalisierter Titel/Alias (lowercase+trimmt) → Datei-Stem der
    jeweiligen Note — Grundlage für `render_portable_note(..., link_mode="file")`,
    damit Wikilinks zwischen zwei Notes desselben Laufs auf die tatsächliche
    (kollisionsfreie) Export-Datei zeigen statt auf den rohen Titel."""
    exported_titles: dict[str, str] = {}
    for note_dict, stem in zip(note_dicts, stems):
        note = note_dict["note"]
        for key in [note["title"], *note.get("aliases", [])]:
            if key and key.strip():
                exported_titles[key.strip().lower()] = stem
    return exported_titles


def run_export(
    drafts: list[AtomicNoteDraft],
    citation: CitationMeta,
    formats,
    export_root: Path,
    *,
    written_files: list[Path] | None = None,
    dry_run: bool = False,
    generated_at: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Exportiert einen Pipeline-Lauf (`drafts`+`citation`) in `formats` unter
    `export_root` (wird inkl. Elternverzeichnissen angelegt).

    `formats`: bereits validierte Formate (s. `parse_export_formats`), in der
    gegebenen Reihenfolge verarbeitet — ein Fehler bei einem späteren Format
    (z.B. fehlende pandoc/typst-Deps bei `docx`) verhindert NICHT, dass zuvor
    verarbeitete Formate (z.B. `json`) bereits geschrieben wurden.

    `written_files`: die vom Vault-Writer tatsächlich geschriebenen `.md`-Pfade
    dieses Laufs (nur für `obsidian-md` relevant — reine Kopie, kein neuer
    Render-Pfad). `dry_run`/fehlende `written_files` → `obsidian-md` wird
    sichtbar übersprungen (Meldung in der Rückgabe), nicht still ignoriert.

    Rückgabe: (geschriebene Dateien, Hinweis-/Skip-Meldungen).
    """
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    messages: list[str] = []

    stem = Path(citation.source_file).stem
    note_stems = _unique_stems([draft.title for draft in drafts])
    collective_title = citation.title or citation.short_label

    # Lazy/einmalig berechnete Zwischenergebnisse — nur gebaut, wenn ein
    # verarbeitetes Format sie tatsächlich braucht (json ohne Export-Deps muss
    # z.B. funktionieren, ohne dass portable_md/export_convert je berührt werden).
    _cache: dict[str, object] = {}

    def _note_dicts() -> list[dict]:
        if "note_dicts" not in _cache:
            _cache["note_dicts"] = [note_to_json_dict(draft, citation, generated_at=generated_at) for draft in drafts]
        return _cache["note_dicts"]  # type: ignore[return-value]

    def _run_dict() -> dict:
        if "run_dict" not in _cache:
            _cache["run_dict"] = run_to_json_dict(drafts, citation, generated_at=generated_at)
        return _cache["run_dict"]  # type: ignore[return-value]

    def _portable_docs() -> tuple[list[str], str]:
        if "portable" not in _cache:
            exported_titles = _build_exported_titles(_note_dicts(), note_stems)
            per_note = [
                render_portable_note(note_dict, exported_titles=exported_titles, link_mode="file")
                for note_dict in _note_dicts()
            ]
            collective = render_portable_run(_run_dict())
            _cache["portable"] = (per_note, collective)
        return _cache["portable"]  # type: ignore[return-value]

    _export_availability: tuple[bool, str] | None = None

    for fmt in formats:
        if fmt == "json":
            for note_stem, note_dict in zip(note_stems, _note_dicts()):
                path = export_root / f"{note_stem}.json"
                path.write_text(dumps(note_dict), encoding="utf-8")
                written.append(path)
            collective_path = export_root / f"{stem}-gesamt.json"
            collective_path.write_text(dumps(_run_dict()), encoding="utf-8")
            written.append(collective_path)

        elif fmt == "portable-md":
            per_note, collective = _portable_docs()
            for note_stem, md in zip(note_stems, per_note):
                path = export_root / f"{note_stem}.md"
                path.write_text(md, encoding="utf-8")
                written.append(path)
            collective_path = export_root / f"{stem}-gesamt.md"
            collective_path.write_text(collective, encoding="utf-8")
            written.append(collective_path)

        elif fmt in _BINARY_FORMATS:
            # Kein Modul-Import von export_convert an dieser Datei-Spitze —
            # sonst bräuchte selbst `--export-format json` die pandoc/typst-
            # Deps installiert. Lazy hier, wo sie wirklich gebraucht werden.
            from generative.pipeline import export_convert

            if _export_availability is None:
                _export_availability = export_convert.export_available()
            available, detail = _export_availability
            if not available:
                raise RuntimeError(
                    f"Export-Format {fmt!r} benötigt pandoc/typst, die nicht verfügbar sind: {detail} — "
                    'installiere sie mit: pip install "atomic-notes[export]"'
                )

            per_note, collective = _portable_docs()
            for draft, note_stem, md in zip(drafts, note_stems, per_note):
                out_path = export_root / f"{note_stem}.{fmt}"
                export_convert.convert_portable_md(
                    md,
                    fmt,
                    out_path,
                    title=draft.title,
                    author=citation.author,
                    date=citation.display_year,
                    lang="de",
                )
                written.append(out_path)
            collective_out = export_root / f"{stem}-gesamt.{fmt}"
            export_convert.convert_portable_md(
                collective,
                fmt,
                collective_out,
                title=collective_title,
                author=citation.author,
                date=citation.display_year,
                lang="de",
            )
            written.append(collective_out)

        elif fmt == "obsidian-md":
            if dry_run or not written_files:
                messages.append(
                    "obsidian-md übersprungen: Dry-Run schreibt keine Notes in den Vault — nichts zu kopieren."
                )
            else:
                for src in written_files:
                    src = Path(src)
                    if not src.is_file():
                        continue
                    dest = export_root / src.name
                    dest.write_bytes(src.read_bytes())
                    written.append(dest)

    return written, messages
