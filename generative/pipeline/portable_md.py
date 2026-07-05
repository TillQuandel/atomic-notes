"""Portabler Markdown-Renderer aus dem F1-Export-Contract (`note_json.py`).

Konsumiert `note_to_json_dict`/`run_to_json_dict`-Dicts und erzeugt CommonMark
+ Standard-Footnotes (pandoc-tauglich), ohne Obsidian-Spezifika: kein YAML-
Frontmatter, keine `[[Wikilinks]]`, keine Callouts. Zielgruppe: F3 (pandoc+
typst → docx/pdf) und F4 (CLI/GUI-Formatwahl) — dieses Modul selbst verdrahtet
nichts in die Pipeline, es ist ein reiner JSON→Markdown-Konsument.

Transformations-Reihenfolge pro Note (siehe Docstrings der Helper unten):
1. Footnote-Vorverarbeitung (Inline `(S. N)` → `[^i]`, defensiv idempotent)
2. Dokumentkopf (H1 + kursive Quellzeile statt YAML)
3. Callout → Standard-Blockquote
4. Wikilink-Auflösung (intern → Link, extern → Klartext)
5. Quellen-Absatz (deterministisch aus source_anchors)
6. optionaler Metadaten-Abschnitt

`render_portable_run` fügt zusätzlich alle Notes eines Laufs zu einem
Sammel-Dokument zusammen (Anker-Links zwischen Notes, Footnote-Offset pro Note).
"""

from __future__ import annotations

import re

from generative.pipeline.vault_writer import collect_anchor_pages, convert_inline_to_footnotes
from generative.schemas.atomic_note import TextAnchor

_HAS_FOOTNOTE_RE = re.compile(r"\[\^\d+\]")
_CALLOUT_RE = re.compile(r"^(>\s*)\[!(\w+)\]([-+]?)\s*(.*)$")
_WIKILINK_RE = re.compile(r"\[\[(?P<target>[^\]|#]+)(?:#(?P<fragment>[^\]|]*))?(?:\|(?P<alias>[^\]]+))?\]\]")
_FN_MARKER_RE = re.compile(r"\[\^(\d+)\](?!:)")
_FN_DEF_RE = re.compile(r"^\[\^(\d+)\]:\s*(.*)$", re.MULTILINE)
_SLUG_STRIP_RE = re.compile(r"[^\w\s-]")


def gfm_anchor_slug(heading: str) -> str:
    """GitHub/pandoc-`gfm_auto_identifiers`-kompatibler Anchor-Slug: lowercase,
    Leerzeichen → `-`, alle Zeichen außer Buchstaben (inkl. Umlauten), Ziffern,
    `-`/`_` entfernt. Muss mit pandocs Auto-Identifiers-Regeln übereinstimmen —
    wird in F3 gegen echtes pandoc (`--to gfm`) verifiziert, nicht hier.
    """
    s = heading.lower()
    s = _SLUG_STRIP_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "-")


def _split_h1(body: str, title: str) -> tuple[str, str]:
    """(h1_line, rest) — H1 aus Body übernommen wenn vorhanden (`# `-Prefix),
    sonst aus `title` erzeugt (Regel 2)."""
    stripped = body.lstrip("\n")
    if stripped.startswith("# "):
        first, _sep, rest = stripped.partition("\n")
        return first.rstrip(), rest.lstrip("\n")
    return f"# {title}", stripped


def _source_line(source: dict) -> str:
    """Kursive Quellzeile unter dem H1: `*Quelle: <author>, <display_year> —
    <title> (<file>)*`. Fehlende Felder werden weggelassen; fehlender Autor
    fällt auf `short_label` zurück (nie „None" im Output)."""
    citation = source.get("citation") or {}
    author = citation.get("author") or citation.get("short_label")
    display_year = citation.get("display_year")
    title = citation.get("title")
    file = source.get("file")

    header = ", ".join(p for p in (author, display_year) if p)
    if title:
        header += f" — {title}"
    if file:
        header += f" ({file})"
    return f"*Quelle: {header}*"


def _convert_callouts(text: str) -> str:
    """Obsidian-Callout-Kopfzeile (`> [!typ]- Header` / `> [!typ] Header`) →
    `> **Header**`; leerer Header wird ersatzlos gestrippt. Folgezeilen bleiben
    unveränderte Blockquote-Zeilen (Regel 3)."""
    out_lines: list[str] = []
    for line in text.split("\n"):
        m = _CALLOUT_RE.match(line)
        if m:
            prefix, _typ, _fold, header = m.groups()
            header = header.strip()
            if header:
                out_lines.append(f"{prefix}**{header}**")
            # leerer Header: Zeile ersatzlos strippen, Rest bleibt Blockquote
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _resolve_wikilinks(text: str, exported_titles: dict[str, str], link_mode: str) -> str:
    """`[[Ziel]]`/`[[Ziel|Anzeige]]`/`[[Ziel#Fragment(|Anzeige)]]` → aufgelöster
    Link (file: `[Anzeige](<Stem>.md)`, anchor: `[Anzeige](#<slug>)`) wenn Ziel
    (normalisiert, Fragment ignoriert) in `exported_titles`, sonst reiner
    Anzeige-Text ohne Klammern (Regel 4)."""

    def repl(m: re.Match) -> str:
        target = m.group("target").strip()
        alias = m.group("alias")
        display = alias.strip() if alias else target
        value = exported_titles.get(target.strip().lower())
        if value is None:
            return display
        if link_mode == "file":
            href = f"{value.replace(' ', '%20')}.md"
        else:
            href = f"#{gfm_anchor_slug(value)}"
        return f"[{display}]({href})"

    return _WIKILINK_RE.sub(repl, text)


def _render_quellen_section(note: dict, source: dict) -> str:
    """`## Quellen` + `*<short_label>: <title>, S. <pages>*` — Seiten aus
    `note.source_anchors` via `collect_anchor_pages` (Regel 8)."""
    citation = source.get("citation") or {}
    short_label = citation.get("short_label") or ""
    title = citation.get("title") or short_label
    anchors = [TextAnchor(**a) for a in note.get("source_anchors", [])]
    pages = collect_anchor_pages(anchors)
    pages_marker = f", S. {', '.join(pages)}" if pages else ""
    return f"## Quellen\n\n*{short_label}: {title}{pages_marker}*"


def _render_metadata_section(note: dict, routing: dict | None) -> str:
    """`## Metadaten` — lesbare `- **Feld:** Wert`-Liste der sonst verborgenen
    Betriebsdaten, nur nicht-leere Felder (Regel 7)."""
    lines = ["## Metadaten", ""]

    if note.get("tags"):
        lines.append(f"- **Tags:** {', '.join(note['tags'])}")
    if note.get("synthesis_confidence"):
        lines.append(f"- **Synthesis-Confidence:** {note['synthesis_confidence']}")
    if note.get("quality_flags"):
        lines.append("- **Quality-Flags:**")
        for flag in note["quality_flags"]:
            lines.append(f"  - {flag}")
    if note.get("aliases"):
        lines.append(f"- **Aliases:** {', '.join(note['aliases'])}")
    if note.get("related"):
        lines.append(f"- **Related:** {', '.join(note['related'])}")
    auto_vault_recommended = note.get("auto_vault_recommended")
    if auto_vault_recommended is not None:
        lines.append(f"- **Auto-Vault-Recommended:** {'ja' if auto_vault_recommended else 'nein'}")
    if note.get("source_status"):
        lines.append(f"- **Source-Status:** {note['source_status']}")
    lines.append(f"- **Critic-Score:** {note.get('critic_score', 0)}")
    lines.append(f"- **Hard-Gates-Pass:** {'ja' if note.get('hard_gates_pass') else 'nein'}")

    if routing:
        auto = routing.get("auto_vault_recommended")
        reason = routing.get("reason", "")
        lines.append(f"- **Vault-Empfehlung:** {'ja' if auto else 'nein'} ({reason})")

    return "\n".join(lines)


def _offset_footnotes(text: str, offset: int) -> tuple[str, int]:
    """Verschiebt alle `[^N]`-Marker/Defs in `text` um `offset` nach oben — für
    Sammel-Dokumente, in denen jede Note eigene Footnotes ab `[^1]` mitbringt
    (Regel 6). Gibt (verschobener Text, höchste neue Nummer) zurück; `offset`
    unverändert wenn `text` keine Footnotes enthält. `vault_writer.renumber_footnotes`
    wird bewusst NICHT genutzt/verändert — das arbeitet dokumentweit ab 1, hier
    wird pro Note nur um einen laufenden Offset verschoben.
    """
    # Marker UND Defs einsammeln: eine Orphan-Def (Def ohne Marker, möglich auf
    # dem „Body bereits konvertiert"-Pfad) muss mitverschoben werden — sonst
    # KeyError beim Def-Rewrite bzw. Nummern-Kollision mit einer Folge-Note.
    used = sorted(
        {int(m.group(1)) for m in _FN_MARKER_RE.finditer(text)} | {int(m.group(1)) for m in _FN_DEF_RE.finditer(text)}
    )
    if not used:
        return text, offset
    mapping = {str(n): str(n + offset) for n in used}
    text = _FN_MARKER_RE.sub(lambda m: f"[^{mapping[m.group(1)]}]", text)
    text = _FN_DEF_RE.sub(lambda m: f"[^{mapping[m.group(1)]}]: {m.group(2)}", text)
    return text, offset + max(used)


def render_portable_note(
    note_json: dict,
    *,
    exported_titles: dict[str, str] | None = None,
    link_mode: str = "file",
    include_metadata: bool = False,
) -> str:
    """Rendert ein einzelnes `note_to_json_dict`-Dict zu portablem Markdown.

    `exported_titles`: Map normalisiertes Ziel (Titel/Alias, lowercase+trimmt)
    → "Ziel" — interpretiert als Datei-Stem im file-mode bzw. als Anchor-Text
    (H1-Text, wird via `gfm_anchor_slug` verschlüsselt) im anchor-mode.
    None/leer = keine internen Ziele auflösbar (alle Wikilinks → Klartext).
    """
    note = note_json["note"]
    source = note_json["source"]
    routing = note_json.get("routing")
    titles_map = exported_titles or {}

    raw_body = note["body"]
    if _HAS_FOOTNOTE_RE.search(raw_body):
        body = raw_body
    else:
        short_label = (source.get("citation") or {}).get("short_label")
        body = convert_inline_to_footnotes(raw_body, short_label, source_file=None)

    h1_line, rest = _split_h1(body, note["title"])
    rest = _convert_callouts(rest)
    h1_line = _resolve_wikilinks(h1_line, titles_map, link_mode)
    rest = _resolve_wikilinks(rest, titles_map, link_mode)

    parts = [h1_line, _source_line(source), rest.strip(), _render_quellen_section(note, source)]
    if include_metadata:
        parts.append(_render_metadata_section(note, routing))

    return "\n\n".join(p for p in parts if p and p.strip()) + "\n"


def render_portable_run(run_json: dict, *, include_metadata: bool = False) -> str:
    """Sammel-Dokument: alle Notes eines Laufs (`run_to_json_dict`) in Reihenfolge,
    `link_mode="anchor"`, `exported_titles` automatisch aus Titel+Aliase aller
    enthaltenen Notes. Notes durch `\n\n---\n\n` getrennt; Footnotes pro Note
    per laufendem Offset renummeriert (Regel 6)."""
    source = run_json["source"]
    run_short_label = (source.get("citation") or {}).get("short_label")

    # 1) exported_titles vorab aus allen Notes bauen (H1-Text je Note — der
    # spätere Anchor-Slug hängt vom GESAMTEN H1 inkl. Untertitel ab, Regel 5).
    exported_titles: dict[str, str] = {}
    for entry in run_json["notes"]:
        note = entry["note"]
        raw_body = note["body"]
        if _HAS_FOOTNOTE_RE.search(raw_body):
            body = raw_body
        else:
            body = convert_inline_to_footnotes(raw_body, run_short_label, source_file=None)
        h1_line, _rest = _split_h1(body, note["title"])
        h1_text = h1_line[2:].strip()
        for key in [note["title"], *note.get("aliases", [])]:
            if key and key.strip():
                exported_titles[key.strip().lower()] = h1_text

    # 2) jede Note einzeln rendern (anchor-mode), dann Footnotes per Offset verschieben.
    rendered_notes: list[str] = []
    offset = 0
    for entry in run_json["notes"]:
        single_note_json = {
            "schema_version": run_json.get("schema_version"),
            "generated_at": run_json.get("generated_at"),
            "agent_version": run_json.get("agent_version"),
            "source": source,
            "note": entry["note"],
            "routing": entry.get("routing"),
        }
        rendered = render_portable_note(
            single_note_json,
            exported_titles=exported_titles,
            link_mode="anchor",
            include_metadata=include_metadata,
        )
        rendered, offset = _offset_footnotes(rendered, offset)
        rendered_notes.append(rendered.rstrip("\n"))

    return "\n\n---\n\n".join(rendered_notes) + "\n"
