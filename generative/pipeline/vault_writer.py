"""Schreibt genehmigte AtomicNoteDraft-Objekte als .md-Dateien in den Vault."""

from __future__ import annotations
import re
import difflib
import hashlib
from datetime import date
from pathlib import Path

import yaml

from generative.config import VAULT, INBOX, LITERATURE_DIR, CRITIC_AUTO_THRESHOLD
from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor
from generative.schemas.citation import CitationMeta
from shared.author_norm import drop_institutional_coauthors


# Schema-MoC Naming: `MoC-<Thema>.md` — Spaces erlaubt, nur FS-unsichere Zeichen ersetzen.
_FS_UNSAFE = re.compile(r'[\\/:*?"<>|]+')


def moc_filename(title: str) -> str:
    safe = _FS_UNSAFE.sub("-", title).strip().strip(".")
    return f"MoC-{safe}.md"


def slugify(title: str) -> str:
    """Note-Filename aus Titel. Vault-Konvention für Inhalts-Notes ist Titlecase mit
    Spaces (`Atomic Notes.md`, `Lewin 3-Phasen-Modell.md`), nicht lowercase-kebab.
    Konvertiert nur FS-unsichere Zeichen, behält Umlaute, collapsed multiple Spaces.
    """
    s = _FS_UNSAFE.sub("-", title)
    s = re.sub(r"\s+", " ", s).strip().strip(".")
    return s


def _yaml_list(items: list[str], indent: str = "  ") -> str:
    """Rendert eine Markdown-/YAML-Liste mit doppelt-quotierten Strings.
    Backslashes und Anführungszeichen werden escapt für YAML-Kompatibilität."""
    if not items:
        return f"{indent}[]"

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    return "\n".join(f'{indent}- "{esc(s)}"' for s in items)


_FILENAME_PATTERN_FULL = re.compile(r"^(?P<author>.+?)\s+-\s+(?P<year>\d{4})\s+-\s+(?P<title>.+?)$")
_FILENAME_PATTERN_NOYEAR = re.compile(r"^(?P<author>.+?)\s+-\s+(?P<title>.+?)$")
_TITLE_LOOKS_BAD = re.compile(r"^[\d\s\.\-]+$|^Microsoft Word")  # Zahlenmüll oder Word-Doc-Header


def parse_filename_fallback(source_file: str) -> dict[str, str]:
    """Filename-Parser für Zotero-Konvention `<Author> - <Year> - <Title>.pdf` (F4).
    Fallback wenn pdf_metadata keine brauchbaren Werte liefert.
    Akzeptiert auch `<Author> - <Title>` (ohne Year).
    Öffentlich, da schichtübergreifend konsumiert (agents/extractor, orchestrator, schemas/citation)."""
    stem = Path(source_file).stem
    m = _FILENAME_PATTERN_FULL.match(stem)
    if m:
        return {
            "Author": drop_institutional_coauthors(m.group("author").strip()),
            "Year": m.group("year"),
            "Title": m.group("title").strip(),
        }
    m = _FILENAME_PATTERN_NOYEAR.match(stem)
    if m:
        return {
            "Author": drop_institutional_coauthors(m.group("author").strip()),
            "Title": m.group("title").strip(),
        }
    return {}


def apply_filename_citation_metadata(pdf_meta: dict, fb: dict) -> None:
    """Befüllt zitierfähige `Author`/`Year` in ``pdf_meta`` aus dem Dateiname-
    Fallback ``fb`` — mutiert ``pdf_meta`` in place.

    Hintergrund: ``pdf_metadata`` liefert keinen (unzuverlässigen) Info-Dict-Autor
    bzw. kein CreationDate-Jahr mehr als Zitier-Quelle. Damit Extractor-Prompt,
    Planner und Quellen-Block einen korrekten Autor/Jahr sehen (statt Platzhalter
    „Autor"), wird der Dateiname-Autor hier vor der Extraktion gemergt.

    Präzedenz:
    - **Author**: fill-if-missing — ein bereits vorhandener (stärkerer) Autor aus
      CrossRef/DOI-Enrichment wird NICHT vom Dateiname überschrieben.
    - **Year**: Filename-Year ist autoritativ für die vorliegende Edition und
      überschreibt ein abweichendes meta-Year (dokumentierte v28/Hiatt-Regel:
      CrossRef gibt bei Mehrfachauflagen oft das Jahr der jüngsten Auflage).
    - Fehlt der Autor überall, bleibt er leer — ehrlich unresolved statt geraten.
    """
    if fb.get("Author") and not pdf_meta.get("Author"):
        pdf_meta["Author"] = fb["Author"]
    if fb.get("Year"):
        pdf_meta["Year"] = fb["Year"]


_PAGE_PREFIX_RE = re.compile(r"^\s*S\.\s*", re.IGNORECASE)


def _strip_page_prefix(value: str) -> str:
    """Entfernt einen führenden `S. `-Prefix aus einem Anker-Page-Wert (Issue #20)."""
    return _PAGE_PREFIX_RE.sub("", value)


def _sort_pages(pages) -> list[str]:
    """Numerisch + range-aware sortierter Dedupe-Kern, geteilt zwischen
    `collect_anchor_pages` (Quelle: `source_anchors`) und `pages_from_body`
    (Quelle: gerenderter Body, Issue #76).

    Numerisch statt lexikografisch sortiert und range-aware (Seiten tragen auch
    "159–160" → int() auf die erste Zahl, sonst mis-sortiert/crasht ein Range).
    (Qwen-Review HIGH, 2. Durchgang.)
    """
    return sorted(
        (p for p in pages if p),
        key=lambda p: (int(m.group()) if (m := re.match(r"\d+", p)) else 10**9, p),
    )


def collect_anchor_pages(note_anchors: list[TextAnchor]) -> list[str]:
    """Extrahiert, dedupliziert und sortiert Seiten-Belege aus Text-Ankern.

    F8: `page` (LLM-exact) ODER `fuzzy_page` (rapidfuzz-Fallback) — beide sind
    valide Seitenbelege. Issue #20: Anker-Werte tragen bereits den `S. `-Prefix
    (Verifier setzt `page_str = f"S. {n}"`) — hier gestrippt, da beide Aufrufer
    ihn selbst voranstellen.

    Issue #76: NICHT mehr von `build_quellen_block` oder
    `portable_md._render_quellen_section` genutzt — `source_anchors` ist der
    Verifier-Stand VOR Critic/Layout/Renumber und kann vom final gerenderten
    Body abdriften (Phantom-Seiten aus verwaisten Ankern, fehlende Seiten aus
    nachträglich ergänzten Fußnoten). Beide Konsumenten lesen jetzt
    `pages_from_body(<finaler Body>)`. Bleibt als eigenständiger, getesteter
    Helper erhalten (kein aktueller Produktions-Konsument mehr) — bei Bedarf
    für Tooling, das direkt auf `source_anchors` zugreifen will, ohne einen
    gerenderten Body zur Hand zu haben.
    """
    _seen_pages = {
        _strip_page_prefix((a.page or a.fuzzy_page).strip())
        for a in note_anchors
        if (a.page or a.fuzzy_page) and (a.page or a.fuzzy_page).strip().lower() not in ("none", "null", "")
    }
    return _sort_pages(_seen_pages)


def build_quellen_block(body: str, source_file: str, citation: CitationMeta | None) -> str:
    """Quellen-Block deterministisch aus CitationMeta + Seiten-Belegen im
    übergebenen (finalen) Body. Kein Halluzinations-Risiko, weil das Modell
    nichts mehr selbst schreibt.

    Issue #76: `body` ist der tatsächlich neben diesem Block gerenderte
    Body-Text — Seiten kommen aus `pages_from_body(body)`, nicht mehr aus
    `note.source_anchors` (das driftete nach Critic/Layout/Renumber vom Body
    ab: Phantom-Seiten aus verwaisten Ankern, fehlende Seiten aus nachträglich
    ergänzten Fußnoten). Aufrufer müssen daher exakt den Body-Stand
    durchreichen, der neben dem Block gerendert wird — nicht `note.body` roh.
    """
    citation = citation or CitationMeta(author=None, year=None, title=None, doi=None, source_file=source_file)
    # F4: Filename-Fallback wenn CitationMeta-Titel leer/unsinnig. CitationMeta
    # garantiert das nicht selbst — build_citation_meta übernimmt exakt die
    # bisherige CrossRef-Override-Logik und lässt einen unkorrigierten "bad"
    # Title (z.B. "Microsoft Word - ...") durch, wenn kein CrossRef-Treffer greift.
    fallback = parse_filename_fallback(source_file)
    raw_title = (citation.title or "").strip()
    if not raw_title or _TITLE_LOOKS_BAD.match(raw_title):
        title = fallback.get("Title", "") or raw_title or Path(source_file).stem
    else:
        title = raw_title

    # Seiten aus dem finalen Body — kein Fallback auf source_anchors: das würde
    # die Phantom-Seiten-Klasse aus Issue #76 wieder öffnen. Trägt der Body
    # keinen Seiten-Beleg, bleibt der Marker ehrlich leer (wie bisher bei
    # leeren Ankern).
    pages = pages_from_body(body)
    pages_str = ", ".join(pages) if pages else ""

    # Quellen-Block: Wikilink zeigt direkt auf die PDF im Vault (Junction
    # `98-system/attachments/literatur/`). Display-Alias `<Author> <Year>` für Lesbarkeit.
    # Kein separater `[PDF](file://...)`-Link mehr (redundant). Kein Year-Doublet
    # mehr (Jahr ist im PDF-Filename und im Alias enthalten).
    short = citation.short_label
    pdf_in_vault = (LITERATURE_DIR / source_file).exists()
    wikilink_unsafe = any(c in source_file for c in ("|", "#", "[", "]"))
    if pdf_in_vault and not wikilink_unsafe:
        link = f"[[{source_file}|{short}]]"
    else:
        link = short  # Klartext-Fallback wenn PDF fehlt oder Filename unsafe
    # Issue #95: Quelle ohne /PageLabels -> Seiten sind physische PDF-Position,
    # keine gedruckte Seite. Gleiche Kennzeichnung wie in convert_inline_to_footnotes.
    page_word = "PDF-S." if citation.physical_pages else "S."
    pages_marker = f", {page_word} {pages_str}" if pages_str else ""
    return f"## Quellen\n\n*Quelle: {link}: {title}{pages_marker}*\n"


# Codex-Finding 1 (2026-05-10): erweitert auf Komma-Listen `(S. N, M)` und
# `(S. N, S. M)`, parallel zu zentralem PAGE_ANCHOR_RE in anchor_patterns.py.
_PAGE_INLINE_RE = re.compile(r"\s*\(S\.\s*(\d+(?:\s*[\-–,]\s*(?:S\.\s*)?\d+)*)\)")
_FN_MARKER_RE = re.compile(r"\[\^(\d+)\](?!:)")
_FN_DEF_LINE_RE = re.compile(r"^\[\^(\d+)\]:\s*(.*)$")


def renumber_footnotes(text: str) -> str:
    """Strippt orphan Footnote-Defs (kein Marker im Body referenziert sie) und
    renumeriert die verbliebenen Marker+Defs sequenziell ab `[^1]`. Wird nach
    Body-Layout-Refactor (z.B. Strip eines redundanten Aufzählungs-Absatzes)
    aufgerufen, damit keine Lücken oder verwaisten Defs übrig bleiben.
    """
    used_in_order: list[str] = []
    seen: set[str] = set()
    for m in _FN_MARKER_RE.finditer(text):
        num = m.group(1)
        if num not in seen:
            used_in_order.append(num)
            seen.add(num)
    if not used_in_order:
        # Keine Marker → alle Defs strippen
        lines = [ln for ln in text.split("\n") if not _FN_DEF_LINE_RE.match(ln)]
        return "\n".join(lines)
    old_to_new = {old: str(i + 1) for i, old in enumerate(used_in_order)}
    text = _FN_MARKER_RE.sub(
        lambda m: f"[^{old_to_new[m.group(1)]}]" if m.group(1) in old_to_new else m.group(0),
        text,
    )
    new_lines: list[str] = []
    for line in text.split("\n"):
        m = _FN_DEF_LINE_RE.match(line)
        if m:
            old = m.group(1)
            if old in old_to_new:
                new_lines.append(f"[^{old_to_new[old]}]: {m.group(2)}")
            # else: orphan def → skip
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def convert_inline_to_footnotes(
    body: str, source_label: str, source_file: str | None = None, physical_pages: bool = False
) -> str:
    """Konvertiert `(S. N)`-Inline-Marker zu `[^i]`-Footnote-Markern. Footnote-Defs
    werden an den Body-Ende als nackter Block angehängt (Reading-Mode rendert sie eh
    am Ende der Note). Block-Quote-Callouts (`> ...`) werden NICHT umgeschrieben —
    deren `S. N`-Angaben gehören zum Quote-Header und bleiben.

    Wenn `source_file` übergeben ist und die PDF unter LITERATURE_DIR (Junction
    `98-system/attachments/literatur/`) auflösbar ist, wird der `S. N`-Teil als
    Obsidian-Wikilink mit `#page=N` gerendert — Klick öffnet Obsidian-internen
    PDF-Viewer auf der richtigen Seite. Bei Page-Range `13–14` zeigt das
    `#page=`-Fragment auf die erste Zahl, das Label behält die Range.

    `physical_pages` (Issue #95): True wenn die Quelle keine `/PageLabels` führt —
    das Label wird dann als `PDF-S. N` statt `S. N` gerendert (ehrliche Kennzeichnung:
    die Seitenzahl ist die physische PDF-Position, keine gedruckte Seite). Nur das
    LABEL ändert sich; das `#page=N`-Fragment bleibt die nackte Zahl (der Obsidian-
    PDF-Viewer braucht den physischen Index — der stimmt hier gerade).
    """
    counter = [0]
    defs: list[str] = []
    label_prefix = "PDF-S." if physical_pages else "S."
    # Filename mit Wikilink-Syntax-Zeichen würde den Wikilink semantisch
    # zerbrechen. Defensiv: Klartext-Fallback. Codex-Finding 1 (`|`, `#`),
    # Gemini-Finding G1 (einzelne `[`, `]`).
    wikilink_unsafe = bool(source_file) and any(c in source_file for c in ("|", "#", "[", "]"))
    pdf_in_vault = source_file is not None and not wikilink_unsafe and (LITERATURE_DIR / source_file).exists()

    def repl(m: re.Match) -> str:
        counter[0] += 1
        i = counter[0]
        # Label: Display-Form. Hyphen → Endash, Whitespace normalisieren,
        # Komma-Listen als ", " trennen.
        raw = m.group(1)
        page_label = re.sub(r"\s*,\s*(?:S\.\s*)?", ", ", raw)
        page_label = re.sub(r"\s*[\-–]\s*", "–", page_label).strip()
        if pdf_in_vault:
            first = re.match(r"\d+", page_label)
            page_anchor = first.group(0) if first else page_label
            page_md = f"[[{source_file}#page={page_anchor}|{label_prefix} {page_label}]]"
        else:
            page_md = f"{label_prefix} {page_label}"
        defs.append(f"[^{i}]: {source_label}, {page_md}.")
        return f"[^{i}]"

    out_lines: list[str] = []
    for line in body.splitlines():
        if line.lstrip().startswith(">"):
            out_lines.append(line)
        else:
            out_lines.append(_PAGE_INLINE_RE.sub(repl, line))
    out = "\n".join(out_lines)
    if defs:
        out = out.rstrip() + "\n\n" + "\n".join(defs)
    return out


# Footnote-Def-Seiten-Wert extrahieren: Wikilink-Form zuerst probieren
# (`[[Datei#page=N|S. <label>]]`, siehe convert_inline_to_footnotes), sonst
# Klartext-Form (`S. <label>.` am Zeilenende). `(?:PDF-)?`-Präfix optional
# (Issue #95): Quellen ohne /PageLabels rendern das Label als `PDF-S. <label>`
# statt `S. <label>` — die Def-Seiten-Extraktion muss beide Formen verstehen,
# sonst verliert der Quellen-Block bei gekennzeichneten Notes seine Seiten.
# Die Klartext-Form (`_DEF_PLAINTEXT_PAGE_RE`) braucht keine Anpassung: `search()`
# findet `S. <label>` auch als Teilstring von `PDF-S. <label>`.
_DEF_WIKILINK_PAGE_RE = re.compile(r"\[\[[^\[\]]*\|\s*(?:PDF-)?S\.\s*([^\]]+)\]\]")
_DEF_PLAINTEXT_PAGE_RE = re.compile(r"S\.\s*(\d[\d,\s\-–]*)\.?\s*$")


def _normalize_page_token(raw: str) -> str:
    """Ein Komma-Split-Teilstück einer Seiten-Angabe normalisieren: führendes
    `S. ` strippen (Komma-Listen im rohen Inline-Ankerformat tragen es pro
    Teilstück, z.B. `(S. 13, S. 15)`), Bindestrich zu Halbgeviertstrich
    (matcht die Normalisierung in `convert_inline_to_footnotes`, falls der Body
    noch nicht konvertiert ist)."""
    token = _strip_page_prefix(raw.strip())
    token = re.sub(r"\s*-\s*", "–", token)
    return token.strip()


def pages_from_body(body: str) -> list[str]:
    """Extrahiert, dedupliziert und sortiert Seiten-Belege aus dem tatsächlich
    übergebenen Body-Text (Issue #76) — Gegenstück zu `collect_anchor_pages`,
    das stattdessen `source_anchors` liest.

    `source_anchors` und der final gerenderte Body können auseinanderdriften:
    Critic/Layout/Renumber entfernen oder ergänzen Fußnoten, ohne
    `source_anchors` nachzuziehen. Zwei belegte Drift-Klassen — Phantom-Seiten
    aus einem Anker ohne Body-Beleg, fehlende Seiten aus einer nachträglich
    ergänzten Fußnote ohne zugehörigen Anker. Quellen-Block-Seiten müssen daher
    direkt aus dem Body kommen.

    Deckt beide Body-Zustände ab, je nachdem in welcher Renderer-Phase der
    Aufrufer steht:
    - bereits footnote-konvertiert: `[^n]: <label>, S. N.` ODER
      `[^n]: <label>, [[Datei#page=N|S. N]].` (beide Def-Formen aus
      `convert_inline_to_footnotes`)
    - noch nicht konvertiert: Inline-`(S. N)`-Anker (`_PAGE_INLINE_RE`)

    Blockquote-Zeilen (`> ...`) werden übersprungen — dortige `S. N`-Angaben
    gehören zum Callout-Header, nicht zu einem Anker (wie in
    `convert_inline_to_footnotes`). Komma-Listen (`S. 13, 15`) werden in
    Einzel-Seiten aufgesplittet, Ranges (`S. 159–160`) bleiben als ein Token
    erhalten — sonst würde eine Range beim numerischen Sortieren mit anderen
    Seiten aus derselben/anderen Fußnoten falsch interleaved.
    """
    tokens: set[str] = set()
    for line in body.splitlines():
        if line.lstrip().startswith(">"):
            continue
        def_match = _FN_DEF_LINE_RE.match(line)
        if def_match:
            content = def_match.group(2)
            m = _DEF_WIKILINK_PAGE_RE.search(content)
            raw_value = m.group(1) if m else None
            if raw_value is None:
                m = _DEF_PLAINTEXT_PAGE_RE.search(content)
                raw_value = m.group(1) if m else None
            if raw_value:
                tokens.update(_normalize_page_token(p) for p in raw_value.split(","))
            continue
        for m in _PAGE_INLINE_RE.finditer(line):
            tokens.update(_normalize_page_token(p) for p in m.group(1).split(","))
    return _sort_pages(tokens)


def _read_proposed_tags_from_inbox(path: Path) -> tuple[list[str], str | None]:
    """Liest proposed-tags + tag-review-status aus existing Inbox-Frontmatter.
    Bewahrt User-Review-State über Re-Runs (Codex-Finding 1): wenn neuer Pipeline-
    Lauf keine `proposed_tags` mehr generiert, soll ein vorheriger Review-Block
    nicht stillschweigend verschwinden.

    Returns ([], None) bei Datei nicht da, Parse-Fehler oder fehlenden Feldern.
    """
    if not path.exists():
        return [], None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], None
    if not text.startswith("---"):
        return [], None
    end = text.find("---", 3)
    if end == -1:
        return [], None
    try:
        import yaml

        fm = yaml.safe_load(text[3:end]) or {}
    except Exception:
        return [], None
    if not isinstance(fm, dict):
        return [], None
    raw = fm.get("proposed-tags") or []
    if not isinstance(raw, list):
        return [], None
    tags = [str(t).strip() for t in raw if isinstance(t, str) and str(t).strip()]
    status = fm.get("tag-review-status")
    return tags, str(status) if status else None


def _render_proposed_tags_block(note: AtomicNoteDraft) -> str:
    """Bootstrap-Block für Frontmatter (Schwäche 4b). Leerer String wenn keine
    Vorschläge — sonst `\\nproposed-tags:\\n  - …\\ntag-review-status: …`. Wird
    sowohl von render_note() als auch render_moc() genutzt — Codex-Finding 5."""
    if not note.proposed_tags:
        return ""
    proposed_yaml = "\n".join(f"  - {t}" for t in note.proposed_tags)
    block = f"\nproposed-tags:\n{proposed_yaml}"
    if note.tag_review_status:
        block += f"\ntag-review-status: {note.tag_review_status}"
    return block


_LEGACY_SECTIONS_RE = re.compile(
    r"\n+##\s+(Quellen?|Confidence-Notiz)\s*\n.*?(?=\n+##\s|\Z)", re.IGNORECASE | re.DOTALL
)


def strip_legacy_sections(body: str) -> str:
    """Entfernt legacy `## Quellen`/`## Confidence-Notiz`-Sektionen aus dem Body.

    Idempotent: vorhandene Quellen-/Confidence-Notiz-Sektionen entfernen, falls noch
    aus alten Pipeline-Versionen im Body vorhanden. Saubere Drafts (post Stabilisierungs-
    Refactor) haben weder noch — dieser Strip ist Defensiv-Code für Cache-Drafts.
    Von `render_note` und `render_moc` geteilt (vorher dupliziert).
    """
    return _LEGACY_SECTIONS_RE.sub("", body).rstrip()


def render_moc(note: AtomicNoteDraft, source_file: str, citation: CitationMeta | None = None) -> str:
    """Hub-Routing: Note als MoC-Note rendern (Schema-MoC).
    Frontmatter: type=moc, cssclasses=[moc], obsidianUIMode=preview. Kein H1, keine
    fixen H2-Sektionen. Body wird übernommen; Quellen-Block am Ende (optional per Schema)
    bleibt zur Traceability erhalten — MoC stammt aus PDF-Pipeline.
    """
    citation = citation or CitationMeta(author=None, year=None, title=None, doi=None, source_file=source_file)
    today = date.today().isoformat()
    title_esc = note.title.replace('"', '\\"')
    aliases_yaml = _yaml_list(note.aliases)
    tags_yaml = "\n".join(f"  - {t}" for t in note.tags) if note.tags else "  []"
    flags_yaml = _yaml_list(note.quality_flags)
    sub_yaml = _yaml_list([f"[[{t}]]" for t in note.hub_subconcepts])

    proposed_block = _render_proposed_tags_block(note)
    frontmatter = f"""---
title: "{title_esc}"
aliases:
{aliases_yaml}
type: moc
cssclasses: [moc]
obsidianUIMode: preview
source-file: "{source_file}"
claude-generated: true
quality-flags:
{flags_yaml}
created: {today}
tags:
{tags_yaml}{proposed_block}
sub-concepts:
{sub_yaml}
---"""

    body = note.body.strip()
    body = strip_legacy_sections(body)
    body = convert_inline_to_footnotes(body, citation.short_label, source_file, citation.physical_pages)

    # v29f: Hub-Body-Layout: H1 → Einleitung (1. Absatz nach H1) → ## Komponenten
    # (nummerierte Liste mit Beschreibung pro Sub-Konzept) → Rest-Absätze (Substanz +
    # Empirie, ohne redundanten Hub-Aufzählungs-Absatz). Beschreibungen werden vom
    # Cross-Draft-Hub aus den H1-Zeilen der Stage-Drafts gezogen.
    sections: list[str] = []
    if note.hub_subconcepts:
        body_split = body.split("\n", 1)
        if body_split and body_split[0].lstrip().startswith("#"):
            h1 = body_split[0]
            rest_after_h1 = body_split[1].lstrip("\n") if len(body_split) > 1 else ""
            paragraphs = rest_after_h1.split("\n\n")
            intro = paragraphs[0].strip() if paragraphs else ""
            remaining = paragraphs[1:] if len(paragraphs) > 1 else []
            # Filter: redundante Hub-Aufzählungs-Absätze raus (Absatz mit ≥3
            # Wikilinks zu hub_subconcepts und Aufzählungs-Charakter).
            sub_set = set(note.hub_subconcepts)
            wikilink_re = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
            cleaned_remaining: list[str] = []
            # Gemini-Finding G4 (2026-05-10): nur reine Aufzählungs-Absätze strippen,
            # nicht jeden Absatz mit ≥3 Sub-Wikilinks. Heuristik: Strip wenn ≥3
            # Sub-Hits UND Wortanzahl niedrig (≤25 Words pro Sub-Hit) — typisch
            # für Aufzählungssätze „X umfasst [[A]], [[B]] und [[C]]". Synthese-
            # Sätze („Das Modell verbindet [[A]], [[B]] und [[C]] zu einem hybriden
            # Ansatz, der ...") überleben.
            for p in remaining:
                hits = sum(1 for m in wikilink_re.finditer(p) if m.group(1).strip() in sub_set)
                if hits >= 3 and len(p.split()) <= hits * 25:
                    continue
                cleaned_remaining.append(p)
            # Komponenten-Liste mit Beschreibung
            list_lines = []
            for i, sc in enumerate(note.hub_subconcepts):
                desc = note.hub_subconcept_descriptions.get(sc, "").strip()
                if desc:
                    list_lines.append(f"{i + 1}. [[{sc}]] — {desc}")
                else:
                    list_lines.append(f"{i + 1}. [[{sc}]]")
            list_md = "## Komponenten\n" + "\n".join(list_lines)
            sections.append(h1)
            if intro:
                sections.append(intro)
            sections.append(list_md)
            if cleaned_remaining:
                sections.append("\n\n".join(cleaned_remaining).strip())
        else:
            # Fallback: kein H1 erkannt → Liste vor Body
            list_md = "## Komponenten\n" + "\n".join(
                f"{i + 1}. [[{sc}]]"
                + (f" — {note.hub_subconcept_descriptions[sc]}" if note.hub_subconcept_descriptions.get(sc) else "")
                for i, sc in enumerate(note.hub_subconcepts)
            )
            sections.append(list_md)
            sections.append(body)
    else:
        sections.append(body)
    # Footnote-Renumbering: nach Body-Layout-Refactor können verwaiste Defs
    # zurückbleiben (z.B. wenn redundanter Aufzählungs-Absatz gestrippt wurde).
    body_combined = renumber_footnotes("\n\n".join(sections))
    rendered = (
        frontmatter
        + "\n"
        + body_combined
        + "\n\n"
        + build_quellen_block(body_combined, source_file, citation).rstrip()
        + "\n"
    )
    return inject_content_hash(rendered)  # #47: auch Hubs hashen (Idempotenz bei Re-Run)


def render_note(note: AtomicNoteDraft, source_file: str, citation: CitationMeta | None = None) -> str:
    if note.action == "hub":
        return render_moc(note, source_file, citation)
    citation = citation or CitationMeta(author=None, year=None, title=None, doi=None, source_file=source_file)
    today = date.today().isoformat()
    related_yaml = _yaml_list(note.related)
    tags_yaml = "\n".join(f"  - {t}" for t in note.tags) if note.tags else "  []"
    flags_yaml = _yaml_list(note.quality_flags)
    aliases_yaml = _yaml_list(note.aliases)

    title_esc = note.title.replace('"', '\\"')

    # F3: confidence-rationale ins Frontmatter (statt Body-Anhang). Nur bei low/medium
    # mit vorhandenem Reasoning. YAML-Doppelquote-Escape für eingebettete Quotes.
    rationale_line = ""
    if note.synthesis_confidence in ("low", "medium") and note.confidence_reasoning:
        rat_esc = note.confidence_reasoning.replace("\\", "\\\\").replace('"', '\\"')
        rationale_line = f'\nconfidence-rationale: "{rat_esc}"'

    # v23: auto-vault-recommended-Marker für Inbox-Reviewer (Tag-basiertes Routing
    # via Auto-Note-Mover ersetzt Pipeline-Pfad-Routing).
    auto_vault_line = ""
    if note.auto_vault_recommended is not None:
        auto_vault_line = f"\nauto-vault-recommended: {'true' if note.auto_vault_recommended else 'false'}"

    # #45: schmales fail-closed-Flag, wenn die Quelle nicht aufgelöst werden konnte.
    # Nur gesetzt → gerendert (kein leeres Metadatum bei aufgelösten Quellen).
    source_status_line = ""
    if note.source_status:
        source_status_line = f"\nsource-status: {note.source_status}"

    # Bootstrap-Schwäche 4b: proposed-tags + tag-review-status nur wenn nicht leer.
    # KEIN Auto-Note-Mover-Routing — User entscheidet beim Inbox-Review ob Tag
    # in tag_registry.yml wandert. Helper auch in render_moc() genutzt (Codex Fix 5).
    proposed_block = _render_proposed_tags_block(note)

    frontmatter = f"""---
title: "{title_esc}"
aliases:
{aliases_yaml}
type: atomic
synthesis-confidence: {note.synthesis_confidence}{rationale_line}{auto_vault_line}{source_status_line}
source-file: "{source_file}"
claude-generated: true
quality-flags:
{flags_yaml}
created: {today}
tags:
{tags_yaml}{proposed_block}
related:
{related_yaml}
---"""

    body = note.body.strip()
    body = strip_legacy_sections(body)

    # v28: `(S. N)` → `[^i]`-Footnotes deterministisch im Renderer (Pipeline-Components
    # wie anchor_repair/verifier arbeiten weiter mit dem Inline-Format im Body-Draft).
    # v30: Page-Wikilink mit `#page=N` wenn PDF im Vault auflösbar.
    # #95: physical_pages kennzeichnet das Label als `PDF-S.` bei Quellen ohne /PageLabels.
    body = convert_inline_to_footnotes(body, citation.short_label, source_file, citation.physical_pages)

    sections: list[str] = [body]
    sections.append(build_quellen_block(body, source_file, citation).rstrip())

    rendered = frontmatter + "\n" + "\n\n".join(sections) + "\n"
    # #47: content-hash ins Frontmatter, damit ein Re-Run erkennt, ob die Datei
    # seither vom Nutzer editiert wurde (dann nicht still überschreiben).
    return inject_content_hash(rendered)


def auto_write_decision(note: AtomicNoteDraft) -> tuple[bool, str]:
    """Auto-Write nach Vault: Score ≥ CRITIC_AUTO_THRESHOLD ∧ Hard-Gates pass → Vault.

    confidence ist kein Routing-Gate mehr — confidence=low ist strukturell unvermeidlich
    für monoquellige, nicht-peer-reviewed Quellen (Adequacy + Methodische-Limits immer fail).
    synthesis_confidence bleibt Frontmatter-Metadatum für den User sichtbar.

    Returns: (auto, reason) — reason erklärt warum nicht-Vault.
    """
    # MoC-Hard-Gate-Lockerung (v14): Hub-Notes sind Pointer-Notes, die Atomic-
    # Hard-Gates (Glance/Future-Self/Quellen) sind dafür nicht passend designed.
    # Eine MoC kann legitim ohne präzisen Glance-Test oder mit weniger Ankern
    # auskommen, wenn der Sub-Konzept-Index (`hub_subconcepts`) substanziell ist.
    # Akzeptanz-Schwelle: Score ≥ 4 + Hard-Gates ignoriert + ≥2 Sub-Konzepte.
    is_strong_hub = note.action == "hub" and note.critic_score >= 4 and len(note.hub_subconcepts) >= 2

    # Edition unverifiziert (Auszug ohne DOI): Auflage/Jahr/Seiten sind nur
    # dateiname-geraten, nicht belegt → nie automatisch in den Vault, immer in die
    # Inbox zur manuellen Bestätigung (oder via --doi pinnen). Vor allen anderen
    # Pfaden, damit auch ein Score-5-Auszug geblockt wird.
    if note.source_status == "edition-unverified":
        return False, "edition unverifiziert (Auszug ohne DOI)"

    # Faithfulness-Gate-Veto (E6, #69): überstimmt auch die Hub-Ausnahme —
    # ein High-Risk-Claim ohne Quellendeckung ist unabhängig vom Note-Typ
    # ein Review-Fall, kein Auto-Vault-Kandidat.
    if note.faithfulness_fail:
        return False, "Faithfulness-Gate: High-Risk-Claim nicht durch Quelle gedeckt — Review nötig"

    if not note.hard_gates_pass and not is_strong_hub:
        return False, "hard-gate fail (Glance/Future-Self/Quellen)"
    # Pfad C: Hub-Note mit Score ≥ 4 und ≥2 Sub-Konzepten → Vault auch ohne HG-pass
    if is_strong_hub:
        return True, "ok"
    # Pfad A: Score ≥ Threshold + Hard-Gates pass → Vault (confidence=low OK)
    if note.critic_score < CRITIC_AUTO_THRESHOLD:
        return False, f"score {note.critic_score}<{CRITIC_AUTO_THRESHOLD}"
    return True, "ok"


def find_existing_in_vault(title: str, aliases: list[str], existing_concepts: dict[str, str]) -> Path | None:
    """Title-/Alias-Match gegen Vault-Index aus context_builder. Existing_concepts
    excludiert bereits 00-inbox/98-system/99-archive/08-dashboards (siehe SKIP_DIRS).
    Match-Reihenfolge: exakter Title, dann jeder Alias. Erster Treffer gewinnt.
    """
    from generative.agents.context_builder import is_dedup_eligible

    candidates = [title.strip().lower()]
    candidates.extend(a.strip().lower() for a in aliases if a)
    for c in candidates:
        rel_path = existing_concepts.get(c)
        if rel_path:
            target = VAULT / rel_path
            # Typ-bewusst: eine `type: literature`/`moc`/`merge-stub`-Note koexistiert per
            # Design mit Konzept-Notes und ist kein Merge-Ziel — überspringen, damit ein
            # weiterer (Alias-)Kandidat noch eine echte Konzept-Note treffen kann.
            if not is_dedup_eligible(target):
                continue
            return target
    return None


def find_existing_in_inbox(source_file: str, title: str, inbox_dir: Path | None = None) -> Path | None:
    """Idempotenz-Check: Inbox-Datei mit identischem source-file + title.
    Findet eigene Pipeline-Drafts aus früherem Run derselben PDF — überschreiben statt
    -2-Suffix anhängen.
    """
    search_dir = inbox_dir if inbox_dir is not None else INBOX
    if not search_dir.exists():
        return None
    title_norm = title.strip().lower()
    matches: list[tuple[Path, str]] = []
    for f in search_dir.glob("*.md"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.startswith("---"):
            continue
        end = text.find("---", 3)
        if end == -1:
            continue
        try:
            fm = yaml.safe_load(text[3:end]) or {}
        except Exception:
            continue
        if fm.get("source-file") == source_file and str(fm.get("title", "")).strip().lower() == title_norm:
            matches.append((f, text))
    if not matches:
        return None
    # #47: bei mehreren Treffern (editiertes Original + pristine Variante aus einem
    # früheren Schutz-Lauf) die pristine Datei wählen — deterministisch, sonst
    # erzeugt jeder Re-Run je nach glob-Reihenfolge eine neue Variante (Churn).
    for f, text in matches:
        if is_pristine_pipeline_note(text):
            return f
    return matches[0][0]


def _read_source_field(note_path: Path) -> str | None:
    """Liest das source-file-Feld aus dem Frontmatter einer bestehenden Note."""
    try:
        text = note_path.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("---"):
            return None
        end = text.find("\n---", 3)
        if end < 0:
            return None
        fm = yaml.safe_load(text[3:end]) or {}
        return str(fm.get("source-file") or fm.get("source_file") or "")
    except Exception:
        return None


def render_merge_stub(
    note: AtomicNoteDraft, source_file: str, existing_path: Path, citation: CitationMeta | None = None
) -> str:
    """v27 MVP — Diff-Stub für menschlichen Merge-Review.

    Voller Attribute-First-Merge (siehe [[Multi-Source-Note-Merge]]) ist v28. v27
    schreibt den neuen Pipeline-Body daneben in die Inbox mit explizitem Verweis auf
    die existierende Note, damit kein -N-Suffix-Duplikat im Vault entsteht und
    SSoT bleibt.
    """
    citation = citation or CitationMeta(author=None, year=None, title=None, doi=None, source_file=source_file)
    today = date.today().isoformat()
    title_esc = note.title.replace('"', '\\"')
    rel_existing = str(existing_path.relative_to(VAULT)).replace("\\", "/")
    existing_link = existing_path.stem
    flags_yaml = _yaml_list(["merge-pending"] + note.quality_flags)
    # Geschwister von Befund D (#45): fail-closed source_status auch auf dem Merge-Pfad
    # rendern — render_note tut es, render_merge_stub ließ es sonst still fallen, sodass
    # eine create-Note mit unauflösbarer Quelle + Vault-Titel-Treffer das Flag verlor.
    source_status_line = f"\nsource-status: {note.source_status}" if note.source_status else ""

    frontmatter = f"""---
title: "MERGE: {title_esc}"
type: merge-stub
merge-target: "[[{existing_link}]]"
merge-target-path: "{rel_existing}"
source-file: "{source_file}"{source_status_line}
claude-generated: true
quality-flags:
{flags_yaml}
created: {today}
tags:
  - merge-pending
---"""

    # Codex-Finding 2 (2026-05-10): Merge-Stub-Body durch dieselbe Footnote-
    # Konvertierung wie render_note routen, damit auch Merge-Stubs Wikilink-
    # Footnotes auf die PDF-Seite haben (v30-Vollständigkeit). In einer Variable
    # gehalten (statt inline), damit build_quellen_block (Issue #76) exakt
    # denselben Body-Stand sieht, der unten tatsächlich gerendert wird.
    converted_body = convert_inline_to_footnotes(
        note.body.strip(),
        citation.short_label,
        source_file,
        citation.physical_pages,
    )
    body_parts = [
        f"# Merge-Stub: {note.title}",
        "",
        f"Pipeline hat das Konzept **{note.title}** aus `{source_file}` extrahiert. "
        f"Eine bestehende Note existiert bereits: [[{existing_link}]] "
        f"([{rel_existing}]({rel_existing})).",
        "",
        "Manueller Merge-Review nötig. Voller Attribute-First-Synthesis-Merge "
        "ist Pipeline v28 (siehe [[Multi-Source-Note-Merge]]).",
        "",
        "## Neuer Pipeline-Body (zur Integration)",
        "",
        converted_body,
        "",
        build_quellen_block(converted_body, source_file, citation).rstrip(),
    ]
    rendered = frontmatter + "\n" + "\n".join(body_parts) + "\n"
    return inject_content_hash(rendered)  # #47: Merge-Stubs hashen → editierte Stubs schützen


def rewrite_merged_related_links(drafts: list[AtomicNoteDraft], existing_concepts: dict[str, str] | None) -> int:
    """Issue #21: Drafts, die beim Schreiben zu Merge-Stubs werden (Title-/Alias-
    Match im Vault), erscheinen unter dem Dateinamen der bestehenden Note
    (`[[<vault-stem>]]`), nicht unter ihrem Draft-Titel. Sibling-Drafts behalten
    aber `related: [[<draft-titel>]]` — das ergibt nach dem Lauf einen toten Link.

    Pre-Pass vor der Write-Schleife: baut eine Map {normalisierter Draft-Titel/Alias
    → Merge-Target-Stem} aus genau den Drafts, die `find_existing_in_vault` trifft,
    und schreibt alle passenden `related`-Einträge der Drafts auf das Target um.
    Gibt die Anzahl umgeschriebener Links zurück.
    """
    if not existing_concepts:
        return 0
    rename_map: dict[str, str] = {}
    for d in drafts:
        existing_vault = find_existing_in_vault(d.title, d.aliases, existing_concepts)
        if existing_vault is None:
            continue
        new_stem = existing_vault.stem
        for key in [d.title, *d.aliases]:
            if key and key.strip():
                rename_map[key.strip().lower()] = new_stem
    if not rename_map:
        return 0
    rewritten = 0
    for d in drafts:
        for i, link in enumerate(d.related):
            target = link.strip().strip("[]").split("|", 1)[0].strip()
            new_stem = rename_map.get(target.lower())
            if new_stem and new_stem != target:
                d.related[i] = f"[[{new_stem}]]"
                rewritten += 1
    return rewritten


_CONTENT_HASH_FIELD = "pipeline-content-hash"
_CONTENT_HASH_RE = re.compile(rf"^{_CONTENT_HASH_FIELD}:.*\n?", re.MULTILINE)


def _strip_content_hash_line(text: str) -> str:
    """Entfernt die pipeline-content-hash-Zeile NUR aus dem Frontmatter.

    Dokumentweit zu strippen würde eine identische Zeile im Body unsichtbar machen
    (Body-Edit würde nicht als Änderung erkannt). Symmetrisch zur Injektion, die
    ausschließlich im Frontmatter einfügt.
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return _CONTENT_HASH_RE.sub("", text[:end]) + text[end:]


def compute_content_hash(text: str) -> str:
    """Stabiler Hash über den Note-Inhalt OHNE die Hash-Zeile selbst.

    Idempotent w.r.t. der Hash-Zeile: hash(text) == hash(text mit injizierter
    Hash-Zeile), sodass Write-Zeit- und Check-Zeit-Hash übereinstimmen.
    """
    base = _strip_content_hash_line(text)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def inject_content_hash(rendered: str) -> str:
    """Fügt `pipeline-content-hash: <hash>` ins Frontmatter ein (#47).

    Der Hash wird über den gerenderten Inhalt (ohne Hash-Zeile) berechnet und als
    eigene Zeile direkt nach dem öffnenden `---` eingefügt — mit trailing Newline,
    damit `_strip_content_hash_line` exakt invers ist (Write-Hash == Check-Hash).
    """
    h = compute_content_hash(rendered)
    if not rendered.startswith("---\n"):
        return rendered
    return f"---\n{_CONTENT_HASH_FIELD}: {h}\n{rendered[4:]}"


def extract_content_hash(text: str) -> str | None:
    """Liest den gespeicherten pipeline-content-hash aus dem Frontmatter."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    m = re.search(rf"^{_CONTENT_HASH_FIELD}:\s*(\S+)\s*$", text[:end], re.MULTILINE)
    return m.group(1) if m else None


def is_pristine_pipeline_note(text: str) -> bool:
    """True wenn die Note seit dem Pipeline-Write unverändert ist (#47).

    Kein gespeicherter Hash (z. B. alte Note vor #47) → konservativ False:
    im Zweifel schützen statt überschreiben.
    """
    stored = extract_content_hash(text)
    if stored is None:
        return False
    return compute_content_hash(text) == stored


def _is_pristine_inbox_file(path: Path) -> bool:
    """is_pristine_pipeline_note auf eine Datei angewendet; unlesbar → schützen."""
    try:
        return is_pristine_pipeline_note(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False


def _free_variant(target_dir: Path, stem: str) -> Path:
    """Erster nicht-existierender `<stem>-<i>.md`-Pfad — garantiert kollisionsfrei.

    Verhindert (anders als der frühere `range(2,20)`-Loop) einen stillen
    Overwrite bei erschöpften Suffixen: Bei echter Erschöpfung wird hart
    abgebrochen statt eine fremde Datei zu überschreiben.
    """
    for i in range(2, 1000):
        candidate = target_dir / f"{stem}-{i}.md"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Keine freie Variante für '{stem}' in {target_dir} (1000 belegt)")


def markdown_overwrite_diff(old_text: str, new_text: str, filename: str = "", max_lines: int = 60) -> str:
    """Schlanker unified Markdown-Diff old→new (#46). Leer wenn keine Änderung.

    Begrenzt auf den Overwrite-Fall — kein Voll-Diff-UI. Lange Diffs werden auf
    max_lines gekappt (mit Kürzungs-Hinweis), damit der Preview schlank bleibt.
    """
    diff = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"{filename} (bestehend)" if filename else "bestehend",
            tofile=f"{filename} (neu)" if filename else "neu",
            n=2,
            lineterm="",
        )
    )
    if not diff:
        return ""
    truncated = False
    if len(diff) > max_lines:
        diff = diff[:max_lines]
        truncated = True
    body = "\n".join(diff)
    if truncated:
        body += "\n… (Diff gekürzt — voller Vergleich beim echten Lauf)"
    return body


def write_note(
    note: AtomicNoteDraft,
    source_file: str,
    dry_run: bool = False,
    citation: CitationMeta | None = None,
    existing_concepts: dict[str, str] | None = None,
    inbox_dir: Path | None = None,
) -> Path:
    """Schreibt Note immer nach 00-inbox/. Auto-Note-Mover-Plugin (Obsidian) routet
    basierend auf Tags zu Zielordner (siehe CLAUDE.md Auto-Note-Mover-Mapping).

    v27 MVP-Verhalten bei Konflikten:
    - Vault-Match (Title/Alias in `04-wissen/`, `01-studium/` etc.) → merge-stub
      statt voller Note. SSoT bleibt, voller Merge ist v28.
    - Inbox-Match aus früherem Lauf derselben PDF (gleicher source_file + title)
      → überschreiben (Idempotenz), kein -2-Suffix.
    - Inbox-Match anderer source_file (gleicher Slug, anderes PDF) → -N-Suffix
      Fallback wie bisher.

    `auto_write_decision` bleibt als Quality-Indicator: Resultat wird als
    Frontmatter-Marker `auto-vault-recommended: true|false` durchgereicht und
    Reason als Quality-Flag — User sieht beim Inbox-Review sofort, was Pipeline
    für Vault-tauglich hält.
    """
    auto, reason = auto_write_decision(note)
    note.auto_vault_recommended = auto
    if not auto:
        note.quality_flags.append(f"vault-empfehlung blockiert: {reason}")

    target_dir = inbox_dir if inbox_dir is not None else INBOX
    is_merge_stub = False
    existing_vault: Path | None = None
    if existing_concepts:
        existing_vault = find_existing_in_vault(note.title, note.aliases, existing_concepts)

    # #2b: cross_reference setzt bei einem echten Konzept-Dup mit ABWEICHENDEM Titel
    # action=extend + extend_path=<Vault-Stem> (z.B. Draft „Information Need" → Vault-Note
    # „Wilson Information Need"). find_existing_in_vault matcht das nicht (Titel/Alias-only),
    # das Signal verpuffte bisher → Vault-Dublette ([[Ungelesenes-Pipeline-Signal]]). extend_path
    # wird jetzt als Merge-Ziel honoriert — typ-sicher: is_dedup_eligible schließt literature/
    # moc/merge-stub aus (das #2a-Gate setzt extend_path ohnehin nur noch für Konzept-Notes).
    # resolve_sibling_dups regelt Intra-Run-Siblings vorher; diese Auflösung greift nur, wenn
    # dort kein Vault-Treffer als Alias hinterlegt wurde.
    if existing_vault is None and existing_concepts and note.action == "extend" and note.extend_path:
        from generative.agents.context_builder import resolve_vault_relpath, is_dedup_eligible

        _rel = resolve_vault_relpath(note.extend_path, existing_concepts)
        if _rel and is_dedup_eligible(VAULT / _rel):
            existing_vault = VAULT / _rel

    if existing_vault is not None:
        # Pre-Merge Source-Check (MVP): Prüfe ob bestehende Note dieselbe Quelle hat.
        # Wenn source-file abweicht → andere Primärquelle → stub markiert als cross-source.
        # Voller Pre-Merge-Validation-LLM-Call ist TODO (v28).
        existing_source = _read_source_field(existing_vault)
        cross_source = (
            existing_source is not None
            and Path(source_file).stem not in existing_source
            and existing_source not in source_file
        )
        is_merge_stub = True
        stub_prefix = "XSOURCE-MERGE" if cross_source else "MERGE"
        if cross_source:
            print(
                f"  [pre-merge] Quellen-Konflikt: neue Quelle '{Path(source_file).stem}' "
                f"vs. bestehende '{existing_source}' — Stub als XSOURCE markiert"
            )
        filename = f"{stub_prefix} - {slugify(note.title)}.md"
        target = target_dir / filename
        # Idempotenz auch für merge-stubs: gleicher source_file + title → überschreiben
        existing_stub = find_existing_in_inbox(source_file, f"MERGE: {note.title}", inbox_dir)
        if existing_stub is not None and _is_pristine_inbox_file(existing_stub):
            target = existing_stub  # unveränderter Stub → idempotent überschreiben
        elif existing_stub is not None:
            # #47: editierter Merge-Stub → nicht überschreiben, neue Version daneben
            target = _free_variant(target_dir, Path(filename).stem)
            print(
                f"  [overwrite-schutz] '{existing_stub.name}' wurde seit dem "
                f"letzten Lauf editiert — neue Version als '{target.name}' "
                f"geschrieben, deine Edits bleiben erhalten."
            )
        elif target.exists():
            target = _free_variant(target_dir, target.stem)
        content = render_merge_stub(note, source_file, existing_vault, citation=citation)
    else:
        # Idempotenz: eigener früherer Run derselben PDF → überschreibe
        existing_inbox = find_existing_in_inbox(source_file, note.title, inbox_dir)
        _base_name = moc_filename(note.title) if note.action == "hub" else slugify(note.title) + ".md"
        if existing_inbox is not None and _is_pristine_inbox_file(existing_inbox):
            # unveränderte Pipeline-Note → idempotent überschreiben (wie bisher)
            target = existing_inbox
        elif existing_inbox is not None:
            # #47: seit dem letzten Lauf editiert → NICHT überschreiben, neue
            # Version daneben; die User-Edits bleiben unangetastet.
            target = _free_variant(target_dir, Path(_base_name).stem)
            print(
                f"  [overwrite-schutz] '{existing_inbox.name}' wurde seit dem "
                f"letzten Lauf editiert — neue Version als '{target.name}' "
                f"geschrieben, deine Edits bleiben erhalten."
            )
            existing_inbox = None  # kein Overwrite → kein proposed-tags-Erhalt
        else:
            target = target_dir / _base_name
            if target.exists():
                target = _free_variant(target_dir, target.stem)
        # Codex-Finding 1: bei Re-Run mit existing Inbox-Datei UND ohne neue
        # Vorschläge bestehenden Review-Block bewahren (sonst verschwindet
        # User-State stillschweigend). Neue Vorschläge überschreiben — der
        # neue Run hat aktuelleres Wissen.
        if existing_inbox is not None and not note.proposed_tags:
            kept_tags, kept_status = _read_proposed_tags_from_inbox(existing_inbox)
            if kept_tags:
                note.proposed_tags = kept_tags
                note.tag_review_status = kept_status or "needs-review"
        content = render_note(note, source_file, citation=citation)

    if dry_run:
        if is_merge_stub:
            marker = f"[Merge-Stub -> {existing_vault.relative_to(VAULT)}]"
        else:
            marker = "[Vault-Empf.]" if auto else f"[Inbox-Review: {reason}]"

        def safe(s):
            return s.encode("ascii", "replace").decode("ascii")

        print(f"  [DRY-RUN] -> Inbox: {target.name}  {marker}")
        print(
            f"    Score: {note.critic_score}/5 | Hard-Gates: {'pass' if note.hard_gates_pass else 'fail'} | Confidence: {note.synthesis_confidence}"
        )
        if note.quality_flags:
            _flags_line = f"    Flags: {', '.join(note.quality_flags)}"
            try:
                print(_flags_line)
            except UnicodeEncodeError:
                print(safe(_flags_line))
        # #46: Overwrite-Fall — target.exists() ⟺ Idempotenz-Re-Run überschreibt
        # eine bestehende Datei. Schlanker Markdown-Diff zeigt, WAS sich ändert.
        if target.exists():
            try:
                _old = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                _old = ""
            _diff = markdown_overwrite_diff(_old, content, target.name)
            if _diff:
                print(f"    [Overwrite-Diff] {target.name}:")
                for _l in _diff.splitlines():
                    # Diff in voller UTF-8-Treue (Umlaute/⚠️) drucken; nur wenn die
                    # Konsole das Encoding nicht kann, auf ASCII-safe zurückfallen.
                    try:
                        print(f"      {_l}")
                    except UnicodeEncodeError:
                        print(f"      {safe(_l)}")
        eval_dir = Path(__file__).resolve().parents[1] / ".cache" / "eval" / "baseline" / Path(source_file).stem
        eval_dir.mkdir(parents=True, exist_ok=True)
        prefix = "merge" if is_merge_stub else ("vault" if auto else "inbox")
        (eval_dir / f"{prefix}__{target.name}").write_text(content, encoding="utf-8")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    def _display(p: Path) -> str:
        try:
            return str(p.relative_to(VAULT))
        except ValueError:
            return str(p)

    if is_merge_stub:
        print(f"  [Merge-Stub] {_display(target)}  -> {_display(existing_vault)}")
    else:
        print(f"  [Inbox] {_display(target)}  ({'vault-empfohlen' if auto else 'review'})")
    return target
