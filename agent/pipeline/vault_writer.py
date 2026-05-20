"""Schreibt genehmigte AtomicNoteDraft-Objekte als .md-Dateien in den Vault."""
from __future__ import annotations
import re
from datetime import date
from pathlib import Path

import yaml

from config import VAULT, WISSEN, INBOX, LITERATURE_DIR, CRITIC_AUTO_THRESHOLD
from schemas.atomic_note import AtomicNoteDraft


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


_FILENAME_PATTERN_FULL = re.compile(
    r"^(?P<author>.+?)\s+-\s+(?P<year>\d{4})\s+-\s+(?P<title>.+?)$"
)
_FILENAME_PATTERN_NOYEAR = re.compile(
    r"^(?P<author>.+?)\s+-\s+(?P<title>.+?)$"
)
_TITLE_LOOKS_BAD = re.compile(r"^[\d\s\.\-]+$|^Microsoft Word")  # Zahlenmüll oder Word-Doc-Header


def _parse_filename_fallback(source_file: str) -> dict[str, str]:
    """Filename-Parser für Zotero-Konvention `<Author> - <Year> - <Title>.pdf` (F4).
    Fallback wenn pdf_metadata keine brauchbaren Werte liefert.
    Akzeptiert auch `<Author> - <Title>` (ohne Year)."""
    stem = Path(source_file).stem
    m = _FILENAME_PATTERN_FULL.match(stem)
    if m:
        return {
            "Author": m.group("author").strip(),
            "Year": m.group("year"),
            "Title": m.group("title").strip(),
        }
    m = _FILENAME_PATTERN_NOYEAR.match(stem)
    if m:
        return {
            "Author": m.group("author").strip(),
            "Title": m.group("title").strip(),
        }
    return {}


def build_quellen_block(note: AtomicNoteDraft, source_file: str,
                        source_meta: dict[str, str] | None) -> str:
    """Quellen-Block deterministisch aus PDF-Metadata + verifizierten Anker-Pages.
    Kein Halluzinations-Risiko, weil das Modell nichts mehr selbst schreibt.
    Public, damit Orchestrator den Block schon vor Critic an draft.body anhängen kann."""
    meta = dict(source_meta or {})
    # F4: Filename-Fallback wenn pdf_metadata leer/unsinnig
    fallback = _parse_filename_fallback(source_file)
    if not meta.get("Author"):
        meta["Author"] = fallback.get("Author", "")
    # v28: PDF-Filename-Year hat Vorrang vor source_meta-Year. CrossRef gibt für
    # mehrfach aufgelegte Bücher oft das Year der jüngsten Auflage zurück (Hiatt-Bug:
    # 2023 statt 2006). Filename-Year ist user-set und entspricht der vorliegenden
    # PDF-Edition — autoritativ für die Quellen-Angabe.
    if fallback.get("Year"):
        meta["Year"] = fallback.get("Year")
    elif not meta.get("Year"):
        meta["Year"] = ""
    raw_title = meta.get("Title", "").strip()
    if not raw_title or _TITLE_LOOKS_BAD.match(raw_title):
        meta["Title"] = fallback.get("Title", "") or raw_title

    # Fallback auf Filename-Stem wenn Metadaten leer — kein "[unbekannt]" im Output
    author = meta.get("Author", "").strip() or fallback.get("Author", "").strip() or Path(source_file).stem
    year = meta.get("Year", "").strip() or fallback.get("Year", "").strip() or ""
    title = (meta.get("Title", "").strip() or fallback.get("Title", "").strip() or Path(source_file).stem)

    # Seiten aus verifizierten Ankern. F8: page (LLM-exact) ODER fuzzy_page
    # (rapidfuzz-Fallback) — beide sind valide Seitenbelege für den Quellen-Block.
    pages = sorted({
        (a.page or a.fuzzy_page).strip()
        for a in note.source_anchors
        if (a.page or a.fuzzy_page) and (a.page or a.fuzzy_page).strip().lower() not in ("none", "null", "")
    })
    pages_str = ", ".join(pages) if pages else ""

    # Quellen-Block: Wikilink zeigt direkt auf die PDF im Vault (Junction
    # `98-system/attachments/literatur/`). Display-Alias `<Author> <Year>` für Lesbarkeit.
    # Kein separater `[PDF](file://...)`-Link mehr (redundant). Kein Year-Doublet
    # mehr (Jahr ist im PDF-Filename und im Alias enthalten).
    short = _short_label({"Author": author, "Year": year}, source_file)
    pdf_in_vault = (LITERATURE_DIR / source_file).exists()
    wikilink_unsafe = any(c in source_file for c in ("|", "#", "[", "]"))
    if pdf_in_vault and not wikilink_unsafe:
        link = f"[[{source_file}|{short}]]"
    else:
        link = short  # Klartext-Fallback wenn PDF fehlt oder Filename unsafe
    pages_marker = f", S. {pages_str}" if pages_str else ""
    return (
        "## Quellen\n\n"
        f"*Quelle: {link}: {title}{pages_marker}*\n"
    )


def _short_label(meta: dict[str, str] | None, source_file: str) -> str:
    """`<Surname> <Year>` für Footnote-Defs. Surname aus `<Last>, <First>` (CrossRef)
    oder `<First> <Last>`. Year aus pdf_metadata oder Filename-Fallback. Fällt auf
    Filename-Stem zurück wenn nichts geht.
    """
    meta = dict(meta or {})
    fallback = _parse_filename_fallback(source_file)
    author = (meta.get("Author") or fallback.get("Author") or "").strip()
    # Filename-Year hat Vorrang (siehe Begründung in build_quellen_block).
    year = (fallback.get("Year") or meta.get("Year") or "").strip()
    if not author:
        return Path(source_file).stem
    # Multi-Author auf erste Surname + et al.
    parts = [p.strip() for p in re.split(r"\s*;\s*|\s+(?:und|and)\s+", author) if p.strip()]
    surname = parts[0].split(",", 1)[0].strip() if "," in parts[0] else parts[0].split()[-1]
    if len(parts) > 1:
        surname = f"{surname} et al."
    return f"{surname} {year}".strip()


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


def convert_inline_to_footnotes(body: str, source_label: str,
                                 source_file: str | None = None) -> str:
    """Konvertiert `(S. N)`-Inline-Marker zu `[^i]`-Footnote-Markern. Footnote-Defs
    werden an den Body-Ende als nackter Block angehängt (Reading-Mode rendert sie eh
    am Ende der Note). Block-Quote-Callouts (`> ...`) werden NICHT umgeschrieben —
    deren `S. N`-Angaben gehören zum Quote-Header und bleiben.

    Wenn `source_file` übergeben ist und die PDF unter LITERATURE_DIR (Junction
    `98-system/attachments/literatur/`) auflösbar ist, wird der `S. N`-Teil als
    Obsidian-Wikilink mit `#page=N` gerendert — Klick öffnet Obsidian-internen
    PDF-Viewer auf der richtigen Seite. Bei Page-Range `13–14` zeigt das
    `#page=`-Fragment auf die erste Zahl, das Label behält die Range.
    """
    counter = [0]
    defs: list[str] = []
    # Filename mit Wikilink-Syntax-Zeichen würde den Wikilink semantisch
    # zerbrechen. Defensiv: Klartext-Fallback. Codex-Finding 1 (`|`, `#`),
    # Gemini-Finding G1 (einzelne `[`, `]`).
    wikilink_unsafe = bool(source_file) and any(
        c in source_file for c in ("|", "#", "[", "]")
    )
    pdf_in_vault = (source_file is not None
                    and not wikilink_unsafe
                    and (LITERATURE_DIR / source_file).exists())

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
            page_md = f"[[{source_file}#page={page_anchor}|S. {page_label}]]"
        else:
            page_md = f"S. {page_label}"
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


def render_moc(note: AtomicNoteDraft, source_file: str,
               source_meta: dict[str, str] | None = None) -> str:
    """Hub-Routing: Note als MoC-Note rendern (Schema-MoC).
    Frontmatter: type=moc, cssclasses=[moc], obsidianUIMode=preview. Kein H1, keine
    fixen H2-Sektionen. Body wird übernommen; Quellen-Block am Ende (optional per Schema)
    bleibt zur Traceability erhalten — MoC stammt aus PDF-Pipeline.
    """
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
    body = re.sub(
        r"\n+##\s+(Quellen?|Confidence-Notiz)\s*\n.*?(?=\n+##\s|\Z)",
        "", body, flags=re.IGNORECASE | re.DOTALL
    ).rstrip()
    body = convert_inline_to_footnotes(body, _short_label(source_meta, source_file), source_file)

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
                    list_lines.append(f"{i+1}. [[{sc}]] — {desc}")
                else:
                    list_lines.append(f"{i+1}. [[{sc}]]")
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
                f"{i+1}. [[{sc}]]" + (f" — {note.hub_subconcept_descriptions[sc]}"
                                       if note.hub_subconcept_descriptions.get(sc) else "")
                for i, sc in enumerate(note.hub_subconcepts)
            )
            sections.append(list_md)
            sections.append(body)
    else:
        sections.append(body)
    # Footnote-Renumbering: nach Body-Layout-Refactor können verwaiste Defs
    # zurückbleiben (z.B. wenn redundanter Aufzählungs-Absatz gestrippt wurde).
    body_combined = renumber_footnotes("\n\n".join(sections))
    return frontmatter + "\n" + body_combined + "\n\n" + build_quellen_block(note, source_file, source_meta).rstrip() + "\n"


def render_note(note: AtomicNoteDraft, source_file: str,
                source_meta: dict[str, str] | None = None) -> str:
    if note.action == "hub":
        return render_moc(note, source_file, source_meta)
    today = date.today().isoformat()
    related_yaml = _yaml_list(note.related)
    tags_yaml = "\n".join(f"  - {t}" for t in note.tags) if note.tags else "  []"
    flags_yaml = _yaml_list(note.quality_flags)
    aliases_yaml = _yaml_list(note.aliases)

    title_esc = note.title.replace('"', '\\"')

    # F3: confidence-rationale ins Frontmatter (statt Body-Anhang). Nur bei low/medium
    # mit vorhandenem Reasoning. YAML-Doppelquote-Escape für eingebettete Quotes.
    rationale_line = ""
    if (note.synthesis_confidence in ("low", "medium")
            and note.confidence_reasoning):
        rat_esc = note.confidence_reasoning.replace("\\", "\\\\").replace('"', '\\"')
        rationale_line = f'\nconfidence-rationale: "{rat_esc}"'

    # v23: auto-vault-recommended-Marker für Inbox-Reviewer (Tag-basiertes Routing
    # via Auto-Note-Mover ersetzt Pipeline-Pfad-Routing).
    auto_vault_line = ""
    if note.auto_vault_recommended is not None:
        auto_vault_line = f"\nauto-vault-recommended: {'true' if note.auto_vault_recommended else 'false'}"

    # Bootstrap-Schwäche 4b: proposed-tags + tag-review-status nur wenn nicht leer.
    # KEIN Auto-Note-Mover-Routing — User entscheidet beim Inbox-Review ob Tag
    # in tag_registry.yml wandert. Helper auch in render_moc() genutzt (Codex Fix 5).
    proposed_block = _render_proposed_tags_block(note)

    frontmatter = f"""---
title: "{title_esc}"
aliases:
{aliases_yaml}
type: atomic
synthesis-confidence: {note.synthesis_confidence}{rationale_line}{auto_vault_line}
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
    # Idempotent: vorhandene Quellen-/Confidence-Notiz-Sektionen entfernen, falls noch
    # aus alten Pipeline-Versionen im Body vorhanden. Saubere Drafts (post Stabilisierungs-
    # Refactor) haben weder noch — dieser Strip ist Defensiv-Code für Cache-Drafts.
    body = re.sub(
        r"\n+##\s+(Quellen?|Confidence-Notiz)\s*\n.*?(?=\n+##\s|\Z)",
        "", body, flags=re.IGNORECASE | re.DOTALL
    ).rstrip()

    # v28: `(S. N)` → `[^i]`-Footnotes deterministisch im Renderer (Pipeline-Components
    # wie anchor_repair/verifier arbeiten weiter mit dem Inline-Format im Body-Draft).
    # v30: Page-Wikilink mit `#page=N` wenn PDF im Vault auflösbar.
    body = convert_inline_to_footnotes(body, _short_label(source_meta, source_file), source_file)

    sections: list[str] = [body]
    sections.append(build_quellen_block(note, source_file, source_meta).rstrip())

    return frontmatter + "\n" + "\n\n".join(sections) + "\n"


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
    is_strong_hub = (note.action == "hub"
                     and note.critic_score >= 4
                     and len(note.hub_subconcepts) >= 2)

    if not note.hard_gates_pass and not is_strong_hub:
        return False, "hard-gate fail (Glance/Future-Self/Quellen)"
    # Pfad C: Hub-Note mit Score ≥ 4 und ≥2 Sub-Konzepten → Vault auch ohne HG-pass
    if is_strong_hub:
        return True, "ok"
    # Pfad A: Score ≥ Threshold + Hard-Gates pass → Vault (confidence=low OK)
    if note.critic_score < CRITIC_AUTO_THRESHOLD:
        return False, f"score {note.critic_score}<{CRITIC_AUTO_THRESHOLD}"
    return True, "ok"


def find_existing_in_vault(title: str, aliases: list[str],
                            existing_concepts: dict[str, str]) -> Path | None:
    """Title-/Alias-Match gegen Vault-Index aus context_builder. Existing_concepts
    excludiert bereits 00-inbox/98-system/99-archive/08-dashboards (siehe SKIP_DIRS).
    Match-Reihenfolge: exakter Title, dann jeder Alias. Erster Treffer gewinnt.
    """
    candidates = [title.strip().lower()]
    candidates.extend(a.strip().lower() for a in aliases if a)
    for c in candidates:
        rel_path = existing_concepts.get(c)
        if rel_path:
            return VAULT / rel_path
    return None


def find_existing_in_inbox(source_file: str, title: str) -> Path | None:
    """Idempotenz-Check: Inbox-Datei mit identischem source-file + title.
    Findet eigene Pipeline-Drafts aus früherem Run derselben PDF — überschreiben statt
    -2-Suffix anhängen.
    """
    if not INBOX.exists():
        return None
    title_norm = title.strip().lower()
    for f in INBOX.glob("*.md"):
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
        if (fm.get("source-file") == source_file
                and str(fm.get("title", "")).strip().lower() == title_norm):
            return f
    return None


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


def render_merge_stub(note: AtomicNoteDraft, source_file: str,
                      existing_path: Path,
                      source_meta: dict[str, str] | None = None) -> str:
    """v27 MVP — Diff-Stub für menschlichen Merge-Review.

    Voller Attribute-First-Merge (siehe [[Multi-Source-Note-Merge]]) ist v28. v27
    schreibt den neuen Pipeline-Body daneben in die Inbox mit explizitem Verweis auf
    die existierende Note, damit kein -N-Suffix-Duplikat im Vault entsteht und
    SSoT bleibt.
    """
    today = date.today().isoformat()
    title_esc = note.title.replace('"', '\\"')
    rel_existing = str(existing_path.relative_to(VAULT)).replace("\\", "/")
    existing_link = existing_path.stem
    flags_yaml = _yaml_list(["merge-pending"] + note.quality_flags)

    frontmatter = f"""---
title: "MERGE: {title_esc}"
type: merge-stub
merge-target: "[[{existing_link}]]"
merge-target-path: "{rel_existing}"
source-file: "{source_file}"
claude-generated: true
quality-flags:
{flags_yaml}
created: {today}
tags:
  - merge-pending
---"""

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
        # Codex-Finding 2 (2026-05-10): Merge-Stub-Body durch dieselbe Footnote-
        # Konvertierung wie render_note routen, damit auch Merge-Stubs Wikilink-
        # Footnotes auf die PDF-Seite haben (v30-Vollständigkeit).
        convert_inline_to_footnotes(
            note.body.strip(),
            _short_label(source_meta, source_file),
            source_file,
        ),
        "",
        build_quellen_block(note, source_file, source_meta).rstrip(),
    ]
    return frontmatter + "\n" + "\n".join(body_parts) + "\n"


def write_note(note: AtomicNoteDraft, source_file: str, dry_run: bool = False,
               source_meta: dict[str, str] | None = None,
               existing_concepts: dict[str, str] | None = None) -> Path:
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

    target_dir = INBOX
    is_merge_stub = False
    existing_vault: Path | None = None
    if existing_concepts:
        existing_vault = find_existing_in_vault(note.title, note.aliases, existing_concepts)

    if existing_vault is not None:
        # Pre-Merge Source-Check (MVP): Prüfe ob bestehende Note dieselbe Quelle hat.
        # Wenn source-file abweicht → andere Primärquelle → stub markiert als cross-source.
        # Voller Pre-Merge-Validation-LLM-Call ist TODO (v28).
        existing_source = _read_source_field(existing_vault)
        cross_source = (existing_source is not None
                        and Path(source_file).stem not in existing_source
                        and existing_source not in source_file)
        is_merge_stub = True
        stub_prefix = "XSOURCE-MERGE" if cross_source else "MERGE"
        if cross_source:
            print(f"  [pre-merge] Quellen-Konflikt: neue Quelle '{Path(source_file).stem}' "
                  f"vs. bestehende '{existing_source}' — Stub als XSOURCE markiert")
        filename = f"{stub_prefix} - {slugify(note.title)}.md"
        target = target_dir / filename
        # Idempotenz auch für merge-stubs: gleicher source_file + title → überschreiben
        existing_stub = find_existing_in_inbox(source_file, f"MERGE: {note.title}")
        if existing_stub is not None:
            target = existing_stub
        elif target.exists():
            base = target.stem
            for i in range(2, 20):
                candidate = target_dir / f"{base}-{i}.md"
                if not candidate.exists():
                    target = candidate
                    break
        content = render_merge_stub(note, source_file, existing_vault,
                                    source_meta=source_meta)
    else:
        # Idempotenz: eigener früherer Run derselben PDF → überschreibe
        existing_inbox = find_existing_in_inbox(source_file, note.title)
        if existing_inbox is not None:
            target = existing_inbox
        else:
            filename = (moc_filename(note.title) if note.action == "hub"
                        else slugify(note.title) + ".md")
            target = target_dir / filename
            if target.exists():
                base = target.stem
                for i in range(2, 20):
                    candidate = target_dir / f"{base}-{i}.md"
                    if not candidate.exists():
                        target = candidate
                        break
        # Codex-Finding 1: bei Re-Run mit existing Inbox-Datei UND ohne neue
        # Vorschläge bestehenden Review-Block bewahren (sonst verschwindet
        # User-State stillschweigend). Neue Vorschläge überschreiben — der
        # neue Run hat aktuelleres Wissen.
        if existing_inbox is not None and not note.proposed_tags:
            kept_tags, kept_status = _read_proposed_tags_from_inbox(existing_inbox)
            if kept_tags:
                note.proposed_tags = kept_tags
                note.tag_review_status = kept_status or "needs-review"
        content = render_note(note, source_file, source_meta=source_meta)

    if dry_run:
        if is_merge_stub:
            marker = f"[Merge-Stub -> {existing_vault.relative_to(VAULT)}]"
        else:
            marker = "[Vault-Empf.]" if auto else f"[Inbox-Review: {reason}]"
        safe = lambda s: s.encode("ascii", "replace").decode("ascii")
        print(f"  [DRY-RUN] -> Inbox: {target.name}  {marker}")
        print(f"    Score: {note.critic_score}/5 | Hard-Gates: {'pass' if note.hard_gates_pass else 'fail'} | Confidence: {note.synthesis_confidence}")
        if note.quality_flags:
            print(f"    Flags: {safe(', '.join(note.quality_flags))}")
        eval_dir = Path(__file__).resolve().parents[1] / ".cache" / "eval" / "baseline" / Path(source_file).stem
        eval_dir.mkdir(parents=True, exist_ok=True)
        prefix = "merge" if is_merge_stub else ("vault" if auto else "inbox")
        (eval_dir / f"{prefix}__{target.name}").write_text(content, encoding="utf-8")
        return target

    target.write_text(content, encoding="utf-8")
    if is_merge_stub:
        print(f"  [Merge-Stub] {target.relative_to(VAULT)}  -> {existing_vault.relative_to(VAULT)}")
    else:
        print(f"  [Inbox] {target.relative_to(VAULT)}  ({'vault-empfohlen' if auto else 'review'})")
    return target
