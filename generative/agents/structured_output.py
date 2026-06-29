"""Structured-Output-Format mit HTML-Kommentar-Sentinels.

Eliminiert das `_fix_unescaped_quotes`-Heuristik-Problem strukturell: freitext-
Felder (body, quote, revision_hint) leben außerhalb des JSON als Heredoc-Sektionen
mit `<!--KEY-->`-Sentinels. Damit kann das Modell beliebige Quotes/Markdown/
Reserved-Chars produzieren, ohne das Parse-Format zu brechen.

Format-Spec siehe Docstrings der einzelnen `parse_*_output`-Funktionen.
Edge-Cases sind durch `tests/test_structured_parser.py` abgedeckt — Format-
Änderungen ohne Test-Update sind ein v16-Wiederholungs-Risiko.
"""

from __future__ import annotations
import re

# Erkannte Sentinels — strikt ALL_CAPS_WITH_UNDERSCORES.
# Lowercase oder unbekannte Namen werden als Body-Content behandelt.
_SENTINELS = {"NOTE", "BODY", "ANCHOR", "QUOTE", "REVISION_HINT", "CONCEPT", "CONTRADICTION", "RELATED", "END"}
_SENTINEL_RE = re.compile(r"^\s*<!--([A-Z_]+)-->\s*$")
# Modelle schreiben gelegentlich `proposed-tags:` statt `proposed_tags:` (Prompt-Drift).
# Bindestriche werden in `parse_header_line()` zu `_` normalisiert, sonst geht der
# Eintrag stillschweigend verloren.
_HEADER_RE = re.compile(r"^([a-z][a-z_-]*):\s*(.*)$")
# Strict single wikilink: `[[Titel]]` — Titel darf keine eckigen Klammern enthalten.
# Verhindert `[[[[Note]]]]` (Doppel-Verschachtelung) und `[[A]] [[B]]` (zwei Links pro Zeile).
_WIKILINK_RE = re.compile(r"^\[\[[^\[\]]+\]\]$")
_NULL_VALUES = {"", "null", "none", "-"}


def _normalize_lines(text: str) -> list[str]:
    """CRLF/CR → LF, dann split."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _split_sentinels(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Splittet Lines an erkannten Sentinels. Erste Section hat name=None
    (Pre-Sentinel-Preamble — typischerweise leer oder Modell-Vorgeplänkel)."""
    result: list[tuple[str | None, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in lines:
        m = _SENTINEL_RE.match(line)
        if m and m.group(1) in _SENTINELS:
            result.append((current_name, current_lines))
            current_name = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    result.append((current_name, current_lines))
    return result


def _parse_header_line(line: str) -> tuple[str, str] | None:
    """Match `^[a-z][a-z_-]*:`. Bindestriche im Key werden zu Underscore normalisiert,
    weil Modelle gelegentlich `proposed-tags:` statt `proposed_tags:` schreiben.
    Wert wird trailing-stripped, Inhalt unverändert."""
    m = _HEADER_RE.match(line)
    if not m:
        return None
    return m.group(1).replace("-", "_"), m.group(2).rstrip()


def _normalize_value(v: str) -> str | None:
    """Empty/null/none/- → None. Sonst stripped Wert."""
    s = v.strip()
    return None if s.lower() in _NULL_VALUES else s


def _parse_bool(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"true", "yes", "1", "t", "y"}


def _parse_int(v: str | None, default: int = 0) -> int:
    """Erste Ziffer-Sequenz aus dem String. `4 (good)` → 4, `4/5` → 4."""
    if v is None or v == "":
        return default
    m = re.search(r"-?\d+", v)
    return int(m.group(0)) if m else default


def parse_headers(content_lines: list[str]) -> tuple[dict[str, str | None], list[str]]:
    """Parst alle Lines des Blocks als `key: value`-Paare. Lines die nicht matchen
    (Erklär-Text, Leerzeilen) werden ignoriert. Returns (headers, ignored_lines)."""
    headers: dict[str, str | None] = {}
    ignored: list[str] = []
    for line in content_lines:
        kv = _parse_header_line(line)
        if kv:
            headers[kv[0]] = _normalize_value(kv[1])
        elif line.strip():
            ignored.append(line)
    return headers, ignored


def _join_body(lines: list[str]) -> str:
    """Heredoc-Body: leading/trailing Newlines strippen, internal Whitespace
    erhalten (Markdown-Indentation, Leerzeilen zwischen Absätzen bleiben)."""
    return "\n".join(lines).strip("\n")


def _split_csv(v: str | None) -> list[str]:
    """Comma-separated value list — Items mit Whitespace stripped, leere weg."""
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def _extract_note_headers(headers: dict[str, str | None]) -> dict:
    """Header-Dict in Note-Schema mappen — Lists comma-split, Strings as-is."""
    result: dict = {}
    if "title" in headers and headers["title"]:
        result["title"] = headers["title"]
    result["aliases"] = _split_csv(headers.get("aliases"))
    result["tags"] = _split_csv(headers.get("tags"))
    # Bootstrap: proposed_tags optional, Bootstrap-Schwäche 4b. Validation findet
    # später im orchestrator/writer statt — Parser nimmt nur entgegen was steht.
    result["proposed_tags"] = _split_csv(headers.get("proposed_tags"))
    for str_field in ("synthesis_confidence", "action", "extend_path"):
        if str_field in headers:
            result[str_field] = headers[str_field]
    return result


def _finalize_anchor(note: dict, anchor: dict, warnings: list[str]) -> None:
    """Validiert Anchor und appended bei vollständigen Daten an note.source_anchors.
    Konvention: page+quote beide nötig; nur eines davon → drop + Warning."""
    page = anchor.get("page")
    quote = anchor.get("quote")
    if not page and not quote:
        warnings.append(f"ANCHOR ohne page UND quote in '{note.get('title', '?')}' — übersprungen")
        return
    if not quote:
        warnings.append(f"ANCHOR ohne QUOTE in '{note.get('title', '?')}' — übersprungen")
        return
    note["source_anchors"].append({"page": page, "quote": quote})


# --- Per-Agent-Parser ---


def parse_extractor_output(text: str) -> tuple[list[dict], list[str]]:
    """Parst Extractor-Output (Liste von Notes mit body + source_anchors).

    Format:
        <!--NOTE-->
        title: ...
        aliases: a, b
        tags: t1, t2
        synthesis_confidence: low
        action: create
        extend_path:
        <!--BODY-->
        <markdown body>
        <!--ANCHOR-->
        page: S. 42
        <!--QUOTE-->
        <literal quote>
        <!--ANCHOR-->
        ...
        <!--NOTE-->
        ...
        <!--END-->

    Returns (notes, warnings). Notes ohne `title` werden mit Warning verworfen,
    ANCHOR ohne QUOTE wird verworfen.
    """
    sections = _split_sentinels(_normalize_lines(text))
    notes: list[dict] = []
    warnings: list[str] = []
    current_note: dict | None = None
    current_anchor: dict | None = None

    def _flush_note():
        nonlocal current_note, current_anchor
        if current_anchor is not None and current_note is not None:
            _finalize_anchor(current_note, current_anchor, warnings)
            current_anchor = None
        if current_note is not None:
            if current_note.get("title"):
                notes.append(current_note)
            else:
                warnings.append("NOTE ohne title übersprungen")
            current_note = None

    for name, content in sections:
        if name is None or name == "END":
            if name == "END":
                break
            continue
        if name == "NOTE":
            _flush_note()
            current_note = {"source_anchors": []}
            headers, _ = parse_headers(content)
            current_note.update(_extract_note_headers(headers))
        elif name == "BODY":
            if current_note is None:
                warnings.append("BODY ohne vorhergehendes NOTE — verworfen")
                continue
            if "body" in current_note:
                warnings.append(f"Duplicate BODY in '{current_note.get('title', '?')}' — last wins")
            current_note["body"] = _join_body(content)
        elif name == "ANCHOR":
            if current_anchor is not None and current_note is not None:
                _finalize_anchor(current_note, current_anchor, warnings)
            if current_note is None:
                warnings.append("ANCHOR ohne vorhergehendes NOTE — verworfen")
                current_anchor = None
                continue
            current_anchor = {}
            headers, _ = parse_headers(content)
            current_anchor["page"] = headers.get("page")
        elif name == "QUOTE":
            if current_anchor is None:
                warnings.append("QUOTE ohne vorhergehendes ANCHOR — verworfen")
                continue
            if "quote" in current_anchor:
                warnings.append("Duplicate QUOTE im ANCHOR — last wins")
            current_anchor["quote"] = _join_body(content) or None
        else:
            warnings.append(f"Unbekannte Sektion {name}")

    _flush_note()
    return notes, warnings


def parse_canonicalizer_output(text: str) -> tuple[dict, list[str]]:
    """Parst Canonicalizer-Output — eine konsolidierte Note.

    Format identisch zu Extractor, aber genau ein <!--NOTE-->-Block.
    Returns ({"title", "aliases", "tags", "body"}, warnings).
    Source-Anchors werden hier deterministisch außerhalb des LLM-Calls gemerged
    (siehe canonicalizer.merge_cluster) — Anchors-Sektionen im Output werden
    ignoriert mit Warning.
    """
    notes, warnings = parse_extractor_output(text)
    if not notes:
        return {}, warnings + ["Canonicalizer: keine Note im Output"]
    if len(notes) > 1:
        warnings.append(f"Canonicalizer: {len(notes)} Notes erwartet 1 — erste verwendet")
    return notes[0], warnings


def parse_verifier_output(text: str) -> tuple[dict, list[str]]:
    """Parst Verifier-Output — Top-Level-Headers + Anchor-Liste mit verified-Flag.

    Format:
        all_verified: true
        <!--ANCHOR-->
        page: S. 5
        verified: true
        <!--QUOTE-->
        <quote text>
        <!--ANCHOR-->
        ...
        <!--END-->

    Returns ({"all_verified": bool, "anchors": [{"page", "quote", "verified"}]},
    warnings).
    """
    sections = _split_sentinels(_normalize_lines(text))
    warnings: list[str] = []
    top_headers: dict[str, str | None] = {}
    anchors: list[dict] = []
    current_anchor: dict | None = None

    def _flush_anchor():
        nonlocal current_anchor
        if current_anchor is None:
            return
        # Verifier: page+verified können auch ohne quote sinnvoll sein
        # (verified=false → page=null, quote optional). Anchor mit gar keinen
        # Daten skippen.
        has_data = any(current_anchor.get(k) is not None for k in ("page", "quote", "verified"))
        if has_data:
            anchors.append(current_anchor)
        else:
            warnings.append("Verifier: leerer ANCHOR übersprungen")
        current_anchor = None

    for name, content in sections:
        if name is None:
            # Top-Level-Header (vor erstem Sentinel)
            h, _ = parse_headers(content)
            top_headers.update(h)
        elif name == "END":
            break
        elif name == "ANCHOR":
            _flush_anchor()
            current_anchor = {"verified": False}
            h, _ = parse_headers(content)
            current_anchor["page"] = h.get("page")
            current_anchor["verified"] = _parse_bool(h.get("verified"))
        elif name == "QUOTE":
            if current_anchor is None:
                warnings.append("Verifier: QUOTE ohne ANCHOR — verworfen")
                continue
            current_anchor["quote"] = _join_body(content) or None
        else:
            warnings.append(f"Verifier: unbekannte Sektion {name}")

    _flush_anchor()
    return {
        "all_verified": _parse_bool(top_headers.get("all_verified")),
        "anchors": anchors,
    }, warnings


def parse_critic_output(text: str) -> tuple[dict, list[str]]:
    """Parst Critic-Output — Test-Booleans + score (Header) + revision_hint (Heredoc).

    Format:
        title_test: true
        glance_test: true
        future_self_test: false
        quellen_test: true
        deletion_test: true
        score: 4
        <!--REVISION_HINT-->
        <text — kann beliebige Quotes enthalten>
        <!--END-->

    Wenn `<!--REVISION_HINT-->` weggelassen wird → revision_hint=None.
    Score wird auf 0–5 geclamped, bei nicht-parsebar default 0.
    """
    sections = _split_sentinels(_normalize_lines(text))
    warnings: list[str] = []
    top_headers: dict[str, str | None] = {}
    revision_hint: str | None = None

    for name, content in sections:
        if name is None:
            h, _ = parse_headers(content)
            top_headers.update(h)
        elif name == "REVISION_HINT":
            joined = _join_body(content)
            revision_hint = joined if joined else None
        elif name == "END":
            break
        else:
            warnings.append(f"Critic: unbekannte Sektion {name}")

    raw_score = _parse_int(top_headers.get("score"), default=0)
    score = max(0, min(5, raw_score))

    return {
        "title_test": _parse_bool(top_headers.get("title_test")),
        "glance_test": _parse_bool(top_headers.get("glance_test")),
        "future_self_test": _parse_bool(top_headers.get("future_self_test")),
        "quellen_test": _parse_bool(top_headers.get("quellen_test")),
        "deletion_test": _parse_bool(top_headers.get("deletion_test")),
        "score": score,
        "revision_hint": revision_hint,
    }, warnings


def parse_planner_output(text: str) -> tuple[dict, list[str]]:
    """Parst Planner-Output — Top-Level-Headers (source_title, source_summary)
    plus Liste von <!--CONCEPT-->-Blöcken.

    Format:
        source_title: ...
        source_summary: 2-Satz-Zusammenfassung
        <!--CONCEPT-->
        title: Konzeptname
        priority: high|medium|low
        chapter: Kapitelname
        action: create|extend|skip
        extend_path:
        <!--CONCEPT-->
        ...
        <!--END-->

    Returns ({"source_title", "source_summary", "concepts": [...]}, warnings).
    Concepts ohne `title` werden mit Warning verworfen.
    """
    sections = _split_sentinels(_normalize_lines(text))
    warnings: list[str] = []
    top_headers: dict[str, str | None] = {}
    concepts: list[dict] = []

    for name, content in sections:
        if name is None:
            h, _ = parse_headers(content)
            top_headers.update(h)
        elif name == "END":
            break
        elif name == "CONCEPT":
            h, _ = parse_headers(content)
            title = h.get("title")
            if not title:
                warnings.append("Planner: CONCEPT ohne title übersprungen")
                continue
            category = (h.get("category") or "conceptual").lower()
            if category not in ("architectural", "operational", "conceptual"):
                warnings.append(f"Planner: ungültige category '{category}' für '{title}' → conceptual")
                category = "conceptual"

            raw_origin = (h.get("origin") or "primary").lower().strip()
            if raw_origin not in {"primary", "extension", "secondary_mention"}:
                warnings.append(f"Planner: ungültiger origin '{raw_origin}' für '{title}' → primary")
                raw_origin = "primary"

            raw_authors = h.get("cited_authors") or ""
            cited_authors = [a.strip() for a in raw_authors.split(",") if a.strip()]

            concepts.append(
                {
                    "title": title,
                    "priority": (h.get("priority") or "medium").lower(),
                    "chapter": h.get("chapter") or "",
                    "action": (h.get("action") or "create").lower(),
                    "extend_path": h.get("extend_path"),
                    "category": category,
                    "origin": raw_origin,
                    "cited_authors": cited_authors,
                }
            )
        else:
            warnings.append(f"Planner: unbekannte Sektion {name}")

    return {
        "source_title": top_headers.get("source_title") or "",
        "source_summary": top_headers.get("source_summary") or "",
        "concepts": concepts,
    }, warnings


def parse_cross_reference_output(text: str) -> tuple[dict, list[str]]:
    """Parst Cross-Reference-Output — duplicate-Headers + CONTRADICTION-Heredocs
    + RELATED-Block mit einer Wikilink-Zeile pro Eintrag.

    Format:
        duplicate_risk: none|low|high
        duplicate_path: <pfad oder leer>
        <!--CONTRADICTION-->
        freitext-widerspruch (kann beliebige Zeichen enthalten)
        <!--CONTRADICTION-->
        weiterer widerspruch
        <!--RELATED-->
        [[Note A]]
        [[Note B]]
        <!--END-->

    CONTRADICTION- und RELATED-Sektionen sind optional. Returns
    ({"duplicate_risk", "duplicate_path", "contradictions", "related"}, warnings).
    Related-Lines die nicht mit `[[` beginnen werden mit Warning verworfen.
    """
    sections = _split_sentinels(_normalize_lines(text))
    warnings: list[str] = []
    top_headers: dict[str, str | None] = {}
    contradictions: list[str] = []
    related: list[str] = []

    for name, content in sections:
        if name is None:
            h, _ = parse_headers(content)
            top_headers.update(h)
        elif name == "END":
            break
        elif name == "CONTRADICTION":
            joined = _join_body(content)
            if joined:
                contradictions.append(joined)
        elif name == "RELATED":
            for line in content:
                s = line.strip()
                if not s:
                    continue
                if _WIKILINK_RE.match(s):
                    related.append(s)
                else:
                    warnings.append(f"Cross-Reference: RELATED-Line ohne saubere Wikilink-Form verworfen: {s[:60]!r}")
        else:
            warnings.append(f"Cross-Reference: unbekannte Sektion {name}")

    return {
        "duplicate_risk": top_headers.get("duplicate_risk") or "none",
        "duplicate_path": top_headers.get("duplicate_path"),
        "contradictions": contradictions,
        "related": related,
    }, warnings
