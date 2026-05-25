"""Context-Builder: scannt Vault → Relevanz-Profil für Planner + Dedup."""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

import yaml

from config import VAULT, WISSEN, SCRIPTS_DIR  # BA_DIR-Import entfernt (v31)

# Vault-Ordner die NICHT als existing_concepts zählen
SKIP_DIRS = {"00-inbox", "98-system", "99-archive", "08-dashboards", ".obsidian", ".trash"}

# Tag-Registry: autoritative Approved-Tags. Wird via Union mit Vault-Frequenz-
# Whitelist gemergt — Registry-Tags sind ab Eintrag sofort zugelassen (Bootstrap),
# bestehende Vault-Tags bleiben unverändert verfügbar.
TAG_REGISTRY_PATH = SCRIPTS_DIR / "tag_registry.yml"


def _read_frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(text[3:end]) or {}
    except Exception:
        return {}


def build_existing_concepts(scan_dirs: list[Path] | None = None) -> dict[str, str]:
    """Gibt {concept_title: file_path} für alle Notes im Vault zurück (ohne SKIP_DIRS).

    Eval-Snapshot-Modus: ENV-Var ``EVAL_VAULT_SNAPSHOT`` zeigt auf JSON-Pfad.
    Wenn die Datei existiert, wird sie geladen statt live-Scan (deterministisch
    für Code-Version-Vergleiche). Existiert sie nicht, wird gescant und der
    Snapshot geschrieben — erster Eval-Run friert den State ein, alle folgenden
    Runs lesen denselben Snapshot.
    """
    snap_path = os.getenv("EVAL_VAULT_SNAPSHOT")
    if snap_path:
        p = Path(snap_path)
        if p.exists():
            print(f"      [eval-snapshot] read concepts from {p.name}", file=sys.stderr)
            return json.loads(p.read_text(encoding="utf-8"))

    if scan_dirs is None:
        # Top-Level-Vault scannen, SKIP_DIRS auslassen
        scan_dirs = [d for d in VAULT.iterdir()
                     if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")]
    concepts: dict[str, str] = {}
    # Gemini-Finding G3 (2026-05-10): Root-Vault-MD-Files (z.B. CLAUDE.md, Home.md)
    # wurden vom Subdir-Scan ignoriert. Explizit mitnehmen — System-Files wie
    # CLAUDE.md/Home.md werden danach via Frontmatter-Type oder Filename-Pattern
    # in höheren Schichten gefiltert, hier ist die Aufgabe nur „alles indexieren".
    files_to_scan: list[Path] = list(VAULT.glob("*.md"))
    for d in scan_dirs:
        if d.exists():
            files_to_scan.extend(d.rglob("*.md"))
    for f in files_to_scan:
        # Sicherheitsnetz: SKIP_DIRS auch in Subpfaden ausschließen
        rel_parts = f.relative_to(VAULT).parts
        if any(p in SKIP_DIRS for p in rel_parts):
            continue
        fm = _read_frontmatter(f)
        title = fm.get("title") or f.stem
        concepts[str(title).lower()] = str(f.relative_to(VAULT))
        for alias in fm.get("aliases", []) or []:
            concepts[str(alias).lower()] = str(f.relative_to(VAULT))
    if snap_path:
        p = Path(snap_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"      [eval-snapshot] wrote concepts to {p.name}", file=sys.stderr)
    return concepts


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:[#|][^\]]*)?\]\]")


def build_concept_links(existing_concepts: dict[str, str]) -> dict[str, set[str]]:
    """Pro Note alle outgoing-Verweise als File-Path-Set.

    Quellen: Wikilinks im Body + `related`-Frontmatter. Targets werden via
    `existing_concepts` (Title + Aliases → file_path) zu kanonischen File-Paths
    aufgelöst, sodass `[[ISP]]` und `[[Information Search Process (Kuhlthau)]]`
    auf denselben Knoten zeigen. Self-Links werden gefiltert.

    Rückgabe: {file_path: {file_path, ...}}. Genutzt vom Hub-Detector
    (Bridging-Coefficient) zur Diskriminierung True-Hub vs. Atomic-mit-Geschwistern.
    """
    seen_files = set(existing_concepts.values())
    file_links: dict[str, set[str]] = {f: set() for f in seen_files}

    for rel_path_str in seen_files:
        path = VAULT / rel_path_str
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        outgoing: set[str] = set()

        # Frontmatter `related`
        fm = _read_frontmatter(path)
        for rel in fm.get("related", []) or []:
            target = re.sub(r"^\[\[|\]\]$", "", str(rel)).strip().lower()
            tgt_path = existing_concepts.get(target)
            if tgt_path and tgt_path != rel_path_str:
                outgoing.add(tgt_path)

        # Body-Wikilinks (alles nach dem zweiten ---)
        body_start = 0
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                body_start = end + 3
        body = text[body_start:]
        for m in _WIKILINK_RE.finditer(body):
            target = m.group(1).strip().lower()
            tgt_path = existing_concepts.get(target)
            if tgt_path and tgt_path != rel_path_str:
                outgoing.add(tgt_path)

        file_links[rel_path_str] = outgoing

    return file_links


def _load_registry_tags() -> set[str]:
    """Lädt approved Tags aus tag_registry.yml. Datei optional — fehlt sie,
    fällt die Whitelist ausschließlich auf Vault-Frequenz zurück.

    Codex-Findings 4+7: strict Schema-Validation (gemeinsame Funktion mit
    _validate_proposed_tags), Top-Level-Dict-Guard gegen malformed YAML."""
    if not TAG_REGISTRY_PATH.exists():
        return set()
    try:
        data = yaml.safe_load(TAG_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, dict):
        return set()
    approved = data.get("approved") or []
    if not isinstance(approved, list):
        return set()
    from agents.extractor import is_valid_schema_tag
    valid: set[str] = set()
    for t in approved:
        if not isinstance(t, str):
            continue
        ts = t.strip().lstrip("#")
        if is_valid_schema_tag(ts):
            valid.add(ts)
    return valid


def build_tag_whitelist(scan_dirs: list[Path] | None = None,
                         min_count: int = 2) -> list[str]:
    """Sammelt approved Tags aus zwei Quellen — Union als Whitelist:
      1. Vault-Frequenz (organisch gewachsene Tags mit min_count≥2)
      2. tag_registry.yml (User-kuratiert, autoritativ — Bootstrap für neue Domains)

    Pipeline-Notes sollen Tags ausschließlich aus dieser Liste wählen, statt
    erfundene Hierarchien zu produzieren. min_count=2 filtert Einmal-Tags
    (Tipp-Fehler, alte Drafts). Registry-Tags brauchen kein min_count — User-Pflege
    ersetzt die Frequenz-Evidenz.
    """
    from collections import Counter
    if scan_dirs is None:
        scan_dirs = [d for d in VAULT.iterdir()
                     if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")]
    # CLAUDE.md-Konvention: lowercase kebab-case, hierarchisch via '/'.
    valid_tag = re.compile(r"^[a-z0-9][a-z0-9\-/]*$")
    counter: Counter = Counter()
    files_to_scan: list[Path] = list(VAULT.glob("*.md"))
    for d in scan_dirs:
        if d.exists():
            files_to_scan.extend(d.rglob("*.md"))
    for f in files_to_scan:
        rel_parts = f.relative_to(VAULT).parts
        if any(p in SKIP_DIRS for p in rel_parts):
            continue
        fm = _read_frontmatter(f)
        tags = fm.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]
        for t in tags:
            ts = str(t).strip().lstrip("#")
            if ts and valid_tag.match(ts):
                counter[ts] += 1
    vault_tags = {t for t, n in counter.items() if n >= min_count}
    return sorted(vault_tags | _load_registry_tags())


# Stoppwörter aus dem Source-Text bei Token-Match ignorieren — sehr häufig und
# diskriminieren schlecht. Liste klein gehalten (kein NLTK-Stopword-Import-Aufwand).
_TAG_SCORE_STOPWORDS = {
    "der", "die", "das", "den", "des", "dem", "ein", "eine", "einer", "einen",
    "und", "oder", "aber", "auch", "noch", "nur", "wie", "was", "wer", "wo",
    "ist", "sind", "war", "waren", "wird", "werden", "hat", "haben", "kann", "können",
    "the", "and", "for", "with", "from", "this", "that", "are", "was", "will",
    "von", "zu", "in", "im", "an", "am", "auf", "mit", "nach", "bei", "um", "über",
}

# Codex-Finding 3 Schwäche 4: generische Tag-Tokens diskriminieren schlecht zwischen
# Domains. „theory" matched fast jede wissenschaftliche Quelle, „method" jeden
# Methoden-Text. Filter aus Tag-Tokens raus, damit nur quellnah-spezifische
# Hits priorisiert werden.
_TAG_GENERIC_TOKENS = {
    "model", "method", "methods", "system", "systems", "theory", "framework",
    "process", "approach", "concept", "concepts", "data", "information",
    "research", "study", "studies", "analysis", "review", "uni", "konzept",
}


def score_tags_for_source(whitelist: list[str], source_text: str,
                            top_n: int = 30) -> tuple[list[str], list[str]]:
    """Source-bezogenes Tag-Ranking. Whitelist bleibt unverändert (Anti-Halluzination),
    aber wird für den Extractor-Prompt in zwei Blöcke geteilt: priorisierte
    Tags (Token-Match mit Source) + übrige.

    Scoring: lexikalisches Match der Tag-Tokens (Split an `/` und `-`) gegen
    Source-Token-Set (case-insensitive, mit Stoppwort-Filter). Häufigkeits-Prior
    bewusst NICHT eingerechnet — Codex-Empfehlung: Häufigkeit als Tie-Breaker
    nur, hier Score-driven Sortierung.

    Returns (priorisiert, übrig). Beide alphabetisch innerhalb des Blocks.
    Priorisiert maximal top_n Tags.
    """
    if not whitelist or not source_text:
        return [], list(whitelist)
    # Unicode-Normalisierung (NFC) + Replacement-Zeichen durch Space ersetzen —
    # schützt vor Mojibake/Latin-1-Fragmenten. Space (nicht Empty) damit Wort-
    # grenzen erhalten bleiben: `change�management` → `change management`.
    import unicodedata
    source_text = unicodedata.normalize("NFC", source_text).replace("�", " ")
    source_tokens = {t.lower() for t in re.findall(r"\b[a-zA-ZäöüÄÖÜß\-]{3,}\b", source_text)}
    source_tokens -= _TAG_SCORE_STOPWORDS
    if not source_tokens:
        return [], list(whitelist)

    scored: list[tuple[int, str]] = []
    for tag in whitelist:
        # Tag in seine Komponenten zerlegen: `change-management/adkar` →
        # {"change", "management", "adkar"}. Tag akzeptieren nur wenn mindestens
        # ein spezifisches (nicht-generisches) Token enthalten ist — `model-theory`
        # alleine matched z.B. nicht, weil beide Tokens generisch sind. Aber
        # `change-management` matched über `change`, weil das spezifisch ist.
        # Scoring zählt alle Tag-Tokens (spezifisch + generisch), damit
        # `change-management-process` gegen Source mit allen drei Tokens den
        # vollen Overlap-Score bekommt.
        tag_tokens = {t for t in re.split(r"[/\-]", tag.lower()) if len(t) >= 3}
        if not tag_tokens or not (tag_tokens - _TAG_GENERIC_TOKENS):
            continue
        overlap = len(tag_tokens & source_tokens)
        if overlap > 0:
            scored.append((overlap, tag))

    scored.sort(key=lambda x: (-x[0], x[1]))
    prioritized = [tag for _, tag in scored[:top_n]]
    rest = [tag for tag in whitelist if tag not in set(prioritized)]
    return prioritized, rest


def build_relevance_profile() -> dict:
    """Vault-Kontext für die Pipeline. KEIN externer Themen-Bias mehr (v31):
    Planner entscheidet aus dem Quellentext heraus, was substantiell ist.
    `existing_concepts` und `tag_whitelist` sind die einzigen kontextuellen
    Inputs — beide sind vault-weite Fakten (was existiert? welche Tags sind
    erlaubt?), kein projektspezifischer Bias.
    """
    return {
        "existing_concepts": build_existing_concepts(),
        "tag_whitelist": build_tag_whitelist(),
    }
