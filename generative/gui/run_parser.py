"""stdout-Parser fuer die Live-GUI.

Uebersetzt die rohen stdout-Zeilen eines `orchestrator`-Laufs in strukturierte
Events fuer das Web-Frontend. Pure Logik (kein I/O), damit test-getrieben.

Events (dict mit "type"):
- stage          : Pipeline-Stufe erreicht (Stepper).  {num, total, label}
- note_progress  : Stage-6 Pro-Note-Marker.             {index, total, title}
- preview        : eine im --dry-run gerenderte Note.    {name, routing, score,
                   hard_gates, confidence, flags, [reason|merge_target]}
- note_written    : eine im Schreib-Lauf geschriebene Note (kein Score/Confidence,
                   die druckt vault_writer nur im Dry-Run). {path, routing,
                   [merge_target]}
- done           : Lauf abgeschlossen.                   {written, dry_run}
- log            : alles uebrige (roher Text).           {text}

Die Marker stammen aus generative/orchestrator.py (Stage-Prints, Pro-Note-Print
orchestrator.py:539) und generative/pipeline/vault_writer.py:874-905 (Dry-Run)
bzw. vault_writer.py:983-986 (Schreib-Lauf: `[Inbox]`/`[Merge-Stub]`-Zeilen).
"""

from __future__ import annotations

import re

# Stabile Stepper-Labels (unabhaengig vom genauen Print-Text im Orchestrator).
STAGES = [
    {"num": 1, "label": "PDF & Chunking"},
    {"num": 2, "label": "Vault-Kontext"},
    {"num": 3, "label": "Quellen-Qualität"},
    {"num": 4, "label": "Planner"},
    {"num": 5, "label": "Extractor"},
    {"num": 6, "label": "Verifier & Critic"},
    {"num": 7, "label": "Vault-Writer"},
    {"num": 8, "label": "Qualitäts-Eval"},
]

# Stage-Marker: KEINE fuehrende Einrueckung, Form "[N/7]" oder "[N.5/7]" / "[8/8]".
_STAGE_RE = re.compile(r"^\[(\d+(?:\.\d+)?(?:-\d+)?)/(\d)\]\s*(.*)$")
# Pro-Note-Marker (Stage 6): mind. ein fuehrendes Leerzeichen, "[i/n] Titel".
_NOTE_RE = re.compile(r"^\s+\[(\d+)/(\d+)\]\s+(.+)$")
# Dry-Run-Block (vault_writer.py:880-883).
_DRYRUN_RE = re.compile(r"^\s+\[DRY-RUN\] -> Inbox:\s+(.+?)\s\s+(\[.+\])\s*$")
_SCORE_RE = re.compile(r"^\s+Score:\s*(\d+)/5\s*\|\s*Hard-Gates:\s*(pass|fail)\s*\|\s*Confidence:\s*(\w+)")
_FLAGS_RE = re.compile(r"^\s+Flags:\s*(.+)$")
# Schreib-Lauf (kein Dry-Run): vault_writer.py:983-986 druckt den tatsaechlich
# geschriebenen Pfad direkt, ohne Score/Confidence-Block.
_WRITTEN_INBOX_RE = re.compile(r"^  \[Inbox\] (.+)  \((vault-empfohlen|review)\)$")
_WRITTEN_MERGE_RE = re.compile(r"^  \[Merge-Stub\] (.+)  -> (.+)$")
# Abschluss (orchestrator.py:1536).
_DONE_RE = re.compile(r"^=== Fertig:\s*(\d+)\s*Notes\s*(\(dry-run\)|geschrieben)\s*===")

# Bekannte Backend-/Setup-Fehlersignaturen (Kleinschreibung-Substrings) — Quelle:
# _subscription_backend.py (_fail_fast_hint), error_hints.py, doctor.py.
_ERROR_SIGNATURES = (
    "nicht eingeloggt",
    "session abgelaufen",
    "rate-limit",
    "rate limit",
    "429",
    "pdftotext nicht gefunden",
    "poppler",
    "kein api-key",
    "api_key",
    "not found",
    "nicht gefunden:",
    "nicht aufrufbar",
    "→ doctor",
    "-> doctor",
)


def _parse_marker(marker: str) -> dict:
    """`[Vault-Empf.]` / `[Inbox-Review: reason]` / `[Merge-Stub -> path]` → dict."""
    inner = marker.strip()[1:-1].strip()  # eckige Klammern weg
    if inner.startswith("Vault-Empf"):
        return {"routing": "vault"}
    if inner.startswith("Inbox-Review:"):
        return {"routing": "inbox", "reason": inner.split(":", 1)[1].strip()}
    if inner.startswith("Merge-Stub"):
        target = inner.split("->", 1)[1].strip() if "->" in inner else ""
        return {"routing": "merge", "merge_target": target}
    return {"routing": "inbox"}


class RunParser:
    """Zeilen-Streaming-Parser mit Pufferung fuer den mehrzeiligen Dry-Run-Block."""

    def __init__(self) -> None:
        self._pending: dict | None = None  # halb-gefuellte preview-Note

    def _flush_pending(self) -> list[dict]:
        if self._pending is None:
            return []
        ev = self._pending
        self._pending = None
        return [ev]

    def feed(self, line: str) -> list[dict]:
        line = line.rstrip("\n")
        if not line.strip():
            return self._flush_pending()

        # 1) Start eines Dry-Run-Preview-Blocks → vorigen Block abschliessen.
        m = _DRYRUN_RE.match(line)
        if m:
            out = self._flush_pending()
            self._pending = {
                "type": "preview",
                "name": m.group(1).strip(),
                "score": None,
                "hard_gates": None,
                "confidence": None,
                "flags": "",
                **_parse_marker(m.group(2)),
            }
            return out

        # 2) Score-Zeile fuellt den laufenden Preview-Block.
        m = _SCORE_RE.match(line)
        if m and self._pending is not None:
            self._pending["score"] = int(m.group(1))
            self._pending["hard_gates"] = m.group(2) == "pass"
            self._pending["confidence"] = m.group(3)
            return []

        # 3) Flags-Zeile (optional) fuellt den laufenden Preview-Block. Roh-String:
        # die Quelle joint mit ", " und ASCII-safed → kein verlässlicher Split.
        m = _FLAGS_RE.match(line)
        if m and self._pending is not None:
            self._pending["flags"] = m.group(1).strip()
            return []

        # Ab hier endet ein etwaiger Preview-Block — zuerst abschliessen.
        prefix = self._flush_pending()

        # 3b) Schreib-Lauf: vault_writer druckt den geschriebenen Pfad direkt
        # (kein DRY-RUN-Block, also kein Score/Confidence). Routing-Vokabular
        # bewusst an die Dry-Run-preview-Events angeglichen (vault/inbox/merge),
        # damit das Frontend dieselbe Badge-Komponente wiederverwenden kann.
        m = _WRITTEN_MERGE_RE.match(line)
        if m:
            return prefix + [
                {
                    "type": "note_written",
                    "path": m.group(1).strip(),
                    "routing": "merge",
                    "merge_target": m.group(2).strip(),
                }
            ]
        m = _WRITTEN_INBOX_RE.match(line)
        if m:
            routing = "vault" if m.group(2) == "vault-empfohlen" else "inbox"
            return prefix + [{"type": "note_written", "path": m.group(1).strip(), "routing": routing}]

        # 4) Pro-Note-Marker (eingerueckt).
        m = _NOTE_RE.match(line)
        if m:
            return prefix + [
                {
                    "type": "note_progress",
                    "index": int(m.group(1)),
                    "total": int(m.group(2)),
                    "title": m.group(3).strip(),
                }
            ]

        # 5) Stage-Marker (nicht eingerueckt).
        m = _STAGE_RE.match(line)
        if m:
            num = int(float(m.group(1).split("-")[0]))  # "4.5"/"4-5" → 4
            return prefix + [{"type": "stage", "num": num, "total": int(m.group(2)), "label": m.group(3).strip()}]

        # 6) Abschluss.
        m = _DONE_RE.match(line)
        if m:
            return prefix + [{"type": "done", "written": int(m.group(1)), "dry_run": m.group(2) == "(dry-run)"}]

        # 7) Bekannte Backend-/Setup-Fehlersignaturen prominent als error_hint
        # hochziehen (sonst gehen sie in hunderten Log-Zeilen unter) — zusätzlich
        # zum normalen Log.
        low = line.lower()
        if any(sig in low for sig in _ERROR_SIGNATURES):
            return prefix + [{"type": "error_hint", "text": line.strip()}, {"type": "log", "text": line}]

        # 8) Sonst: roher Log.
        return prefix + [{"type": "log", "text": line}]

    def flush(self) -> list[dict]:
        """Am Stream-Ende: noch offenen Preview-Block ausgeben."""
        return self._flush_pending()
