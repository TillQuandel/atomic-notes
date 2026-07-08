"""Zweisprachiger Runtime-UX-String-Katalog (#157).

Menschgerichtete CLI-/doctor-/Fehler-Texte in EN (Default) und DE, umschaltbar
per ``ATOMIC_AGENT_UI_LANGUAGE`` (``en``|``de``; unbekannte Werte → ``en``).

Bewusst schlicht: ein flaches dict ``{key: {"en": str, "de": str}}`` plus
``msg(key, **fmt)``. Kein gettext/babel, keine .po-Dateien.

NICHT hier lokalisiert (Parser-Vertrag, ``gui/run_parser.py``): die von der GUI
geparsten stdout-Marker (Stage-Banner ``[n/7]``, ``=== Fertig:``, ``note_written``,
``run_summary`` …) sowie die Lauf-Logs bleiben sprachinvariant. ``msg`` bedient nur
argparse-Hilfen, doctor-Checks und error_hints.
"""

from __future__ import annotations

import os

_ENV = "ATOMIC_AGENT_UI_LANGUAGE"
_DEFAULT = "en"
_SUPPORTED = ("en", "de")

# Sprachinvarianter doctor-Verweis (Kommando-Name, kein zu übersetzender Text).
DOCTOR_POINTER = "→ atomic-notes doctor"


def lang() -> str:
    """Aktive UI-Sprache aus der Umgebung (Laufzeit gelesen). Unbekannt → ``en``."""
    value = os.environ.get(_ENV, _DEFAULT).strip().lower()
    return value if value in _SUPPORTED else _DEFAULT


def msg(key: str, **fmt: object) -> str:
    """Lokalisierten String zu ``key`` liefern (aktive Sprache, Fallback ``en``).

    KeyError bei unbekanntem ``key`` (fail-fast). Platzhalter via ``str.format``.
    """
    entry = STRINGS[key]  # KeyError = unbekannter Key (fail-fast)
    template = entry.get(lang()) or entry[_DEFAULT]
    return template.format(**fmt) if fmt else template


# ---------------------------------------------------------------------------
# Katalog. Konvention der Keys: <bereich>.<name>. Platzhalter in {name}-Form.
# ---------------------------------------------------------------------------
STRINGS: dict[str, dict[str, str]] = {
    # --- cli.py: Top-Level-Entry-Point -------------------------------------
    "cli.usage": {
        "en": (
            "atomic-notes — PDF → verified atomic notes for Obsidian\n"
            "\n"
            "Usage:\n"
            "  atomic-notes run --source <pdf> [orchestrator flags]   start the pipeline\n"
            "  atomic-notes run --help                                all pipeline flags\n"
            "  atomic-notes doctor                                    preflight checks\n"
            "  atomic-notes gui [--port N] [--no-browser]             local web GUI\n"
            "\n"
            "Environment:\n"
            "  ATOMIC_AGENT_BACKEND        LLM backend: 'subscription' (default, Claude-Code\n"
            "                              CLI, no API key) or 'litellm' (Anthropic/OpenAI via\n"
            "                              API key; local Ollama via OLLAMA_API_BASE).\n"
            "                              Check the setup with: atomic-notes doctor\n"
            "  ATOMIC_AGENT_UI_LANGUAGE    UI language: 'en' (default) or 'de'.\n"
        ),
        "de": (
            "atomic-notes — PDF → geprüfte Atomic Notes für Obsidian\n"
            "\n"
            "Verwendung:\n"
            "  atomic-notes run --source <pdf> [Orchestrator-Flags]   Pipeline starten\n"
            "  atomic-notes run --help                                alle Pipeline-Flags\n"
            "  atomic-notes doctor                                    Preflight-Checks\n"
            "  atomic-notes gui [--port N] [--no-browser]             lokale Web-GUI\n"
            "\n"
            "Umgebung:\n"
            "  ATOMIC_AGENT_BACKEND        LLM-Backend: 'subscription' (Default, Claude-Code-CLI,\n"
            "                              kein API-Key) oder 'litellm' (Anthropic/OpenAI per\n"
            "                              API-Key; lokales Ollama via OLLAMA_API_BASE).\n"
            "                              Setup prüfen mit: atomic-notes doctor\n"
            "  ATOMIC_AGENT_UI_LANGUAGE    UI-Sprache: 'en' (Default) oder 'de'.\n"
        ),
    },
    "cli.port_expects_portnumber": {
        "en": "--port expects a port number",
        "de": "--port erwartet eine Portnummer",
    },
    "cli.port_expects_number": {
        "en": "--port expects a number, not '{raw}'",
        "de": "--port erwartet eine Zahl, nicht '{raw}'",
    },
    "cli.port_out_of_range": {
        "en": "--port out of range 1–65535: {port}",
        "de": "--port außerhalb 1–65535: {port}",
    },
    "cli.invalid_argument": {
        "en": "Invalid argument: {exc}",
        "de": "Ungültiges Argument: {exc}",
    },
    "cli.gui_deps_missing": {
        "en": "GUI dependencies missing. Install with: pip install -e '.[gui]'",
        "de": "GUI-Dependencies fehlen. Installation: pip install -e '.[gui]'",
    },
    "cli.unknown_command": {
        "en": "Unknown command: {cmd}",
        "de": "Unbekanntes Kommando: {cmd}",
    },
    # --- orchestrator.py: run-Parser (argparse) ----------------------------
    "orch.description": {
        "en": "Atomic Note Multi-Agent Pipeline",
        "de": "Atomic-Note-Multi-Agent-Pipeline",
    },
    "orch.arg.source": {
        "en": "Path to the PDF file",
        "de": "Pfad zur PDF-Datei",
    },
    "orch.arg.doi": {
        "en": "DOI for the quality check (optional)",
        "de": "DOI für Qualitäts-Check (optional)",
    },
    "orch.arg.dry_run": {
        "en": "Do not write to the vault",
        "de": "Kein Schreiben in Vault",
    },
    "orch.arg.by_chapter": {
        "en": "Run planner and extractor chapter by chapter (for large books)",
        "de": "Planner und Extractor kapitelweise ausführen (für große Bücher)",
    },
    "orch.arg.no_llm": {
        "en": (
            "Stage-6 agents (verifier/crossref/critic) without an LLM — "
            "FOSS alternatives (BM25, embeddings, regex). "
            "Extractor and planner still run with an LLM."
        ),
        "de": (
            "Stage-6-Agents (Verifier/CrossRef/Critic) ohne LLM — "
            "FOSS-Alternativen (BM25, Embeddings, Regex). "
            "Extractor + Planner laufen weiterhin mit LLM."
        ),
    },
    "orch.arg.target_tag": {
        "en": (
            "Tag hint for auto-note-mover routing out of 00-inbox/. "
            "Appended to every note in addition to inferred tags. "
            "Mapping in CLAUDE.md (e.g. 'job', 'bike', 'private/fitness', "
            "'bachelorarbeit'). Without --target-tag notes stay in the inbox "
            "when tag inference yields no routing tag."
        ),
        "de": (
            "Tag-Hint für Auto-Note-Mover-Routing aus 00-inbox/. "
            "Wird allen Notes zusätzlich zu inferierten Tags angehängt. "
            "Mapping in CLAUDE.md (z.B. 'job', 'bike', 'private/fitness', "
            "'bachelorarbeit'). Ohne --target-tag bleiben Notes in Inbox "
            "wenn Tag-Inferenz keinen Routing-Tag liefert."
        ),
    },
    "orch.arg.llm_fallback": {
        "en": "Use an LLM (Haiku) for PDF enrichment when CrossRef finds nothing",
        "de": "LLM (Haiku) für PDF-Enrichment nutzen wenn CrossRef nichts findet",
    },
    "orch.arg.fresh_run": {
        "en": (
            "Set the LLM cache namespace to the current run ID — "
            "no cache hit from earlier runs. Needed for model comparisons "
            "and real quality measurements. Retries within the run "
            "stay cached."
        ),
        "de": (
            "LLM-Cache-Namespace auf aktuelle Run-ID setzen — "
            "kein Cache-Hit aus früheren Runs. Nötig für Modell-Vergleiche "
            "und echte Qualitäts-Messungen. Retries innerhalb des Runs "
            "bleiben gecacht."
        ),
    },
    "orch.arg.save_drafts": {
        "en": "Save drafts after stage 5 (extractor) as JSON — for the A/B comparison LLM vs. no-LLM in stage 6.",
        "de": "Drafts nach Stage 5 (Extractor) als JSON speichern — für A/B-Vergleich LLM vs. no-LLM in Stage 6.",
    },
    "orch.arg.load_drafts": {
        "en": (
            "Load drafts from --save-drafts and skip stages 1–5. --source is then ignored (the source is in the state)."
        ),
        "de": (
            "Drafts aus --save-drafts laden und Stage 1–5 überspringen. "
            "--source wird dann ignoriert (Quelle steht im State)."
        ),
    },
    "orch.arg.inbox_dir": {
        "en": (
            "Target folder instead of 00-inbox/ (created if missing). "
            "Useful for A/B comparisons: --inbox-dir 00-inbox/ab-llm/"
        ),
        "de": (
            "Zielordner statt 00-inbox/ (wird erstellt falls nicht vorhanden). "
            "Nützlich für A/B-Vergleiche: --inbox-dir 00-inbox/ab-llm/"
        ),
    },
    "orch.arg.export_format": {
        "en": (
            "Comma list of additional export formats (output project F4): core set "
            "json, obsidian-md, portable-md, docx, pdf, html — odt/epub can be enabled. "
            "Planned, not-yet-enabled formats: {future}. "
            "Without this flag no additional export (unchanged behavior)."
        ),
        "de": (
            "Kommaliste zusätzlicher Export-Formate (Output-Projekt F4): Kern-Set "
            "json, obsidian-md, portable-md, docx, pdf, html — odt/epub zuschaltbar. "
            "Geplante, noch nicht aktivierbare Formate: {future}. "
            "Ohne dieses Flag kein zusätzlicher Export (unverändertes Verhalten)."
        ),
    },
    "orch.arg.export_dir": {
        "en": "Target folder for --export-format (default: generative/.cache/exports/<pdf-stem>/).",
        "de": "Zielordner für --export-format (Default: generative/.cache/exports/<pdf-stem>/).",
    },
    "orch.error.source_required": {
        "en": "--source is required (except with --load-drafts)",
        "de": "--source ist erforderlich (außer mit --load-drafts)",
    },
    # --- doctor.py: Preflight-Checks ---------------------------------------
    "doctor.poppler_hint": {
        "en": (
            "install poppler: Ubuntu/Debian `sudo apt install poppler-utils`, "
            "Windows `choco install poppler` (or scoop), macOS `brew install poppler` — "
            "then open a new shell so PATH takes effect."
        ),
        "de": (
            "poppler installieren: Ubuntu/Debian `sudo apt install poppler-utils`, "
            "Windows `choco install poppler` (oder scoop), macOS `brew install poppler` — "
            "danach neue Shell öffnen, damit PATH greift."
        ),
    },
    "doctor.tool_missing_detail": {
        "en": "{tool} not in PATH",
        "de": "{tool} nicht im PATH",
    },
    "doctor.subscription_cli_missing_detail": {
        "en": "claude CLI '{bin}' not in PATH",
        "de": "claude-CLI '{bin}' nicht im PATH",
    },
    "doctor.subscription_cli_missing_hint": {
        "en": (
            "install the Claude-Code CLI: `npm install -g @anthropic-ai/claude-code`, "
            "then start `claude` once and log in (Pro/Max plan). "
            "Alternative without the CLI: ATOMIC_AGENT_BACKEND=litellm + API key."
        ),
        "de": (
            "Claude-Code-CLI installieren: `npm install -g @anthropic-ai/claude-code`, "
            "danach einmal `claude` starten und einloggen (Pro/Max-Abo). "
            "Alternative ohne CLI: ATOMIC_AGENT_BACKEND=litellm + API-Key."
        ),
    },
    "doctor.subscription_not_logged_in_detail": {
        "en": "claude CLI found ({cli}), but {credentials} is missing",
        "de": "claude-CLI gefunden ({cli}), aber {credentials} fehlt",
    },
    "doctor.subscription_not_logged_in_hint": {
        "en": (
            "probably not logged in: start `claude` once and complete the "
            "login. (Heuristic — if you authenticated differently, "
            "double-check this with a real run.)"
        ),
        "de": (
            "Vermutlich nicht eingeloggt: einmal `claude` starten und den "
            "Login durchlaufen. (Heuristik — falls anders authentifiziert, "
            "Check mit einem echten Lauf gegenprüfen.)"
        ),
    },
    "doctor.subscription_ok_detail": {
        "en": "claude CLI: {cli}, credentials file present (login not live-verified)",
        "de": "claude-CLI: {cli}, Credentials-Datei vorhanden (Login nicht live verifiziert)",
    },
    "doctor.litellm_no_key_detail": {
        "en": "no provider key in the environment",
        "de": "kein Provider-Key in der Umgebung",
    },
    "doctor.litellm_no_key_hint": {
        "en": (
            "set an API key, e.g. ANTHROPIC_API_KEY or OPENAI_API_KEY (in .env or the environment). Checked: {vars}"
        ),
        "de": ("API-Key setzen, z. B. ANTHROPIC_API_KEY oder OPENAI_API_KEY (in .env oder Umgebung). Geprüft: {vars}"),
    },
    "doctor.litellm_ok_detail": {
        "en": "set: {vars}",
        "de": "gesetzt: {vars}",
    },
    "doctor.backend_unknown_detail": {
        "en": "unknown backend value",
        "de": "unbekannter Backend-Wert",
    },
    "doctor.backend_unknown_hint": {
        "en": "set ATOMIC_AGENT_BACKEND to 'subscription' (default) or 'litellm'.",
        "de": "ATOMIC_AGENT_BACKEND auf 'subscription' (Default) oder 'litellm' setzen.",
    },
    "doctor.vault_missing_detail": {
        "en": "vault path does not exist: {vault}",
        "de": "Vault-Pfad existiert nicht: {vault}",
    },
    "doctor.vault_missing_hint": {
        "en": (
            "set ATOMIC_AGENT_VAULT_PATH to the Obsidian vault (or any target folder) — in .env or the environment."
        ),
        "de": (
            "ATOMIC_AGENT_VAULT_PATH auf den Obsidian-Vault (oder einen "
            "beliebigen Zielordner) setzen — in .env oder Umgebung."
        ),
    },
    "doctor.vault_not_writable_detail": {
        "en": "vault not writable: {vault} ({err})",
        "de": "Vault nicht beschreibbar: {vault} ({err})",
    },
    "doctor.vault_not_writable_hint": {
        "en": "check the folder's write permissions.",
        "de": "Schreibrechte des Ordners prüfen.",
    },
    "doctor.import_installed_detail": {
        "en": "{module} installed",
        "de": "{module} installiert",
    },
    "doctor.import_missing_detail": {
        "en": "{module} missing",
        "de": "{module} fehlt",
    },
    "doctor.hint_pypdf": {
        "en": "pip install pypdf (PDF metadata enrichment)",
        "de": "pip install pypdf (PDF-Metadaten-Enrichment)",
    },
    "doctor.hint_sentence_transformers": {
        "en": "pip install sentence-transformers (embeddings/entity resolution)",
        "de": "pip install sentence-transformers (Embeddings/Entity-Resolution)",
    },
    "doctor.export_hint": {
        "en": (
            'pip install "atomic-notes[export]"  (export formats docx/pdf/html are '
            "optional; the core pipeline runs without them)"
        ),
        "de": (
            'pip install "atomic-notes[export]"  (Export-Formate docx/pdf/html sind optional; Kern-Pipeline läuft ohne)'
        ),
    },
    "doctor.mark_ok": {"en": "OK ", "de": "OK "},
    "doctor.mark_required_fail": {"en": "FAIL ", "de": "FEHLT"},
    "doctor.mark_warn": {"en": "WARN", "de": "WARN"},
    "doctor.summary_failed": {
        "en": "doctor: {failed} of {total} checks failed.",
        "de": "doctor: {failed} von {total} Checks fehlgeschlagen.",
    },
    "doctor.summary_ok": {
        "en": "doctor: all required checks passed{suffix}.",
        "de": "doctor: alle erforderlichen Checks ok{suffix}.",
    },
    "doctor.summary_warn_suffix": {
        "en": " ({n} warning(s) on optional checks)",
        "de": " ({n} Warnung(en) bei optionalen Checks)",
    },
    # --- error_hints.py: handlungsanleitende Fehlermeldungen ---------------
    "error.scanned_pdf.problem_thin": {
        "en": (
            "contains barely any extractable text (only ~{wpp:.0f} words/page) — "
            "probably a scanned or poorly extractable PDF"
        ),
        "de": (
            "enthält kaum extrahierbaren Text (nur ~{wpp:.0f} Wörter/Seite) — "
            "vermutlich ein gescanntes oder schlecht extrahierbares PDF"
        ),
    },
    "error.scanned_pdf.problem_empty": {
        "en": "contains no extractable text — probably a scanned PDF",
        "de": "enthält keinen extrahierbaren Text — vermutlich ein gescanntes PDF",
    },
    "error.scanned_pdf.body": {
        "en": (
            "  [Warning] '{pdf}' {problem}. The pipeline needs text and otherwise "
            "produces empty/thin notes.\n"
            "  Next step: run OCR into a new file, e.g. "
            "`ocrmypdf '{pdf}' '{out}'`, then start again with '{out}'."
        ),
        "de": (
            "  [Warnung] '{pdf}' {problem}. Die Pipeline braucht Text und liefert "
            "sonst leere/dünne Notes.\n"
            "  Nächster Schritt: OCR in eine neue Datei ausführen, z. B. "
            "`ocrmypdf '{pdf}' '{out}'`, dann mit '{out}' erneut starten."
        ),
    },
    "error.pdftotext.no_stderr": {
        "en": "<no stderr output>",
        "de": "<kein stderr ausgegeben>",
    },
    "error.pdftotext.body": {
        "en": (
            "pdftotext could not read the PDF.\n"
            "  Check that poppler-utils is installed and the PDF is not corrupted "
            "({doctor}).\n"
            "  Original error: {detail}"
        ),
        "de": (
            "pdftotext konnte das PDF nicht lesen.\n"
            "  Prüfe, ob poppler-utils installiert und das PDF nicht beschädigt ist "
            "({doctor}).\n"
            "  Original-Fehler: {detail}"
        ),
    },
    "error.pdftotext.encrypted": {
        "en": (
            "\n  The PDF is password-protected/encrypted — remove the protection, "
            "e.g. `qpdf --decrypt 'input.pdf' 'output.pdf'`, then start again."
        ),
        "de": (
            "\n  Das PDF ist passwortgeschützt/verschlüsselt — Schutz entfernen, "
            "z. B. `qpdf --decrypt 'input.pdf' 'output.pdf'`, dann erneut starten."
        ),
    },
    "error.litellm.base": {
        "en": "litellm backend error ({agent}/{model}): {detail}",
        "de": "litellm-Backend-Fehler ({agent}/{model}): {detail}",
    },
    "error.litellm.auth": {
        "en": (
            "\n  Looks like a problem with the API key/backend. Check the "
            "provider key (e.g. ANTHROPIC_API_KEY/OPENAI_API_KEY) and "
            "ATOMIC_AGENT_BACKEND ({doctor})."
        ),
        "de": (
            "\n  Sieht nach einem Problem mit dem API-Key/Backend aus. Prüfe den "
            "Provider-Key (z. B. ANTHROPIC_API_KEY/OPENAI_API_KEY) und "
            "ATOMIC_AGENT_BACKEND ({doctor})."
        ),
    },
    "error.litellm.generic": {
        "en": "\n  Check backend/network/model name ({doctor}).",
        "de": "\n  Prüfe Backend/Netzwerk/Modellnamen ({doctor}).",
    },
}
