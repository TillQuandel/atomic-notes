"""FastAPI-App fuer die Live-GUI der Atomic-Agent-Pipeline.

Eine eigenstaendige, lokale Web-GUI (neben dem read-only Eval-Dashboard):
PDF waehlen -> Lauf starten -> Live-Fortschritt pro Pipeline-Stufe streamen
(SSE) -> im Dry-Run die erzeugten Notes mit Confidence/Score als Preview zeigen.

Stack (lt. Plan „atomic-notes Frontend-Stack-Entscheidung"): FastAPI + HTMX/SSE
+ vanilla CSS, kein React/npm. Der eigentliche Lauf laeuft als Subprocess
(generative/gui/runner.py); diese App orchestriert nur Start + Event-Stream.
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
import zipfile
from collections.abc import Iterator, Callable
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from generative.gui import gui_settings, run_history, runner

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"

# P4 (Run-Historie): Default-Ablageort fuer Lauf-Records — Modul-Konstante
# (statt Inline-Default in create_app), damit Tests sie isoliert per
# monkeypatch auf ein tmp_path umbiegen koennen (s. tests/conftest.py), ohne
# jeden bestehenden create_app(...)-Aufruf einzeln anzufassen.
_DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / ".cache" / "gui" / "runs"

# P4: mehr als so viele Records im runs_dir -> aelteste werden geloescht.
_MAX_RUN_RECORDS = 50

# P2 (Einstellungs-Defaults): GUI-eigene Settings-Datei, analog _DEFAULT_RUNS_DIR
# -- Modul-Konstante, damit Tests sie per monkeypatch isolieren koennen (s.
# tests/conftest.py) und create_app(settings_path=...) sie injizieren kann.
_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / ".cache" / "gui" / "settings.json"

# Endungen, die als „PDF-Kandidat" gelistet werden.
_PDF_GLOB = "*.pdf"

# S1: Host-Header-Allowlist gegen DNS-Rebinding — kombiniert mit
# `/api/outputs/file` (liefert jede `.md` im Vault) waere ein fehlender Check
# ein Exfiltrationspfad. Nur lokale Hosts erlaubt (Server bindet ohnehin nur an
# 127.0.0.1).
_DEFAULT_ALLOWED_HOSTS = ["127.0.0.1", "localhost", "::1"]

# Same-Origin-Hosts: Die GUI bindet nur an 127.0.0.1. Ein Browser sendet bei
# Cross-Origin-POSTs einen `Origin`-Header — fehlt er (curl/TestClient/Beacon
# same-origin), ist es kein CSRF-Vektor.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# P1 (Lauf-Einstellungen): Whitelist der `POST /api/run`-`options`. Backend-/
# Profil-Wertepruefung teilt sich `gui_settings.validate_backend/validate_profile`
# (P2 hat dieselbe Pruefung fuer `PUT /api/settings` -- ausfaktoriert statt
# dupliziert, s. gui_settings.py). `inbox_dir` (B3): freier Export-Ordner statt
# Vault-Inbox -- nur im Schreib-Modus wirksam (s. `start_run`/`build_run_spec`).
_OPTION_KEYS = frozenset({"backend", "profile", "no_llm", "inbox_dir"})


def _validate_run_options(options) -> tuple[dict, str | None]:
    """Normalisiert+prueft `options` aus `POST /api/run`.

    Rueckgabe: (normalisierte_optionen, fehlermeldung). Fehlt `options` oder ist
    es leer, ist das Ergebnis `({}, None)` — identisch zum heutigen Verhalten
    (kein Env-Override, kein `--no-llm`).
    """
    if not options:
        return {}, None
    if not isinstance(options, dict):
        return {}, "options muss ein Objekt sein."
    unknown = set(options) - _OPTION_KEYS
    if unknown:
        return {}, f"Unbekannte Option(en): {', '.join(sorted(unknown))}"

    normalized: dict = {}
    backend, error = gui_settings.validate_backend(options.get("backend"))
    if error:
        return {}, error
    if backend is not None:
        normalized["backend"] = backend

    profile, error = gui_settings.validate_profile(options.get("profile"))
    if error:
        return {}, error
    if profile is not None:
        normalized["profile"] = profile

    no_llm = options.get("no_llm")
    if no_llm is not None:
        if not isinstance(no_llm, bool):
            return {}, "no_llm muss ein Boolean sein."
        if no_llm:
            normalized["no_llm"] = True

    # B3 (Output-Ziel waehlbar): leerer String = kein Export-Wunsch (wie bei
    # backend/profile) -- Existenz-/Verzeichnis-Pruefung passiert serverseitig
    # in `start_run` (nur dort ist bekannt, ob es ein Schreib-Lauf ist).
    inbox_dir = options.get("inbox_dir")
    if inbox_dir is not None:
        if not isinstance(inbox_dir, str):
            return {}, "inbox_dir muss ein String sein."
        if inbox_dir.strip():
            normalized["inbox_dir"] = inbox_dir

    return normalized, None


def _is_same_origin(request: Request) -> bool:
    """CSRF-Schutz: Cross-Origin-Browser-Requests an mutierende Endpunkte abweisen."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    from urllib.parse import urlparse

    return urlparse(origin).hostname in _LOCAL_HOSTS


def _active_vault(session: "RunSession | None", state_vault_path: Path) -> Path:
    """R1 (Race, KRITISCH, B2): die session-gebundenen Output-Endpunkte
    (`/api/outputs`, `/api/outputs/file`, `/api/outputs/archive`) validieren
    gegen den Vault-SNAPSHOT des (letzten) Laufs (`session.vault_path`,
    gesetzt bei Lauf-Start in `start_run`), NICHT gegen den moeglicherweise
    zwischenzeitlich per `PUT /api/vault` gewechselten globalen
    `app.state.vault_path`. Sonst wuerde ein Download gegen Vault B validieren,
    waehrend der Lauf tatsaechlich in Vault A geschrieben hat. Ohne Session
    (noch nie gelaufen): Fallback auf den aktuellen State-Vault."""
    if session is not None and getattr(session, "vault_path", None) is not None:
        return session.vault_path
    return state_vault_path


def _active_export_dir(session: "RunSession | None") -> Path | None:
    """B3 (analog `_active_vault`): der Export-Ordner-Snapshot DIESES Laufs --
    es gibt (anders als beim Vault) keinen persistenten globalen State dafuer,
    nur den per-Lauf-Snapshot in `session.export_dir`. Ohne Session (noch nie
    gelaufen) oder Vault-Inbox-Modus: `None` (keine zusaetzliche erlaubte
    Wurzel, unveraendertes Verhalten vor B3)."""
    if session is not None:
        return getattr(session, "export_dir", None)
    return None


def _is_within(path: str, root: Path) -> bool:
    """True, wenn `path` (aufgelöst) innerhalb von `root` liegt."""
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _output_items_from_events(
    events: list[dict],
    *,
    pdf: str | None,
    vault_path: Path,
    preview_root: Path,
    export_dir: Path | None = None,
) -> list[dict]:
    """Aggregiert die Ergebnisliste (`GET /api/outputs`) aus den Events der
    aktuellen/letzten RunSession.

    Dry-Run: aus `preview`-Events (Name/Routing/Score/Confidence/Flags); der
    Download-Pfad ist die eval-Kopie unter `preview_root/<pdf-stem>/…`, sofern
    sie existiert (kein Erfinden — fehlt sie, bleibt `path` weg, L5).
    Schreib-Lauf: aus `note_written`-Events (vault_writer-stdout) — dort gibt es
    kein Score/Confidence, die druckt vault_writer nur im Dry-Run.

    `export_dir` (B3): wurde der Lauf mit einem freien Export-Ordner statt der
    Vault-Inbox gestartet, druckt `vault_writer._display` einen ABSOLUTEN Pfad
    (er liegt nicht unter `VAULT`, s. `orchestrator.py`/`vault_writer.py`
    B3-Vorbedingungen) — `candidate` (unten) ist dann bereits dieser absolute
    Pfad (ein absoluter RHS-Operand bei `Path.__truediv__` verdraengt `vault_root`).
    Ohne den zusaetzlichen Export-Ordner-Check faellt so ein Pfad hier weg.
    """
    items: list[dict] = []
    stem = Path(pdf).stem if pdf else ""
    vault_root = Path(vault_path).resolve()
    preview_base = Path(preview_root).resolve()
    export_base = Path(export_dir).resolve() if export_dir is not None else None
    for ev in events:
        if ev.get("type") == "preview":
            item: dict = {"title": ev["name"], "routing": ev["routing"]}
            if ev.get("score") is not None:
                item["score"] = ev["score"]
            if ev.get("confidence") is not None:
                item["confidence"] = ev["confidence"]
            if ev.get("merge_target"):
                item["merge_target"] = ev["merge_target"]
            if ev.get("flags"):
                item["flags"] = ev["flags"]
            if stem:
                candidate = (preview_base / stem / f"{ev['routing']}__{ev['name']}").resolve()
                if candidate.is_relative_to(preview_base) and candidate.exists():
                    item["path"] = str(candidate)
            items.append(item)
        elif ev.get("type") == "note_written":
            item = {"title": Path(ev["path"]).name, "routing": ev["routing"]}
            if ev.get("merge_target"):
                item["merge_target"] = ev["merge_target"]
            candidate = (vault_root / ev["path"]).resolve()
            if candidate.is_relative_to(vault_root):
                item["path"] = str(candidate)
            elif export_base is not None and candidate.is_relative_to(export_base):
                item["path"] = str(candidate)
            items.append(item)
    return items


def _archive_filename(pdf: str | None) -> str:
    """ZIP-Dateiname fuer `GET /api/outputs/archive` aus dem PDF-Stem der
    aktuellen Session (C1). Ohne Session/PDF: Fallback `"outputs.zip"`."""
    if not pdf:
        return "outputs.zip"
    stem = Path(pdf).stem
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", stem)
    return f"{safe}-outputs.zip"


def _run_summary_from_events(events: list[dict]) -> dict:
    """Extrahiert das `run_summary`-Event (P5: Final-Report Zeit/Tokens) fuer den
    Historie-Record. Hoechstens ein solches Event pro Lauf (s. run_parser.py).
    Fehlt es (z.B. Crash vor dem Final-Report), ist das Ergebnis `{}` — der
    Aufrufer laesst die Record-Felder dann schlicht weg (kein Erfinden, L5).
    """
    for ev in events:
        if ev.get("type") == "run_summary":
            result: dict = {}
            if "duration_s" in ev:
                result["duration_s"] = ev["duration_s"]
            if ev.get("tokens"):
                result["tokens"] = ev["tokens"]
            return result
    return {}


def _validate_output_path(
    path: str, *, vault_path: Path, preview_root: Path, export_dir: Path | None = None
) -> Path | None:
    """Pfad-Whitelist (L4) fuer `/api/outputs/file` + `/api/outputs/archive`:
    nur `.md`-Dateien unterhalb `vault_path`, oder beliebige Dateien unterhalb
    `preview_root` (die eval-Kopien der Dry-Run-Vorschau, bereits auf `.md`
    beschraenkt), oder `.md`-Dateien unterhalb `export_dir` (B3: der zur
    Laufzeit gewaehlte freie Export-Ordner -- gleiche `.md`-Beschraenkung wie
    beim Vault, da der Export-Ordner ebenso ein beliebiger lokaler Ordner ist).
    `resolve()` neutralisiert Symlink-Escapes. Alles andere: `None` -> Aufrufer
    antwortet 403.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, ValueError):
        return None
    vault_root = Path(vault_path).resolve()
    preview_base = Path(preview_root).resolve()
    if resolved.is_relative_to(vault_root) and resolved.suffix == ".md":
        return resolved
    if resolved.is_relative_to(preview_base):
        return resolved
    if export_dir is not None:
        export_base = Path(export_dir).resolve()
        if resolved.is_relative_to(export_base) and resolved.suffix == ".md":
            return resolved
    return None


class RunSession:
    """Ein laufender (oder abgeschlossener) Pipeline-Lauf. Single-Run zur Zeit."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.finished = False
        self.cancelled = False
        self.pdf: str | None = None
        self.dry_run: bool | None = None
        self.options: dict = {}
        self._proc = None  # vom Runner registriertes Popen-Handle (für Cancel)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.started_at: float | None = None
        # P4 (Run-Historie): vom Aufrufer (start_run-Handler) gesetzter Kontext
        # fuer den Record-Write am Lauf-Ende. `runs_dir=None` deaktiviert das
        # Schreiben (z.B. wenn RunSession direkt ohne GUI-Kontext genutzt wird).
        self.vault_path: Path | None = None
        # B3 (Output-Ziel waehlbar): Snapshot des per `options.inbox_dir`
        # gewaehlten Export-Ordners fuer DIESEN Lauf (Schreib-Modus) -- analog
        # `vault_path` oben. `None` = Vault-Inbox (Default) oder Dry-Run
        # (inbox_dir wird dort ignoriert, s. `start_run`).
        self.export_dir: Path | None = None
        self.preview_root: Path | None = None
        self.runs_dir: Path | None = None
        self.clock: Callable[[], float] = time.time

    def register_proc(self, proc) -> None:
        """Vom Runner aufgerufen, sobald der Subprocess läuft — ermöglicht Cancel."""
        with self._lock:
            self._proc = proc

    def start(self, run_iter: Iterator[dict]) -> None:
        self._thread = threading.Thread(target=self._consume, args=(run_iter,), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Laufenden Subprocess beenden (Stop-Button / Tab-Close). Best-effort."""
        self.cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # pragma: no cover - Prozess schon weg
                pass

    def _consume(self, run_iter: Iterator[dict]) -> None:
        self.started_at = self.clock()
        try:
            for ev in run_iter:
                if ev.get("type") in ("exited", "error"):
                    # Terminal-Event: Historie-Record SYNCHRON schreiben,
                    # BEVOR das Event in `self.events` sichtbar wird. Sonst
                    # kann ein Beobachter (SSE-Client via `_event_stream`,
                    # der auf genau dieses Event wartet und dann sofort
                    # zurueckkehrt) `/api/runs` abfragen, bevor der Record auf
                    # Platte liegt — Race zwischen zwei Threads, `finished`
                    # allein reicht nicht (der SSE-Stream endet bereits beim
                    # Lesen des Terminal-Events, nicht erst bei `finished`).
                    self._write_history_record(ev)
                with self._lock:
                    self.events.append(ev)
        except Exception as exc:  # Lauf-Crash als error-Event sichtbar machen
            err_ev = {"type": "error", "message": str(exc)}
            self._write_history_record(err_ev)
            with self._lock:
                self.events.append(err_ev)
        finally:
            with self._lock:
                self.finished = True

    def _write_history_record(self, terminal_ev: dict) -> None:
        if self.runs_dir is None:
            return
        finished_at = self.clock()
        rc = terminal_ev.get("returncode") if terminal_ev.get("type") == "exited" else None
        notes = _output_items_from_events(
            self.events,
            pdf=self.pdf,
            vault_path=self.vault_path,
            preview_root=self.preview_root,
            export_dir=self.export_dir,
        )
        summary = _run_summary_from_events(self.events)
        record = run_history.build_run_record(
            run_id=run_history.make_run_id(finished_at),
            started_at=self.started_at,
            finished_at=finished_at,
            source_pdf=self.pdf,
            dry_run=self.dry_run,
            options=self.options,
            rc=rc,
            notes=notes,
            duration_s=summary.get("duration_s"),
            tokens=summary.get("tokens"),
        )
        try:
            run_history.write_run_record(record, self.runs_dir)
            run_history.prune_old_records(self.runs_dir, keep=_MAX_RUN_RECORDS)
        except OSError as exc:  # Historie darf einen Lauf nie zum Absturz bringen (L5: sichtbar, nicht still)
            logger.warning("Konnte Run-Historie nicht schreiben: %s", exc)

    @property
    def active(self) -> bool:
        return not self.finished


def _default_run_factory(
    pdf: str,
    dry_run: bool,
    register=None,
    options: dict | None = None,
    vault_path: Path | None = None,
) -> Iterator[dict]:
    """`vault_path` (B2, Punkt 3): der Subprocess bekommt `ATOMIC_AGENT_VAULT_PATH`
    fuer den zur Laufzeit gewaehlten Vault. `create_app` reicht hier bei jedem
    Lauf-Start den AKTUELLEN `app.state.vault_path` durch (s. Wrapper-Closure
    unten) -- nicht den urspruenglichen `create_app(vault_path=...)`-Parameter."""
    argv, env_overrides = runner.build_run_spec(pdf, dry_run=dry_run, options=options, vault_path=vault_path)
    yield from runner.iter_run_events(argv, env=env_overrides, on_proc=register)


def create_app(
    *,
    run_factory: Callable[..., Iterator[dict]] | None = None,  # (pdf, dry_run, register, options)
    pdf_dirs: list[Path] | None = None,
    vault_path: Path | None = None,
    backend: str | None = None,
    uploads_dir: Path | None = None,
    doctor_fn: Callable[[], list] | None = None,
    litellm_check_fn: Callable[[], object] | None = None,
    preview_root: Path | None = None,
    runs_dir: Path | None = None,
    settings_path: Path | None = None,
    clock: Callable[[], float] | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    _injected_run_factory = run_factory
    if doctor_fn is None:
        from generative.doctor import run_all as doctor_fn
    if litellm_check_fn is None:
        from generative.doctor import check_backend as _check_backend

        litellm_check_fn = lambda: _check_backend("litellm")  # noqa: E731
    if preview_root is None:
        preview_root = Path(__file__).resolve().parents[1] / ".cache" / "eval" / "baseline"
    preview_root = Path(preview_root)
    if runs_dir is None:
        runs_dir = _DEFAULT_RUNS_DIR
    runs_dir = Path(runs_dir)
    if settings_path is None:
        settings_path = _DEFAULT_SETTINGS_PATH
    settings_path = Path(settings_path)
    clock = clock or time.time

    if pdf_dirs is None or vault_path is None or backend is None:
        from generative import config as _cfg

        if pdf_dirs is None:
            _repo = Path(__file__).resolve().parents[2]  # …/atomic-notes
            pdf_dirs = [_repo / "examples", getattr(_cfg, "LITERATURE_DIR", None)]
        if vault_path is None:
            vault_path = _cfg.VAULT
            # P6/B2: eine zuvor per `PUT /api/vault` gewaehlte + persistierte
            # Vault-Wahl ueberschreibt beim Server-Start den config.VAULT-Default
            # -- aber nur, wenn der gespeicherte Pfad noch existiert und ein
            # Verzeichnis ist (fail-open lesend, L5: ein kaputter/veralteter
            # Eintrag darf den Server-Start nicht verhindern).
            _stored_settings, _ = gui_settings.read_settings(settings_path)
            _stored_vault = _stored_settings.get("vault_path")
            if _stored_vault:
                _candidate = Path(_stored_vault)
                if _candidate.is_dir():
                    vault_path = _candidate
                else:
                    logger.warning(
                        "Gespeicherter vault_path existiert nicht (mehr) oder ist kein "
                        "Verzeichnis, verwende Default: %s",
                        _stored_vault,
                    )
        if backend is None:
            backend = _cfg.BACKEND
    pdf_dirs = [Path(d) for d in pdf_dirs if d]
    if uploads_dir is None:
        import tempfile

        uploads_dir = Path(tempfile.gettempdir()) / "atomic-notes-gui-uploads"
    uploads_dir = Path(uploads_dir)
    # #2: Lauf-Quellen auf gelistete PDF-Verzeichnisse + Upload-Ablage begrenzen —
    # ein existierender Pfad allein genügt nicht (sonst beliebige lokale Datei).
    _allowed_roots = [d.resolve() for d in pdf_dirs] + [uploads_dir.resolve()]

    app = FastAPI(title="atomic-notes GUI")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or _DEFAULT_ALLOWED_HOSTS)
    app.state.session = None
    app.state.session_lock = threading.Lock()
    # B2: mutabler Vault-State -- die nicht-session-gebundenen Pruefungen
    # (/api/run-Vault-Check, /api/doctor) lesen ab hier `app.state.vault_path`
    # statt der eingefrorenen Closure-Variable `vault_path`, damit ein Wechsel
    # per `PUT /api/vault` sofort greift. `.resolve()`: kanonischer absoluter
    # Pfad, konsistent mit der Validierung in `PUT /api/vault` weiter unten.
    app.state.vault_path = Path(vault_path).resolve()

    if _injected_run_factory is not None:
        run_factory = _injected_run_factory
    else:

        def run_factory(pdf: str, dry_run: bool, register=None, options: dict | None = None) -> Iterator[dict]:
            # B2 (Punkt 3, Subprocess-Override): `app.state.vault_path` wird
            # erst HIER beim tatsaechlichen Lauf-Start gelesen (nicht beim
            # create_app-Aufruf) -- kann also einen zwischenzeitlichen Wechsel
            # per `PUT /api/vault` mitnehmen.
            return _default_run_factory(
                pdf, dry_run, register=register, options=options, vault_path=app.state.vault_path
            )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/app.css")
    def css():
        return StreamingResponse(iter([(_STATIC / "app.css").read_bytes()]), media_type="text/css")

    @app.get("/app.js")
    def js():
        return StreamingResponse(iter([(_STATIC / "app.js").read_bytes()]), media_type="text/javascript")

    @app.get("/api/pdfs")
    def list_pdfs() -> JSONResponse:
        seen: dict[str, str] = {}
        for d in pdf_dirs:
            if d and d.exists():
                for p in sorted(d.glob(_PDF_GLOB)):
                    seen.setdefault(p.name, str(p))
        return JSONResponse({"pdfs": [{"name": n, "path": p} for n, p in seen.items()]})

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        """Per Drag-and-Drop/Dialog hochgeladenes PDF server-seitig ablegen.

        Der Originalname (Basename, ohne Traversal) bleibt erhalten — die
        Pipeline leitet Metadaten u.a. aus dem Dateinamen ab. `--source` fährt
        anschliessend gegen den zurückgegebenen Pfad.
        """
        if not _is_same_origin(request):
            return JSONResponse({"error": "Cross-Origin-Request abgelehnt."}, status_code=403)
        raw = file.filename or "upload.pdf"
        safe_name = Path(raw.replace("\\", "/")).name
        if not safe_name.lower().endswith(".pdf"):
            return JSONResponse({"error": "Nur PDF-Dateien werden akzeptiert."}, status_code=400)
        data = await file.read()
        if not data:
            return JSONResponse({"error": "Leere Datei."}, status_code=400)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target = (uploads_dir / safe_name).resolve()
        if not target.is_relative_to(uploads_dir.resolve()):
            return JSONResponse({"error": "ungültiger Dateiname"}, status_code=400)
        target.write_bytes(data)
        return JSONResponse({"name": safe_name, "path": str(target)})

    @app.get("/api/doctor")
    def doctor() -> JSONResponse:
        checks = []
        for chk in doctor_fn():
            checks.append(
                {
                    "name": getattr(chk, "name", "?"),
                    "ok": bool(getattr(chk, "ok", False)),
                    "detail": getattr(chk, "detail", ""),
                    "hint": getattr(chk, "hint", ""),
                    "required": bool(getattr(chk, "required", True)),
                }
            )
        # R2 (doctor, KRITISCH, B2): `doctor_fn()` (Default: `doctor.run_all()`)
        # enthaelt bereits einen "vault"-Check, der prueft aber INTERN
        # `config.VAULT` -- die Modul-Import-Konstante aus config.py, NICHT den
        # zur Laufzeit per `PUT /api/vault` gewechselten `app.state.vault_path`.
        # Nach einem Vault-Wechsel waere dieser eingebettete Check dauerhaft
        # stale (zeigt nie den neuen State-Vault) und wuerde `ok` fälschlich
        # blockieren/freigeben. Deshalb: aus der `ok`-Gating-Menge ausklammern
        # und durch eine eigene, frische Re-Validierung des TATSAECHLICH
        # gewaehlten Vaults ersetzen. Der alte Eintrag bleibt in `checks`
        # sichtbar (Transparenz/Debugging), zaehlt aber nicht mehr mit --
        # bewusst dokumentierte Grenze (L6): sein `detail`-Text kann irreführend
        # veraltet sein, solange `doctor_fn` selbst nicht B2-aware ist.
        checks_ok = all(c["ok"] for c in checks if c["required"] and c["name"] != "vault")
        # P1 Doctor-Gating: litellm-Verfuegbarkeit unabhaengig vom aktuell
        # konfigurierten Server-Backend pruefen (sonst faellt der Key-Check weg,
        # sobald der Server per Default auf "subscription" laeuft) — reine
        # Wiederverwendung von doctor.check_backend, keine neue Logik.
        litellm_check = litellm_check_fn()
        litellm_available = bool(getattr(litellm_check, "ok", False))
        current_vault = app.state.vault_path
        vault_exists = current_vault.is_dir()
        ok = checks_ok and vault_exists
        response = {
            "backend": backend,
            "vault": str(current_vault),
            "vault_exists": vault_exists,
            "ok": ok,
            "checks": checks,
            "litellm_available": litellm_available,
        }
        if not litellm_available:
            response["litellm_hint"] = getattr(litellm_check, "hint", "") or getattr(litellm_check, "detail", "")
        return JSONResponse(response)

    @app.post("/api/run")
    async def start_run(request: Request) -> JSONResponse:
        if not _is_same_origin(request):
            return JSONResponse({"error": "Cross-Origin-Request abgelehnt."}, status_code=403)
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "Ungültiger JSON-Body."}, status_code=400)
        # G1: gueltiges JSON, das kein Objekt ist (null/[]), liefe sonst in
        # body.get(...) -> AttributeError -> 500. Explizit als 400 abweisen.
        if not isinstance(body, dict):
            return JSONResponse({"error": "Ungültiger JSON-Body."}, status_code=400)
        pdf = body.get("pdf", "")
        dry_run = bool(body.get("dry_run", True))
        options, options_error = _validate_run_options(body.get("options"))
        if options_error:
            return JSONResponse({"error": options_error}, status_code=422)
        # B3 (Output-Ziel waehlbar): `inbox_dir` ist nur im SCHREIB-Modus
        # wirksam -- im Dry-Run wird er ignoriert (kein Fehler, kein Export-
        # Ordner-Snapshot, s. `_active_export_dir`/`RunSession.export_dir`).
        export_dir: Path | None = None
        if options.get("inbox_dir") and not dry_run:
            try:
                export_dir = Path(options["inbox_dir"]).resolve()
            except (OSError, ValueError):
                return JSONResponse({"error": f"Ungültiger Export-Ordner: {options['inbox_dir']}"}, status_code=400)
            if not export_dir.is_dir():
                return JSONResponse({"error": f"Export-Ordner nicht gefunden: {options['inbox_dir']}"}, status_code=400)
        if not pdf or not Path(pdf).exists():
            return JSONResponse({"error": f"PDF nicht gefunden: {pdf}"}, status_code=400)
        # #2: Quelle muss unter einem erlaubten Root liegen (gelistet/hochgeladen).
        if not any(_is_within(pdf, root) for root in _allowed_roots):
            return JSONResponse({"error": "PDF liegt ausserhalb der erlaubten Verzeichnisse."}, status_code=400)
        # Server-seitige Revalidierung (Client-Gate könnte umgangen/veraltet sein):
        # der Vault wird auch im Dry-Run gebraucht (Context-Builder scannt ihn).
        # B2: liest den GEWAEHLTEN Vault (`app.state.vault_path`), nicht die
        # eingefrorene Closure-Variable -- folgt also einem Wechsel per PUT /api/vault.
        if not app.state.vault_path.exists():
            return JSONResponse({"error": f"Vault nicht gefunden: {app.state.vault_path}"}, status_code=400)
        with app.state.session_lock:
            if app.state.session is not None and app.state.session.active:
                return JSONResponse({"error": "Es läuft bereits ein Pipeline-Lauf."}, status_code=409)
            session = RunSession()
            session.pdf = pdf
            session.dry_run = dry_run
            session.options = options
            # P4/B2 (R1): Vault-SNAPSHOT dieses Laufs -- Downloads/Ergebnislisten
            # validieren spaeter gegen GENAU diesen Wert (s. `_active_vault`),
            # auch wenn `app.state.vault_path` inzwischen per PUT /api/vault
            # weitergewechselt wurde.
            session.vault_path = app.state.vault_path
            session.export_dir = export_dir
            session.preview_root = preview_root
            session.runs_dir = runs_dir
            session.clock = clock
            # Iterator MIT der Proc-Registrierung der Session erzeugen → Cancel
            # kann den Subprocess später terminieren.
            run_iter = run_factory(pdf, dry_run, session.register_proc, options)
            app.state.session = session
            session.start(run_iter)
        return JSONResponse({"status": "started", "pdf": pdf, "dry_run": dry_run, "options": options})

    @app.get("/api/status")
    def status() -> JSONResponse:
        """Erlaubt einer frisch geladenen Seite, einen bereits laufenden Lauf zu
        erkennen und sich anzuhängen (Stop-Button + Stream-Reattach)."""
        s = app.state.session
        if s is None or not s.active:
            return JSONResponse({"active": False})
        return JSONResponse(
            {
                "active": True,
                "pdf": getattr(s, "pdf", None),
                "dry_run": getattr(s, "dry_run", None),
                "options": getattr(s, "options", {}),
            }
        )

    @app.post("/api/cancel")
    def cancel_run(request: Request) -> JSONResponse:
        if not _is_same_origin(request):
            return JSONResponse({"error": "Cross-Origin-Request abgelehnt."}, status_code=403)
        session = app.state.session
        if session is None or not session.active:
            return JSONResponse({"error": "Kein aktiver Lauf."}, status_code=409)
        session.cancel()
        return JSONResponse({"status": "cancelling"})

    @app.get("/api/stream")
    def stream() -> StreamingResponse:
        session = app.state.session
        if session is None:
            return StreamingResponse(
                iter(['event: log\ndata: {"text": "kein Lauf"}\n\n']), media_type="text/event-stream"
            )
        return StreamingResponse(_event_stream(session), media_type="text/event-stream")

    @app.get("/api/preview")
    def preview(pdf_stem: str, name: str) -> JSONResponse:
        """Gerenderten Markdown-Body einer Dry-Run-Note liefern (eval-Kopie).

        pdf_stem/name werden auf reine Dateinamen reduziert (kein Traversal); der
        aufgelöste Pfad muss innerhalb des baseline-Roots liegen.

        R3 (Preview-Staleness, MITTEL, B2 -- bewusst NICHT geloest, out of
        scope): `preview_root` ist vault-UNABHAENGIG, nur nach `pdf_stem`
        gekeyed (s. `_output_items_from_events`). Nach einem Vault-Wechsel kann
        derselbe PDF-Stem also alte Preview-Kopien aus einem FRUEHEREN Vault
        liefern, bevor ein frischer Lauf sie ueberschreibt. Bekannte, dokumentierte
        Grenze -- kein Namespacing pro Vault/Run gebaut (L5).
        """
        base = preview_root.resolve()
        safe_stem = Path(pdf_stem).name
        safe_name = Path(name).name
        if not safe_stem or not safe_name:
            return JSONResponse({"error": "ungültiger Pfad"}, status_code=400)
        eval_dir = (base / safe_stem).resolve()
        if not eval_dir.is_relative_to(base):
            return JSONResponse({"error": "ungültiger Pfad"}, status_code=400)
        for prefix in ("vault", "inbox", "merge"):
            f = (eval_dir / f"{prefix}__{safe_name}").resolve()
            if f.is_relative_to(base) and f.exists():
                return JSONResponse({"name": safe_name, "body": f.read_text(encoding="utf-8")})
        return JSONResponse({"error": "nicht gefunden"}, status_code=404)

    @app.get("/api/outputs")
    def outputs() -> JSONResponse:
        """Ergebnisliste des aktuellen/letzten Laufs (P3). Nur GET, nicht
        mutierend -> kein Origin-Check (L4-Ausnahme, wie /api/preview)."""
        session = app.state.session
        events = session.events if session is not None else []
        pdf = getattr(session, "pdf", None) if session is not None else None
        dry_run = getattr(session, "dry_run", None) if session is not None else None
        active_vault = _active_vault(session, app.state.vault_path)
        active_export_dir = _active_export_dir(session)
        items = _output_items_from_events(
            events, pdf=pdf, vault_path=active_vault, preview_root=preview_root, export_dir=active_export_dir
        )
        return JSONResponse({"items": items, "dry_run": dry_run})

    @app.get("/api/outputs/file")
    def outputs_file(path: str):
        session = app.state.session
        active_vault = _active_vault(session, app.state.vault_path)
        active_export_dir = _active_export_dir(session)
        resolved = _validate_output_path(
            path, vault_path=active_vault, preview_root=preview_root, export_dir=active_export_dir
        )
        if resolved is None:
            return JSONResponse({"error": "Pfad nicht erlaubt"}, status_code=403)
        if not resolved.is_file():
            return JSONResponse({"error": "nicht gefunden"}, status_code=404)
        return FileResponse(resolved, filename=resolved.name, media_type="text/markdown")

    @app.get("/api/outputs/archive")
    def outputs_archive() -> StreamingResponse:
        session = app.state.session
        events = session.events if session is not None else []
        pdf = getattr(session, "pdf", None) if session is not None else None
        active_vault = _active_vault(session, app.state.vault_path)
        active_export_dir = _active_export_dir(session)
        items = _output_items_from_events(
            events, pdf=pdf, vault_path=active_vault, preview_root=preview_root, export_dir=active_export_dir
        )
        buf = io.BytesIO()
        used: dict[str, int] = {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                raw = item.get("path")
                if not raw:
                    continue
                resolved = _validate_output_path(
                    raw, vault_path=active_vault, preview_root=preview_root, export_dir=active_export_dir
                )
                if resolved is None or not resolved.is_file():
                    continue
                base = resolved.name
                count = used.get(base, 0)
                used[base] = count + 1
                arcname = base if count == 0 else f"{Path(base).stem}-{count + 1}{Path(base).suffix}"
                zf.write(resolved, arcname=arcname)
        headers = {"Content-Disposition": f'attachment; filename="{_archive_filename(pdf)}"'}
        return StreamingResponse(iter([buf.getvalue()]), media_type="application/zip", headers=headers)

    @app.get("/api/runs")
    def list_runs() -> JSONResponse:
        """Run-Historie (P4), neueste zuerst, max. `_MAX_RUN_RECORDS`. Nur GET,
        nicht mutierend -> kein Origin-Check (L4-Ausnahme, wie /api/outputs)."""
        records = run_history.list_run_records(runs_dir, limit=_MAX_RUN_RECORDS)
        return JSONResponse({"runs": records})

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        record = run_history.read_run_record(run_id, runs_dir)
        if record is None:
            return JSONResponse({"error": "Lauf nicht gefunden"}, status_code=404)
        return JSONResponse(record)

    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        """Zuletzt gespeicherte Lauf-Einstellungen (P2). Nur GET, nicht
        mutierend -> kein Origin-Check (L4-Ausnahme, wie /api/outputs)."""
        data, warning = gui_settings.read_settings(settings_path)
        body = dict(data)
        if warning:
            body["warning"] = warning
        return JSONResponse(body)

    @app.put("/api/settings")
    async def put_settings(request: Request) -> JSONResponse:
        """Speichert die Lauf-Einstellungen (P2) -- immer das vollstaendige
        Objekt, kein Merge mit der vorherigen Datei (Aufrufer schickt den
        kompletten aktuellen Formular-Zustand)."""
        if not _is_same_origin(request):
            return JSONResponse({"error": "Cross-Origin-Request abgelehnt."}, status_code=403)
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "Ungültiger JSON-Body."}, status_code=400)
        normalized, error = gui_settings.validate_settings(body)
        if error:
            return JSONResponse({"error": error}, status_code=422)
        gui_settings.write_settings(normalized, settings_path)
        return JSONResponse(normalized)

    @app.get("/api/vault")
    def get_vault() -> JSONResponse:
        """Aktuell gewaehlter Vault (B2). Nur GET, nicht mutierend -> kein
        Origin-Check (L4-Ausnahme, wie /api/outputs bzw. /api/settings)."""
        return JSONResponse({"vault": str(app.state.vault_path)})

    @app.put("/api/vault")
    async def put_vault(request: Request) -> JSONResponse:
        """Ziel-Vault zur Laufzeit wechseln (B2). Kein OS-Dialog -- reines
        Pfad-Textfeld, serverseitig validiert (existiert + ist Verzeichnis).

        Bewusst OHNE Whitelist/erlaubte-Wurzeln-Pruefung (anders als
        `/api/outputs/file`): das ist genau der Zweck dieses Endpunkts -- der
        Nutzer waehlt hier explizit einen beliebigen lokalen Ordner als neuen
        Arbeits-Vault, analog einem nativen "Ordner waehlen"-Dialog. Traversal-
        /Symlink-Eingaben werden nicht durch eine Sonderregel geblockt, sondern
        laufen durch dieselbe `resolve()`+`is_dir()`-Pruefung wie jeder andere
        Pfad -- ein nicht existierendes Ziel oder eine Datei statt Verzeichnis
        scheitert dort ohnehin (400).
        """
        if not _is_same_origin(request):
            return JSONResponse({"error": "Cross-Origin-Request abgelehnt."}, status_code=403)
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "Ungültiger JSON-Body."}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Ungültiger JSON-Body."}, status_code=400)
        raw_path = body.get("path")
        if not raw_path or not isinstance(raw_path, str):
            return JSONResponse({"error": "path fehlt oder ist kein String."}, status_code=400)
        with app.state.session_lock:
            # R1 (Race, KRITISCH): waehrend ein Lauf aktiv ist, bleibt der
            # gewaehlte Vault fest -- ein Wechsel greift erst fuer den
            # NAECHSTEN Lauf (der laufende Subprocess/die Session-Downloads
            # haengen bereits am alten Vault, s. `_active_vault`).
            if app.state.session is not None and app.state.session.active:
                return JSONResponse(
                    {"error": "Vault-Wechsel während eines aktiven Laufs nicht möglich."}, status_code=409
                )
            try:
                resolved = Path(raw_path).resolve()
            except (OSError, ValueError):
                return JSONResponse({"error": f"Ungültiger Pfad: {raw_path}"}, status_code=400)
            if not resolved.is_dir():
                return JSONResponse({"error": f"Verzeichnis nicht gefunden: {raw_path}"}, status_code=400)
            app.state.vault_path = resolved
            # Persistenz (P2-Mechanik wiederverwendet): mit den bestehenden
            # Settings mergen statt sie zu ersetzen -- anders als `PUT
            # /api/settings` (dort schickt der Client immer das komplette
            # Formular-Objekt), hier soll der Vault-Wechsel bereits gespeicherte
            # Lauf-Einstellungen (backend/profile/…) nicht loeschen.
            stored, _warning = gui_settings.read_settings(settings_path)
            stored = dict(stored)
            stored["vault_path"] = str(resolved)
            gui_settings.write_settings(stored, settings_path)
        return JSONResponse({"vault": str(app.state.vault_path)})

    return app


def _event_stream(session: RunSession) -> Iterator[str]:
    import time

    i = 0
    while True:
        while i < len(session.events):
            ev = session.events[i]
            i += 1
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            # NICHT auf `done` enden — der Orchestrator druckt danach noch
            # Routing-Report + Stage-8-Eval. Terminal ist `exited` (Subprocess
            # beendet) bzw. `error` (RunSession-Exception).
            if ev["type"] in ("exited", "error"):
                return
        if session.finished and i >= len(session.events):
            return
        time.sleep(0.05)


def serve(port: int = 8052, open_browser: bool = True) -> None:  # pragma: no cover
    """Startet den uvicorn-Server und oeffnet den Browser (CLI-Entry)."""
    import uvicorn

    app = create_app()
    if open_browser:
        import webbrowser
        from threading import Timer

        Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"[gui] atomic-notes GUI → http://127.0.0.1:{port}  (Strg+C zum Beenden)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
