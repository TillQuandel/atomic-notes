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
import threading
import time
import zipfile
from collections.abc import Iterator, Callable
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from generative.gui import run_history, runner
from generative.runtime_config import PRESETS

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).parent / "static"

# P4 (Run-Historie): Default-Ablageort fuer Lauf-Records — Modul-Konstante
# (statt Inline-Default in create_app), damit Tests sie isoliert per
# monkeypatch auf ein tmp_path umbiegen koennen (s. tests/conftest.py), ohne
# jeden bestehenden create_app(...)-Aufruf einzeln anzufassen.
_DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / ".cache" / "gui" / "runs"

# P4: mehr als so viele Records im runs_dir -> aelteste werden geloescht.
_MAX_RUN_RECORDS = 50

# Endungen, die als „PDF-Kandidat" gelistet werden.
_PDF_GLOB = "*.pdf"

# Same-Origin-Hosts: Die GUI bindet nur an 127.0.0.1. Ein Browser sendet bei
# Cross-Origin-POSTs einen `Origin`-Header — fehlt er (curl/TestClient/Beacon
# same-origin), ist es kein CSRF-Vektor.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# P1 (Lauf-Einstellungen): Whitelist der `POST /api/run`-`options`. Backend-Werte
# sind, wie in doctor.check_backend/config.BACKEND verifiziert, genau diese zwei;
# Profil-Whitelist wird bewusst aus runtime_config.PRESETS importiert statt
# hart dupliziert (Plan P1 Schritt 1).
_BACKENDS = frozenset({"subscription", "litellm"})
_OPTION_KEYS = frozenset({"backend", "profile", "no_llm"})


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
    backend = options.get("backend")
    if backend not in (None, ""):
        if backend not in _BACKENDS:
            return {}, f"Unbekannter Backend-Wert: {backend!r} (erlaubt: {', '.join(sorted(_BACKENDS))})"
        normalized["backend"] = backend

    profile = options.get("profile")
    if profile not in (None, ""):
        if profile not in PRESETS:
            return {}, f"Unbekanntes Profil: {profile!r} (erlaubt: {', '.join(sorted(PRESETS))})"
        normalized["profile"] = profile

    no_llm = options.get("no_llm")
    if no_llm is not None:
        if not isinstance(no_llm, bool):
            return {}, "no_llm muss ein Boolean sein."
        if no_llm:
            normalized["no_llm"] = True

    return normalized, None


def _is_same_origin(request: Request) -> bool:
    """CSRF-Schutz: Cross-Origin-Browser-Requests an mutierende Endpunkte abweisen."""
    origin = request.headers.get("origin")
    if origin is None:
        return True
    from urllib.parse import urlparse

    return urlparse(origin).hostname in _LOCAL_HOSTS


def _is_within(path: str, root: Path) -> bool:
    """True, wenn `path` (aufgelöst) innerhalb von `root` liegt."""
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _output_items_from_events(
    events: list[dict], *, pdf: str | None, vault_path: Path, preview_root: Path
) -> list[dict]:
    """Aggregiert die Ergebnisliste (`GET /api/outputs`) aus den Events der
    aktuellen/letzten RunSession.

    Dry-Run: aus `preview`-Events (Name/Routing/Score/Confidence/Flags); der
    Download-Pfad ist die eval-Kopie unter `preview_root/<pdf-stem>/…`, sofern
    sie existiert (kein Erfinden — fehlt sie, bleibt `path` weg, L5).
    Schreib-Lauf: aus `note_written`-Events (vault_writer-stdout) — dort gibt es
    kein Score/Confidence, die druckt vault_writer nur im Dry-Run.
    """
    items: list[dict] = []
    stem = Path(pdf).stem if pdf else ""
    vault_root = Path(vault_path).resolve()
    preview_base = Path(preview_root).resolve()
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
            items.append(item)
    return items


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


def _validate_output_path(path: str, *, vault_path: Path, preview_root: Path) -> Path | None:
    """Pfad-Whitelist (L4) fuer `/api/outputs/file` + `/api/outputs/archive`:
    nur `.md`-Dateien unterhalb `vault_path`, oder beliebige Dateien unterhalb
    `preview_root` (die eval-Kopien der Dry-Run-Vorschau, bereits auf `.md`
    beschraenkt). `resolve()` neutralisiert Symlink-Escapes. Alles andere:
    `None` -> Aufrufer antwortet 403.
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
            self.events, pdf=self.pdf, vault_path=self.vault_path, preview_root=self.preview_root
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


def _default_run_factory(pdf: str, dry_run: bool, register=None, options: dict | None = None) -> Iterator[dict]:
    argv, env_overrides = runner.build_run_spec(pdf, dry_run=dry_run, options=options)
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
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    run_factory = run_factory or _default_run_factory
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
    clock = clock or time.time

    if pdf_dirs is None or vault_path is None or backend is None:
        from generative import config as _cfg

        if pdf_dirs is None:
            _repo = Path(__file__).resolve().parents[2]  # …/atomic-notes
            pdf_dirs = [_repo / "examples", getattr(_cfg, "LITERATURE_DIR", None)]
        if vault_path is None:
            vault_path = _cfg.VAULT
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
    app.state.session = None
    app.state.session_lock = threading.Lock()

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
        ok = all(c["ok"] for c in checks if c["required"])
        # P1 Doctor-Gating: litellm-Verfuegbarkeit unabhaengig vom aktuell
        # konfigurierten Server-Backend pruefen (sonst faellt der Key-Check weg,
        # sobald der Server per Default auf "subscription" laeuft) — reine
        # Wiederverwendung von doctor.check_backend, keine neue Logik.
        litellm_check = litellm_check_fn()
        litellm_available = bool(getattr(litellm_check, "ok", False))
        response = {
            "backend": backend,
            "vault": str(Path(vault_path)),
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
        body = await request.json()
        pdf = body.get("pdf", "")
        dry_run = bool(body.get("dry_run", True))
        options, options_error = _validate_run_options(body.get("options"))
        if options_error:
            return JSONResponse({"error": options_error}, status_code=422)
        if not pdf or not Path(pdf).exists():
            return JSONResponse({"error": f"PDF nicht gefunden: {pdf}"}, status_code=400)
        # #2: Quelle muss unter einem erlaubten Root liegen (gelistet/hochgeladen).
        if not any(_is_within(pdf, root) for root in _allowed_roots):
            return JSONResponse({"error": "PDF liegt ausserhalb der erlaubten Verzeichnisse."}, status_code=400)
        # Server-seitige Revalidierung (Client-Gate könnte umgangen/veraltet sein):
        # der Vault wird auch im Dry-Run gebraucht (Context-Builder scannt ihn).
        if not Path(vault_path).exists():
            return JSONResponse({"error": f"Vault nicht gefunden: {vault_path}"}, status_code=400)
        with app.state.session_lock:
            if app.state.session is not None and app.state.session.active:
                return JSONResponse({"error": "Es läuft bereits ein Pipeline-Lauf."}, status_code=409)
            session = RunSession()
            session.pdf = pdf
            session.dry_run = dry_run
            session.options = options
            # P4: Kontext fuer den Historie-Record am Lauf-Ende (s. RunSession._write_history_record).
            session.vault_path = vault_path
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
        return JSONResponse({"active": True, "pdf": getattr(s, "pdf", None), "dry_run": getattr(s, "dry_run", None)})

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
        items = _output_items_from_events(events, pdf=pdf, vault_path=vault_path, preview_root=preview_root)
        return JSONResponse({"items": items, "dry_run": dry_run})

    @app.get("/api/outputs/file")
    def outputs_file(path: str):
        resolved = _validate_output_path(path, vault_path=vault_path, preview_root=preview_root)
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
        items = _output_items_from_events(events, pdf=pdf, vault_path=vault_path, preview_root=preview_root)
        buf = io.BytesIO()
        used: dict[str, int] = {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                raw = item.get("path")
                if not raw:
                    continue
                resolved = _validate_output_path(raw, vault_path=vault_path, preview_root=preview_root)
                if resolved is None or not resolved.is_file():
                    continue
                base = resolved.name
                count = used.get(base, 0)
                used[base] = count + 1
                arcname = base if count == 0 else f"{Path(base).stem}-{count + 1}{Path(base).suffix}"
                zf.write(resolved, arcname=arcname)
        headers = {"Content-Disposition": 'attachment; filename="outputs.zip"'}
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
