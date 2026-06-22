"""FastAPI-App fuer die Live-GUI der Atomic-Agent-Pipeline.

Eine eigenstaendige, lokale Web-GUI (neben dem read-only Eval-Dashboard):
PDF waehlen -> Lauf starten -> Live-Fortschritt pro Pipeline-Stufe streamen
(SSE) -> im Dry-Run die erzeugten Notes mit Confidence/Score als Preview zeigen.

Stack (lt. Plan „atomic-notes Frontend-Stack-Entscheidung"): FastAPI + HTMX/SSE
+ vanilla CSS, kein React/npm. Der eigentliche Lauf laeuft als Subprocess
(generative/gui/runner.py); diese App orchestriert nur Start + Event-Stream.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Callable
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from generative.gui import runner

_STATIC = Path(__file__).parent / "static"

# Endungen, die als „PDF-Kandidat" gelistet werden.
_PDF_GLOB = "*.pdf"


class RunSession:
    """Ein laufender (oder abgeschlossener) Pipeline-Lauf. Single-Run zur Zeit."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.finished = False
        self.cancelled = False
        self._proc = None  # vom Runner registriertes Popen-Handle (für Cancel)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

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
        try:
            for ev in run_iter:
                with self._lock:
                    self.events.append(ev)
        except Exception as exc:  # pragma: no cover - Defensive: Lauf-Crash sichtbar machen
            with self._lock:
                self.events.append({"type": "error", "message": str(exc)})
        finally:
            with self._lock:
                self.finished = True

    @property
    def active(self) -> bool:
        return not self.finished


def _default_run_factory(pdf: str, dry_run: bool, register=None) -> Iterator[dict]:
    yield from runner.iter_run_events(runner.build_argv(pdf, dry_run=dry_run), on_proc=register)


def create_app(
    *,
    run_factory: Callable[[str, bool], Iterator[dict]] | None = None,
    pdf_dirs: list[Path] | None = None,
    vault_path: Path | None = None,
    backend: str | None = None,
    uploads_dir: Path | None = None,
    doctor_fn: Callable[[], list] | None = None,
) -> FastAPI:
    run_factory = run_factory or _default_run_factory
    if doctor_fn is None:
        from generative.doctor import run_all as doctor_fn

    if pdf_dirs is None or vault_path is None or backend is None:
        from generative import config as _cfg
        if pdf_dirs is None:
            _repo = Path(__file__).resolve().parents[2]  # …/atomic-notes
            pdf_dirs = [_repo / "examples",
                        getattr(_cfg, "LITERATURE_DIR", None)]
        if vault_path is None:
            vault_path = _cfg.VAULT
        if backend is None:
            backend = _cfg.BACKEND
    pdf_dirs = [Path(d) for d in pdf_dirs if d]
    if uploads_dir is None:
        import tempfile
        uploads_dir = Path(tempfile.gettempdir()) / "atomic-notes-gui-uploads"
    uploads_dir = Path(uploads_dir)

    app = FastAPI(title="atomic-notes GUI")
    app.state.session = None
    app.state.session_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/app.css")
    def css():
        return StreamingResponse(iter([(_STATIC / "app.css").read_bytes()]),
                                 media_type="text/css")

    @app.get("/app.js")
    def js():
        return StreamingResponse(iter([(_STATIC / "app.js").read_bytes()]),
                                 media_type="text/javascript")

    @app.get("/api/pdfs")
    def list_pdfs() -> JSONResponse:
        seen: dict[str, str] = {}
        for d in pdf_dirs:
            if d and d.exists():
                for p in sorted(d.glob(_PDF_GLOB)):
                    seen.setdefault(p.name, str(p))
        return JSONResponse({"pdfs": [{"name": n, "path": p} for n, p in seen.items()]})

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:
        """Per Drag-and-Drop/Dialog hochgeladenes PDF server-seitig ablegen.

        Der Originalname (Basename, ohne Traversal) bleibt erhalten — die
        Pipeline leitet Metadaten u.a. aus dem Dateinamen ab. `--source` fährt
        anschliessend gegen den zurückgegebenen Pfad.
        """
        raw = file.filename or "upload.pdf"
        safe_name = Path(raw.replace("\\", "/")).name
        if not safe_name.lower().endswith(".pdf"):
            return JSONResponse({"error": "Nur PDF-Dateien werden akzeptiert."},
                                status_code=400)
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
            checks.append({
                "name": getattr(chk, "name", "?"),
                "ok": bool(getattr(chk, "ok", False)),
                "detail": getattr(chk, "detail", ""),
                "hint": getattr(chk, "hint", ""),
                "required": bool(getattr(chk, "required", True)),
            })
        ok = all(c["ok"] for c in checks if c["required"])
        return JSONResponse({
            "backend": backend,
            "vault": str(Path(vault_path)),
            "ok": ok,
            "checks": checks,
        })

    @app.post("/api/run")
    async def start_run(request: Request) -> JSONResponse:
        body = await request.json()
        pdf = body.get("pdf", "")
        dry_run = bool(body.get("dry_run", True))
        if not pdf or not Path(pdf).exists():
            return JSONResponse({"error": f"PDF nicht gefunden: {pdf}"}, status_code=400)
        with app.state.session_lock:
            if app.state.session is not None and app.state.session.active:
                return JSONResponse({"error": "Es läuft bereits ein Pipeline-Lauf."},
                                    status_code=409)
            session = RunSession()
            # Iterator MIT der Proc-Registrierung der Session erzeugen → Cancel
            # kann den Subprocess später terminieren.
            run_iter = run_factory(pdf, dry_run, session.register_proc)
            app.state.session = session
            session.start(run_iter)
        return JSONResponse({"status": "started", "pdf": pdf, "dry_run": dry_run})

    @app.post("/api/cancel")
    def cancel_run() -> JSONResponse:
        session = app.state.session
        if session is None or not session.active:
            return JSONResponse({"error": "Kein aktiver Lauf."}, status_code=409)
        session.cancel()
        return JSONResponse({"status": "cancelling"})

    @app.get("/api/stream")
    def stream() -> StreamingResponse:
        session = app.state.session
        if session is None:
            return StreamingResponse(iter(["event: log\ndata: {\"text\": \"kein Lauf\"}\n\n"]),
                                     media_type="text/event-stream")
        return StreamingResponse(_event_stream(session), media_type="text/event-stream")

    @app.get("/api/preview")
    def preview(pdf_stem: str, name: str) -> JSONResponse:
        """Gerenderten Markdown-Body einer Dry-Run-Note liefern (eval-Kopie).

        pdf_stem/name werden auf reine Dateinamen reduziert (kein Traversal); der
        aufgelöste Pfad muss innerhalb des baseline-Roots liegen.
        """
        base = (Path(__file__).resolve().parents[1] / ".cache" / "eval" / "baseline").resolve()
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
