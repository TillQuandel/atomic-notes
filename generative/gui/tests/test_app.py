"""Endpoint-Tests fuer die Live-GUI (FastAPI TestClient).

Der echte Orchestrator-Lauf (Subprocess, Minuten, LLM-Calls) wird per
Dependency-Injection durch eine `fake_run`-Generator-Funktion ersetzt, die
echte Event-Dicts yieldet — keine Mock-Bibliothek.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from generative.gui.app import create_app


def fake_doctor():
    from generative.doctor import CheckResult

    return [
        CheckResult(name="pdftotext", ok=True, detail="pdftotext: /usr/bin"),
        CheckResult(name="backend (subscription)", ok=True, detail="CLI ok"),
        CheckResult(name="vault", ok=True, detail="/vault"),
        CheckResult(name="pypdf", ok=True, detail="ok", required=False),
    ]


def fake_run(pdf, dry_run, register=None):
    yield {"type": "started", "argv": ["fake"]}
    yield {"type": "stage", "num": 1, "total": 7, "label": "PDF & Chunking"}
    yield {
        "type": "preview",
        "name": "a.md",
        "routing": "vault",
        "score": 5,
        "hard_gates": True,
        "confidence": "high",
        "flags": "",
    }
    yield {"type": "done", "written": 1, "dry_run": dry_run}
    yield {"type": "exited", "returncode": 0}


@pytest.fixture
def client(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    uploads = tmp_path / "uploads"
    vault = tmp_path / "vault"
    vault.mkdir()
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=uploads,
        doctor_fn=fake_doctor,
    )
    c = TestClient(app)
    c._uploads = uploads  # für Upload-Tests
    return c, pdf


def test_index_serves_html(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "atomic-notes" in r.text.lower()


def test_list_pdfs(client):
    c, pdf = client
    r = c.get("/api/pdfs")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["pdfs"]]
    assert "beispiel.pdf" in names


def test_doctor_runs_real_checks(client):
    c, _ = client
    r = c.get("/api/doctor")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "subscription"
    assert "vault" in body
    assert body["ok"] is True  # alle required-Checks grün
    names = [chk["name"] for chk in body["checks"]]
    assert "vault" in names and "pdftotext" in names
    assert all({"name", "ok", "detail", "hint", "required"} <= set(chk) for chk in body["checks"])


def test_doctor_ok_false_when_required_check_fails(tmp_path):
    from generative.doctor import CheckResult

    def failing_doctor():
        return [
            CheckResult(name="backend (subscription)", ok=False, detail="CLI nicht eingeloggt", hint="claude login"),
            CheckResult(name="pypdf", ok=True, detail="ok", required=False),
        ]

    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=failing_doctor,
    )
    body = TestClient(app).get("/api/doctor").json()
    assert body["ok"] is False  # required-Fehler → Start sperren


def test_run_rejects_unknown_pdf(client):
    c, _ = client
    r = c.post("/api/run", json={"pdf": "C:/does/not/exist.pdf", "dry_run": True})
    assert r.status_code == 400


def test_run_then_stream_yields_events(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    stream = c.get("/api/stream")
    assert stream.status_code == 200
    body = stream.text
    # SSE-Frames enthalten die geparsten Events.
    assert "event: stage" in body
    assert "event: preview" in body
    assert "event: done" in body
    # Stream endet erst auf `exited` (nicht auf `done`).
    assert "event: exited" in body
    assert body.rstrip().endswith('data: {"type": "exited", "returncode": 0}')
    # Letztes Note-Event korrekt durchgereicht.
    done_payloads = [ln for ln in body.splitlines() if ln.startswith("data:") and '"done"' in ln]
    assert any(json.loads(ln[len("data:") :].strip()).get("written") == 1 for ln in done_payloads)


def test_preview_rejects_path_traversal(client):
    c, _ = client
    r = c.get("/api/preview", params={"pdf_stem": "../../../etc", "name": "../secret.md"})
    # Traversal darf nicht in einen Lesezugriff ausserhalb des Cache-Roots münden.
    assert r.status_code in (400, 404)


def test_upload_pdf_saves_and_returns_path(client):
    c, _ = client
    r = c.post("/api/upload", files={"file": ("Mein Dokument.pdf", b"%PDF-1.4 echtes", "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Mein Dokument.pdf"
    saved = Path(body["path"])
    assert saved.exists()
    assert saved.read_bytes() == b"%PDF-1.4 echtes"
    # Liegt im uploads_dir — und der Originalname (Stem) bleibt erhalten
    # (Pipeline nutzt den Dateinamen für Metadaten-Fallback).
    assert saved.parent == c._uploads
    assert saved.stem == "Mein Dokument"


def test_upload_rejects_non_pdf(client):
    c, _ = client
    r = c.post("/api/upload", files={"file": ("notiz.txt", b"kein pdf", "text/plain")})
    assert r.status_code == 400


def test_upload_sanitizes_filename_no_traversal(client):
    c, _ = client
    r = c.post("/api/upload", files={"file": ("../../evil.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    saved = Path(r.json()["path"])
    # Kein Entkommen aus uploads_dir.
    assert saved.parent == c._uploads
    assert saved.name == "evil.pdf"


def test_run_rejected_while_active(client, monkeypatch):
    c, pdf = client

    # Langsamer Lauf: blockiert, bis das Test-Event gesetzt wird.
    import threading

    gate = threading.Event()

    def slow_run(pdf, dry_run, register=None):
        yield {"type": "started", "argv": ["slow"]}
        gate.wait(timeout=5)
        yield {"type": "done", "written": 0, "dry_run": dry_run}

    app = create_app(run_factory=slow_run, pdf_dirs=[pdf.parent], vault_path=pdf.parent, backend="subscription")
    cc = TestClient(app)
    r1 = cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r1.status_code == 200
    r2 = cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r2.status_code == 409  # bereits ein Lauf aktiv
    gate.set()


def test_cancel_terminates_active_run(tmp_path):
    import threading

    gate = threading.Event()
    terminated = {"v": False}

    class FakeProc:
        def poll(self):
            return 1 if terminated["v"] else None

        def terminate(self):
            terminated["v"] = True
            gate.set()  # entsperrt den Lauf, simuliert Subprocess-Tod

    def slow_run(pdf, dry_run, register=None):
        if register:
            register(FakeProc())
        yield {"type": "started", "argv": ["slow"]}
        gate.wait(timeout=5)
        yield {"type": "exited", "returncode": 1}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=slow_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    cc = TestClient(app)
    assert cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True}).status_code == 200
    r = cc.post("/api/cancel")
    assert r.status_code == 200
    assert terminated["v"] is True  # Subprocess wurde terminiert


def test_cancel_without_active_run_409(client):
    c, _ = client
    assert c.post("/api/cancel").status_code == 409


def test_run_revalidates_vault_server_side(tmp_path):
    # B: Auch wenn der Client das Gate umgeht, lehnt der Server einen Lauf ohne
    # existierenden Vault ab (Fehlervermeidung statt Mid-Run-Crash).
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path / "nicht-da",
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    r = TestClient(app).post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r.status_code == 400
    assert "vault" in r.json()["error"].lower()


def test_run_factory_exception_surfaces_as_error_event(tmp_path):
    # E: Wirft der Lauf, muss das als error-Event im Stream ankommen (schliesst
    # den bisher ungetesteten _consume-Exception-Pfad).
    def boom(pdf, dry_run, register=None):
        yield {"type": "started", "argv": ["boom"]}
        raise RuntimeError("kaputt")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=boom, pdf_dirs=[tmp_path], vault_path=tmp_path, backend="subscription", uploads_dir=tmp_path / "u"
    )
    cc = TestClient(app)
    assert cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True}).status_code == 200
    body = cc.get("/api/stream").text
    assert "event: error" in body
    assert "kaputt" in body


def test_preview_returns_body_for_existing_eval_copy(tmp_path):
    # F: erfolgreicher Lesepfad von /api/preview (eval-Kopie vorhanden).
    base = tmp_path / "baseline"
    (base / "meinpdf").mkdir(parents=True)
    (base / "meinpdf" / "vault__Konzept.md").write_text("# Konzept\nKörper", encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        preview_root=base,
    )
    r = TestClient(app).get("/api/preview", params={"pdf_stem": "meinpdf", "name": "Konzept.md"})
    assert r.status_code == 200
    assert r.json()["body"] == "# Konzept\nKörper"


def test_run_rejects_cross_origin(client):
    # #1: Ein Cross-Origin-POST (CSRF aus fremdem Browser-Tab) wird abgelehnt,
    # bevor irgendein Lauf startet — auch wenn der Browser den Request absetzt.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_run_allows_same_origin(client):
    # Same-Origin (127.0.0.1) bleibt erlaubt.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}, headers={"Origin": "http://127.0.0.1:8052"})
    assert r.status_code == 200


def test_upload_rejects_cross_origin(client):
    c, _ = client
    r = c.post(
        "/api/upload",
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403


def test_cancel_rejects_cross_origin(client):
    c, _ = client
    r = c.post("/api/cancel", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_run_rejects_pdf_outside_allowed_dirs(tmp_path):
    # #2: Ein existierender Pfad ausserhalb pdf_dirs/uploads_dir (z.B. beliebige
    # lokale Datei via CSRF/abgelaufenem Client-State) wird serverseitig abgelehnt.
    allowed = tmp_path / "pdfs"
    allowed.mkdir()
    outside = tmp_path / "geheim.pdf"
    outside.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[allowed],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "uploads",
        doctor_fn=fake_doctor,
    )
    r = TestClient(app).post("/api/run", json={"pdf": str(outside), "dry_run": True})
    assert r.status_code == 400


def test_run_accepts_pdf_from_uploads_dir(tmp_path):
    # Hochgeladene PDFs (in uploads_dir) bleiben gültige Lauf-Quellen.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    up = uploads / "doc.pdf"
    up.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path / "pdfs"],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=uploads,
        doctor_fn=fake_doctor,
    )
    r = TestClient(app).post("/api/run", json={"pdf": str(up), "dry_run": True})
    assert r.status_code == 200


def test_status_reports_no_active_run_initially(client):
    c, _ = client
    body = c.get("/api/status").json()
    assert body["active"] is False


def test_status_reports_active_run(tmp_path):
    import threading

    gate = threading.Event()

    def slow_run(pdf, dry_run, register=None):
        yield {"type": "started", "argv": ["slow"]}
        gate.wait(timeout=5)
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=slow_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    cc = TestClient(app)
    cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    body = cc.get("/api/status").json()
    assert body["active"] is True
    assert body["pdf"].endswith("x.pdf")
    assert body["dry_run"] is True
    gate.set()
