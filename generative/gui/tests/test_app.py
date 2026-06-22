"""Endpoint-Tests fuer die Live-GUI (FastAPI TestClient).

Der echte Orchestrator-Lauf (Subprocess, Minuten, LLM-Calls) wird per
Dependency-Injection durch eine `fake_run`-Generator-Funktion ersetzt, die
echte Event-Dicts yieldet — keine Mock-Bibliothek.
"""
import json

import pytest
from fastapi.testclient import TestClient

from generative.gui.app import create_app


def fake_run(pdf, dry_run):
    yield {"type": "started", "argv": ["fake"]}
    yield {"type": "stage", "num": 1, "total": 7, "label": "PDF & Chunking"}
    yield {"type": "preview", "name": "a.md", "routing": "vault", "score": 5,
           "hard_gates": True, "confidence": "high", "flags": ""}
    yield {"type": "done", "written": 1, "dry_run": dry_run}
    yield {"type": "exited", "returncode": 0}


@pytest.fixture
def client(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    app = create_app(run_factory=fake_run, pdf_dirs=[tmp_path],
                     vault_path=tmp_path / "vault", backend="subscription")
    return TestClient(app), pdf


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


def test_doctor_reports_backend_and_vault(client):
    c, _ = client
    r = c.get("/api/doctor")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "subscription"
    assert "vault" in body
    assert "vault_exists" in body


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
    assert any(json.loads(ln[len("data:"):].strip()).get("written") == 1
               for ln in done_payloads)


def test_preview_rejects_path_traversal(client):
    c, _ = client
    r = c.get("/api/preview", params={"pdf_stem": "../../../etc", "name": "../secret.md"})
    # Traversal darf nicht in einen Lesezugriff ausserhalb des Cache-Roots münden.
    assert r.status_code in (400, 404)


def test_run_rejected_while_active(client, monkeypatch):
    c, pdf = client

    # Langsamer Lauf: blockiert, bis das Test-Event gesetzt wird.
    import threading
    gate = threading.Event()

    def slow_run(pdf, dry_run):
        yield {"type": "started", "argv": ["slow"]}
        gate.wait(timeout=5)
        yield {"type": "done", "written": 0, "dry_run": dry_run}

    app = create_app(run_factory=slow_run, pdf_dirs=[pdf.parent],
                     vault_path=pdf.parent, backend="subscription")
    cc = TestClient(app)
    r1 = cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r1.status_code == 200
    r2 = cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r2.status_code == 409  # bereits ein Lauf aktiv
    gate.set()
