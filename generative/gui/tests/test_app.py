"""Endpoint-Tests fuer die Live-GUI (FastAPI TestClient).

Der echte Orchestrator-Lauf (Subprocess, Minuten, LLM-Calls) wird per
Dependency-Injection durch eine `fake_run`-Generator-Funktion ersetzt, die
echte Event-Dicts yieldet — keine Mock-Bibliothek.
"""

import io
import json
import zipfile
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


def fake_run(pdf, dry_run, register=None, options=None):
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

    def slow_run(pdf, dry_run, register=None, options=None):
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

    def slow_run(pdf, dry_run, register=None, options=None):
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
    def boom(pdf, dry_run, register=None, options=None):
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


def test_run_without_options_key_behaves_like_before(client):
    # Rueckwaertskompatibilitaet: Payload ohne `options` verhaelt sich wie heute.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    assert r.status_code == 200
    assert r.json().get("options") == {}


def test_run_with_empty_options_dict_behaves_like_missing(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {}})
    assert r.status_code == 200
    assert r.json().get("options") == {}


def test_run_options_unknown_key_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"foo": "bar"}})
    assert r.status_code == 422
    assert "foo" in r.json()["error"]


def test_run_options_unknown_backend_value_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"backend": "openai-direct"}})
    assert r.status_code == 422
    assert "backend" in r.json()["error"].lower()


def test_run_options_unknown_profile_value_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"profile": "turbo"}})
    assert r.status_code == 422
    assert "profil" in r.json()["error"].lower()


def test_run_options_no_llm_wrong_type_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"no_llm": "yes"}})
    assert r.status_code == 422


def test_run_options_valid_values_return_200(client):
    c, pdf = client
    r = c.post(
        "/api/run",
        json={
            "pdf": str(pdf),
            "dry_run": True,
            "options": {"backend": "litellm", "profile": "fast", "no_llm": True},
        },
    )
    assert r.status_code == 200
    assert r.json()["options"] == {"backend": "litellm", "profile": "fast", "no_llm": True}


def test_run_options_rejects_cross_origin(client):
    # L4: Der bestehende Origin-Check greift auch, wenn `options` mitgeschickt wird.
    c, pdf = client
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"backend": "litellm"}},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403


def test_run_forwards_normalized_options_to_run_factory(tmp_path):
    captured = {}

    def capturing_run(pdf, dry_run, register=None, options=None):
        captured["options"] = options
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=capturing_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app)
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"backend": "litellm", "profile": "fast"}},
    )
    assert r.status_code == 200
    # /api/stream blockiert (Polling-Loop), bis der Lauf-Thread fertig ist —
    # danach ist `captured` garantiert befuellt (keine Race-Condition).
    c.get("/api/stream")
    assert captured["options"] == {"backend": "litellm", "profile": "fast"}


def test_doctor_reports_litellm_available_true_when_check_ok(tmp_path):
    from generative.doctor import CheckResult

    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        litellm_check_fn=lambda: CheckResult(name="backend (litellm)", ok=True, detail="gesetzt: ANTHROPIC_API_KEY"),
    )
    body = TestClient(app).get("/api/doctor").json()
    assert body["litellm_available"] is True


def test_doctor_reports_litellm_unavailable_with_hint(tmp_path):
    from generative.doctor import CheckResult

    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        litellm_check_fn=lambda: CheckResult(
            name="backend (litellm)", ok=False, detail="kein Key", hint="API-Key setzen"
        ),
    )
    body = TestClient(app).get("/api/doctor").json()
    assert body["litellm_available"] is False
    assert "API-Key setzen" in body.get("litellm_hint", "")


def test_doctor_litellm_available_present_with_default_check(client):
    # Ohne injizierten litellm_check_fn greift die echte doctor.check_backend-Logik.
    c, _ = client
    body = c.get("/api/doctor").json()
    assert "litellm_available" in body
    assert isinstance(body["litellm_available"], bool)


def _drain(client):
    """Blockiert (Stream lesen), bis der Lauf-Thread fertig ist — s. bestehendes
    Muster in test_run_forwards_normalized_options_to_run_factory."""
    client.get("/api/stream")


# --- P3: GET /api/outputs, /api/outputs/file, /api/outputs/archive --------


def test_outputs_empty_without_any_run(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    r = TestClient(app).get("/api/outputs")
    assert r.status_code == 200
    assert r.json() == {"items": [], "dry_run": None}


def test_outputs_lists_preview_items_after_dry_run(client):
    c, pdf = client
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    body = c.get("/api/outputs").json()
    assert body["dry_run"] is True
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["title"] == "a.md"
    assert item["routing"] == "vault"
    assert item["score"] == 5
    assert item["confidence"] == "high"
    # leere Flags / kein merge_target / keine eval-Kopie auf Platte -> keine Keys erfinden (L5).
    assert "flags" not in item
    assert "merge_target" not in item
    assert "path" not in item


def test_outputs_includes_path_when_eval_copy_exists(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    preview_root = tmp_path / "baseline"
    (preview_root / "beispiel").mkdir(parents=True)
    eval_file = preview_root / "beispiel" / "vault__a.md"
    eval_file.write_text("# a\nKoerper", encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        preview_root=preview_root,
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    item = c.get("/api/outputs").json()["items"][0]
    assert Path(item["path"]) == eval_file.resolve()


def test_outputs_write_mode_items_have_no_score_or_confidence(tmp_path):
    vault = tmp_path / "vault"
    (vault / "00-inbox").mkdir(parents=True)
    (vault / "00-inbox" / "Foo.md").write_text("# Foo", encoding="utf-8")
    (vault / "00-inbox" / "Bar.md").write_text("# Bar", encoding="utf-8")
    (vault / "00-inbox" / "MERGE - Baz.md").write_text("# Baz", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "note_written", "path": "00-inbox/Bar.md", "routing": "inbox"}
        yield {
            "type": "note_written",
            "path": "00-inbox/MERGE - Baz.md",
            "routing": "merge",
            "merge_target": "04-wissen/Baz.md",
        }
        yield {"type": "done", "written": 3, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    body = c.get("/api/outputs").json()
    assert body["dry_run"] is False
    items = body["items"]
    assert [i["routing"] for i in items] == ["vault", "inbox", "merge"]
    assert items[2]["merge_target"] == "04-wissen/Baz.md"
    for item in items:
        assert "score" not in item
        assert "confidence" not in item
    assert items[0]["path"] == str((vault / "00-inbox" / "Foo.md").resolve())
    assert items[1]["path"] == str((vault / "00-inbox" / "Bar.md").resolve())
    assert items[2]["path"] == str((vault / "00-inbox" / "MERGE - Baz.md").resolve())


def test_outputs_empty_run_zero_notes_no_crash(tmp_path):
    def empty_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "done", "written": 0, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=empty_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    r = c.get("/api/outputs")
    assert r.status_code == 200
    assert r.json()["items"] == []


def _write_mode_app(tmp_path):
    vault = tmp_path / "vault"
    (vault / "00-inbox").mkdir(parents=True)
    (vault / "00-inbox" / "Foo.md").write_text("# Foo\nInhalt", encoding="utf-8")
    (vault / "00-inbox" / "image.png").write_bytes(b"\x89PNG")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "done", "written": 1, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    return c, vault, pdf


def test_outputs_file_downloads_vault_md(tmp_path):
    c, vault, _ = _write_mode_app(tmp_path)
    target = vault / "00-inbox" / "Foo.md"
    r = c.get("/api/outputs/file", params={"path": str(target)})
    assert r.status_code == 200
    # Byte-Vergleich gegen das tatsaechlich Geschriebene (Windows uebersetzt
    # write_text-Newlines zu CRLF — der Download muss exakt widerspiegeln,
    # was auf der Platte liegt, kein eigener Newline-Vergleich).
    assert r.content == target.read_bytes()
    assert "attachment" in r.headers["content-disposition"]
    assert "Foo.md" in r.headers["content-disposition"]


def test_outputs_file_downloads_preview_eval_copy(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    preview_root = tmp_path / "baseline"
    (preview_root / "beispiel").mkdir(parents=True)
    eval_file = preview_root / "beispiel" / "vault__a.md"
    eval_file.write_text("# a\nKoerper", encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        preview_root=preview_root,
    )
    c = TestClient(app)
    r = c.get("/api/outputs/file", params={"path": str(eval_file)})
    assert r.status_code == 200
    assert r.content == eval_file.read_bytes()


def test_outputs_file_rejects_path_traversal(tmp_path):
    c, vault, _ = _write_mode_app(tmp_path)
    outside = tmp_path / "secret.md"
    outside.write_text("geheim", encoding="utf-8")
    traversal = str(vault / "00-inbox" / ".." / ".." / "secret.md")
    r = c.get("/api/outputs/file", params={"path": traversal})
    assert r.status_code == 403


def test_outputs_file_rejects_absolute_foreign_path(tmp_path):
    c, _vault, _ = _write_mode_app(tmp_path)
    outside = tmp_path / "anderswo.md"
    outside.write_text("fremd", encoding="utf-8")
    r = c.get("/api/outputs/file", params={"path": str(outside)})
    assert r.status_code == 403


def test_outputs_file_rejects_non_md_in_vault(tmp_path):
    c, vault, _ = _write_mode_app(tmp_path)
    r = c.get("/api/outputs/file", params={"path": str(vault / "00-inbox" / "image.png")})
    assert r.status_code == 403


def test_outputs_file_rejects_symlink_escape(tmp_path):
    c, vault, _ = _write_mode_app(tmp_path)
    outside_target = tmp_path / "geheim.md"
    outside_target.write_text("geheim", encoding="utf-8")
    link = vault / "00-inbox" / "link.md"
    try:
        link.symlink_to(outside_target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks ohne Sonderrechte auf dieser Plattform nicht erstellbar")
    r = c.get("/api/outputs/file", params={"path": str(link)})
    assert r.status_code == 403


def test_outputs_file_missing_returns_404(tmp_path):
    c, vault, _ = _write_mode_app(tmp_path)
    r = c.get("/api/outputs/file", params={"path": str(vault / "00-inbox" / "nicht-da.md")})
    assert r.status_code == 404


def test_outputs_archive_returns_zip_with_expected_entries(tmp_path):
    vault = tmp_path / "vault"
    (vault / "00-inbox").mkdir(parents=True)
    (vault / "00-inbox" / "Foo.md").write_text("# Foo", encoding="utf-8")
    (vault / "00-inbox" / "Bar.md").write_text("# Bar", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "note_written", "path": "00-inbox/Bar.md", "routing": "inbox"}
        yield {"type": "done", "written": 2, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    r = c.get("/api/outputs/archive")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == ["Bar.md", "Foo.md"]
    assert zf.read("Foo.md") == b"# Foo"
    assert zf.read("Bar.md") == b"# Bar"


def test_outputs_archive_empty_run_returns_empty_zip_no_crash(tmp_path):
    def empty_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "done", "written": 0, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=empty_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    r = c.get("/api/outputs/archive")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.namelist() == []


def test_outputs_archive_dedupes_colliding_basenames(tmp_path):
    vault = tmp_path / "vault"
    (vault / "00-inbox").mkdir(parents=True)
    (vault / "00-inbox" / "sub").mkdir(parents=True)
    (vault / "00-inbox" / "Foo.md").write_text("eins", encoding="utf-8")
    (vault / "00-inbox" / "sub" / "Foo.md").write_text("zwei", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "note_written", "path": "00-inbox/sub/Foo.md", "routing": "inbox"}
        yield {"type": "done", "written": 2, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    r = c.get("/api/outputs/archive")
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = sorted(zf.namelist())
    assert names == ["Foo-2.md", "Foo.md"]
    contents = {n: zf.read(n) for n in names}
    assert set(contents.values()) == {b"eins", b"zwei"}


def test_outputs_endpoints_have_no_origin_check(tmp_path):
    # L4: nur GET, nicht mutierend -> kein CSRF-Vektor, kein Origin-Gate noetig.
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    r = TestClient(app).get("/api/outputs", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200


def test_status_reports_active_run(tmp_path):
    import threading

    gate = threading.Event()

    def slow_run(pdf, dry_run, register=None, options=None):
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
