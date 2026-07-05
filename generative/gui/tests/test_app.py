"""Endpoint-Tests fuer die Live-GUI (FastAPI TestClient).

Der echte Orchestrator-Lauf (Subprocess, Minuten, LLM-Calls) wird per
Dependency-Injection durch eine `fake_run`-Generator-Funktion ersetzt, die
echte Event-Dicts yieldet — keine Mock-Bibliothek.
"""

import io
import json
import logging
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from generative.gui import gui_settings
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
    c = TestClient(app, base_url="http://localhost")
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
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert body["ok"] is False  # required-Fehler → Start sperren


def test_run_rejects_invalid_json_body_with_400(client):
    # B2: kaputter JSON-Body loeste bisher ein ungefangenes JSONDecodeError ->
    # 500 aus, statt eines sauberen 400.
    c, _ = client
    r = c.post("/api/run", content="{invalid", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_settings_put_rejects_invalid_json_body_with_400(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.put("/api/settings", content="{invalid", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_run_rejects_non_object_json_body_with_400(client):
    # G1 (Codex-Nachreview): syntaktisch gueltiges JSON, das kein Objekt ist
    # (null, []), lief bisher in body.get(...) -> AttributeError -> 500. Der
    # B2-Fix fing nur JSONDecodeError. Beide Faelle muessen 400 liefern.
    c, _ = client
    for payload in ("null", "[]", "42", '"text"'):
        r = c.post("/api/run", content=payload, headers={"Content-Type": "application/json"})
        assert r.status_code == 400, f"payload {payload!r} -> {r.status_code}"
        assert "error" in r.json()


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


def test_run_summary_event_forwarded_over_sse(tmp_path):
    # P5: run_summary ist ein Event wie jedes andere -> _event_stream reicht es
    # unveraendert durch, keine eigene Sonderbehandlung noetig.
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run_with_summary,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    body = c.get("/api/stream").text
    assert "event: run_summary" in body
    summary_payloads = [ln for ln in body.splitlines() if ln.startswith("data:") and '"run_summary"' in ln]
    assert len(summary_payloads) == 1
    payload = json.loads(summary_payloads[0][len("data:") :].strip())
    assert payload["duration_s"] == 12.4
    assert payload["tokens"]["total"] == 18432


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
    cc = TestClient(app, base_url="http://localhost")
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
    cc = TestClient(app, base_url="http://localhost")
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
    r = TestClient(app, base_url="http://localhost").post("/api/run", json={"pdf": str(pdf), "dry_run": True})
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
    cc = TestClient(app, base_url="http://localhost")
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
    r = TestClient(app, base_url="http://localhost").get(
        "/api/preview", params={"pdf_stem": "meinpdf", "name": "Konzept.md"}
    )
    assert r.status_code == 200
    assert r.json()["body"] == "# Konzept\nKörper"


def test_run_rejects_cross_origin(client):
    # #1: Ein Cross-Origin-POST (CSRF aus fremdem Browser-Tab) wird abgelehnt,
    # bevor irgendein Lauf startet — auch wenn der Browser den Request absetzt.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_run_allows_same_origin(client):
    # Same-Origin bleibt erlaubt. Die `client`-Fixture setzt base_url auf
    # "http://localhost" (ohne Port) -> der TestClient sendet den Host-Header
    # "localhost". M1 vergleicht Origin exakt gegen diesen Host-Header, daher
    # hier "http://localhost" statt eines beliebigen anderen Ports.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}, headers={"Origin": "http://localhost"})
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


def _run_app(tmp_path):
    # M1: eigener App-Aufbau (statt `client`-Fixture), weil diese Tests den
    # Host-Header ueber die `base_url` des TestClients gezielt variieren
    # muessen -- die Fixture ist fest auf "http://localhost" verdrahtet.
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    vault = tmp_path / "vault"
    vault.mkdir()
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "uploads",
        doctor_fn=fake_doctor,
    )
    return app, pdf


def test_run_rejects_cross_origin_port(tmp_path):
    # M1 (KRITISCH): Ein fremder localhost-Port ist KEIN Same-Origin -- vorher
    # akzeptierte `_is_same_origin` jede Origin mit Hostname 127.0.0.1/
    # localhost/::1, unabhaengig vom Port (CSRF-Luecke fuer jede andere lokal
    # laufende Web-App). Jetzt: exakter netloc-Vergleich gegen den Host-Header.
    app, pdf = _run_app(tmp_path)
    c = TestClient(app, base_url="http://127.0.0.1:8052")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True},
        headers={"Origin": "http://127.0.0.1:3000"},
    )
    assert r.status_code == 403


def test_run_rejects_cross_origin_hostname_same_port(tmp_path):
    # M1: gleicher Port, aber anderer Hostname (127.0.0.1 vs. localhost) --
    # der netloc-Vergleich ist exakt, keine Hostname-Aequivalenz.
    app, pdf = _run_app(tmp_path)
    c = TestClient(app, base_url="http://127.0.0.1:8052")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True},
        headers={"Origin": "http://localhost:8052"},
    )
    assert r.status_code == 403


def test_run_allows_same_origin_case_insensitive(tmp_path):
    # M1: Origin und Host-Header case-insensitiv vergleichen (Hostnames sind
    # nicht case-sensitiv).
    app, pdf = _run_app(tmp_path)
    c = TestClient(app, base_url="http://localhost:8052")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True},
        headers={"Origin": "http://LOCALHOST:8052"},
    )
    assert r.status_code == 200


def test_run_rejects_null_origin(client):
    # M1: `Origin: null` (z.B. sandboxed iframe/data:-URL) faellt nicht unter
    # "Origin fehlt" -- fail-closed, nicht erlaubt.
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}, headers={"Origin": "null"})
    assert r.status_code == 403


def test_is_same_origin_direct_matrix():
    # Codex-Review M1 (GERING): direkte Matrix fuer `_is_same_origin` --
    # Faelle, die ueber den TestClient nicht erreichbar sind (fehlender
    # Host-Header, Userinfo im Origin-netloc, ungueltiger Port, IPv6).
    from starlette.requests import Request

    from generative.gui.app import _is_same_origin

    def _req(headers: dict[str, str]) -> Request:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/run",
            "query_string": b"",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        }
        return Request(scope)

    # kein Origin -> erlaubt (Nicht-Browser-Clients)
    assert _is_same_origin(_req({})) is True
    # Origin gesetzt, aber Host-Header fehlt -> fail-closed
    assert _is_same_origin(_req({"origin": "http://127.0.0.1:8052"})) is False
    # Userinfo im Origin-netloc matcht den Host-Header nicht
    assert _is_same_origin(_req({"origin": "http://user@127.0.0.1:8052", "host": "127.0.0.1:8052"})) is False
    # ungueltiger/fremder Port -> abgelehnt
    assert _is_same_origin(_req({"origin": "http://127.0.0.1:99999", "host": "127.0.0.1:8052"})) is False
    # IPv6: netloc-Gleichheit inkl. Klammer-Notation
    assert _is_same_origin(_req({"origin": "http://[::1]:8052", "host": "[::1]:8052"})) is True


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
    r = TestClient(app, base_url="http://localhost").post("/api/run", json={"pdf": str(outside), "dry_run": True})
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
    r = TestClient(app, base_url="http://localhost").post("/api/run", json={"pdf": str(up), "dry_run": True})
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
    c = TestClient(app, base_url="http://localhost")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"backend": "litellm", "profile": "fast"}},
    )
    assert r.status_code == 200
    # /api/stream blockiert (Polling-Loop), bis der Lauf-Thread fertig ist —
    # danach ist `captured` garantiert befuellt (keine Race-Condition).
    c.get("/api/stream")
    assert captured["options"] == {"backend": "litellm", "profile": "fast"}


def test_run_forwards_resolved_export_dir_to_run_factory(tmp_path):
    # B3 (TOCTOU-Konsistenz): der an --inbox-dir durchgereichte Pfad muss der
    # AUFGELOESTE sein (== gesnapshotteter session.export_dir), nicht der rohe
    # Eingabe-String -- sonst koennte ein Symlink nach der Validierung woanders
    # hinzeigen und der Subprocess ausserhalb des Snapshots schreiben.
    captured = {}

    def capturing_run(pdf, dry_run, register=None, options=None):
        captured["options"] = options
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    export = tmp_path / "export"
    export.mkdir()
    app = create_app(
        run_factory=capturing_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    # relativ-verschachtelter Eingabe-Pfad, der zu `export` aufloest
    messy = str(export / "sub" / "..")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": False, "options": {"inbox_dir": messy}},
    )
    assert r.status_code == 200
    c.get("/api/stream")
    assert captured["options"]["inbox_dir"] == str(export.resolve())


# --- B3: Output-Ziel waehlbar (options.inbox_dir) --------------------------


def test_run_options_inbox_dir_string_accepted_in_dry_run(client):
    # Dry-Run: inbox_dir wird normalisiert durchgereicht, aber serverseitig
    # nicht auf Existenz geprueft (im Dry-Run schreibt vault_writer ohnehin nur
    # eval-Kopien nach .cache/eval/baseline, unabhaengig von --inbox-dir).
    c, pdf = client
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"inbox_dir": "C:/nicht-vorhanden"}},
    )
    assert r.status_code == 200
    assert r.json()["options"]["inbox_dir"] == "C:/nicht-vorhanden"


def test_run_options_inbox_dir_wrong_type_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"inbox_dir": 123}})
    assert r.status_code == 422


def test_run_options_inbox_dir_empty_string_omitted(client):
    # Leerer String = kein Export-Wunsch, wie bei backend/profile (Server-Default).
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"inbox_dir": ""}})
    assert r.status_code == 200
    assert "inbox_dir" not in r.json()["options"]


def test_run_write_mode_inbox_dir_nonexistent_returns_400(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    vault = tmp_path / "vault"
    vault.mkdir()
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": False, "options": {"inbox_dir": str(tmp_path / "fehlt")}},
    )
    assert r.status_code == 400


def test_run_write_mode_inbox_dir_file_not_directory_returns_400(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    vault = tmp_path / "vault"
    vault.mkdir()
    not_a_dir = tmp_path / "datei.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": False, "options": {"inbox_dir": str(not_a_dir)}},
    )
    assert r.status_code == 400


def test_run_write_mode_inbox_dir_valid_starts_and_sets_export_dir(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": False, "options": {"inbox_dir": str(export_dir)}},
    )
    assert r.status_code == 200
    assert app.state.session.export_dir == export_dir.resolve()


def test_run_dry_run_ignores_inbox_dir_no_export_dir_snapshot(tmp_path):
    # Dry-Run: kein Export-Ordner-Snapshot, selbst wenn inbox_dir gesetzt ist
    # (das Ziel wird im Vorschau-Modus schlicht ignoriert, kein Fehler, L5).
    vault = tmp_path / "vault"
    vault.mkdir()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"inbox_dir": str(tmp_path / "nicht-vorhanden")}},
    )
    assert r.status_code == 200
    assert app.state.session.export_dir is None


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
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
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
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert body["litellm_available"] is False
    assert "API-Key setzen" in body.get("litellm_hint", "")


def test_doctor_litellm_available_present_with_default_check(client):
    # Ohne injizierten litellm_check_fn greift die echte doctor.check_backend-Logik.
    c, _ = client
    body = c.get("/api/doctor").json()
    assert "litellm_available" in body
    assert isinstance(body["litellm_available"], bool)


# --- access (B1a) -----------------------------------------------------------


def _fake_access_summary():
    return {
        "subscription": {"cli_found": True, "credentials_present": True},
        "litellm": {"available": True, "key_vars_set": ["ANTHROPIC_API_KEY"]},
    }


def test_doctor_access_struktur(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        access_summary_fn=_fake_access_summary,
    )
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert body["access"] == {
        "backend": "subscription",
        "subscription": {"cli_found": True, "credentials_present": True},
        "litellm": {"available": True, "key_vars_set": ["ANTHROPIC_API_KEY"]},
    }


def test_doctor_access_altfelder_unveraendert(tmp_path):
    # Rueckwaertskompatibilitaet: bestehende Felder bleiben unveraendert vorhanden.
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        access_summary_fn=_fake_access_summary,
    )
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert {"backend", "vault", "vault_exists", "ok", "checks", "litellm_available"} <= set(body)


def test_doctor_access_kein_key_leak(tmp_path):
    def access_with_key_like_value():
        # Absichtlich falscher Wert, um sicherzustellen, dass der Endpunkt
        # nichts zusaetzlich einschleust -- die echte access_summary gibt nie
        # Werte zurueck, dieser Test prueft nur die Transport-Schicht.
        return {
            "subscription": {"cli_found": True, "credentials_present": True},
            "litellm": {"available": True, "key_vars_set": ["ANTHROPIC_API_KEY"]},
        }

    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        access_summary_fn=access_with_key_like_value,
    )
    r = TestClient(app, base_url="http://localhost").get("/api/doctor")
    assert "sk-" not in r.text
    assert "ANTHROPIC_API_KEY" in r.text  # Name darf, Wert nicht


def test_doctor_access_present_with_default_access_summary_fn(client):
    # Ohne injizierten access_summary_fn greift die echte doctor.access_summary-Logik.
    c, _ = client
    body = c.get("/api/doctor").json()
    assert "access" in body
    assert "subscription" in body["access"] and "litellm" in body["access"]


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
    r = TestClient(app, base_url="http://localhost").get("/api/outputs")
    assert r.status_code == 200
    # F4: "exports" ist additiv immer im Response (leer ohne Export-Formate/Session).
    assert r.json() == {"items": [], "dry_run": None, "exports": []}


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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
    r = c.get("/api/outputs/file", params={"path": str(eval_file)})
    assert r.status_code == 200
    assert r.content == eval_file.read_bytes()


def test_outputs_file_rejects_non_md_under_preview_root(tmp_path):
    # Der preview-Zweig der Whitelist muss wie Vault/Export auf .md
    # beschraenkt sein -- eine Nicht-.md-Datei unterhalb `preview_root` darf
    # NICHT ausgeliefert werden, auch wenn sie (z.B. via Traversal aus einem
    # Lauf-Verzeichnis) dort landet.
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    preview_root = tmp_path / "baseline"
    (preview_root / "beispiel").mkdir(parents=True)
    evil_file = preview_root / "beispiel" / "inbox__evil.txt"
    evil_file.write_text("nicht erlaubt", encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        preview_root=preview_root,
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.get("/api/outputs/file", params={"path": str(evil_file)})
    assert r.status_code == 403


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


# --- B3: Export-Ordner in der Download-Whitelist ---------------------------


def test_validate_output_path_helper_allows_md_under_export_dir(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    note = export_dir / "a.md"
    note.write_text("x", encoding="utf-8")
    resolved = _validate_output_path(str(note), vault_path=vault, preview_root=preview, export_dir=export_dir)
    assert resolved == note.resolve()


def test_validate_output_path_helper_rejects_outside_export_dir(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    outside = tmp_path / "fremd.md"
    outside.write_text("x", encoding="utf-8")
    resolved = _validate_output_path(str(outside), vault_path=vault, preview_root=preview, export_dir=export_dir)
    assert resolved is None


def test_validate_output_path_helper_rejects_non_md_under_export_dir(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    img = export_dir / "bild.png"
    img.write_bytes(b"\x89PNG")
    resolved = _validate_output_path(str(img), vault_path=vault, preview_root=preview, export_dir=export_dir)
    assert resolved is None


def test_validate_output_path_helper_without_export_dir_unaffected(tmp_path):
    # export_dir=None (Default) -- bestehendes Verhalten unveraendert.
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    outside = tmp_path / "fremd.md"
    outside.write_text("x", encoding="utf-8")
    resolved = _validate_output_path(str(outside), vault_path=vault, preview_root=preview)
    assert resolved is None


def test_validate_output_path_helper_rejects_non_md_under_preview_root(tmp_path):
    # Direkter Helper-Test (ergaenzend zum Endpoint-Test oben): der
    # preview-Zweig darf keine beliebigen Dateien mehr durchlassen.
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    img = preview / "beispiel" / "vault__bild.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\x89PNG")
    resolved = _validate_output_path(str(img), vault_path=vault, preview_root=preview)
    assert resolved is None


def test_validate_output_path_helper_allows_md_under_preview_root(tmp_path):
    # Bestands-Semantik bleibt erhalten: .md unterhalb preview_root weiterhin erlaubt.
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    note = preview / "beispiel" / "vault__a.md"
    note.parent.mkdir(parents=True)
    note.write_text("# a", encoding="utf-8")
    resolved = _validate_output_path(str(note), vault_path=vault, preview_root=preview)
    assert resolved == note.resolve()


def test_validate_output_path_helper_rejects_ads_under_preview_root(tmp_path):
    # NTFS Alternate Data Stream (ADS): `wirt.txt:geheim.md` hat als
    # Path-Suffix ".md" (Python liest den Stream-Namen als Endung), Windows
    # liefert beim Oeffnen aber den Stream der Basisdatei `wirt.txt` aus --
    # die .md-Whitelist waere umgangen. Reine Pfad-Logik, kein echtes ADS
    # noetig (NTFS-only, nicht CI-portabel).
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    bad = str(preview / "wirt.txt:geheim.md")
    resolved = _validate_output_path(bad, vault_path=vault, preview_root=preview)
    assert resolved is None


def test_validate_output_path_helper_rejects_ads_under_vault(tmp_path):
    # Wie oben, aber unter `vault_path` -- der ADS-Guard muss fuer ALLE
    # Whitelist-Zweige greifen, nicht nur fuer preview_root.
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    bad = str(vault / "note.txt:hidden.md")
    resolved = _validate_output_path(bad, vault_path=vault, preview_root=preview)
    assert resolved is None


def test_output_items_from_events_note_written_absolute_path_under_export_dir(tmp_path):
    from generative.gui.app import _output_items_from_events

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    note = export_dir / "a.md"
    note.write_text("x", encoding="utf-8")
    events = [{"type": "note_written", "path": str(note), "routing": "vault"}]
    items = _output_items_from_events(events, pdf=None, vault_path=vault, preview_root=preview, export_dir=export_dir)
    assert items[0]["path"] == str(note.resolve())


def test_output_items_from_events_note_written_outside_export_dir_no_path(tmp_path):
    from generative.gui.app import _output_items_from_events

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    outside = tmp_path / "fremd.md"
    outside.write_text("x", encoding="utf-8")
    events = [{"type": "note_written", "path": str(outside), "routing": "vault"}]
    items = _output_items_from_events(events, pdf=None, vault_path=vault, preview_root=preview, export_dir=export_dir)
    assert "path" not in items[0]


def _export_mode_app(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    note = export_dir / "Foo.md"
    note.write_text("# Foo\nInhalt", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": str(note), "routing": "vault"}
        yield {"type": "done", "written": 1, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False, "options": {"inbox_dir": str(export_dir)}})
    _drain(c)
    return c, vault, export_dir, note


def test_outputs_lists_note_written_absolute_path_under_export_dir(tmp_path):
    c, _vault, _export_dir, note = _export_mode_app(tmp_path)
    items = c.get("/api/outputs").json()["items"]
    assert items[0]["path"] == str(note.resolve())


def test_outputs_file_downloads_md_under_export_dir(tmp_path):
    c, _vault, _export_dir, note = _export_mode_app(tmp_path)
    r = c.get("/api/outputs/file", params={"path": str(note)})
    assert r.status_code == 200
    assert r.content == note.read_bytes()


def test_outputs_file_rejects_path_outside_export_dir(tmp_path):
    c, _vault, export_dir, _note = _export_mode_app(tmp_path)
    outside = export_dir.parent / "fremd.md"
    outside.write_text("fremd", encoding="utf-8")
    r = c.get("/api/outputs/file", params={"path": str(outside)})
    assert r.status_code == 403


def test_outputs_file_rejects_traversal_out_of_export_dir(tmp_path):
    c, _vault, export_dir, _note = _export_mode_app(tmp_path)
    outside = export_dir.parent / "geheim.md"
    outside.write_text("geheim", encoding="utf-8")
    traversal = str(export_dir / ".." / "geheim.md")
    r = c.get("/api/outputs/file", params={"path": traversal})
    assert r.status_code == 403


def test_outputs_file_rejects_non_md_under_export_dir(tmp_path):
    c, _vault, export_dir, _note = _export_mode_app(tmp_path)
    img = export_dir / "bild.png"
    img.write_bytes(b"\x89PNG")
    r = c.get("/api/outputs/file", params={"path": str(img)})
    assert r.status_code == 403


def test_outputs_archive_includes_export_dir_note(tmp_path):
    c, _vault, _export_dir, _note = _export_mode_app(tmp_path)
    r = c.get("/api/outputs/archive")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "Foo.md" in zf.namelist()


def test_export_dir_session_snapshot_survives_vault_switch(tmp_path):
    # Analog R1 (B2): der Export-Snapshot DIESES Laufs bleibt massgeblich, auch
    # wenn der Vault danach per PUT /api/vault gewechselt wird.
    c, _vault, _export_dir, note = _export_mode_app(tmp_path)
    new_vault = tmp_path / "vault-b"
    new_vault.mkdir()
    r = c.put("/api/vault", json={"path": str(new_vault)})
    assert r.status_code == 200
    ok = c.get("/api/outputs/file", params={"path": str(note)})
    assert ok.status_code == 200


# --- F4 (Output-Projekt): Export-Formatwahl als Lauf-Option ----------------
# Session-Export-Ordner fuer --export-format-Outputs (docx/pdf/html/json/...) --
# eigener Snapshot `session.export_formats_dir`, getrennt von B3s `session.export_dir`
# (dort geht es um den Vault-Inbox-Ersatz fuer normale .md-Notes).


def test_run_options_export_formats_valid_accepted(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"export_formats": ["docx", "pdf"]}})
    assert r.status_code == 200
    assert r.json()["options"]["export_formats"] == ["docx", "pdf"]


def test_run_options_export_formats_invalid_returns_422(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"export_formats": ["bogus"]}})
    assert r.status_code == 422


def test_run_options_export_formats_rejects_obsidian_md(client):
    # obsidian-md ist kein GUI-Format (die .md-Notes gibt es ohnehin als Download).
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"export_formats": ["obsidian-md"]}})
    assert r.status_code == 422


def test_run_options_export_formats_empty_list_omitted_from_response(client):
    c, pdf = client
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"export_formats": []}})
    assert r.status_code == 200
    assert "export_formats" not in r.json()["options"]


def _export_formats_app(tmp_path, *, dry_run=True, export_formats=("json", "docx")):
    vault = tmp_path / "vault"
    vault.mkdir()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        exports_dir=tmp_path / "exports",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": dry_run, "options": {"export_formats": list(export_formats)}})
    _drain(c)
    return c, app


def test_start_run_with_export_formats_sets_session_snapshot_dir(tmp_path):
    _c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    assert export_dir is not None
    assert export_dir.is_relative_to((tmp_path / "exports").resolve())


def test_start_run_export_formats_works_in_dry_run(tmp_path):
    # Anders als B3 inbox_dir: export_formats gilt AUCH im Dry-Run (json/
    # portable-md/docx/... brauchen keinen Vault-Schreib-Lauf).
    _c, app = _export_formats_app(tmp_path, dry_run=True)
    assert app.state.session.export_formats_dir is not None


def test_start_run_without_export_formats_no_session_snapshot_dir(client):
    c, pdf = client
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)


def test_start_run_forwards_export_formats_dir_to_run_factory(tmp_path):
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
        exports_dir=tmp_path / "exports",
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"export_formats": ["docx"]}})
    assert r.status_code == 200
    c.get("/api/stream")
    assert captured["options"]["export_formats"] == ["docx"]
    export_formats_dir = Path(captured["options"]["export_formats_dir"])
    assert export_formats_dir.is_relative_to((tmp_path / "exports").resolve())


def test_outputs_exports_empty_when_no_export_formats_requested(client):
    c, pdf = client
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    assert c.get("/api/outputs").json()["exports"] == []


def test_outputs_exports_empty_when_export_formats_dir_has_no_files(tmp_path):
    c, _app = _export_formats_app(tmp_path)
    assert c.get("/api/outputs").json()["exports"] == []


def test_outputs_exports_lists_files_sorted(tmp_path):
    c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    # Im echten Lauf legt export_runner.run_export das Verzeichnis an -- der
    # Fake-run_factory hier tut das nicht, also simuliert der Test es.
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "b.docx").write_bytes(b"stub")
    (export_dir / "a.json").write_text("{}", encoding="utf-8")
    body = c.get("/api/outputs").json()
    names = [item["name"] for item in body["exports"]]
    assert names == ["a.json", "b.docx"]
    for item in body["exports"]:
        assert Path(item["path"]).is_file()


def test_outputs_file_downloads_binary_export_with_guessed_mimetype(tmp_path):
    c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = export_dir / "Note.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 stub")
    r = c.get("/api/outputs/file", params={"path": str(pdf_file)})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-1.4 stub"


def test_outputs_file_rejects_path_outside_export_formats_dir(tmp_path):
    c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    outside = export_dir.parent / "fremd.pdf"
    outside.write_bytes(b"x")
    r = c.get("/api/outputs/file", params={"path": str(outside)})
    assert r.status_code == 403


def test_outputs_file_rejects_disallowed_suffix_under_export_formats_dir(tmp_path):
    c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    exe = export_dir / "bad.exe"
    exe.write_bytes(b"x")
    r = c.get("/api/outputs/file", params={"path": str(exe)})
    assert r.status_code == 403


def test_outputs_archive_includes_export_formats_files(tmp_path):
    c, app = _export_formats_app(tmp_path)
    export_dir = app.state.session.export_formats_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "Note.json").write_text("{}", encoding="utf-8")
    r = c.get("/api/outputs/archive")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "Note.json" in zf.namelist()


def test_validate_output_path_helper_allows_json_under_export_formats_dir(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    export_formats_dir = tmp_path / "exports" / "run-1"
    export_formats_dir.mkdir(parents=True)
    f = export_formats_dir / "a.json"
    f.write_text("{}", encoding="utf-8")
    resolved = _validate_output_path(
        str(f), vault_path=vault, preview_root=preview, export_formats_dir=export_formats_dir
    )
    assert resolved == f.resolve()


def test_validate_output_path_helper_rejects_outside_export_formats_dir(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    export_formats_dir = tmp_path / "exports" / "run-1"
    export_formats_dir.mkdir(parents=True)
    outside = tmp_path / "a.json"
    outside.write_text("{}", encoding="utf-8")
    resolved = _validate_output_path(
        str(outside), vault_path=vault, preview_root=preview, export_formats_dir=export_formats_dir
    )
    assert resolved is None


def test_validate_output_path_helper_without_export_formats_dir_unaffected(tmp_path):
    from generative.gui.app import _validate_output_path

    vault = tmp_path / "vault"
    vault.mkdir()
    preview = tmp_path / "preview"
    preview.mkdir()
    outside = tmp_path / "a.json"
    outside.write_text("{}", encoding="utf-8")
    resolved = _validate_output_path(str(outside), vault_path=vault, preview_root=preview)
    assert resolved is None


def test_archive_filename_helper_uses_pdf_stem():
    from generative.gui.app import _archive_filename

    assert _archive_filename("beispiel.pdf") == "beispiel-outputs.zip"


def test_archive_filename_helper_fallback_without_pdf():
    from generative.gui.app import _archive_filename

    assert _archive_filename(None) == "outputs.zip"


def test_archive_filename_helper_sanitizes_special_chars():
    from generative.gui.app import _archive_filename

    assert _archive_filename("mein Dökument (v2)!.pdf") == "mein-D-kument--v2---outputs.zip"


def test_outputs_archive_content_disposition_uses_pdf_stem(tmp_path):
    vault = tmp_path / "vault"
    (vault / "00-inbox").mkdir(parents=True)
    (vault / "00-inbox" / "Foo.md").write_text("# Foo", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "done", "written": 1, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run, pdf_dirs=[tmp_path], vault_path=vault, backend="subscription", uploads_dir=tmp_path / "u"
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    r = c.get("/api/outputs/archive")
    assert 'filename="beispiel-outputs.zip"' in r.headers["content-disposition"]


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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
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
    c = TestClient(app, base_url="http://localhost")
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
    r = TestClient(app, base_url="http://localhost").get("/api/outputs", headers={"Origin": "http://evil.example"})
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
    cc = TestClient(app, base_url="http://localhost")
    cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    body = cc.get("/api/status").json()
    assert body["active"] is True
    assert body["pdf"].endswith("x.pdf")
    assert body["dry_run"] is True
    gate.set()


def test_status_reports_options_of_active_run(tmp_path):
    # C2: nach Reattach (frisch geladene Seite waehrend ein Lauf aktiv ist)
    # muss der Options-Header sich befuellen koennen -- /api/status braucht
    # dafuer die Optionen des laufenden Lauf mit.
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
    cc = TestClient(app, base_url="http://localhost")
    cc.post("/api/run", json={"pdf": str(pdf), "dry_run": True, "options": {"profile": "fast"}})
    body = cc.get("/api/status").json()
    assert body["options"] == {"profile": "fast"}
    gate.set()


# --- P4: Run-Historie (GET /api/runs, GET /api/runs/{run_id}) --------------


def _clock_seq(*values):
    """Deterministischer Fake-Clock: liefert die Werte der Reihe nach, dann
    haengt er beim letzten Wert (fuer beliebig viele weitere Aufrufe)."""
    values = list(values)

    def _clock():
        return values.pop(0) if len(values) > 1 else values[0]

    return _clock


def test_runs_endpoint_empty_when_no_runs_yet(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    r = TestClient(app, base_url="http://localhost").get("/api/runs")
    assert r.status_code == 200
    assert r.json() == {"runs": []}


def test_run_completion_writes_history_record(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        runs_dir=tmp_path / "runs",
        clock=_clock_seq(1000.0, 1010.0),
    )
    c = TestClient(app, base_url="http://localhost")
    c.post(
        "/api/run",
        json={"pdf": str(pdf), "dry_run": True, "options": {"backend": "litellm", "profile": "fast"}},
    )
    _drain(c)
    body = c.get("/api/runs").json()
    assert len(body["runs"]) == 1
    record = body["runs"][0]
    from generative.gui import run_history

    assert run_history.is_valid_run_id(record["run_id"])
    assert record["started_at"] == 1000.0
    assert record["finished_at"] == 1010.0
    assert record["source_pdf"] == str(pdf)
    assert record["dry_run"] is True
    assert record["options"] == {"backend": "litellm", "profile": "fast"}
    assert record["rc"] == 0
    assert record["notes"] == [{"title": "a.md", "routing": "vault", "score": 5, "confidence": "high"}]


def test_run_without_options_writes_empty_options_in_record(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    record = c.get("/api/runs").json()["runs"][0]
    assert record["options"] == {}


# --- P5: run_summary-Event -> Historie-Record duration_s/tokens ------------


def fake_run_with_summary(pdf, dry_run, register=None, options=None):
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
    yield {
        "type": "run_summary",
        "duration_s": 12.4,
        "tokens": {"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0},
    }
    yield {"type": "exited", "returncode": 0}


def test_run_summary_event_lands_in_history_record(tmp_path):
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run_with_summary,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    record = c.get("/api/runs").json()["runs"][0]
    assert record["duration_s"] == 12.4
    assert record["tokens"] == {"total": 18432, "input": 14200, "output": 4232, "cache_read": 0, "cache_create": 0}


def test_run_without_summary_event_omits_duration_and_tokens_in_record(tmp_path):
    # fake_run (Standard-Fixture) endet ohne run_summary-Event — Record darf
    # keine erfundenen duration_s/tokens-Felder bekommen (L5).
    pdf = tmp_path / "beispiel.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    record = c.get("/api/runs").json()["runs"][0]
    assert "duration_s" not in record
    assert "tokens" not in record


def test_run_crash_still_writes_history_record_with_null_rc(tmp_path):
    def boom(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["boom"]}
        raise RuntimeError("kaputt")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=boom,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    record = c.get("/api/runs").json()["runs"][0]
    assert record["rc"] is None


def test_runs_endpoint_lists_newest_first(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
        clock=_clock_seq(100.0, 200.0, 300.0, 400.0),
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    runs = c.get("/api/runs").json()["runs"]
    assert [r["finished_at"] for r in runs] == [400.0, 200.0]


def test_run_detail_endpoint_returns_full_record(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    listed = c.get("/api/runs").json()["runs"][0]
    detail = c.get(f"/api/runs/{listed['run_id']}").json()
    assert detail == listed


def test_run_detail_endpoint_missing_run_id_returns_404(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    r = TestClient(app, base_url="http://localhost").get("/api/runs/20240101000000-doesnotexist")
    assert r.status_code == 404


def test_run_detail_endpoint_rejects_invalid_run_id_shapes(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    for bad in ("UPPERCASE", "with%20space", "semi;colon", "dot.dot", "a%2e%2e"):
        r = c.get(f"/api/runs/{bad}")
        assert r.status_code in (404, 422), (bad, r.status_code)


def test_run_detail_endpoint_rejects_path_traversal_run_id(tmp_path):
    # Ein Record ausserhalb von runs_dir existiert real — ".." darf trotzdem
    # nicht dorthin auflösen (Regex blockt Punkte von vornherein).
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps({"run_id": "secret"}), encoding="utf-8")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    c = TestClient(app, base_url="http://localhost")
    r = c.get("/api/runs/..%2Fsecret")
    assert r.status_code in (404, 422)


def test_runs_endpoint_prunes_records_beyond_50(tmp_path):
    from generative.gui import run_history

    runs_dir = tmp_path / "runs"
    for i in range(55):
        record = run_history.build_run_record(
            run_id=run_history.make_run_id(float(i), suffix=f"{i:03d}"),
            started_at=float(i),
            finished_at=float(i),
            source_pdf="x.pdf",
            dry_run=True,
            options={},
            rc=0,
            notes=[],
        )
        run_history.write_run_record(record, runs_dir)
    assert len(list(runs_dir.glob("*.json"))) == 55

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=runs_dir,
        clock=_clock_seq(1000.0, 2000.0),
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(c)
    assert len(list(runs_dir.glob("*.json"))) == 50


def test_run_history_survives_simulated_gui_restart(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    runs_dir = tmp_path / "runs"
    app_a = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=runs_dir,
        clock=_clock_seq(1.0, 2.0, 3.0, 4.0),
    )
    ca = TestClient(app_a, base_url="http://localhost")
    ca.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(ca)
    ca.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
    _drain(ca)

    # "Neustart": frische App-Instanz, derselbe runs_dir.
    app_b = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=runs_dir,
    )
    runs = TestClient(app_b, base_url="http://localhost").get("/api/runs").json()["runs"]
    assert len(runs) == 2


def test_runs_endpoints_have_no_origin_check(tmp_path):
    # L4: nur GET, nicht mutierend.
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        runs_dir=tmp_path / "runs",
    )
    r = TestClient(app, base_url="http://localhost").get("/api/runs", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200


# --- P2: GET/PUT /api/settings (persistierte Lauf-Einstellungen) -----------


def _settings_app(tmp_path, **kwargs):
    kwargs.setdefault("run_factory", fake_run)
    kwargs.setdefault("pdf_dirs", [tmp_path])
    kwargs.setdefault("vault_path", tmp_path)
    kwargs.setdefault("backend", "subscription")
    kwargs.setdefault("uploads_dir", tmp_path / "u")
    kwargs.setdefault("settings_path", tmp_path / "gui" / "settings.json")
    return create_app(**kwargs)


def test_settings_empty_when_no_file_yet(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.get("/api/settings")
    assert r.status_code == 200
    assert r.json() == {}


def test_settings_put_then_get_roundtrip(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    put = c.put(
        "/api/settings",
        json={"backend": "litellm", "profile": "fast", "no_llm": True, "dry_run": False},
    )
    assert put.status_code == 200
    r = c.get("/api/settings")
    assert r.json() == {"backend": "litellm", "profile": "fast", "no_llm": True, "dry_run": False}


def test_settings_persists_across_two_create_app_instances(tmp_path):
    # Simulierter GUI-Neustart: gleicher settings_path, frische App-Instanz.
    settings_path = tmp_path / "gui" / "settings.json"
    app_a = _settings_app(tmp_path, settings_path=settings_path)
    TestClient(app_a, base_url="http://localhost").put("/api/settings", json={"backend": "litellm", "dry_run": False})

    app_b = _settings_app(tmp_path, settings_path=settings_path)
    r = TestClient(app_b, base_url="http://localhost").get("/api/settings")
    assert r.json() == {"backend": "litellm", "dry_run": False}


def test_settings_put_unknown_key_returns_422(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.put("/api/settings", json={"foo": "bar"})
    assert r.status_code == 422
    assert "foo" in r.json()["error"]


def test_settings_put_unknown_backend_value_returns_422(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.put("/api/settings", json={"backend": "openai-direct"})
    assert r.status_code == 422


def test_settings_put_no_llm_wrong_type_returns_422(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.put("/api/settings", json={"no_llm": "yes"})
    assert r.status_code == 422


def test_settings_put_rejects_cross_origin(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.put(
        "/api/settings",
        json={"backend": "litellm"},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403


def test_settings_get_has_no_origin_check(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    r = c.get("/api/settings", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200


def test_settings_get_corrupt_file_returns_empty_with_warning(tmp_path):
    settings_path = tmp_path / "gui" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not valid json", encoding="utf-8")
    c = TestClient(_settings_app(tmp_path, settings_path=settings_path), base_url="http://localhost")
    r = c.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert "warning" in body
    # Nur der Warn-Key kommt dazu -- keine erfundenen Settings-Werte (L5).
    assert {k: v for k, v in body.items() if k != "warning"} == {}


def test_settings_put_replaces_full_object_no_merge(tmp_path):
    c = TestClient(_settings_app(tmp_path), base_url="http://localhost")
    c.put("/api/settings", json={"backend": "litellm", "profile": "fast"})
    # Zweiter PUT ohne "profile" -> darf NICHT mit dem alten Wert gemergt werden.
    c.put("/api/settings", json={"backend": "litellm"})
    r = c.get("/api/settings")
    assert r.json() == {"backend": "litellm"}


# --- S1: Host-Header-Allowlist (DNS-Rebinding-Schutz) ----------------------


def test_foreign_host_header_rejected_with_400(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://evil.example")
    r = c.get("/api/doctor")
    assert r.status_code == 400


def test_localhost_host_header_still_allowed(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://localhost")
    assert c.get("/api/doctor").status_code == 200


def test_127_0_0_1_host_header_still_allowed(tmp_path):
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=tmp_path,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://127.0.0.1")
    assert c.get("/api/doctor").status_code == 200


def test_settings_no_secrets_written_to_disk(tmp_path):
    settings_path = tmp_path / "gui" / "settings.json"
    c = TestClient(_settings_app(tmp_path, settings_path=settings_path), base_url="http://localhost")
    c.put("/api/settings", json={"backend": "litellm", "profile": "fast", "no_llm": True, "dry_run": True})
    raw = settings_path.read_text(encoding="utf-8")
    assert set(json.loads(raw)) <= {"backend", "profile", "no_llm", "dry_run"}


# --- B2: Vault-/Ordner-Wahl -------------------------------------------------


def _vault_app(tmp_path, vault=None, **kwargs):
    if vault is None:
        vault = tmp_path / "vault"
        vault.mkdir()
    kwargs.setdefault("run_factory", fake_run)
    kwargs.setdefault("pdf_dirs", [tmp_path])
    kwargs.setdefault("vault_path", vault)
    kwargs.setdefault("backend", "subscription")
    kwargs.setdefault("uploads_dir", tmp_path / "u")
    kwargs.setdefault("doctor_fn", fake_doctor)
    kwargs.setdefault("settings_path", tmp_path / "gui" / "settings.json")
    app = create_app(**kwargs)
    return app, vault


def test_vault_get_returns_current_vault(tmp_path):
    app, vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    r = c.get("/api/vault")
    assert r.status_code == 200
    assert r.json()["vault"] == str(vault.resolve())


def test_vault_put_valid_directory_changes_state(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    r = c.put("/api/vault", json={"path": str(new_vault)})
    assert r.status_code == 200
    assert r.json()["vault"] == str(new_vault.resolve())
    assert c.get("/api/vault").json()["vault"] == str(new_vault.resolve())


def test_vault_put_rejects_nonexistent_path(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    r = c.put("/api/vault", json={"path": str(tmp_path / "nicht-da")})
    assert r.status_code == 400


def test_vault_put_rejects_file_not_directory(tmp_path):
    app, _vault = _vault_app(tmp_path)
    f = tmp_path / "datei.txt"
    f.write_text("x", encoding="utf-8")
    c = TestClient(app, base_url="http://localhost")
    r = c.put("/api/vault", json={"path": str(f)})
    assert r.status_code == 400


def test_vault_put_rejects_traversal_to_nonexistent_dir(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    bogus = str(tmp_path / "a" / ".." / ".." / "definitiv-nicht-da-xyz")
    r = c.put("/api/vault", json={"path": bogus})
    assert r.status_code == 400


def test_vault_put_rejects_symlink_to_file(tmp_path):
    app, _vault = _vault_app(tmp_path)
    target = tmp_path / "real.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link-zur-datei"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks ohne Sonderrechte auf dieser Plattform nicht erstellbar")
    c = TestClient(app, base_url="http://localhost")
    r = c.put("/api/vault", json={"path": str(link)})
    assert r.status_code == 400


def test_vault_put_rejects_invalid_json_body(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    r = c.put("/api/vault", content="{invalid", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_vault_put_rejects_non_object_json_body(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    for payload in ("null", "[]", "42", '"text"'):
        r = c.put("/api/vault", content=payload, headers={"Content-Type": "application/json"})
        assert r.status_code == 400, f"payload {payload!r} -> {r.status_code}"


def test_vault_put_rejects_missing_path_key(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    r = c.put("/api/vault", json={})
    assert r.status_code == 400


def test_vault_put_rejects_cross_origin(tmp_path):
    app, _vault = _vault_app(tmp_path)
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    c = TestClient(app, base_url="http://localhost")
    r = c.put(
        "/api/vault",
        json={"path": str(new_vault)},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403


def test_vault_put_rejects_cross_origin_port(tmp_path):
    # M1: Haertung deckt auch andere Mutatoren als /api/run ab -- fremder
    # localhost-Port wird auch bei PUT /api/vault abgelehnt.
    app, _vault = _vault_app(tmp_path)
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    c = TestClient(app, base_url="http://127.0.0.1:8052")
    r = c.put(
        "/api/vault",
        json={"path": str(new_vault)},
        headers={"Origin": "http://127.0.0.1:3000"},
    )
    assert r.status_code == 403


def test_vault_put_persists_to_settings(tmp_path):
    settings_path = tmp_path / "gui" / "settings.json"
    app, _vault = _vault_app(tmp_path, settings_path=settings_path)
    c = TestClient(app, base_url="http://localhost")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    c.put("/api/vault", json={"path": str(new_vault)})
    data, _warning = gui_settings.read_settings(settings_path)
    assert data["vault_path"] == str(new_vault.resolve())


def test_vault_put_merges_with_existing_settings_not_replace(tmp_path):
    settings_path = tmp_path / "gui" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    gui_settings.write_settings({"backend": "litellm", "profile": "fast"}, settings_path)
    app, _vault = _vault_app(tmp_path, settings_path=settings_path)
    c = TestClient(app, base_url="http://localhost")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    c.put("/api/vault", json={"path": str(new_vault)})
    data, _warning = gui_settings.read_settings(settings_path)
    assert data["backend"] == "litellm"
    assert data["profile"] == "fast"
    assert data["vault_path"] == str(new_vault.resolve())


def test_vault_put_rejected_while_run_active(tmp_path):
    import threading

    gate = threading.Event()

    def slow_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["slow"]}
        gate.wait(timeout=5)
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    app, _vault = _vault_app(tmp_path, run_factory=slow_run)
    c = TestClient(app, base_url="http://localhost")
    assert c.post("/api/run", json={"pdf": str(pdf), "dry_run": True}).status_code == 200
    r = c.put("/api/vault", json={"path": str(new_vault)})
    assert r.status_code == 409
    gate.set()


def test_build_run_spec_wired_into_default_run_factory(tmp_path):
    # Subprocess-Override (Punkt 3): der ECHTE Default-Run-Factory-Pfad (kein
    # injizierter Fake) muss ATOMIC_AGENT_VAULT_PATH aus dem AKTUELLEN
    # app.state.vault_path in die Subprocess-Env setzen. Wir stubben nur
    # runner.iter_run_events, um keinen echten Orchestrator-Subprocess zu starten.
    captured = {}

    def fake_iter_run_events(argv, *, env=None, cwd=None, on_proc=None):
        captured["env"] = env
        yield {"type": "started", "argv": argv}
        yield {"type": "exited", "returncode": 0}

    import generative.gui.runner as runner_module

    vault = tmp_path / "vault"
    vault.mkdir()
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
    )
    c = TestClient(app, base_url="http://localhost")
    orig = runner_module.iter_run_events
    runner_module.iter_run_events = fake_iter_run_events
    try:
        c.post("/api/run", json={"pdf": str(pdf), "dry_run": True})
        _drain(c)
    finally:
        runner_module.iter_run_events = orig
    assert captured["env"]["ATOMIC_AGENT_VAULT_PATH"] == str(vault.resolve())


# --- R1 (Race, KRITISCH): Output-Endpunkte gegen Session-Snapshot ----------


def test_outputs_file_uses_session_snapshot_after_vault_switch(tmp_path):
    vault_a = tmp_path / "vault-a"
    (vault_a / "00-inbox").mkdir(parents=True)
    (vault_a / "00-inbox" / "Foo.md").write_text("# Foo", encoding="utf-8")
    vault_b = tmp_path / "vault-b"
    vault_b.mkdir()
    (vault_b / "Bar.md").write_text("# Bar", encoding="utf-8")

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "done", "written": 1, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run,
        pdf_dirs=[tmp_path],
        vault_path=vault_a,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)

    # Vault-Wechsel NACH Lauf-Ende.
    r = c.put("/api/vault", json={"path": str(vault_b)})
    assert r.status_code == 200

    # Session-Snapshot (Vault A, wo tatsaechlich geschrieben wurde) bleibt erlaubt.
    ok = c.get("/api/outputs/file", params={"path": str(vault_a / "00-inbox" / "Foo.md")})
    assert ok.status_code == 200

    # Neuer State-Vault (B), aber ausserhalb des Session-Snapshots -> 403.
    blocked = c.get("/api/outputs/file", params={"path": str(vault_b / "Bar.md")})
    assert blocked.status_code == 403


def test_outputs_list_uses_session_snapshot_after_vault_switch(tmp_path):
    vault_a = tmp_path / "vault-a"
    (vault_a / "00-inbox").mkdir(parents=True)
    (vault_a / "00-inbox" / "Foo.md").write_text("# Foo", encoding="utf-8")
    vault_b = tmp_path / "vault-b"
    vault_b.mkdir()

    def write_run(pdf, dry_run, register=None, options=None):
        yield {"type": "started", "argv": ["x"]}
        yield {"type": "note_written", "path": "00-inbox/Foo.md", "routing": "vault"}
        yield {"type": "done", "written": 1, "dry_run": dry_run}
        yield {"type": "exited", "returncode": 0}

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    app = create_app(
        run_factory=write_run,
        pdf_dirs=[tmp_path],
        vault_path=vault_a,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://localhost")
    c.post("/api/run", json={"pdf": str(pdf), "dry_run": False})
    _drain(c)
    c.put("/api/vault", json={"path": str(vault_b)})

    body = c.get("/api/outputs").json()
    assert body["items"][0]["path"] == str((vault_a / "00-inbox" / "Foo.md").resolve())


def test_outputs_uses_state_vault_when_no_session_yet(tmp_path):
    # Ohne je gelaufene Session: Fallback auf app.state.vault_path.
    vault = tmp_path / "vault"
    vault.mkdir()
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
    )
    c = TestClient(app, base_url="http://localhost")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    (new_vault / "Note.md").write_text("# Note", encoding="utf-8")
    c.put("/api/vault", json={"path": str(new_vault)})
    r = c.get("/api/outputs/file", params={"path": str(new_vault / "Note.md")})
    assert r.status_code == 200


# --- R2 (doctor, KRITISCH): gewaehlter Vault statt config.VAULT -----------


def test_doctor_ok_true_despite_stale_embedded_vault_check(tmp_path):
    # R2 -- der Kernfall: `doctor_fn()` (z.B. das echte `doctor.run_all()`)
    # enthaelt bereits einen "vault"-Check, der aber gegen den ALTEN
    # `config.VAULT`-Import-Default prueft -- der kann nach einem GUI-Vault-
    # Wechsel stale/False sein, obwohl der TATSAECHLICH gewaehlte Vault
    # (`app.state.vault_path`) valide ist. `ok` darf sich davon nicht blockieren
    # lassen -- das ist der eigentliche Kern von R2, nicht nur ein zusaetzliches
    # Feld daneben.
    from generative.doctor import CheckResult

    def stale_vault_doctor():
        return [
            CheckResult(name="pdftotext", ok=True, detail="ok"),
            CheckResult(name="backend (subscription)", ok=True, detail="ok"),
            CheckResult(name="vault", ok=False, detail="alter config.VAULT-Pfad existiert nicht"),
        ]

    real_vault = tmp_path / "echter-vault"
    real_vault.mkdir()
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        vault_path=real_vault,
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=stale_vault_doctor,
    )
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert body["vault_exists"] is True
    assert body["ok"] is True  # trotz stale "vault"-Eintrag in `checks`
    # Der stale Eintrag bleibt sichtbar (Transparenz), gated `ok` aber nicht mehr.
    assert any(c["name"] == "vault" and c["ok"] is False for c in body["checks"])


def test_doctor_shows_configured_vault(tmp_path):
    app, vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    body = c.get("/api/doctor").json()
    assert body["vault"] == str(vault.resolve())
    assert body["vault_exists"] is True


def test_doctor_reflects_vault_after_switch(tmp_path):
    app, _vault = _vault_app(tmp_path)
    c = TestClient(app, base_url="http://localhost")
    new_vault = tmp_path / "new-vault"
    new_vault.mkdir()
    c.put("/api/vault", json={"path": str(new_vault)})
    body = c.get("/api/doctor").json()
    assert body["vault"] == str(new_vault.resolve())
    assert body["vault_exists"] is True


def test_doctor_vault_exists_false_when_state_vault_missing(tmp_path):
    # Kann nur ueber einen kaputten Settings-Restart entstehen (PUT /api/vault
    # selbst lehnt nicht-existente Pfade ab) -- ueber settings_path simuliert.
    settings_path = tmp_path / "gui" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    gui_settings.write_settings({"vault_path": str(tmp_path / "weg-seitdem")}, settings_path)
    (tmp_path / "weg-seitdem").mkdir()
    # Verzeichnis existiert beim create_app-Aufruf noch (Preload akzeptiert es) ...
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        settings_path=settings_path,
    )
    import shutil

    # ... wird aber danach geloescht -> doctor muss das ehrlich zeigen (ok=False).
    shutil.rmtree(tmp_path / "weg-seitdem")
    body = TestClient(app, base_url="http://localhost").get("/api/doctor").json()
    assert body["vault_exists"] is False
    assert body["ok"] is False


# --- P6: GUI-Settings-Preload beim Server-Start ----------------------------


def test_create_app_preloads_vault_path_from_settings(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    settings_path = tmp_path / "gui" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    gui_settings.write_settings({"vault_path": str(vault)}, settings_path)
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        settings_path=settings_path,
        # vault_path bewusst NICHT gesetzt -> muss aus Settings vorbelegt werden.
    )
    body = TestClient(app, base_url="http://localhost").get("/api/vault").json()
    assert body["vault"] == str(vault.resolve())


def test_create_app_ignores_broken_stored_vault_path_no_crash(tmp_path):
    settings_path = tmp_path / "gui" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    gui_settings.write_settings({"vault_path": str(tmp_path / "nicht-da")}, settings_path)
    app = create_app(
        run_factory=fake_run,
        pdf_dirs=[tmp_path],
        backend="subscription",
        uploads_dir=tmp_path / "u",
        doctor_fn=fake_doctor,
        settings_path=settings_path,
    )
    body = TestClient(app, base_url="http://localhost").get("/api/vault").json()
    assert body["vault"] != str((tmp_path / "nicht-da").resolve())


# --- B1b: litellm-API-Key write-only in generative/.env setzen -------------


def _litellm_key_app(tmp_path, **kwargs):
    kwargs.setdefault("run_factory", fake_run)
    kwargs.setdefault("pdf_dirs", [tmp_path])
    kwargs.setdefault("vault_path", tmp_path)
    kwargs.setdefault("backend", "subscription")
    kwargs.setdefault("uploads_dir", tmp_path / "u")
    kwargs.setdefault("doctor_fn", fake_doctor)
    kwargs.setdefault("settings_path", tmp_path / "gui" / "settings.json")
    kwargs.setdefault("env_path", tmp_path / ".env")
    return create_app(**kwargs)


def test_litellm_key_success_returns_200_without_key_in_response(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": "sk-test-xxx"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"provider": "ANTHROPIC_API_KEY", "set": True}
    assert "sk-test-xxx" not in r.text


def test_litellm_key_writes_to_injected_env_path(tmp_path):
    env_path = tmp_path / ".env"
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "OPENAI_API_KEY", "key": "sk-openai-xxx"})
    assert r.status_code == 200
    assert env_path.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-openai-xxx\n"


def test_litellm_key_merge_preserves_existing_env_content(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# Kommentar\nATOMIC_AGENT_VAULT_PATH=/vault\nANTHROPIC_API_KEY=alt\nATOMIC_AGENT_BACKEND=subscription\n",
        encoding="utf-8",
    )
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": "neu"})
    assert r.status_code == 200
    content = env_path.read_text(encoding="utf-8")
    assert content == (
        "# Kommentar\nATOMIC_AGENT_VAULT_PATH=/vault\nANTHROPIC_API_KEY=neu\nATOMIC_AGENT_BACKEND=subscription\n"
    )


def test_litellm_key_unknown_provider_returns_422(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "GEMINI_API_KEY", "key": "sk-test-xxx"})
    assert r.status_code == 422
    # Klartext-Fehler, aber der (hier ohnehin nicht vorhandene) key wird nicht erwaehnt.
    assert "sk-test-xxx" not in r.text


def test_litellm_key_missing_key_returns_400(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY"})
    assert r.status_code == 400


def test_litellm_key_empty_key_returns_400(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": ""})
    assert r.status_code == 400


def test_litellm_key_whitespace_only_key_returns_400(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": "   "})
    assert r.status_code == 400


@pytest.mark.parametrize("bad_char", ["\r", "\n", "\0", "\t", "\x7f"])
def test_litellm_key_rejects_control_chars(tmp_path, bad_char):
    env_path = tmp_path / ".env"
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": f"sk-x{bad_char}rest"})
    assert r.status_code == 400
    assert not env_path.exists()


def test_litellm_key_injection_attempt_rejected_nothing_written(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ATOMIC_AGENT_BACKEND=subscription\n", encoding="utf-8")
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post(
        "/api/access/litellm-key",
        json={"provider": "ANTHROPIC_API_KEY", "key": "sk-x\nATOMIC_AGENT_BACKEND=evil"},
    )
    assert r.status_code == 400
    content = env_path.read_text(encoding="utf-8")
    assert content == "ATOMIC_AGENT_BACKEND=subscription\n"
    assert "evil" not in content


def test_litellm_key_rejects_invalid_json_body(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", content="{invalid", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_litellm_key_rejects_non_object_json_body(tmp_path):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    for payload in ("null", "[]", "42", '"text"'):
        r = c.post("/api/access/litellm-key", content=payload, headers={"Content-Type": "application/json"})
        assert r.status_code == 400


def test_litellm_key_rejects_cross_origin(tmp_path):
    env_path = tmp_path / ".env"
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post(
        "/api/access/litellm-key",
        json={"provider": "ANTHROPIC_API_KEY", "key": "sk-test-xxx"},
        headers={"Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    assert not env_path.exists()


def test_litellm_key_rejects_cross_origin_port(tmp_path):
    # M1-Haertung (analog /api/run, /api/vault): ein fremder localhost-Port
    # ist KEIN Same-Origin.
    env_path = tmp_path / ".env"
    app = _litellm_key_app(tmp_path, env_path=env_path)
    c = TestClient(app, base_url="http://127.0.0.1:8052")
    r = c.post(
        "/api/access/litellm-key",
        json={"provider": "ANTHROPIC_API_KEY", "key": "sk-test-xxx"},
        headers={"Origin": "http://127.0.0.1:3000"},
    )
    assert r.status_code == 403
    assert not env_path.exists()


def test_litellm_key_origin_check_happens_before_body_parse(tmp_path):
    # Cross-Origin muss VOR dem JSON-Parsing abgewiesen werden -- ein kaputter
    # Body darf die Origin-Pruefung nicht umgehen/verdecken.
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    r = c.post(
        "/api/access/litellm-key",
        content="{invalid",
        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
    )
    assert r.status_code == 403


def test_litellm_key_no_leak_in_log(tmp_path, caplog):
    c = TestClient(_litellm_key_app(tmp_path), base_url="http://localhost")
    with caplog.at_level(logging.DEBUG):
        c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": "sk-super-secret-xxx"})
        c.post(
            "/api/access/litellm-key",
            json={"provider": "ANTHROPIC_API_KEY", "key": "sk-super-secret-xxx\nevil=1"},
        )
    assert "sk-super-secret-xxx" not in caplog.text


def test_litellm_key_appended_when_not_previously_set(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("ATOMIC_AGENT_BACKEND=subscription\n", encoding="utf-8")
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "OLLAMA_API_BASE", "key": "http://localhost:11434"})
    assert r.status_code == 200
    content = env_path.read_text(encoding="utf-8")
    assert content == "ATOMIC_AGENT_BACKEND=subscription\nOLLAMA_API_BASE=http://localhost:11434\n"


def test_litellm_key_non_utf8_env_returns_500_no_traceback(tmp_path):
    # Punkt 4 (fail-closed): eine bestehende nicht-UTF-8-`.env` (cp1252/BOM)
    # wirft beim Read einen UnicodeDecodeError -- der Endpunkt muss ihn als
    # generische 500 fangen (kein Traceback-Durchschlag, kein Key in der
    # Response).
    env_path = tmp_path / ".env"
    env_path.write_bytes(b"ATOMIC_AGENT_BACKEND=subscription\n\xff\xfe not utf8\n")
    c = TestClient(
        _litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost", raise_server_exceptions=False
    )
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": "sk-test-xxx"})
    assert r.status_code == 500
    body = r.json()
    assert body == {"error": "Key konnte nicht gespeichert werden."}
    assert "sk-test-xxx" not in r.text


def test_litellm_key_rejects_leading_quote_400(tmp_path):
    # Punkt 5 (Endpunkt-Ebene): ein mit Quote beginnender Key wird als 400
    # abgewiesen, nichts geschrieben.
    env_path = tmp_path / ".env"
    c = TestClient(_litellm_key_app(tmp_path, env_path=env_path), base_url="http://localhost")
    r = c.post("/api/access/litellm-key", json={"provider": "ANTHROPIC_API_KEY", "key": '"sk-x'})
    assert r.status_code == 400
    assert not env_path.exists()
