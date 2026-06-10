"""Tests für atomic-notes doctor (generative/doctor.py) — Preflight-Checks als pure Funktionen."""
from __future__ import annotations

from pathlib import Path

from generative import doctor


# --- Poppler ---

def test_pdftotext_fehlt():
    r = doctor.check_tool("pdftotext", which=lambda n: None)
    assert r.ok is False
    assert "poppler" in r.hint.lower()


def test_pdftotext_gefunden():
    r = doctor.check_tool("pdftotext", which=lambda n: "C:/poppler/bin/pdftotext.exe")
    assert r.ok is True
    assert "pdftotext" in r.detail


# --- Backend: subscription ---

def test_subscription_cli_fehlt(tmp_path):
    r = doctor.check_backend("subscription", which=lambda n: None, home=tmp_path, env={})
    assert r.ok is False
    assert "claude" in r.hint.lower()
    assert "litellm" in r.hint.lower()  # Alternative nennen


def test_subscription_nicht_eingeloggt(tmp_path):
    r = doctor.check_backend(
        "subscription", which=lambda n: "/usr/bin/claude", home=tmp_path, env={}
    )
    assert r.ok is False
    assert "einlogg" in r.hint.lower() or "login" in r.hint.lower()


def test_subscription_eingeloggt(tmp_path):
    cred = tmp_path / ".claude" / ".credentials.json"
    cred.parent.mkdir()
    cred.write_text("{}", encoding="utf-8")
    r = doctor.check_backend(
        "subscription", which=lambda n: "/usr/bin/claude", home=tmp_path, env={}
    )
    assert r.ok is True


# --- Backend: litellm ---

def test_litellm_ohne_key(tmp_path):
    r = doctor.check_backend("litellm", which=lambda n: None, home=tmp_path, env={})
    assert r.ok is False
    assert "key" in r.hint.lower()


def test_litellm_mit_key(tmp_path):
    r = doctor.check_backend(
        "litellm", which=lambda n: None, home=tmp_path,
        env={"ANTHROPIC_API_KEY": "sk-test"},
    )
    assert r.ok is True


# --- Vault ---

def test_vault_fehlt(tmp_path):
    r = doctor.check_vault(tmp_path / "gibt-es-nicht")
    assert r.ok is False
    assert "ATOMIC_AGENT_VAULT_PATH" in r.hint


def test_vault_vorhanden_und_beschreibbar(tmp_path):
    r = doctor.check_vault(tmp_path)
    assert r.ok is True


# --- Gesamtlauf ---

def test_main_exit_0_wenn_alles_ok(monkeypatch, capsys):
    ok = doctor.CheckResult(name="x", ok=True, detail="gut")
    monkeypatch.setattr(doctor, "run_all", lambda: [ok])
    assert doctor.main() == 0
    assert "x" in capsys.readouterr().out


def test_main_exit_1_und_hint_bei_fehlschlag(monkeypatch, capsys):
    bad = doctor.CheckResult(name="pdftotext", ok=False, detail="fehlt", hint="poppler installieren")
    monkeypatch.setattr(doctor, "run_all", lambda: [bad])
    assert doctor.main() == 1
    out = capsys.readouterr().out
    assert "poppler installieren" in out
