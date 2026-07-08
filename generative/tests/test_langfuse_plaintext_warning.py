"""S7 (#150): Klartext-Warnung des Langfuse-Backends.

Bei einem nicht-lokalen http://-LANGFUSE_HOST gehen die Basic-Auth-Keys und die
gesendeten Trace-Metadaten im Klartext raus (keine Roh-Prompts — s.
langfuse_backend.write) -> einmalige Warnung. Lokale oder https-Hosts warnen nicht.
"""

from __future__ import annotations

import logging

import pytest

from generative.agents import langfuse_backend
from generative.agents.langfuse_backend import LangfuseBackend


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    # Modul-weites "einmal warnen"-Flag pro Test zuruecksetzen.
    monkeypatch.setattr(langfuse_backend, "_PLAINTEXT_WARNED", False)


def _construct(monkeypatch, host: str):
    monkeypatch.setenv("LANGFUSE_HOST", host)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    LangfuseBackend()


def test_warns_on_http_remote_host(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="generative.agents.langfuse_backend"):
        _construct(monkeypatch, "http://langfuse.example.com")
    assert any("Klartext" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "host",
    ["http://localhost:3000", "http://127.0.0.1:3000", "https://cloud.langfuse.com", "https://langfuse.example.com"],
)
def test_no_warning_for_local_or_https(monkeypatch, caplog, host):
    with caplog.at_level(logging.WARNING, logger="generative.agents.langfuse_backend"):
        _construct(monkeypatch, host)
    assert not any("Klartext" in r.message for r in caplog.records)


def test_warns_only_once(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="generative.agents.langfuse_backend"):
        _construct(monkeypatch, "http://langfuse.example.com")
        _construct(monkeypatch, "http://langfuse.example.com")
    assert sum("Klartext" in r.message for r in caplog.records) == 1
