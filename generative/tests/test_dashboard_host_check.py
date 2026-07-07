"""S1 (#150): Host-Header-Allowlist des Eval-Dashboard-Servers.

Der Dashboard-Server ist ein nackter BaseHTTPRequestHandler (keine
TrustedHostMiddleware wie die GUI) — ohne Host-Check waere er per
DNS-Rebinding aus einem Opfer-Browser lesbar. Wir starten den echten Server
auf einem Ephemeral-Port und schicken Requests mit gefaelschtem Host-Header.
"""

from __future__ import annotations

import http.client
import threading
from http.server import HTTPServer

import pytest

from generative.eval_dashboard_server import Handler, _host_allowed


@pytest.fixture
def server_port():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def _get(port: int, path: str, host: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host)
        conn.endheaders()
        return conn.getresponse().status
    finally:
        conn.close()


# --- Unit: _host_allowed ---------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "127.0.0.1:8051", "localhost:8051", "[::1]:8051", "::1", "LOCALHOST"],
)
def test_host_allowed_accepts_local(host):
    assert _host_allowed(host) is True


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8051", "attacker.com", "", None, "0.0.0.0:8051"])
def test_host_allowed_rejects_foreign_and_missing(host):
    assert _host_allowed(host) is False


# --- Integration: echter Server --------------------------------------------


def test_foreign_host_header_gets_403(server_port):
    assert _get(server_port, "/", "evil.example") == 403


def test_missing_host_header_gets_403(server_port):
    assert _get(server_port, "/", "") == 403


def test_localhost_host_header_passes_check(server_port):
    # Unbekannter Pfad -> 404, aber NICHT 403 -> Host-Check bestanden.
    assert _get(server_port, "/nichtvorhanden", "localhost:1234") == 404
