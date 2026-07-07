"""Tests für #146 H2: klare RAM-Fehlermeldung beim ML-Modell-Laden.

torch/sentence-transformers scheitern unter RAM-Druck typischerweise schon beim
DLL-Load (`OSError [WinError 1455]: Die Auslagerungsdatei ist zu klein`) oder
mit MemoryError — vor dem Fix schlug das als nackter Traceback mit rc=1 durch
(erste Berührung: Entity-Resolution Stage 1, `orchestrator.py`). Erwartung:
`embeddings._model()` übersetzt Ladefehler in einen RuntimeError mit
actionabler Meldung (RAM-Hinweis), verschluckt sie aber NICHT (kein fail-soft
— Embeddings sind funktional nötig).

Dazu: Guard-Test für die stdout/stderr-UTF-8-Rekonfiguration am
Orchestrator-Import — Umgebungen ohne reconfigure-fähige Streams (pythonw
setzt sys.stdout=None, Test-Runner ersetzen Streams) dürfen nicht sterben.
"""

from __future__ import annotations

import sys
import types

import pytest

from generative.pipeline import embeddings


def _fake_sentence_transformers(monkeypatch, exc: BaseException) -> None:
    """Ersetzt das sentence_transformers-Modul durch einen Fake, dessen
    Konstruktor `exc` wirft — simuliert den Erstlade-Fehler ohne echte
    ML-Deps/Downloads. Setzt auch den Modell-Singleton zurück."""

    def _boom(*args, **kwargs):
        raise exc

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(embeddings, "_MODEL", None)


# ---- _model(): Ladefehler → RuntimeError mit RAM-Hinweis -----------------


def test_oserror_1455_raises_ram_hint(monkeypatch):
    """WinError 1455 (Auslagerungsdatei zu klein) beim torch-DLL-Load →
    RuntimeError mit RAM-Hinweis statt nacktem OSError-Traceback."""
    original = OSError(1455, "Die Auslagerungsdatei ist zu klein für diesen Vorgang")
    _fake_sentence_transformers(monkeypatch, original)

    with pytest.raises(RuntimeError, match="RAM-Druck") as excinfo:
        embeddings._model()

    msg = str(excinfo.value)
    assert "paraphrase-multilingual-MiniLM-L12-v2" in msg
    assert "1455" in msg  # Original-Fehler bleibt sichtbar
    assert "erneut versuchen" in msg
    assert excinfo.value.__cause__ is original  # Ursache verkettet, nicht verschluckt


def test_memoryerror_raises_ram_hint(monkeypatch):
    _fake_sentence_transformers(monkeypatch, MemoryError())

    with pytest.raises(RuntimeError, match="RAM-Druck"):
        embeddings._model()


def test_importerror_passes_through_unchanged(monkeypatch):
    """Fehlende Dependency ist KEIN RAM-Problem — ImportError darf nicht in
    die RAM-Meldung umgeschrieben werden."""
    _fake_sentence_transformers(monkeypatch, ImportError("No module named 'torch'"))

    with pytest.raises(ImportError):
        embeddings._model()


def test_load_failure_is_not_swallowed(monkeypatch):
    """Kein fail-soft: _model() darf nach Ladefehler nichts zurückgeben."""
    _fake_sentence_transformers(monkeypatch, OSError(1455, "page file too small"))

    with pytest.raises(Exception):
        embeddings._model()
    assert embeddings._MODEL is None  # kein kaputter Singleton gecacht


# ---- load_ml_model-Helper (auch vom NLI-CrossEncoder-Pfad genutzt) --------


def test_helper_success_passthrough():
    sentinel = object()
    assert embeddings.load_ml_model(lambda: sentinel, "x") is sentinel


def test_helper_wraps_runtimeerror():
    def _factory():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="RAM-Druck"):
        embeddings.load_ml_model(_factory, "some-model")


# ---- Orchestrator-Import: reconfigure-Guard -------------------------------


def test_reconfigure_guard_survives_streams_without_reconfigure(monkeypatch):
    """Ersetzte Streams (Test-Runner, StringIO) haben kein reconfigure —
    der Guard darf nicht mit AttributeError sterben."""
    from generative import orchestrator

    class _NoReconfigure:
        def write(self, s):  # minimales Stream-Interface
            return len(s)

    monkeypatch.setattr(sys, "stdout", _NoReconfigure())
    monkeypatch.setattr(sys, "stderr", _NoReconfigure())
    orchestrator._reconfigure_streams_utf8()  # darf nicht werfen


def test_reconfigure_guard_survives_none_streams(monkeypatch):
    """pythonw setzt sys.stdout/sys.stderr auf None — Guard darf nicht sterben."""
    from generative import orchestrator

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    orchestrator._reconfigure_streams_utf8()  # darf nicht werfen
