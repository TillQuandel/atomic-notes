"""Tests für den thread-lokalen Call-Record in base.py (Issue #17).

Damit ein Stage-6-Crash-Handler Prompt + rohen Output des gecrashten Calls
eindeutig zuordnen kann (auch bei N parallelen Notes via to_thread), merkt sich
call_claude_full pro Thread den letzten Call.
"""

import threading

import generative.agents.base as base
from generative.agents.base import CallResult


def test_last_call_record_set_on_success(monkeypatch):
    monkeypatch.setattr(base, "_backend_call_full", lambda prompt, **k: CallResult(text="roher output"))
    base.clear_last_call_record()

    base.call_claude_full("mein prompt", agent="verifier", use_cache=False)

    rec = base.get_last_call_record()
    assert rec is not None
    assert rec["agent"] == "verifier"
    assert rec["prompt"] == "mein prompt"
    assert rec["raw_output"] == "roher output"
    assert rec["error"] is None


def test_last_call_record_set_on_runtime_error(monkeypatch):
    def boom(prompt, **k):
        raise RuntimeError("backend crashed")

    monkeypatch.setattr(base, "_backend_call_full", boom)
    base.clear_last_call_record()

    try:
        base.call_claude_full("crash prompt", agent="critic", use_cache=False)
    except RuntimeError:
        pass

    rec = base.get_last_call_record()
    assert rec is not None
    assert rec["agent"] == "critic"
    assert rec["prompt"] == "crash prompt"
    assert rec["error"] == "backend crashed"
    assert rec["raw_output"] == ""


def test_last_call_record_is_thread_local(monkeypatch):
    monkeypatch.setattr(base, "_backend_call_full", lambda prompt, **k: CallResult(text="x"))
    base.clear_last_call_record()
    captured = {}

    def worker():
        base.call_claude_full("p", agent="cross_reference", use_cache=False)
        captured["thread"] = base.get_last_call_record()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # Der Worker-Thread sieht seinen eigenen Record ...
    assert captured["thread"]["agent"] == "cross_reference"
    # ... der Hauptthread (der keinen Call machte) bleibt unberührt.
    assert base.get_last_call_record() is None
