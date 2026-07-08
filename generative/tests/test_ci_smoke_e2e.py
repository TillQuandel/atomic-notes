"""CI-Smoke-E2E: `orchestrator.main()` mit gestubbtem LLM-Backend (Issue #97).

Kontext: Im CI läuft kein einziger echter `orchestrator.main()`-Durchlauf — die
kanonische Suite testet pure Helper, die echten E2E-Tests (test_e2e_baseline.py)
sind `@pytest.mark.slow` und global deselektiert. Drei dokumentierte Wiring-Bugs
(q_title-NameError, fb-NameError, quality-Modul-Shadowing) erreichten master
trotz hunderter grüner Tests, weil nichts den vollen Stage-Wiring-Pfad exerziert.

Dieser Test ruft `main()` DIREKT (in-process, kein Subprozess) auf dem echten
Beispiel-PDF auf: echtes pdftotext-Parsing, aber LLM-Backend/Embeddings/Netz
deterministisch gestubbt. Läuft in der kanonischen Suite (kein slow-Marker),
Laufzeit-Budget < 60s.
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import pytest

from generative import orchestrator
from generative.agents import base as agents_base
from generative.agents import context_builder
from generative.agents import quality as quality_agent
from generative import embeddings as embeddings_mod
from generative.schemas.atomic_note import QualityReport

pytestmark = pytest.mark.skipif(
    shutil.which("pdftotext") is None,
    reason="pdftotext (poppler) nicht installiert — CI hat es auf allen 3 OS (ci.yml), lokal ggf. nachinstallieren.",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PDF = REPO_ROOT / "examples" / "zettelkasten-primer.pdf"

# Echte, wörtliche Zitate aus examples/zettelkasten-primer.pdf (per
# `pdf_chunker.pdf_to_text` verifiziert) — der Verifier-Tier-1-Prepass matcht
# exact-substring gegen den echten Chunk-Text; ein erfundenes Zitat bliebe
# unresolved und würde die Anker-Verifikation nicht wirklich exerzieren.
_QUOTE_ATOMIC = "An atomic note captures exactly one idea."
_QUOTE_PROGRESSIVE = (
    "Progressive summarization is a reading and note-taking strategy that processes source material in layers."
)


class _NetworkGuardTripped(RuntimeError):
    """Wird geworfen wenn der Smoke-Test einen echten HTTP-Versuch abfängt."""


class _FakeEmbeddingModel:
    """Deterministisches Hash-basiertes Fake statt echtem MiniLM-Download.

    Zero-centered (Byte-128) statt roher 0..255-Werte: reale Sentence-Embeddings
    streuen um 0, nicht rein positiv — sonst wäre die Cosine zwischen zwei
    UNÄHNLICHEN Titeln systematisch zu hoch und würde Stage-1-Blocking der
    Entity-Resolution (ER_TITLE_COSINE_THRESHOLD=0.93) verfälschen.
    """

    _DIM = 16

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIM

    def encode(self, sentences, **_kwargs):
        import hashlib

        import numpy as np

        vecs = []
        for s in sentences:
            h = hashlib.sha256(s.encode("utf-8")).digest()
            v = np.frombuffer(h[: self._DIM], dtype=np.uint8).astype(float) - 128.0
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs)


_PLANNER_RESPONSE = """\
source_title: A Short Primer on Atomic Note-Taking
source_summary: Erklaert Atomic Notes und Progressive Summarization im Zettelkasten-Kontext.
<!--CONCEPT-->
title: Atomic Note
priority: high
chapter: What Is an Atomic Note?
action: create
extend_path:
category: conceptual
origin: primary
cited_authors:
<!--CONCEPT-->
title: Progressive Summarization
priority: high
chapter: Progressive Summarization
action: create
extend_path:
category: conceptual
origin: primary
cited_authors:
<!--END-->
"""


def _extractor_response(title: str, quote: str) -> str:
    """Baut eine minimale, aber schema-gültige Extractor-Antwort mit einem
    Inline-`(S. 1)`-Anker im Body UND einem strukturierten ANCHOR/QUOTE-Block
    mit wörtlichem PDF-Zitat (Issue-#97-Akzeptanzkriterium)."""
    return f"""\
<!--NOTE-->
title: {title}
aliases: {title}
tags:
proposed_tags:
synthesis_confidence: low
action: create
extend_path:
<!--BODY-->
# {title}: aus der Quelle destillierte Kerncharakteristik

{quote} (S. 1)
<!--ANCHOR-->
page: S. 1
<!--QUOTE-->
{quote}
<!--END-->
"""


def _fake_check_quality(doi=None, author=None, year=None, title=None) -> QualityReport:
    """Quality-Agent-Fixture mit Title-Match-DOI-Pfad (Issue #97 Bug 1).

    `doi_from_title_match=True` + ein absichtlich abweichender `crossref_title`
    lässt `crossref_override_blocked()` in main() True zurückgeben — exakt der
    Zweig, in dem der historische `q_title`-NameError lebte (nur bei PDFs mit
    Title-Match-DOI-Pfad erreichbar, sonst nie exerziert).
    """
    return QualityReport(
        peer_reviewed=None,
        citation_count=None,
        retracted=False,
        flags=["ℹ️ DOI per Title-Match gefunden: 10.9999/stub-doi"],
        crossref_title="Ein völlig anderer, nicht verwandter CrossRef-Titel",
        crossref_author=None,
        crossref_year=None,
        doi_from_title_match=True,
    )


def _make_backends(network_calls: list[str]):
    """Agent-abhängiger Fixture-Router. Unbekannte Agents machen den Test ROT
    (fail-closed) statt still eine Fallback-Antwort zu liefern — deckt genau
    die Klasse Wiring-Bug ab, die dieser Smoke-Test fangen soll: ein Stage-Call
    der im echten Lauf unerwartet das LLM erreicht."""

    def _dispatch(prompt: str, *, agent: str, **_kwargs) -> agents_base.CallResult:
        if agent == "planner":
            return agents_base.CallResult(text=_PLANNER_RESPONSE)
        if agent == "extractor":
            if "- Atomic Note (Priorität" in prompt:
                return agents_base.CallResult(text=_extractor_response("Atomic Note", _QUOTE_ATOMIC))
            if "- Progressive Summarization (Priorität" in prompt:
                return agents_base.CallResult(text=_extractor_response("Progressive Summarization", _QUOTE_PROGRESSIVE))
            raise AssertionError(f"extractor-Prompt ohne erkanntes Fixture-Konzept:\n{prompt[:500]}")
        raise AssertionError(
            f"Unerwarteter Agent-Call '{agent}' — --no-llm soll Stage-6-Agents "
            "(Verifier/CrossRef/Critic) auf den FOSS-Modus schalten. Der Test-Stub "
            "kennt nur 'planner' und 'extractor'; ein weiterer Agent-Call zeigt "
            "entweder einen Wiring-Bug (Stage faellt nicht in FOSS-Modus) oder "
            "eine unvollstaendige Test-Fixture an."
        )

    def _sync_backend(prompt: str, *, model: str, agent: str = "unknown", **kwargs) -> agents_base.CallResult:
        return _dispatch(prompt, agent=agent, model=model, **kwargs)

    async def _async_backend(prompt: str, *, model: str, agent: str = "unknown", **kwargs) -> agents_base.CallResult:
        return _dispatch(prompt, agent=agent, model=model, **kwargs)

    return _sync_backend, _async_backend


def _install_network_guard(monkeypatch, network_calls: list[str]) -> None:
    """Blockiert JEDEN echten HTTP-Versuch (urllib — der einzige Netz-Client im
    Pipeline-Code, siehe Recherche Issue #97). Zeichnet Versuche in
    `network_calls` auf statt sich auf die Exception-Propagation zu verlassen —
    manche Aufrufer (z.B. quality._http_json) fangen `except Exception` breit
    ab und würden einen bloß geworfenen Fehler stillschweigend schlucken."""

    def _blocked_urlopen(request, *args, **kwargs):
        url = getattr(request, "full_url", None) or str(request)
        network_calls.append(url)
        raise _NetworkGuardTripped(f"Netz-Zugriff im Smoke-Test blockiert: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)


def test_ci_smoke_e2e_full_stage_wiring(tmp_path, monkeypatch, capsys):
    """Voller Stage-0..7-Wiring-Pfad inkl. CrossRef-Override-Zweig, ohne Netz/LLM.

    Akzeptanzkriterium (Issue #97): mind. 1 Note im Dry-Run verarbeitet, kein
    Crash, kein echter Netzwerk-/unerwarteter LLM-Call, Laufzeit < 60s.
    """
    network_calls: list[str] = []
    _install_network_guard(monkeypatch, network_calls)

    # Kein Dashboard-Autostart / Version-Bump (main() liest das zur Laufzeit).
    monkeypatch.setenv("ATOMIC_AGENT_GUI", "1")
    # Stage 8 (Qualitäts-Eval) aus — bewusst nicht Ziel dieses Smokes (Issue-Text:
    # "inhaltliche Qualität prüfen ... bleibt bei den slow-Baselines").
    monkeypatch.setenv("ATOMIC_AGENT_PROFILE", "fast")
    # Defensiv aus, falls in Tills Shell global gesetzt — sonst laedt cross_reference
    # ein zweites sentence-transformers-Modell (CrossEncoder) bzw. Phoenix versucht
    # einen lokalen Server zu erreichen.
    monkeypatch.delenv("ENABLE_NLI_VALIDATION", raising=False)
    monkeypatch.delenv("ATOMIC_AGENT_TRACING", raising=False)
    monkeypatch.delenv("ENABLE_ACRONYM_LLM_FALLBACK", raising=False)

    # Embeddings: Fake-Modell statt MiniLM-Download (Rechercheauftrag Issue #97).
    # Patch auf Modul-Ebene — embed_body/embed_title/_default_semantic_presence
    # lesen `_model` als Modul-Attribut bzw. lazy-importieren es pro Call.
    monkeypatch.setattr(embeddings_mod, "_model", lambda: _FakeEmbeddingModel())

    # Vault-Scan (Stage 2, context_builder.build_relevance_profile): leerer
    # tmp-Vault statt Tills echtem Obsidian-Vault — deterministisch, schnell,
    # kein Read auf privaten Vault-Content in einem Test.
    fake_vault = tmp_path / "fake-vault"
    fake_vault.mkdir()
    monkeypatch.setattr(context_builder, "VAULT", fake_vault)

    # Quality-Agent (Stage 3): Title-Match-DOI-Fixture, siehe Docstring oben.
    monkeypatch.setattr(quality_agent, "check_quality", _fake_check_quality)

    # Stage 0 (PDF-Enrichment): das Beispiel-PDF hat kein Author/Year im
    # Info-Dict → Enrichment würde real CrossRef/arXiv/PubMed/OpenLibrary/Google
    # Books abfragen. Deterministisch auf "nichts gefunden" stubben (rename=False
    # ohnehin erzwungen, dies spart nur den echten Netz-Versuch).
    import generative.tools.pdf_enrich as pdf_enrich_mod

    monkeypatch.setattr(pdf_enrich_mod, "enrich", lambda *a, **k: None)

    sync_backend, async_backend = _make_backends(network_calls)
    monkeypatch.setattr(agents_base, "_backend_call_full", sync_backend)
    monkeypatch.setattr(agents_base, "_backend_call_full_async", async_backend)

    # PDF-Kopie unter EIGENEM Namen: write_note schreibt Dry-Run-eval-Kopien
    # hart nach `.cache/eval/baseline/<pdf-stem>/` (vault_writer, nicht
    # parametrisierbar) — mit dem Original-Namen würde dieser Test bei jedem
    # lokalen Suite-Lauf die ECHTE zettelkasten-primer-Baseline mit Stub-Notes
    # überschreiben (empirisch passiert, 2026-07-06). Eigener Stub-Namespace
    # `ci-smoke-fixture` + Cleanup im finally unten.
    smoke_pdf = tmp_path / "ci-smoke-fixture.pdf"
    shutil.copyfile(EXAMPLE_PDF, smoke_pdf)
    smoke_eval_dir = REPO_ROOT / "generative" / ".cache" / "eval" / "baseline" / "ci-smoke-fixture"

    inbox_dir = tmp_path / "inbox"
    argv = [
        "--source",
        str(smoke_pdf),
        "--dry-run",
        "--no-llm",
        "--inbox-dir",
        str(inbox_dir),
    ]

    # main() setzt globalen Backend-Laufzeit-State (set_llm_runtime_config) und
    # gibt ihn nie frei — ein State-Leak, der bei in-process-main()-Aufrufen (im
    # Gegensatz zu den subprocess-basierten slow-E2E-Tests) in nachfolgende
    # Tests derselben Session durchsickert (empirisch gefunden: test_phoenix_span.py
    # brach danach, weil dessen Backend-Lambdas plötzlich call_timeout_sec/
    # timeout_retries-Kwargs bekamen). Muss unabhängig vom Testausgang zurückgesetzt werden.
    t0 = time.time()
    try:
        orchestrator.main(argv)
    finally:
        agents_base.clear_llm_runtime_config()
        # Stub-Baseline-Namespace aufräumen (kein Artefakt-Müll im eval-Cache).
        shutil.rmtree(smoke_eval_dir, ignore_errors=True)
    elapsed_s = time.time() - t0

    captured = capsys.readouterr()
    out = captured.out
    err = captured.err

    # Netz-Blockade als Beweis: kein einziger echter HTTP-Versuch.
    assert network_calls == [], f"Echter Netz-Zugriff im Smoke-Test aufgezeichnet: {network_calls}"

    # Kein Stage-6-Crash (würde als Crash-Report + [Stage-6-Crash]-Zeile auf stderr landen).
    assert "[Stage-6-Crash]" not in err, f"Stage-6-Crash im Lauf:\n{err}"
    # Keine Note wurde vom Extractor-Async-Pfad still verworfen (asyncio.gather
    # schluckt Exceptions aus run_per_concept als [WARN]/[extractor-empty] statt
    # sie zu propagieren — ein stiller Drop hier waere ein Fixture- oder Wiring-Fund).
    assert "[WARN] Extractor" not in err, f"Extractor-Task fehlgeschlagen:\n{err}"
    assert "[extractor-empty]" not in err, f"Extractor lieferte keinen verwertbaren Output:\n{err}"

    # Mindestens 1 Note im Dry-Run verarbeitet.
    assert "[DRY-RUN]" in out, f"Kein Dry-Run-Output — Pipeline lief nicht durch:\n{out[-2000:]}"
    assert "=== Fertig:" in out
    import re

    m = re.search(r"=== Fertig:\s+(\d+)\s+Notes", out)
    assert m is not None, f"Kein 'Fertig'-Report gefunden:\n{out[-1000:]}"
    assert int(m.group(1)) >= 1

    # Bug-1-Zweig (q_title / CrossRef-Override) nachweislich durchlaufen —
    # nicht nur der triviale False-Fall (siehe _build_citation in orchestrator.py).
    assert "CrossRef-Override verworfen" in out, (
        "Der historische q_title-Bug-Zweig (crossref_override_blocked) wurde nicht "
        f"durchlaufen — Quality-Fixture greift nicht:\n{out[:2000]}"
    )

    assert elapsed_s < 60, f"Smoke-Test-Laufzeit {elapsed_s:.1f}s ueber Budget (< 60s)"
