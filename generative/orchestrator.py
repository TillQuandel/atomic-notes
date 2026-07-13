#!/usr/bin/env python3
"""
atomic-agent — Multi-Agenten-Pipeline: Quelle → Atomic Notes im Vault.

Usage:
    atomic-notes run --source path/to/file.pdf
    atomic-notes run --source path/to/file.pdf --dry-run
    atomic-notes run --source path/to/file.pdf --doi 10.1234/xyz
    (äquivalent: python -m generative.orchestrator …)

Ablauf:
    1. Input-Pipeline: PDF → Text → Chunks
    2. Context-Builder: Vault-Scan → Relevanz-Profil
    3. Quality-Agent: CrossRef/OpenAlex → QualityReport  (parallel zu 2)
    4. Planner: TOC+Intro → ConceptPlan
    5. Extractor × N Chunks: Chunk → Draft-Notes         (parallel)
    6. Verifier, Cross-Reference, Critic pro Note        (sequenziell pro Note)
    7. Vault-Writer: Note → 04-wissen/ oder 00-inbox/
"""

from __future__ import annotations
import argparse
import asyncio
import contextlib
import dataclasses
import json
import os
import sys
import threading
import traceback
from pathlib import Path

_TRACER = None  # gesetzt von _setup_phoenix_tracing wenn Phoenix läuft
_PROVIDER = None  # TracerProvider von register() — für force_flush am Prozess-Ende


@contextlib.contextmanager
def _span(name: str, **attrs):
    """OTel-Stage-Span wenn Phoenix aktiv, sonst no-op."""
    if _TRACER is None:
        yield
        return
    with _TRACER.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        yield span


# Windows-Terminal-Codepage ignoriert PYTHONIOENCODING für bestimmte Print-Pfade.
# UTF-8 verhindert UnicodeEncodeError (rc=1) bei gepipetem Output ohne
# PYTHONIOENCODING, sobald ER-/Stage-6-Logzeilen Nicht-cp1252-Zeichen
# (⊂, ↔, ✓, ⚠️) drucken — #146 H1.
def _reconfigure_streams_utf8() -> None:
    """stdout/stderr defensiv auf UTF-8 umstellen. Guard: Umgebungen ohne
    reconfigure-fähige Streams (pythonw setzt sys.stdout=None, Test-Runner
    ersetzen Streams) dürfen den Import nicht mit AttributeError beenden."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


_reconfigure_streams_utf8()

from generative.agents import (
    context_builder,
    quality,
    planner,
    extractor,
    background_extractor,
    verifier,
    cross_reference,
    confidence,
    critic,
    canonicalizer,
)
from generative import embeddings
from generative.pipeline import (
    pdf_chunker,
    vault_writer,
    acronym_fix,
    anchor_repair,
    boilerplate_dedup,
    citation_check,
    export_runner,
    figure_alt,
    routing_report,
)
from generative.pipeline.page_index import build_page_index
from generative.schemas.atomic_note import AtomicNoteDraft, ConceptPlan
from generative.schemas.citation import CitationMeta, build_citation_meta, crossref_override_blocked
from generative.schemas.run_context import RunContext
from shared.path_safety import resolve_source_path
from generative.config import (
    AGENT_VERSION,
    CRITIC_AUTO_THRESHOLD,
    ER_BODY_COSINE_THRESHOLD,
    ER_TITLE_COSINE_THRESHOLD,
    ER_MAX_TOKEN_DIFF,
    ER_HUB_GENERIC_TOKENS,
    ENABLE_ENTITY_RESOLUTION,
    ENABLE_BACKGROUND_EXTRACTOR,
    MAX_CONCURRENT_CALLS,
    ENABLE_LLM_DEDUP,
    ER_AMBIGUOUS_LOWER,
    MODEL_LLM_DEDUP,
    MAX_CHUNKS_SHORT_DOC,
    MAX_PAGES_SHORT_DOC,
    REDUNDANT_SIBLING_COSINE_THRESHOLD,
    ENABLE_FAITHFULNESS_GATE,
    is_maintainer,
)
from generative.runtime_config import (
    load_runtime_config,
    cap_actionable_concepts,
    RunBudget,
    refine_accepted,
    should_attempt_refine,
    LEGACY,
)

LARGE_DOC_THRESHOLD = 15


def _extract_primary_authors(citation: CitationMeta | None) -> list[str]:
    """Normalisierte Autor-Nachnamen aus CitationMeta für Planner-origin-Klassifikation.

    Unterstützt: "Lastname, F." / "Firstname Lastname" / "A & B" / "A et al."
    Gibt Liste von Nachnamen zurück, leer bei fehlendem/unbekanntem Author-Feld.
    """
    if not citation or not citation.author:
        return []
    import re

    raw = citation.author
    if not raw or raw.strip() in ("?", "unknown", ""):
        return []
    # "et al." vorab entfernen (kann am Ende stehen oder ein Segment sein)
    raw = re.sub(r"\s*,?\s*et\s+al\.?", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return []
    # Trenne bei " & ", "and", ";" oder Komma gefolgt von Großbuchstaben (Trenn-Komma)
    parts = re.split(r"\s*(?:&|and|;)\s*|\s*,\s*(?=[A-Z])", raw)
    authors = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # "Lastname, F." Format — Nachname vor dem Komma
        if "," in part:
            lastname = part.split(",")[0].strip()
        else:
            # "Firstname Lastname" → letztes Wort; "Lastname" allein → direkt
            lastname = part.split()[-1].strip() if " " in part else part
        # Initialen (einzelner Buchstabe ± Punkt) überspringen
        if re.match(r"^[A-Za-z]\.?$", lastname):
            continue
        if lastname:
            authors.append(lastname)
    return authors


def _background_extractor_by_chapter_skip_line(gate_enabled: bool) -> str | None:
    """#102: Sichtbarkeits-Fix für den --by-chapter-Pfad. Stage 4.5 (Background-
    Extractor) läuft dort NIE (Background-Calls würden sich pro Kapitel
    multiplizieren) — bewusste Kosten-Entscheidung (Variante a), keine
    Wiring-Lücke. Variante (b), echte Verdrahtung pro Kapitel, ist bewusst
    nicht gebaut (eigenes Feature-Issue bei Bedarf).

    Gibt die Log-Zeile nur zurück wenn das Gate an ist — sonst doppelt-
    verwirrend, weil dann ohnehin nichts liefe (analog zum Single-Doc-
    else-Zweig, der ebenfalls nur bei Bedarf meldet).
    """
    if not gate_enabled:
        return None
    return "[4.5/7] Background-Extractor: übersprungen im --by-chapter-Modus (bewusst — Kosten pro Kapitel)"


async def run_extractors_per_concept(
    full_text: str,
    concept_plan: ConceptPlan,
    existing_concepts: dict,
    citation: CitationMeta | None = None,
    tag_whitelist: list[str] | None = None,
    background_map: dict[str, list[str]] | None = None,
    related_mentions: list[str] | None = None,
    max_concurrent_calls: int | None = None,
) -> tuple[list[AtomicNoteDraft], dict, int, list[tuple[str, str]]]:
    """Pro Konzept ein Extractor-Call mit den relevanten Textstellen aus ALLEN Chunks.

    Konzepte mit action='skip' werden übersprungen. Konzepte ohne Treffer im Volltext
    werden vor dem LLM-Call verworfen (zusätzlicher Halluzinations-Schutz neben
    planner.filter_hallucinated).

    max_concurrent_calls: aus RuntimeConfig gespeist (#101); None → Legacy-Fallback
    auf die feste Konstante MAX_CONCURRENT_CALLS.

    Returns: (drafts, concept_map, dropped, failures).
    concept_map[concept.title] = (concept, ctext) für Self-Refine-Loop (Milestone 3.6).
    dropped = len(tasks) - len(drafts) (Fehler UND legitime Leer-Extraktionen, für n_dropped).
    failures = [(concept_title, error)] NUR für Calls, die mit Exception starben (#210,
    Timeout/CLI-Fehler nach Retries) — NICHT für None-Rückgaben (Konzept zu schwach im
    Text). Trennt echten Verlust vom erwarteten Leer-Fall, damit Summary/Exit-Code nur
    echte Ausfälle melden.
    """
    sem = asyncio.Semaphore(max_concurrent_calls if max_concurrent_calls is not None else MAX_CONCURRENT_CALLS)

    async def _run_with_sem(concept, ctext):
        async with sem:
            bg = (background_map or {}).get(concept.title)
            return await extractor.run_per_concept(
                concept=concept,
                concept_text=ctext,
                existing_concepts=existing_concepts,
                citation=citation,
                tag_whitelist=tag_whitelist,
                background_context=bg,
                related_mentions=related_mentions,
            )

    tasks: list = []
    concept_for_idx: list = []  # parallele Liste für besseres Logging
    contexts: list = []  # parallele Liste mit (concept, ctext) für concept_map
    for c in concept_plan.concepts:
        if c.action == "skip" or c.origin == "secondary_mention":
            continue
        # Search-Terms: Konzept-Titel + ggf. Aliase aus Title (Kuhlthau, ISP, …)
        search_terms = [c.title]
        # Heuristisch: Tokens des Titels die nicht Stoppwörter sind
        from generative.agents.cross_reference import _tokens

        search_terms.extend(t for t in _tokens(c.title) if len(t) >= 4)
        # Fenster sammeln
        from generative.pipeline.pdf_chunker import concept_text_window

        # window_words=400 = neue Option-D-Semantik (Fenster-Größe für Sliding-Window-Scoring),
        # nicht mehr ±expansion wie vor 2026-05-17.
        ctext = concept_text_window(full_text, search_terms, window_words=400)
        if not ctext.strip():
            print(f"      [skip] '{c.title}' nicht im Volltext gefunden (Halluzinations-Schutz)", file=sys.stderr)
            # #197 Nachbesserung: bisher stummer Pre-Call-Drop (Konzept nicht im
            # Volltext) → Funnel-Event. Konfliktfrei zu #216 (das diesen Block nicht anfasst).
            _trace_stage_outcome(
                c.title, "extractor", "dropped", drop_reason="empty_extraction", detail="not in fulltext"
            )
            continue
        tasks.append(_run_with_sem(c, ctext))
        concept_for_idx.append(c.title)
        contexts.append((c, ctext))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    drafts: list[AtomicNoteDraft] = []
    concept_map: dict = {}  # draft.title -> (concept, ctext)
    failures: list[tuple[str, str]] = []  # #210: (title, error) nur für harte Call-Ausfälle
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [WARN] Extractor '{concept_for_idx[i]}' fehlgeschlagen: {r}", file=sys.stderr)
            # #197 Nachbesserung: harter Call-Ausfall → Funnel-Event. ACHTUNG:
            # #216 fügt in genau diesem Zweig `failures.append(...)` hinzu — beim
            # Merge beide Zeilen behalten (semantisch unabhängig).
            _trace_stage_outcome(
                concept_for_idx[i], "extractor", "dropped", drop_reason="call_failed", detail=str(r)[:120]
            )
            failures.append((concept_for_idx[i], str(r)))
        elif r is None:
            # #197 Nachbesserung: leere Extraktion → Funnel-Event (vorher nur
            # von run_per_concept als [extractor-empty] auf stderr geloggt).
            _trace_stage_outcome(concept_for_idx[i], "extractor", "dropped", drop_reason="empty_extraction")
        else:
            r.refine_key = contexts[i][0].title  # plan title als stabiler Fallback-Key (Bug #5)
            drafts.append(r)
            concept_map[r.title] = contexts[i]
            concept_map[contexts[i][0].title] = contexts[i]  # plan title als zusätzlicher Key
    dropped = len(tasks) - len(drafts)
    if dropped:
        print(f"      [extractor-empty] {dropped}/{len(tasks)} Konzepte stumm weggefallen", file=sys.stderr)
    return drafts, concept_map, dropped, failures


# --- #210: Timeout-Verlust im Run-Summary sichtbar machen (kein stilles Verschlucken) ---
# Exit-Code-Konvention: 0 = voller Erfolg, 1 = harter Abbruch (unbehandelte Exception,
# z.B. Planner-Timeout nach Retries → kein Plan → Traceback), 3 = Lauf abgeschlossen,
# aber >=1 Konzept endgültig durch Timeout/CLI-Fehler verloren. 3 hält den Teilverlust
# von einem harten Absturz unterscheidbar (CI/Wrapper können darauf reagieren).
_EXIT_EXTRACTOR_LOSS = 3


def format_extractor_failure_report(failures: list[tuple[str, str]], n_attempted: int) -> list[str]:
    """Warn-Block-Zeilen für Konzepte, die beim Extrahieren mit Exception starben.

    Rein (keine Seiteneffekte) → unit-testbar ohne die volle Pipeline. Leere Liste
    liefert [] (kein Block). Der Aufrufer druckt die Zeilen nach stderr.
    """
    if not failures:
        return []
    lines = [
        f"⚠️  {len(failures)} von {n_attempted} Konzept(en) beim Extrahieren verloren "
        "(Timeout/CLI-Fehler, auch nach Retry) — NICHT in den geschriebenen Notes enthalten:"
    ]
    for title, err in failures:
        reason = "Timeout" if "timeout" in (err or "").lower() else "Fehler"
        lines.append(f"    - {title}: {reason} — {(err or '')[:160]}")
    return lines


def extractor_failure_exit_code(failures: list[tuple[str, str]]) -> int:
    """Exit-Code 3 wenn Konzepte endgültig verloren gingen, sonst 0."""
    return _EXIT_EXTRACTOR_LOSS if failures else 0


def _normalize(title: str) -> str:
    """Normalisiert Titel für Dedup-Vergleich: Kleinbuchstaben, Satzzeichen entfernen."""
    import re

    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def _llm_dedup_batch(pairs: list[tuple[int, int]], drafts: list[AtomicNoteDraft]) -> set[tuple[int, int]]:
    """Stage 2.5: Haiku-Batch-Call für ambiguous Cosine-Zone.

    Schickt alle ambiguous Paare in einem einzigen Call. Antwortformat:
      1: SAME
      2: DIFFERENT
      ...
    Gibt Menge der als SAME bewerteten (i,j)-Paare zurück.
    """
    if not pairs:
        return set()

    import re as _re

    def _first_sentences(body: str, n: int = 2, maxlen: int = 300) -> str:
        body = (body or "").replace("\n", " ").strip()
        parts = body.split(". ")
        return (". ".join(parts[:n]) + ("." if len(parts) > 1 else ""))[:maxlen]

    lines = []
    for idx, (i, j) in enumerate(pairs, 1):
        a, b = drafts[i], drafts[j]
        lines.append(
            f'Pair {idx}:\n  A: "{a.title}" — {_first_sentences(a.body)}\n  B: "{b.title}" — {_first_sentences(b.body)}'
        )

    prompt = (
        "Decide for each pair whether A and B describe the EXACT SAME concept and should be merged "
        "into one note, or are DIFFERENT concepts that must stay separate.\n\n"
        "Rules:\n"
        "- SAME: identical topic, same scope, just differently worded or translated\n"
        "- DIFFERENT: different level of abstraction, different aspect, different entities, or loosely related\n"
        "- CONSERVATIVE: If in doubt, choose DIFFERENT. Two separate notes are better than losing distinct information.\n\n"
        "Format — exactly one line per pair, same order, no other text:\n"
        "1: SAME\n"
        "2: DIFFERENT\n\n"
        "Pairs:\n" + "\n\n".join(lines) + "\n\nYour answer:"
    )

    from generative.agents.base import call_claude_async

    try:
        result = await call_claude_async(prompt, model=MODEL_LLM_DEDUP)
        raw = result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        print(f"      [er-stage2.5] LLM-Call fehlgeschlagen: {e}\nPrompt[:300]: {prompt[:300]}", file=sys.stderr)
        return set()

    same_pairs: set[tuple[int, int]] = set()
    for idx_str, verdict in _re.findall(r"^\s*(\d+)\s*:\s*(SAME|DIFFERENT)", raw, _re.MULTILINE | _re.IGNORECASE):
        idx = int(idx_str) - 1
        if 0 <= idx < len(pairs) and verdict.upper() == "SAME":
            same_pairs.add(pairs[idx])

    print(f"      [er-stage2.5] {len(same_pairs)}/{len(pairs)} Paare als SAME bewertet", file=sys.stderr)
    return same_pairs


def er_stage1_decision(a: set[str], b: set[str]) -> tuple[str, int]:
    """Pure predicate für ER-Stage-1-Blocking. Entscheidet ob ein Title-Token-Paar
    in die Embedding-Stage darf. Returns (verdict, token_diff).

    verdict ∈ {
        "accept",            # Paar geht zur Body-Cosine-Stage
        "skip-mono",         # eine Seite < 2 Tokens (zu wenig Signal)
        "skip-no-subset",    # keine Seite ist Subset der anderen
        "skip-token-diff",   # |longer\\shorter| > ER_MAX_TOKEN_DIFF
        "skip-hub-generic",  # kürzere Tokens-Menge ⊆ ER_HUB_GENERIC_TOKENS
    }

    Asymmetrie absichtlich: Author-Suffix („Five Laws" ⊂ „Five Laws (Bates)",
    diff=1) wird akzeptiert, Hub-Sub-Verhältnis („Information Need" ⊂ „Wilson
    Information Need Model", diff≥2 ODER shorter=hub-generic) wird verworfen.
    """
    if len(a) < 2 or len(b) < 2:
        return "skip-mono", 0
    if not (a <= b or b <= a):
        return "skip-no-subset", 0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    diff = len(longer - shorter)
    if diff > ER_MAX_TOKEN_DIFF:
        return "skip-token-diff", diff
    if shorter <= ER_HUB_GENERIC_TOKENS:
        return "skip-hub-generic", diff
    return "accept", diff


async def entity_resolution(
    drafts: list[AtomicNoteDraft], max_concurrent_calls: int | None = None
) -> list[AtomicNoteDraft]:
    """4-Stage Entity-Resolution-Pipeline (Christen 2012, GraphRAG-Pattern):

    1. **Blocking** — paarweiser Title-Token-Subset-Test (er_stage1_decision) als
       HARD-Constraint. Spart Embedding-Calls für offensichtlich verschiedene Konzepte
       (ISP Phase X vs. Bates Five Laws → kein Body-Vergleich nötig).
    2. **Similarity** — für gefilterte Paare: Body-Embedding-Cosine via
       sentence-transformers. Cosine ≥ ER_BODY_COSINE_THRESHOLD = Cluster-Edge.
       Body-Inhalt ist semantisch viel präziser als Title-Tokens
       (ISP-Phase-Varianten haben verschiedene Bodies → cosine niedrig,
       'HIB' und 'HIB (Bates)' aus demselben PDF haben ~identische Bodies).
    3. **Clustering** — Connected Components via Union-Find auf den Edges.
    4. **Canonicalization** — pro Multi-Member-Cluster ein LLM-Merge-Call
       (canonicalizer.merge_cluster) der alle Bodies zu einem konsolidiert.
       Anker werden deterministisch konkateniert, nicht LLM-geschrieben.

    Verlustarm: Body-Inhalt aller Cluster-Mitglieder geht in den Merge-Call ein.
    Token-effizient: 1 LLM-Call pro Cluster statt N. Debugbar: jede Stage loggt
    eigene Trace-Zeile.
    """
    from generative.agents.cross_reference import _tokens

    n = len(drafts)
    if n <= 1:
        return drafts
    if not ENABLE_ENTITY_RESOLUTION:
        print("      [er] disabled via ENABLE_ENTITY_RESOLUTION=0", file=sys.stderr)
        return drafts

    # Stage 1: Blocking — Title-Token-Subset als HARD-Constraint, plus Hub-Schutz
    # (Codex-Cross-Review 2026-05-09) ODER semantic Title-Cosine (v35).
    # Nur Title-Varianten desselben Konzepts (eine Tokens-Menge ist Subset der anderen
    # ODER Cosine-Similarity hoch) dürfen ins Embedding-Stage.
    # Verhindert dass distinkte Geschwister-Konzepte mit ähnlichen Bodies gemergt werden.
    token_sets = [_tokens(d.title) for d in drafts]
    title_embs = [embeddings.embed_title(d.title) for d in drafts]
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            verdict, diff = er_stage1_decision(a, b)

            # Pfad A: Token-Subset-Blocking (deterministisch)
            if verdict == "accept":
                candidate_pairs.append((i, j))
                continue

            # Pfad B: Semantic Title-Cosine Fallback (v35). Adressiert die Lücke
            # bei null Token-Overlap (z.B. EN-Original vs DE-Übersetzung).
            t_cos = embeddings.cosine(title_embs[i], title_embs[j])
            if t_cos >= ER_TITLE_COSINE_THRESHOLD:
                candidate_pairs.append((i, j))
                print(
                    f"      [er-stage1] semantic-accept cos={t_cos:.3f} '{drafts[i].title}' ↔ '{drafts[j].title}'",
                    file=sys.stderr,
                )
                continue

            if verdict in ("skip-mono", "skip-no-subset"):
                continue
            ti, tj = (i, j) if len(a) <= len(b) else (j, i)
            if verdict == "skip-token-diff":
                print(
                    f"      [er-stage1-rejected] token-diff={diff} '{drafts[ti].title}' ⊂ '{drafts[tj].title}'",
                    file=sys.stderr,
                )
            elif verdict == "skip-hub-generic":
                print(
                    f"      [er-stage1-rejected] hub-generic '{drafts[ti].title}' ⊂ '{drafts[tj].title}'",
                    file=sys.stderr,
                )
    if not candidate_pairs:
        return drafts

    # Stage 2: Similarity — Body-Embedding-Cosine
    # Embeddings einmal pro Draft berechnen (auch wenn ein Draft in mehreren Paaren
    # vorkommt). lru_cache wäre nett — hier inline-Cache via dict.
    print(f"      [er-stage1] {len(candidate_pairs)} Block-Kandidaten von {n * (n - 1) // 2} Paaren", file=sys.stderr)
    body_embs: dict[int, object] = {}
    for i in {idx for pair in candidate_pairs for idx in pair}:
        body_embs[i] = embeddings.embed_body(drafts[i].body)

    # Stage 3: Clustering — Union-Find über Cosine-Edges
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_count = 0
    ambiguous_pairs: list[tuple[int, int]] = []

    for i, j in candidate_pairs:
        c = embeddings.cosine(body_embs[i], body_embs[j])
        if c >= ER_BODY_COSINE_THRESHOLD:
            union(i, j)
            edge_count += 1
            print(
                f"      [er-stage2] cluster-edge cos={c:.3f} '{drafts[i].title}' ↔ '{drafts[j].title}'", file=sys.stderr
            )
        elif ENABLE_LLM_DEDUP and ER_AMBIGUOUS_LOWER <= c < ER_BODY_COSINE_THRESHOLD:
            ambiguous_pairs.append((i, j))
            print(f"      [er-stage2] ambiguous cos={c:.3f} '{drafts[i].title}' ↔ '{drafts[j].title}'", file=sys.stderr)

    # Stage 2.5: LLM-Dedup für ambiguous Zone — in Chunks à 25 Paare
    if ambiguous_pairs:
        _BATCH = 25
        print(
            f"      [er-stage2.5] {len(ambiguous_pairs)} ambiguous Paare → Haiku ({(_BATCH - 1 + len(ambiguous_pairs)) // _BATCH} Batch(es))",
            file=sys.stderr,
        )
        for chunk_start in range(0, len(ambiguous_pairs), _BATCH):
            chunk = ambiguous_pairs[chunk_start : chunk_start + _BATCH]
            llm_same = await _llm_dedup_batch(chunk, drafts)
            for i, j in llm_same:
                union(i, j)
                edge_count += 1
                print(f"      [er-stage2.5] LLM-SAME '{drafts[i].title}' ↔ '{drafts[j].title}'", file=sys.stderr)

    if edge_count == 0:
        return drafts

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    # Stage 4: Canonicalization — pro Multi-Member-Cluster ein LLM-Merge-Call.
    # Single-Member-Cluster bleiben unverändert.
    multi_clusters = [members for members in clusters.values() if len(members) > 1]
    if not multi_clusters:
        return drafts

    print(f"      [er-stage4] {len(multi_clusters)} Cluster zu mergen", file=sys.stderr)
    # #151, Punkt 5: Merge-Calls unter dieselbe max_concurrent_calls-Semaphore wie der
    # Extraktor — sonst gathern sie ungebremst am Concurrency-Limit vorbei (Opus-Merges,
    # potenziell viele Cluster). max_concurrent_calls=None → Legacy-Fallback auf die Konstante.
    _sem = asyncio.Semaphore(max_concurrent_calls if max_concurrent_calls is not None else MAX_CONCURRENT_CALLS)

    async def _merge_with_sem(members: list[int]) -> AtomicNoteDraft:
        async with _sem:
            return await canonicalizer.merge_cluster([drafts[k] for k in members])

    merge_tasks = [_merge_with_sem(members) for members in multi_clusters]
    merged_results = await asyncio.gather(*merge_tasks, return_exceptions=True)

    # Resultate zurück in die Draft-Liste einsetzen
    consumed: set[int] = set()
    result: list[AtomicNoteDraft] = []
    cluster_idx_to_merged: dict[int, AtomicNoteDraft] = {}
    for members, merged in zip(multi_clusters, merged_results):
        if isinstance(merged, Exception):
            print(f"      [er-stage4] Merge fehlgeschlagen: {merged} — Repräsentant behalten", file=sys.stderr)
            merged = drafts[members[0]]
        merged.refine_key = drafts[members[0]].refine_key  # plan title für concept_map-Lookup erhalten (Bug #5)
        cluster_idx_to_merged[members[0]] = merged
        consumed.update(members[1:])  # nicht-Repräsentanten verwerfen
        # #197 Nachbesserung: strukturell derselbe Vorgang wie resolve_sibling_dups
        # (ein Draft verschwindet in einen Survivor) — bekam bisher kein Event.
        for _k in members[1:]:
            _trace_stage_outcome(
                drafts[_k].title, "dedup", "dropped", drop_reason="entity_resolution_merge", detail=merged.title
            )
        print(f"      [er-stage4] '{merged.title}' ← {[drafts[k].title for k in members]}", file=sys.stderr)

    for i, d in enumerate(drafts):
        if i in consumed:
            continue
        if i in cluster_idx_to_merged:
            result.append(cluster_idx_to_merged[i])
        else:
            result.append(d)
    return result


# --- #197 Schritt 1: Stage-Outcome-Events an den Gate-Punkten ---------------
# EIN konsistentes Trace-Event pro Note, das ihren Weg durch die Gates
# maschinenlesbar macht (Grundlage für den Gate-Funnel, #197 Schritt 3).
# Ergänzt das bestehende Event-Vokabular (note_outcome/score_result/
# anchor_stats/plan_stats) um genau eine Klasse — kein zweites Tracing-System,
# derselbe trace_event()-Pfad. Verifier und Critic tragen ihr per-Note-Urteil
# bereits via anchor_stats bzw. score_result; dieses Event füllt die bisher
# stummen Gates: Faithfulness (kein Event) und alle Drop-Klassen (Artifact,
# Stage-6-Crash, Exact-/Sibling-Dedup), die vorher nur als aggregierte Zähler
# oder stderr-Prints existierten und aus dem Trace nicht rekonstruierbar waren.
def _trace_stage_outcome(
    title: str,
    stage: str,
    outcome: str,
    drop_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Schreibt ein `stage_outcome`-Event.

    stage: verifier|critic|faithfulness|dedup|artifact|stage6|…
    outcome: passed|downgraded|dropped|skipped
    drop_reason: maschinenlesbarer Code (None bei outcome=passed)
    detail: optionaler menschenlesbarer Zusatz (z.B. Survivor-Titel, step/phase)

    Lazy Import wie die übrigen orchestrator-Trace-Aufrufe; trace_event greift
    zur Laufzeit auf das aktive Backend zu (tests biegen es auf tmp um).
    """
    from generative.agents.base import trace_event as _te

    payload: dict = {"title": title, "stage": stage, "outcome": outcome}
    if drop_reason is not None:
        payload["drop_reason"] = drop_reason
    if detail is not None:
        payload["detail"] = detail
    _te("orchestrator", "stage_outcome", payload)


# --- Stage-6-Crash-Handling (Issue #17) ------------------------------------
# Eine Note, die in Stage 6 (Verifier/Cross-Reference/Critic) crasht, wird NICHT
# als unverifizierter Draft geschrieben, sondern gedroppt + als JSON-Crash-Report
# diagnostizierbar abgelegt. Siehe pipeline/crash_report.py.

_STAGE6_PHASE = threading.local()  # "initial" | "refine" pro to_thread-Worker


def _current_phase() -> str:
    return getattr(_STAGE6_PHASE, "value", "initial")


class _Stage6Failure:
    """Sentinel-Ergebnis eines gecrashten Stage-6-Note-Laufs (statt Exception)."""

    def __init__(self, idx: int, payload: dict):
        self.idx = idx
        self.payload = payload


def _run_note_pipeline_guarded(i, n_total, draft, *args, _run_meta=None, **kwargs):
    """Läuft im to_thread-Worker. Fängt jeden Stage-6-Crash und baut — im selben
    Thread, in dem der Call-Record gesetzt wurde — einen vollständigen Crash-Payload.
    Gibt (i, draft) bei Erfolg oder _Stage6Failure(i, payload) bei Crash zurück.
    """
    from generative.agents.base import get_last_call_record, clear_last_call_record

    clear_last_call_record()
    _STAGE6_PHASE.value = "initial"
    try:
        return _run_note_pipeline(i, n_total, draft, *args, **kwargs)
    except Exception as e:
        rec = get_last_call_record() or {}
        payload = {
            "title": draft.title,
            "step": rec.get("agent", "unknown"),
            "exception": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "prompt": rec.get("prompt", ""),
            "raw_output": rec.get("raw_output", ""),
            "draft_body": draft.body or "",
            "phase": _current_phase(),
            "run_meta": _run_meta or {},
        }
        return _Stage6Failure(i, payload)


def _collect_stage6_results(results, failed_dir: Path):
    """Trennt Stage-6-Ergebnisse: erfolgreiche Drafts (idx-sortiert) vs. Crashes.
    Schreibt pro Crash einen JSON-Report nach failed_dir. Gibt (survived, crashes).
    """
    from generative.pipeline.crash_report import write_crash_report

    survived_by_idx: dict[int, AtomicNoteDraft] = {}
    crashes: list[_Stage6Failure] = []
    for res in results:
        if isinstance(res, _Stage6Failure):
            write_crash_report(failed_dir, res.payload)
            crashes.append(res)
            _trace_stage_outcome(
                res.payload.get("title", "?"),
                "stage6",
                "dropped",
                drop_reason="stage6_crash",
                detail=f"{res.payload.get('step', '?')}/{res.payload.get('phase', '?')}",
            )
        elif isinstance(res, BaseException):
            # Crash außerhalb des guarded Wrappers — defensiv, ohne per-Note-Payload.
            # #197 Nachbesserung: konsistent zum _Stage6Failure-Zweig instrumentieren —
            # Crash-Report (statt nur einer stderr-Zeile, damit der Drop diagnostizierbar
            # bleibt) + stage_outcome-Event (sonst verschwindet die Note lautlos aus dem
            # Funnel). Ohne Payload bleibt der Titel unbekannt ("?").
            print(
                f"  [WARN] Stage-6 unerwartet fehlgeschlagen (BaseException {type(res).__name__}): {res}",
                file=sys.stderr,
            )
            write_crash_report(
                failed_dir,
                {
                    "title": "?",
                    "step": "stage6",
                    "phase": _current_phase(),
                    "exception": f"{type(res).__name__}: {res}",
                    "traceback": "".join(traceback.format_exception(type(res), res, res.__traceback__)),
                },
            )
            _trace_stage_outcome("?", "stage6", "dropped", drop_reason="stage6_crash", detail=type(res).__name__)
        else:
            idx, d = res
            survived_by_idx[idx] = d
    survived = [survived_by_idx[i] for i in sorted(survived_by_idx)]
    return survived, crashes


def _apply_faithfulness_gate(draft: AtomicNoteDraft, page_index: dict | None, citation: CitationMeta) -> None:
    """Faithfulness-Gate-Anwendung (E6, #69) — mutiert `draft` in-place, sonst pure.

    Aus `_run_note_pipeline` extrahiert, damit der Kleber (welche Felder/Flags
    gesetzt werden) isoliert testbar ist, ohne die volle Pipeline mocken zu
    müssen (Lehre: Orchestrator-Wiring-Bugs entstehen im Kleber).

    Enthält den vollen Guard: Flag default aus bis Human-Kalibrierung (#123);
    nur für `action=="create"` — Merge-Stubs/Extends tragen Verwaltungstext,
    der Junk-Claims erzeugt (Kalibrierungs-Report E5b §Konsequenzen), Hubs
    sind Übersichts-Notes. Lazy Import: bei Flag aus wird kein ML-Modul
    importiert.
    """
    if not (ENABLE_FAITHFULNESS_GATE and page_index and draft.action == "create"):
        if ENABLE_FAITHFULNESS_GATE and page_index and draft.action != "create":
            # Skip sichtbar machen — ein stilles Gate wäre die dokumentierte
            # Anti-Klasse „Signal gesetzt, aber nie gelesen" in neuer Form
            # (realer Hrastinski-E2E 2026-07-05: alle Notes via cross_reference
            # auf extend gedreht, Gate skippte kommentarlos).
            print(f"      [faithfulness] skipped (action={draft.action})")
            _trace_stage_outcome(
                draft.title, "faithfulness", "skipped", drop_reason="action_not_create", detail=draft.action
            )
        elif ENABLE_FAITHFULNESS_GATE and not page_index and draft.action == "create":
            # #197 Nachbesserung: Gate aktiv + create-Note, aber leerer page_index
            # (PDF ohne [S. N]-Marker — dokumentierter Fall). Das Gate kann nicht
            # greifen; ohne Event wäre dieser Skip im Funnel unsichtbar.
            print("      [faithfulness] skipped (kein Page-Index)")
            _trace_stage_outcome(draft.title, "faithfulness", "skipped", drop_reason="no_page_index")
        return

    from generative.pipeline.faithfulness_gate import run_faithfulness_gate

    gate = run_faithfulness_gate(draft.body, page_index, citation)
    if gate.failed:
        draft.faithfulness_fail = True
        _trace_stage_outcome(
            draft.title, "faithfulness", "downgraded", drop_reason="faithfulness_fail", detail=f"{gate.n_failed} failed"
        )
    else:
        _trace_stage_outcome(draft.title, "faithfulness", "passed", detail=f"{gate.n_supported} supported")
    for v in gate.verdicts:
        if v.status.startswith("failed_"):
            e_txt = f" e={v.entailment:.2f}" if v.entailment is not None else ""
            draft.quality_flags.append(f"⚠️ Faithfulness: {v.status}{e_txt} — {v.claim.text[:100]}")
    if gate.n_abstained:
        draft.quality_flags.append(f"Faithfulness: {gate.n_abstained} Claim(s) abstained (nicht prüfbar — kein Fail)")
    print(f"      [faithfulness] {gate.n_supported} supported / {gate.n_failed} failed / {gate.n_abstained} abstained")


def _run_note_pipeline(
    i: int,
    n_total: int,
    draft: AtomicNoteDraft,
    initial_drafts: list[AtomicNoteDraft],
    existing_concepts: dict,
    concept_links: dict,
    chunk_map: dict,
    full_text: str,
    acronym_dict: dict,
    concept_map: dict,
    quality_report,
    citation: CitationMeta,
    source_path: Path,
    tag_whitelist: list,
    all_hub_concepts: dict | None = None,
    all_run_concept_links: dict | None = None,
    background_map: dict | None = None,
    related_mentions: list[str] | None = None,
    runtime_config=None,  # None → LEGACY-Fallback; refine_budget=None → unbegrenztes Budget
    refine_budget: RunBudget | None = None,
    page_index: dict | None = None,  # E6, #69: nur gesetzt wenn ENABLE_FAITHFULNESS_GATE
) -> tuple[int, AtomicNoteDraft]:
    """Stage-6-Pipeline für eine einzelne Note. Läuft in asyncio.to_thread().

    Gibt (i, draft) zurück. Wirft bei schwerem Fehler eine Exception.
    initial_drafts ist ein Snapshot aller Drafts vor Stage-6 — wird für den
    siblings-Index genutzt (konsistent, da Body-Änderungen aus der Stage
    das Hub-Routing nicht beeinflussen).
    """
    print(f"  [{i + 1}/{n_total}] {draft.title}")

    _STOP_V = frozenset(
        {
            "the",
            "of",
            "and",
            "in",
            "a",
            "an",
            "for",
            "on",
            "to",
            "is",
            "as",
            "und",
            "der",
            "die",
            "das",
            "von",
            "mit",
            "für",
            "auf",
            "bei",
            "im",
        }
    )
    _title_tokens = [t for t in draft.title.lower().split() if t not in _STOP_V and len(t) >= 3]
    source_chunk = pdf_chunker.concept_text_window(
        full_text, [draft.title] + _title_tokens, window_words=400, max_chars=8000
    )
    if not source_chunk.strip():
        source_chunk = next(
            (text[:12000] for _ct, text in chunk_map.items() if any(w in draft.body[:500] for w in _ct.split()[:3])),
            list(chunk_map.values())[0][:12000] if chunk_map else full_text[:6000],
        )

    per_draft_dict = dict(acronym_dict)
    per_draft_dict.update(acronym_fix.llm_fallback_resolve(draft.body, acronym_dict))
    new_body, expanded = acronym_fix.expand_acronyms(draft.body, per_draft_dict)
    if expanded:
        print(f"      [acronym-fix] {', '.join(expanded)} aufgelöst")
        draft.body = new_body

    # Post-Extraction-Cleanup: Kapitel/Abschnitt-Verweise entfernen die der Extractor
    # trotz Prompt-Verbot produziert. Verhindert Future-Self-Hard-Gate-Fail.
    _CHAPTER_REF_RE = __import__("re").compile(
        r"\b(in|siehe|vgl\.?)\s+(Kapitel|Abschnitt|Section|Chapter)\s+\d+\w*",
        __import__("re").IGNORECASE,
    )
    cleaned, n_refs = _CHAPTER_REF_RE.subn("", draft.body)
    if n_refs:
        draft.body = cleaned
        print(f"      [chapter-ref-fix] {n_refs} Kapitel-Verweis(e) entfernt", file=__import__("sys").stderr)

    draft = verifier.run(draft, source_chunk)
    siblings = {d.title: d for d in initial_drafts if d.title != draft.title}
    draft = cross_reference.run(draft, existing_concepts, siblings=siblings)

    new_body, repaired = anchor_repair.repair_trailing_anchors(draft.body)
    if repaired:
        print(f"      [anchor-repair] {repaired} Schlusssatz-Anker vererbt")
        draft.body = new_body

    has_corrob = len(draft.related) >= 1
    draft = confidence.run(
        draft,
        has_vault_corroboration=has_corrob,
        peer_reviewed=bool(quality_report.peer_reviewed),
        citation_count=quality_report.citation_count,
    )

    # Vorkalkulierte Hub-Maps aus process_all_notes_async nutzen (O(N) statt O(N²))
    hub_concepts = all_hub_concepts if all_hub_concepts is not None else {**existing_concepts}
    run_concept_links = all_run_concept_links if all_run_concept_links is not None else dict(concept_links)

    draft = critic.run(draft, existing_concepts=hub_concepts, concept_links=run_concept_links)

    # Self-Refine (Milestone 3.6 + v8): Retry bei knapp gescheiterten Notes
    refine_trigger_b = (
        draft.critic_score >= CRITIC_AUTO_THRESHOLD and not draft.hard_gates_pass
    )  # nur noch für synthesized_hint; Refine-Gating macht should_attempt_refine unten
    fs_violations = [f for f in draft.quality_flags if f.startswith("⚠️ Future-Self:")]
    synthesized_hint = None
    if not draft.revision_hint and refine_trigger_b and fs_violations:
        synthesized_hint = "Hard-Gate-Fail trotz Score-Pass — Future-Self-Verstöße deterministisch erkannt."

    # Score=4 + Hint: kein Retry (Gemini-Review 2026-05-18: 0% Erfolgsrate, Vault-Note braucht
    # keinen Retry). Hint als Metadatum für spätere Analyse speichern.
    if (
        draft.critic_score == CRITIC_AUTO_THRESHOLD
        and draft.revision_hint
        and "critic_improvement_hint" not in (draft.quality_flags or [])
    ):
        draft.quality_flags.append(f"critic_improvement_hint: {draft.revision_hint[:120]}")

    # Bug #5: concept_map-Lookup mit refine_key-Fallback (nach ER kann draft.title abweichen)
    _refine_map_key = draft.title if draft.title in concept_map else draft.refine_key
    _policy = runtime_config.refine if runtime_config is not None else LEGACY.refine
    refine_decision = should_attempt_refine(
        draft,
        _policy,
        auto_threshold=CRITIC_AUTO_THRESHOLD,
        has_concept_context=_refine_map_key in concept_map,
        synthesized_hint=synthesized_hint,
    )
    _should_attempt = refine_decision.attempt
    _budget_ok = (refine_budget is None or refine_budget.try_consume()) if _should_attempt else False
    if _should_attempt and not _budget_ok:
        print("      [refine] übersprungen: Run-Budget ausgeschöpft")
    if _should_attempt and _budget_ok:
        base_hint = draft.revision_hint or synthesized_hint
        augmented_hint = base_hint + (
            "\n\nKonkrete Future-Self-Verstöße (deterministisch, alle entfernen):\n"
            + "\n".join(f"- {v.replace('⚠️ Future-Self: ', '')}" for v in fs_violations)
            if fs_violations
            else ""
        )
        hint_source = "Critic-Hint" if draft.revision_hint else "synth"
        print(
            f"      [refine] Score {draft.critic_score} + {hint_source} — 1 Retry"
            + (f" + {len(fs_violations)} Regex-Violations" if fs_violations else "")
        )
        concept_obj, ctext = concept_map[_refine_map_key]
        try:
            # asyncio.run() ist in Threads (kein Event-Loop) erlaubt
            _bg = (background_map or {}).get(concept_obj.title) or (background_map or {}).get(
                draft.title
            )  # Bug #6: plan title bevorzugen
            refined = asyncio.run(
                extractor.run_per_concept(
                    concept=concept_obj,
                    concept_text=ctext,
                    existing_concepts=existing_concepts,
                    citation=citation,
                    revision_hint=augmented_hint,
                    tag_whitelist=tag_whitelist,
                    background_context=_bg,
                    related_mentions=related_mentions,  # Bug #7: beim Retry übergeben
                    current_draft_body=draft.body,  # Bug #1: gezieltes Überarbeiten statt Neugenerierung
                )
            )
        except Exception as e:
            print(f"      [refine] Retry fehlgeschlagen: {e}")
            refined = None
        if refined is not None:
            refined.quality_flags.extend(quality_report.flags)
            refined_dict = dict(acronym_dict)
            refined_dict.update(acronym_fix.llm_fallback_resolve(refined.body, acronym_dict))
            new_body, expanded = acronym_fix.expand_acronyms(refined.body, refined_dict)
            if expanded:
                print(f"      [acronym-fix] (refine) {', '.join(expanded)} aufgelöst")
                refined.body = new_body
            _STAGE6_PHASE.value = "refine"  # Crash ab hier wird als refine-Phase getaggt (Issue #17)
            refined = verifier.run(refined, source_chunk)
            refined = cross_reference.run(refined, existing_concepts, siblings=siblings)
            new_body, repaired = anchor_repair.repair_trailing_anchors(refined.body)
            if repaired:
                print(f"      [anchor-repair] (refine) {repaired} Schlusssatz-Anker vererbt")
                refined.body = new_body
            refined = confidence.run(
                refined,
                has_vault_corroboration=(len(refined.related) >= 1),
                peer_reviewed=bool(quality_report.peer_reviewed),
                citation_count=quality_report.citation_count,
            )
            refined = critic.run(refined, existing_concepts=hub_concepts, concept_links=run_concept_links)
            better = refine_accepted(refined, auto_threshold=CRITIC_AUTO_THRESHOLD)
            if better:
                print(
                    f"      [refine] Score {draft.critic_score}/{draft.hard_gates_pass} → "
                    f"{refined.critic_score}/{refined.hard_gates_pass} ✓"
                )
                draft = refined
            else:
                print(
                    f"      [refine] Score {refined.critic_score}/{refined.hard_gates_pass} ≤ "
                    f"{draft.critic_score}/{draft.hard_gates_pass}, Original behalten"
                )

    # Faithfulness-Gate (E6, #69): prüft High-Risk-Claims gegen das PDF-Seitenfenster.
    # Ein Einhängepunkt für Erst- UND Refine-Pfad, da draft hier final ist.
    _apply_faithfulness_gate(draft, page_index, citation)

    # #45: Routing-Grund + konkrete Quality-Flags auch im echten Lauf sichtbar
    # (bisher erschienen die Flags nur im --dry-run).
    print(f"      {routing_report.routing_status_line(draft)}")

    return i, draft


async def process_all_notes_async(
    drafts: list[AtomicNoteDraft],
    existing_concepts: dict,
    concept_links: dict,
    chunk_map: dict,
    full_text: str,
    acronym_dict: dict,
    concept_map: dict,
    quality_report,
    citation: CitationMeta,
    source_path: Path,
    tag_whitelist: list,
    background_map: dict | None = None,
    related_mentions: list[str] | None = None,
    runtime_config=None,
    refine_budget: RunBudget | None = None,
    failed_dir: Path | None = None,
) -> list[AtomicNoteDraft]:
    """Stage-6-Pipeline für alle Notes parallel via asyncio.to_thread() + Semaphore."""
    # #101: aus RuntimeConfig gespeist statt fester Konstante; runtime_config=None
    # (Legacy-Aufrufpfad) fällt auf MAX_CONCURRENT_CALLS zurück.
    _max_concurrent = runtime_config.max_concurrent_calls if runtime_config is not None else MAX_CONCURRENT_CALLS
    sem = asyncio.Semaphore(_max_concurrent)
    initial_drafts = list(drafts)
    n_total = len(drafts)

    # O(N)-Vorkalkulation statt O(N²): hub_concepts + run_concept_links einmal berechnen.
    # Gemini-Review 2026-05-13: im sequenziellen Code war das bereits O(N²·M) —
    # bei Parallelisierung wird es durch Race-freie Vorkalkulation O(N·M).
    # Alle Drafts als Siblings (inkl. self) — self_keys-Mechanismus in critic.hub_test
    # schließt die Note selbst aus → Ergebnis identisch zu per-Note-Berechnung.
    all_hub_concepts: dict = dict(existing_concepts)
    for d in initial_drafts:
        all_hub_concepts.setdefault(d.title.lower(), f"<sibling:{d.title}>")
        for alias in d.aliases or []:
            all_hub_concepts.setdefault(alias.lower(), f"<sibling:{d.title}>")

    all_run_concept_links: dict = dict(concept_links)
    for sib_draft in initial_drafts:
        sib_path = f"<sibling:{sib_draft.title}>"
        sib_self = {sib_draft.title.lower()} | {a.lower() for a in (sib_draft.aliases or [])}
        sub_keys = critic.hub_test(sib_draft.body or "", all_hub_concepts, self_keys=sib_self)
        outgoing: set[str] = set()
        for k in sub_keys:
            tgt_path = all_hub_concepts.get(k.lower())
            if tgt_path and tgt_path != sib_path:
                outgoing.add(tgt_path)
        all_run_concept_links[sib_path] = outgoing

    from generative.agents.base import _RUN_ID
    from generative.config import CACHE_DIR, BACKEND

    if failed_dir is None:
        failed_dir = CACHE_DIR / "failed" / _RUN_ID
    run_meta = {"run_id": _RUN_ID, "pdf": source_path.name, "backend": BACKEND}

    # Faithfulness-Gate (E6, #69): EINMAL pro Lauf gebaut, nicht pro Note.
    # Nur bei aktivem Flag — build_page_index selbst ist ML-frei, aber ohne
    # Flag hat kein Aufrufer Verwendung dafür.
    page_index = build_page_index(full_text) if ENABLE_FAITHFULNESS_GATE else None

    async def _with_sem(i: int, draft: AtomicNoteDraft):
        async with sem:
            return await asyncio.to_thread(
                _run_note_pipeline_guarded,
                i,
                n_total,
                draft,
                initial_drafts,
                existing_concepts,
                concept_links,
                chunk_map,
                full_text,
                acronym_dict,
                concept_map,
                quality_report,
                citation,
                source_path,
                tag_whitelist,
                all_hub_concepts,
                all_run_concept_links,
                background_map,
                related_mentions,
                runtime_config,
                refine_budget,
                page_index=page_index,
                _run_meta=run_meta,
            )

    results = await asyncio.gather(
        *[_with_sem(i, d) for i, d in enumerate(drafts)],
        return_exceptions=True,
    )

    survived, crashes = _collect_stage6_results(results, failed_dir)

    if crashes:
        print(
            f"\n  [Stage-6-Crash] {len(crashes)} Note(s) verworfen (unverifiziert, nicht geschrieben):", file=sys.stderr
        )
        for c in crashes:
            p = c.payload
            print(f"    - {p['title']} | {p['step']}/{p['phase']} | {p['exception']}", file=sys.stderr)
        print(f"    Crash-Reports: {failed_dir}", file=sys.stderr)

    return survived


_ABSENCE_PHRASES = (
    "nicht behandelt",
    "nicht vorkommt",
    "kommt nicht vor",
    "nicht diskutiert",
    "nicht thematisiert",
    "keine erwähnung",
    "behandelt nicht",
    "erwähnt nicht",
    "thematisiert nicht",
    "not discussed",
    "not covered",
    "not mentioned",
    "not addressed",
    "abwesenheit statt wissen",
    "dokumentiert abwesenheit",
)


def _drop_artifacts(drafts: list[AtomicNoteDraft]) -> list[AtomicNoteDraft]:
    """Verwirft Abwesenheits-Noten (Extraction-Artefakte) ohne LLM-Call.

    Tritt auf wenn der Extractor die 'weglassen'-Instruktion ignoriert und stattdessen
    eine Note schreibt die dokumentiert, dass ein Konzept nicht im Quelltext vorkommt.
    MERGE-Stubs (action='extend') werden nicht angefasst.
    """
    kept: list[AtomicNoteDraft] = []
    dropped: list[str] = []
    for draft in drafts:
        if draft.action == "extend":
            kept.append(draft)
            continue
        body_lower = (draft.body or "").lower()
        if any(phrase in body_lower for phrase in _ABSENCE_PHRASES):
            dropped.append(draft.title)
            _trace_stage_outcome(draft.title, "artifact", "dropped", drop_reason="absence_artifact")
        else:
            kept.append(draft)
    if dropped:
        print(f"      [artifact-drop] {len(dropped)} Abwesenheits-Artefakt(e) verworfen: {', '.join(dropped)}")
    return kept


def dedup_exact(drafts: list[AtomicNoteDraft], existing_concepts: dict[str, str]) -> list[AtomicNoteDraft]:
    """Exact-Match-Dedup: identischer normalisierter Titel innerhalb der Drafts +
    Vault-Match-Umflag (action=create → action=extend bei Vault-Treffer).

    Fuzzy-/Semantic-Cluster läuft separat in entity_resolution() — diese Funktion
    deckt nur den deterministischen Fall ab, sodass die teure ER-Pipeline nur
    auf bereits exact-deduplizierten Drafts läuft.
    """
    seen: set[str] = set()
    result: list[AtomicNoteDraft] = []
    for d in drafts:
        key = _normalize(d.title)
        if key in seen:
            _trace_stage_outcome(d.title, "dedup", "dropped", drop_reason="exact_dup")
            continue
        exact_match = existing_concepts.get(d.title.lower().strip())
        if exact_match and d.action == "create":
            d.action = "extend"
            d.extend_path = exact_match
        seen.add(key)
        result.append(d)
    return result


def resolve_sibling_dups(
    drafts: list[AtomicNoteDraft], existing_concepts: dict[str, str] | None = None
) -> tuple[list[AtomicNoteDraft], int]:
    """Intra-Run-Sibling-Dedup (Befund D).

    cross_reference erkennt zwei Near-Dup-Drafts EINES Laufs (dup_risk=high) und setzt
    action=extend + extend_path=<Sibling-Titel>. Da der Sibling keine Vault-Datei ist,
    verpufft das Signal beim Writer (write_note routet nur über find_existing_in_vault,
    nicht über extend_path) und BEIDE Notes werden als Vollnoten geschrieben. Diese
    Funktion wertet genau dieses bereits gesetzte extend-Signal aus und kollabiert solche
    Geschwister deterministisch zu EINER Note — VOR dem Schreiben.

    Bewusst KEIN Eingriff ins Title-Blocking von entity_resolution: hier wird nur das vom
    LLM bereits gefällte Dup-Urteil interpretiert, kein neuer Body-Cosine-Pass (der echte,
    distinkt betitelte Geschwister fälschlich mergen könnte) und kein zusätzlicher LLM-Call.

    Survivor pro Cluster: höchster critic_score, Tie → längerer Body, Tie → norm-Titel
    (ordnungsunabhängig deterministisch). Verlustarm: related-Links + source_anchors der
    gedroppten Drafts wandern in den Survivor (related auf MAX_RELATED gedeckelt); gedroppte
    Titel/Aliase werden Survivor-Aliase, sodass [[…]]-Links auf den gedroppten Titel auf den
    Survivor auflösen (kein Dead-Link).

    Vault-Erhalt (Cross-Model-Review Codex 2026-06-23, HIGH#2): hat IRGENDEIN Cluster-Member
    eine Vault-Dublette (action=extend mit extend_path auf eine reale Vault-Note), erbt der
    Survivor diesen Bezug (action=extend + Vault-Stem als Alias → title-/alias-basierter
    Writer findet die Vault-Note). Sonst wird ein dangling Intra-Run-extend auf 'create'
    zurückgesetzt.

    Präkondition: dedup_exact lief vorher → alle Drafts haben unique normalisierte Titel
    (keine norm-Title-Kollisionen, Codex MED#3).
    """
    from generative.agents.cross_reference import MAX_RELATED

    n = len(drafts)
    if n <= 1:
        return drafts, 0

    norm_title = [_normalize(d.title) for d in drafts]
    title_to_idx = {nt: i for i, nt in enumerate(norm_title)}

    # Vault-Index normalisieren (Keys = Titel, Values = Pfade) — für Vault-Dubletten-Erkennung
    ec = existing_concepts or {}
    vault_norms = {_normalize(k) for k in ec} | {_normalize(Path(v).stem) for v in ec.values()}

    def _vault_target(ep: str | None) -> bool:
        """True wenn extend_path auf eine reale Vault-Note zeigt (Titel- ODER Stem-Match)."""
        if not ep:
            return False
        return _normalize(ep) in vault_norms or _normalize(Path(ep).stem) in vault_norms

    def _match_idx(ep: str, self_i: int) -> int | None:
        """In-Run-Draft, dessen Titel zu extend_path passt — per direkt-normalize ODER
        Path-Stem (Codex MED#4: bare Titel mit '/' dürfen nicht über stem zerlegt werden)."""
        for key in (_normalize(ep), _normalize(Path(ep).stem)):
            j = title_to_idx.get(key)
            if j is not None and j != self_i:
                return j
        return None

    # Union-Find über Intra-Run-extend-Kanten (gleiches Cluster-Pattern wie entity_resolution)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, d in enumerate(drafts):
        if d.action != "extend" or not d.extend_path:
            continue
        j = _match_idx(d.extend_path, i)
        if j is not None:  # extend_path trifft einen anderen In-Run-Draft → Geschwister
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def _link_norm(link: str) -> str:
        s = link.strip()
        if s.startswith("[[") and s.endswith("]]"):
            s = s[2:-2]
        s = s.split("|", 1)[0].split("#", 1)[0].strip()  # Alias- und Heading-Anker abtrennen
        return _normalize(Path(s).stem)

    drop_idx: set[int] = set()
    for members in clusters.values():
        if len(members) <= 1:  # Multi-Member-Cluster entsteht nur durch eine Sibling-Kante
            continue
        survivor = max(members, key=lambda m: (drafts[m].critic_score, len(drafts[m].body or ""), norm_title[m]))
        s = drafts[survivor]
        alias_norms = {_normalize(a) for a in s.aliases} | {norm_title[survivor]}

        def _absorb_alias(name: str) -> None:
            if name and _normalize(name) not in alias_norms:
                s.aliases.append(name)
                alias_norms.add(_normalize(name))

        for m in members:
            if m == survivor:
                continue
            d = drafts[m]
            drop_idx.add(m)
            _trace_stage_outcome(d.title, "dedup", "dropped", drop_reason="sibling_neardup", detail=s.title)
            for alias in [d.title, *d.aliases]:
                _absorb_alias(alias)
            s.source_anchors.extend(d.source_anchors)
            for link in d.related:
                if link not in s.related:
                    s.related.append(link)

        # Vault-Erhalt: hat ein Cluster-Member eine reale Vault-Dublette, erbt der Survivor
        # sie. Das eigene Vault-Ziel des Survivors hat Vorrang (Mistral-Review 2026-06-23),
        # sonst der erste Member in Index-Reihenfolge.
        _vault_order = [survivor] + [m for m in sorted(members) if m != survivor]
        vault_ep = next(
            (
                drafts[m].extend_path
                for m in _vault_order
                if drafts[m].action == "extend" and _vault_target(drafts[m].extend_path)
            ),
            None,
        )
        if vault_ep is not None:
            s.action = "extend"
            s.extend_path = vault_ep
            _absorb_alias(Path(vault_ep).stem)  # Writer findet Vault-Note via Alias
        elif s.action == "extend" and s.extend_path:  # dangling Intra-Run-extend
            s.action = "create"
            s.extend_path = None

        # Self-Links (auf Survivor-Titel oder absorbierte Aliase) entfernen, dann deckeln
        s.related = [link for link in s.related if _link_norm(link) not in alias_norms][:MAX_RELATED]

    kept = [d for i, d in enumerate(drafts) if i not in drop_idx]
    return kept, len(drop_idx)


def flag_redundant_siblings(
    drafts: list[AtomicNoteDraft],
    threshold: float | None = None,
    body_cosine_fn=None,
) -> tuple[list[AtomicNoteDraft], int]:
    """#8: seiteneffekt-freier Flag bei hoher Body-Überlappung zwischen DISTINKTEN Notes.

    Zwei empirische Gates (Ebner-Audit 2026-06-23) zeigten: Geschwister-Notes EINES Laufs
    mit hoher Body-Cosine (gemessen 0.967) sind weder mergebar (distinkte Konzepte:
    Kirkpatrick-Modell = Theorie vs. Satisfaction-Learning-Dissoziation = Befund) noch
    satz-strippbar (Redundanz paraphrasiert, nicht dupliziert — exakt 0/10, fuzzy≥0.93 nur
    1/10 Sätze). Der einzige verlustfreie Eingriff ist ein Flag, der den menschlichen
    Reviewer auf die Überlappung hinweist ("Kontext kürzen/verlinken"). KEIN Merge, KEIN
    Strip, KEIN Kollabieren — die Notes bleiben unverändert, nur quality_flags wächst.

    Läuft NACH resolve_sibling_dups + dedup_hub_subconcepts (echte Dups und Hub→Sub schon
    behandelt) und VOR dem Writer (Flag landet via _yaml_list im Frontmatter, im Inbox-
    Review sichtbar). Nur create-Drafts werden paarweise verglichen: extend-Drafts gehören
    dem Merge-Pfad (resolve_sibling_dups / write_note) und werden hier nicht doppelt geflaggt.

    body_cosine_fn(i, j) ist injizierbar (deterministische Tests, vgl. filter_hallucinated);
    Default berechnet Body-Embeddings einmal und nutzt embeddings.cosine.
    """
    if threshold is None:
        threshold = REDUNDANT_SIBLING_COSINE_THRESHOLD

    # Nur create-Drafts mit nicht-leerem Body: extend gehört dem Merge-Pfad; ein leerer Body
    # kann nicht redundant sein (Cosine 0) und würde nur das Embedding-Modell unnötig laden.
    candidates = [i for i, d in enumerate(drafts) if d.action == "create" and (d.body or "").strip()]
    if len(candidates) < 2:
        return drafts, 0

    if body_cosine_fn is None:
        body_embs = {i: embeddings.embed_body(drafts[i].body or "") for i in candidates}

        def body_cosine_fn(i, j):
            return embeddings.cosine(body_embs[i], body_embs[j])

    def _add_flag(draft: AtomicNoteDraft, other_title: str, cos: float) -> None:
        marker = f"Überlappung mit [[{other_title}]]"
        if any(marker in f for f in draft.quality_flags):  # idempotent
            return
        draft.quality_flags.append(
            f"⚠️ Hohe inhaltliche {marker} (Body-cos={cos:.2f}) — beim Review Kontext kürzen/verlinken"
        )

    n_pairs = 0
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            i, j = candidates[a], candidates[b]
            cos = body_cosine_fn(i, j)
            if cos >= threshold:
                _add_flag(drafts[i], drafts[j].title, cos)
                _add_flag(drafts[j], drafts[i].title, cos)
                n_pairs += 1
    return drafts, n_pairs


def _auto_start_dashboard() -> None:
    """Startet den Dashboard-Server im Hintergrund falls er noch nicht läuft."""
    import socket
    import subprocess

    try:
        with socket.create_connection(("localhost", 8051), timeout=0.5):
            return  # Läuft bereits
    except OSError:
        pass
    server_py = Path(__file__).parent / "eval_dashboard_server.py"
    if server_py.exists():
        subprocess.Popen(
            [sys.executable, str(server_py), "--port", "8051"],
            cwd=Path(__file__).parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  [dashboard] Server gestartet: http://localhost:8051")


def inline_eval_enabled(runtime_config) -> bool:
    """Whether Stage-8 inline quality evaluation should run for this process."""
    return bool(runtime_config.inline_eval)


def dry_run_eval_targets(written: list[tuple[Path, bool]], cache_note_dir: Path) -> list[Path]:
    """Eval-Dateien des aktuellen Dry-Runs — nur die vault-empfohlenen Notes DIESES Laufs.

    Verhindert, dass Stage-8 veraltete `vault__*.md` aus früheren Läufen im selben
    Cache-Ordner mit-evaluiert (sonst Kontamination von quality_history.jsonl und der
    Run-End-Mittelwerte). Spiegelt das Live-Verhalten, das nur die Run-Notes wertet.

    `written`: (target, is_auto)-Paare aus dem Vault-Writer. Der Dry-Run schreibt
    vault-empfohlene Notes als `vault__<target.name>` ins Cache-Verzeichnis.
    """
    files: list[Path] = []
    for target, is_auto in written:
        if not is_auto:
            continue
        eval_file = cache_note_dir / f"vault__{target.name}"
        if eval_file.exists():
            files.append(eval_file)
    return files


def run_stage8_eval(
    note_files: list[Path],
    source_path: Path,
    run_meta: dict,
    *,
    fresh_run: bool = False,
) -> tuple[list[dict], int, int]:
    """Stage-8-Kernschleife: bewertet jede Note gegen die Quell-PDF.

    Re-Eval-Hash-Guard: eine Note deren pipeline-content-hash (#47) + eval_version +
    pipeline_version bereits in quality_history.jsonl stehen, wird nicht erneut vom
    Judge bewertet — das Alt-Ergebnis wird fuer die Lauf-Aggregation wiederverwendet
    (spart ~2 Sonnet-Calls / ~26k Input-Tokens pro unveraenderter Note). `fresh_run`
    (--fresh-run) bypassed den Guard bewusst — er ist semantisch ein Cache, und
    --fresh-run fordert garantiert frische Ergebnisse an.

    Gibt (eval_results, evaluated_count, reused_count) zurueck; druckt eine sichtbare
    Zusammenfassungszeile (nie ein stiller Skip).
    """
    from generative import eval_quality_v4 as _eq

    eval_results: list[dict] = []
    evaluated_count = 0
    reused_count = 0

    # #151, Punkt 5: der 10er-Cap war bisher stumm — ab der 11. Note wurde nie
    # evaluiert, ohne Meldung. Jetzt sichtbar machen, wenn er greift.
    _EVAL_CAP = 10
    if len(note_files) > _EVAL_CAP:
        print(f"[8/8] Stage 8: evaluiere {_EVAL_CAP} von {len(note_files)} Notes (Cap)…")

    for note_path in note_files[:_EVAL_CAP]:
        note_hash: str | None = None
        try:
            note_text = note_path.read_text(encoding="utf-8")
        except OSError:
            note_text = None
        if note_text is not None:
            note_hash = vault_writer.extract_content_hash(note_text)

        cached = None
        if note_hash and not fresh_run:
            cached = _eq.find_cached_eval(note_hash, _eq.EVAL_VERSION, AGENT_VERSION)

        if cached is not None:
            result = dict(cached)
            result.update(run_meta)
            eval_results.append(result)
            reused_count += 1
            continue

        result = _eq.eval_note(note_path, source_path, pipeline_version=AGENT_VERSION, content_hash=note_hash)
        result.update(run_meta)
        _eq.save_result(result)
        eval_results.append(result)
        evaluated_count += 1

    print(f"[8/8] {evaluated_count} evaluiert, {reused_count} unveraendert uebernommen (Hash-Guard)")
    return eval_results, evaluated_count, reused_count


def _iter_raw_version_strings():
    """Roh-Versions-Strings aller bekannten Stempel-Senken (#191): pipeline_runs-DB,
    quality_history*.jsonl (inkl. Archiv-Dateien) und Baseline-Log-Dateinamen.
    Jede Quelle fail-safe — eine fehlende/kaputte Senke darf den Bump nicht stoppen."""
    import re as _re

    try:
        from generative import db as _db

        for r in _db.query_pipeline_runs():
            yield str(r.get("pipeline_version") or "")
        # Archivierte WIP-Läufe (#193): Nummern bleiben verbrannt.
        yield from _db.query_archived_pipeline_versions()
    except Exception:
        pass
    try:
        from generative.config import QUALITY_HISTORY

        for f in QUALITY_HISTORY.parent.glob("quality_history*.jsonl"):
            text = f.read_text(encoding="utf-8", errors="replace")
            yield from _re.findall(r'"(?:pipeline_)?version":\s*"([^"]+)"', text)
        for f in (QUALITY_HISTORY.parent / "eval" / "baseline").glob("*.log"):
            m = _re.search(r"_(v[\d.]+)(?:_run\d+)?\.log$", f.name)
            if m:
                yield m.group(1)
    except Exception:
        pass


def _known_pipeline_versions() -> set[str]:
    """Alle je gestempelten generativen Pipeline-Versionen — Basis für den
    kollisionssicheren Bump. Filtert extraktive Versionen und Nicht-Versions-Junk."""
    import re as _re

    return {v for v in _iter_raw_version_strings() if _re.fullmatch(r"v\d+\.\d+\.\d+", v)}


def _auto_version_bump(base_dir: Path | None = None) -> None:
    """Erhöht AGENT_VERSION wenn sich Pipeline-Code seit letztem Run geändert hat.

    Neue Version = Patch+1 der MAXIMAL bekannten Version (config, State,
    DB/History/Logs via `_known_pipeline_versions`) statt config+1 — verwaiste
    WIP-Branch-Stempel (v0.3.141/142-Vorfall, #191) werden nie wiederverwendet.
    `base_dir` nur für Tests; Default ist das echte Paket-Verzeichnis.
    """
    import hashlib
    import json as _json
    import re as _re

    global AGENT_VERSION

    if base_dir is None:
        base_dir = Path(__file__).parent
    state_file = base_dir / ".cache" / "pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Hash aller relevanten Python-Dateien
    tracked = [
        base_dir / "agents",
        base_dir / "pipeline",
        base_dir / "orchestrator.py",
        base_dir / "config.py",
    ]

    def _tracked_hash() -> str:
        h = hashlib.md5()
        for p in sorted(f for d in tracked for f in ([d] if d.is_file() else d.rglob("*.py")) if f.is_file()):
            h.update(p.read_bytes())
        return h.hexdigest()

    current_hash = _tracked_hash()

    state = {}
    if state_file.exists():
        try:
            state = _json.loads(state_file.read_text())
        except Exception:
            pass

    if state.get("code_hash") == current_hash:
        return  # Kein Bump nötig

    cfg_path = base_dir / "config.py"
    cfg_text = cfg_path.read_text(encoding="utf-8")
    m = _re.search(r'AGENT_VERSION\s*=\s*"v(\d+)\.(\d+)\.(\d+)"', cfg_text)
    if m:
        candidates = {f"v{m.group(1)}.{m.group(2)}.{m.group(3)}", str(state.get("last_version") or "")}
        candidates |= _known_pipeline_versions()
        candidates = {v for v in candidates if _re.fullmatch(r"v\d+\.\d+\.\d+", v)}
        base = max(candidates, key=lambda v: tuple(int(x) for x in v[1:].split(".")))
        bmaj, bmin, bpatch = (int(x) for x in base[1:].split("."))
        new_ver = f"v{bmaj}.{bmin}.{bpatch + 1}"
        cfg_path.write_text(
            cfg_text.replace(m.group(0), f'AGENT_VERSION = "{new_ver}"'),
            encoding="utf-8",
        )
        # AGENT_VERSION im laufenden Prozess aktualisieren — config-Attribut UND
        # das from-Import-Global dieses Moduls, sonst stempelt der laufende Run
        # (DB, Stage-8-Eval, quality_history) weiter die alte Version (#191).
        from generative import config as _cfg

        _cfg.AGENT_VERSION = new_ver
        AGENT_VERSION = new_ver
        print(f"  [version] Code geändert → {new_ver}")
        # Post-Bump-Hash speichern: der Bump selbst ändert config.py — mit dem
        # Prä-Bump-Hash würde JEDER folgende Lauf erneut bumpen.
        current_hash = _tracked_hash()
    else:
        new_ver = AGENT_VERSION

    state["code_hash"] = current_hash
    state["last_version"] = new_ver
    state_file.write_text(_json.dumps(state, indent=2))


def _phoenix_exe(venv: Path) -> Path:
    """Pfad zum Phoenix-Server-Binary im venv (Windows: Scripts/, POSIX: bin/)."""
    win = Path(venv) / "Scripts" / "phoenix.exe"
    posix = Path(venv) / "bin" / "phoenix"
    if win.exists():
        return win
    if posix.exists():
        return posix
    # Default nach Plattform, auch wenn (noch) nicht vorhanden — Aufrufer prüft .exists().
    return win if os.name == "nt" else posix


def _phoenix_server_running(port: int) -> bool:
    """True wenn auf localhost:port ein Server lauscht."""
    import socket

    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def _ensure_phoenix_server(port: int | None = None, venv: Path | None = None, timeout: float | None = None) -> bool:
    """Startet den Phoenix-Server (detached) falls er nicht auf `port` lauscht.

    Idempotent bei sequentiellen Läufen: läuft der Server schon, passiert nichts
    (Folgeläufe zahlen 0s). Gibt True zurück sobald der Port erreichbar ist, sonst
    False (venv/Binary fehlt oder Start-Timeout). Fehler bleiben graceful — die
    Pipeline läuft dann ohne Traces weiter.

    Timeout default 60s (ENV ATOMIC_AGENT_PHOENIX_TIMEOUT): Phoenix braucht warm
    ~10s, kalt (DB-Migration) und unter der CPU-Last der gleichzeitig startenden
    Pipeline messbar länger — 30s schnitt den ersten Lauf zu früh ab.

    Nicht concurrent-safe (best effort): starten zwei Läufe exakt gleichzeitig mit
    totem Port, spawnen beide ein `phoenix serve`; der zweite scheitert am Port-bind
    und beendet sich, beide sehen am Ende den offenen Port. Für ein Tracing-Hilfsmittel
    akzeptabel — ein File-Lock wäre unverhältnismäßig.
    """
    import subprocess
    import time
    from generative import config

    port = config.PHOENIX_PORT if port is None else port
    venv = config.PHOENIX_VENV if venv is None else venv
    if timeout is None:
        timeout = float(os.environ.get("ATOMIC_AGENT_PHOENIX_TIMEOUT", "60"))
    if _phoenix_server_running(port):
        return True
    exe = _phoenix_exe(venv)
    if not exe.exists():
        print(f"[phoenix] Server-Binary nicht gefunden ({exe}) — Pipeline läuft ohne Traces")
        return False
    flags = 0
    kwargs = {}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP → Server überlebt den Pipe-Prozess.
        flags = 0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(
            [str(exe), "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except OSError as e:
        # z.B. WinError 193 bei beschädigtem Binary trotz exists() — graceful bleiben.
        print(f"[phoenix] Server-Start fehlgeschlagen ({e}) — Pipeline läuft ohne Traces")
        return False
    print(f"  [phoenix] Server wird gestartet (Port {port})…")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _phoenix_server_running(port):
            print(f"  [phoenix] Server bereit → http://localhost:{port}")
            return True
        time.sleep(0.5)
    print(f"[phoenix] Server-Start-Timeout ({timeout:.0f}s) — Pipeline läuft ohne Traces")
    return False


def _setup_phoenix_tracing() -> None:
    """Startet (falls nötig) den Phoenix-Server und sendet OTel-Traces an ihn.

    Nur aktiv bei ENV ATOMIC_AGENT_TRACING=phoenix. Kein Fehler wenn Phoenix
    nicht startbar ist — Pipeline läuft normal ohne Traces.

    Die LLM-Calls werden manuell in agents/base.py instrumentiert (gilt für
    beide Backends: claude-CLI-Subprocess UND litellm). Daher KEIN
    Auto-Instrumentor — der würde bei BACKEND=litellm doppelte Spans erzeugen
    und beim CLI-Subprocess-Default ohnehin nichts sehen.
    """
    global _TRACER, _PROVIDER
    if os.getenv("ATOMIC_AGENT_TRACING") != "phoenix":
        return
    # Server sicherstellen, BEVOR Spans verdrahtet werden — ein toter OTLP-Endpoint
    # ließe den SimpleSpanProcessor pro LLM-Call einen fehlschlagenden POST feuern.
    if not _ensure_phoenix_server():
        return
    try:
        # Rohes OpenTelemetry statt phoenix.otel.register: Die Pipeline läuft im
        # System-Python, wo `import phoenix` an einem sqlean-Paketkonflikt crasht.
        # Der Phoenix-SERVER läuft als separater Prozess; wir müssen hier nur
        # OTLP-Spans an localhost:6006 schicken — das braucht KEIN phoenix-Paket.
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # SimpleSpanProcessor: sofortiger Export pro Span. Robuster als
        # BatchSpanProcessor bei kurzlebigen CLI-Runs (kein Flush-Verlust).
        # Resource openinference.project.name → Phoenix gruppiert in dieses Projekt
        # (statt "default"); entspricht register(project_name=...).
        _PROVIDER = TracerProvider(resource=Resource.create({"openinference.project.name": "atomic-agent"}))
        _PROVIDER.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")))
        trace.set_tracer_provider(_PROVIDER)
        _TRACER = trace.get_tracer("atomic-agent")
        # LLM-Call-Instrumentierung in base.py explizit aktivieren (nur hier, nur
        # bei aktivem Tracing — kein impliziter Proxy-Tracer).
        from generative.agents.base import set_llm_tracer

        set_llm_tracer(_TRACER)
        import atexit

        atexit.register(lambda: _PROVIDER and _PROVIDER.force_flush())
        print("[phoenix] Tracing aktiv → http://localhost:6006")
    except Exception as e:
        print(f"[phoenix] Tracing nicht verfügbar ({e}) — Pipeline läuft ohne Traces")


def _build_citation(
    pdf_meta: dict, quality_report, q_title: str | None, source_name: str, physical_pages: bool = False
) -> CitationMeta:
    """Baut die kanonische CitationMeta + druckt die CrossRef-Override-Diagnose.

    Aufgerufen sowohl im Normalpfad (`_run_extraction_stages`, vor dem Planner)
    als auch im `--load-drafts`-Pfad (Stage 1–5 dort übersprungen, `citation` muss
    trotzdem vor Stage 6 stehen). Deterministisch/idempotent — dieselbe Eingabe
    liefert dieselbe CitationMeta (Analogie zum bestehenden Muster, `q_title`/
    `parse_filename_fallback` bei Bedarf erneut abzuleiten statt durchzureichen).

    `physical_pages` (#95): vom Aufrufer via `pdf_chunker.pdf_uses_physical_pages(
    source_path)` ermittelt — hier nicht selbst berechnet, weil dieser Helper nur
    `source_name` (Anzeigename) kennt, nicht den tatsächlichen `Path`.
    """
    citation = build_citation_meta(pdf_meta, quality_report, q_title, source_name, physical_pages=physical_pages)
    if crossref_override_blocked(quality_report, q_title):
        print(
            f"      [quality] CrossRef-Override verworfen (schwacher Titel-Match): "
            f"'{quality_report.crossref_title[:60]}'"
        )
    return citation


@dataclasses.dataclass(frozen=True)
class _PlanExtractResult:
    """Ergebnis EINER `_plan_and_extract`-Invokation (ein Text-Scope).

    Benannte Felder statt Positions-Tupel — dieselbe Anti-Wiring-Bug-Motivation
    wie RunContext, nur für die Planner→Extractor-Kette. `kept_actionable` speist
    die kumulative Budget-Dekrementierung des by-chapter-Pfads; `related` sammelt
    der by-chapter-Pfad über Kapitel dedupliziert, der Normalpfad nutzt es direkt.
    """

    drafts: list
    concept_map: dict
    dropped: int
    failures: list
    related: list
    kept_actionable: int
    background_map: dict


def _plan_and_extract(
    *,
    plan_text: str,
    hall_text: str,
    extract_text: str,
    relevance_profile: dict,
    existing_concepts: dict,
    citation: CitationMeta | None,
    tag_whitelist: list,
    runtime_config,
    cap_budget: int | None,
    cap_label: str,
    hall_ellipsis: str,
    by_chapter: bool,
    source_name: str,
    n_chunks: int,
) -> _PlanExtractResult:
    """Gemeinsame Planner→(Cap)→Background→Extractor-Kette für Normal- und
    by-chapter-Pfad (#152). Die bug-anfällige Verdrahtung (welcher Text in
    planner.run/filter_hallucinated/run_extractors_per_concept, welche
    background_map) lebt hier an EINER Stelle; nur die pfad-spezifischen
    Ausgabetexte und die Cap-Semantik werden parametrisiert — Verhalten pro Pfad
    bleibt exakt erhalten (siehe Divergenz-Liste im PR):

    - Textbasis (Div. 3): `plan_text` (Planner), `hall_text` (Halluzinations-
      Filter), `extract_text` (Extractor) getrennt übergeben.
    - Cap (Div. 2): `cap_budget`/`cap_label` — by-chapter reicht das pro Kapitel
      dekrementierte `remaining_concepts` herein und dekrementiert selbst via
      `kept_actionable`; der Normalpfad cappt einmalig gegen `max_concepts`.
    - Background (Div. 1): by-chapter fährt IMMER `background_map={}` (bewusst,
      #102); der Normalpfad gated `background_extractor.run(plan)` über
      `ENABLE_BACKGROUND_EXTRACTOR` inkl. Stage-4.5-Log.
    - Spans (Div. 7): Planner- und Extractor-`_span` liegen jetzt hier, der
      by-chapter-Pfad ERBT sie dadurch (Bookkeeping ohne model-Feld, brechen keine
      Aggregation); der BackgroundExtractor-Span bleibt normalpfad-only, weil
      by-chapter den Background-Extractor gar nicht fährt.
    - Ellipsis (Div. 6): `hall_ellipsis` bewahrt die pfad-eigene Schreibweise der
      Halluzinations-Zeile ("..." by-chapter / "…" Normalpfad).
    """
    primary_authors = _extract_primary_authors(citation)
    with _span("Planner", pdf=source_name, n_chunks=n_chunks):
        plan = planner.run(plan_text, relevance_profile, primary_authors=primary_authors)
        plan, hallucinated = planner.filter_hallucinated(plan, hall_text)
    if hallucinated:
        print(
            f"      {len(hallucinated)} halluzinierte Konzepte verworfen: "
            f"{', '.join(hallucinated[:3])}{hall_ellipsis if len(hallucinated) > 3 else ''}"
        )
    if runtime_config is not None:
        plan.concepts, _capped = cap_actionable_concepts(plan.concepts, cap_budget)
        if _capped:
            print(
                f"      [runtime-config] {cap_label}={cap_budget} "
                f"-> {len(_capped)} Konzept(e) übersprungen: "
                f"{', '.join(c.title for c in _capped[:3])}"
                f"{'…' if len(_capped) > 3 else ''}"
            )

    related = [c.title for c in plan.concepts if c.origin == "secondary_mention"]
    actionable = [c for c in plan.concepts if c.action != "skip" and c.origin != "secondary_mention"]

    if by_chapter:
        if not actionable:
            print("      Keine Konzepte fuer dieses Kapitel")
            # Wie der frühere `continue`: kein Extractor-Call, und `related` dieses
            # Kapitels wird NICHT akkumuliert (related=[]), kept=0 dekrementiert nicht.
            return _PlanExtractResult(
                drafts=[],
                concept_map={},
                dropped=0,
                failures=[],
                related=[],
                kept_actionable=0,
                background_map={},
            )
        print(
            f"      {len(actionable)} Konzepte: "
            f"{', '.join(c.title for c in actionable[:4])}{'...' if len(actionable) > 4 else ''}"
        )
        # #102: hart leer statt background_extractor.run() pro Kapitel — bewusst
        # (Kosten-Multiplikation), Sichtbarkeit via Skip-Zeile im by-chapter-Zweigkopf.
        background_map = {}
    else:
        if related:
            print(
                f"      {len(related)} Sekundär-Erwähnungen → Related Mentions: "
                f"{', '.join(related[:3])}{'…' if len(related) > 3 else ''}"
            )
        print(f"      {len(actionable)} Konzepte geplant ({len(plan.concepts)} total)")
        for c in actionable:
            print(f"      [{c.priority:6s}] {c.action:6s} — {c.title}")

        # --- Schritt 4.5: Background-Extractor (nur Normalpfad) ---
        if ENABLE_BACKGROUND_EXTRACTOR:
            print("[4.5/7] Background-Extractor: Trainingswissen pro Konzept…")
            with _span("BackgroundExtractor", pdf=source_name):
                background_map = background_extractor.run(plan)
        else:
            print("[4.5/7] Background-Extractor: deaktiviert (ENABLE_BACKGROUND_EXTRACTOR=0)")
            background_map = {}

        # --- Schritt 5: Extractor ---
        print(f"\n[5/7] Extractor: {len(actionable)} Konzepte parallel verarbeiten…")

    with _span("Extractor", pdf=source_name, n_concepts=len(actionable)):
        drafts, concept_map, dropped, failures = asyncio.run(
            run_extractors_per_concept(
                extract_text,
                plan,
                existing_concepts,
                citation=citation,
                tag_whitelist=tag_whitelist,
                background_map=background_map,
                related_mentions=related,
                max_concurrent_calls=(runtime_config.max_concurrent_calls if runtime_config is not None else None),
            )
        )

    if not by_chapter:
        print(f"      {len(drafts)} Draft-Notes extrahiert")

    return _PlanExtractResult(
        drafts=drafts,
        concept_map=concept_map,
        dropped=dropped,
        failures=failures,
        related=related,
        kept_actionable=len(actionable),
        background_map=background_map,
    )


def _run_extraction_stages(
    args, source_path: Path, runtime_config=None
):  # main() übergibt immer einen RuntimeConfig; None = kein Runtime-Config / Capping deaktiviert
    """Stages 0–5: PDF extract → planning → extraction.

    Returns:
        RunContext — benannte Felder statt 19er-Positions-Tupel (#152). Der
        `--load-drafts`-Pfad (`_load_draft_state`) liefert dieselbe Struktur.
        `extractor_failures` (#210): [(concept_title, error)] für Konzepte, deren
        Extractor-Call mit Exception (Timeout/CLI-Fehler nach Retries) starb.
    """
    from generative.agents.base import trace_run_start as _trace_run_start
    from generative.config import MODEL_CONFIG as _MODEL_CONFIG

    _trace_run_start(_MODEL_CONFIG)

    # --- Schritt 1: PDF → Text + Metadata → Chunks ---
    print("[1/7] PDF extrahieren und chunken…")
    text = pdf_chunker.pdf_to_text(source_path)
    word_count = len(text.split())
    print(f"      {word_count} Wörter")
    # #48/M4 + #27/G6: gescanntes/textloses ODER zu dünnes PDF aktiv melden (sonst
    # leerer/dünner Output ohne Erklärung) + handlungsanleitender OCR-Hinweis.
    # Das Gate warnt nur (fail-open) und bricht den Lauf nicht ab.
    text_quality = pdf_chunker.assess_text_quality(text)
    if text_quality.is_empty or text_quality.is_thin:
        from generative.error_hints import scanned_pdf_hint

        print(
            scanned_pdf_hint(
                source_path.name,
                words_per_page=text_quality.words_per_page if text_quality.is_thin else None,
            )
        )
    chunks = pdf_chunker.split_by_chapters(text)
    pdf_meta_early = pdf_chunker.pdf_metadata(source_path) or {}
    try:
        source_pages = int(pdf_meta_early.get("Pages") or 0)
    except (TypeError, ValueError):
        source_pages = 0
    if (
        len(chunks) > MAX_CHUNKS_SHORT_DOC
        and 0 < source_pages <= MAX_PAGES_SHORT_DOC
        and not getattr(args, "by_chapter", False)
    ):
        print(
            f"      [chunk-cap] {len(chunks)} Chunks bei {source_pages} S. → "
            f"Fallback auf Word-Count-Split (max {MAX_CHUNKS_SHORT_DOC})"
        )
        chunks = pdf_chunker._split_by_words(text)
    print(f"      {len(chunks)} Chunks")
    if len(chunks) > LARGE_DOC_THRESHOLD and not getattr(args, "by_chapter", False):
        print(f"      [WARN] {len(chunks)} Chunks - großes Dokument. Erwäge --by-chapter für Bücher.")
    acronym_dict = acronym_fix.extract_acronym_pairs(text)
    if acronym_dict:
        print(
            f"      [schwartz-hearst] {len(acronym_dict)} Akronyme aus Quelle: "
            f"{', '.join(list(acronym_dict.keys())[:8])}"
            f"{'...' if len(acronym_dict) > 8 else ''}"
        )
    overview = pdf_chunker.extract_overview(text)
    pdf_meta = pdf_meta_early
    if pdf_meta:
        meta_line = (
            f"{pdf_meta.get('Title', '?')[:60]} | "
            f"{pdf_meta.get('Author', '?')[:40]} | "
            f"{pdf_meta.get('Year', '?')} | {pdf_meta.get('Pages', '?')} S."
        )
        print(f"      Metadata: {meta_line}")

    # --- Stage 0: PDF-Enrichment bei fehlenden Metadaten ---
    _has_author = bool(pdf_meta.get("Author") or pdf_meta.get("author")) if pdf_meta else False
    _has_year = bool(pdf_meta.get("Year") or pdf_meta.get("year")) if pdf_meta else False
    if not (_has_author and _has_year):
        print("[0/7] PDF-Enrichment — keine Metadaten im Dateinamen erkannt…")
        try:
            from generative.tools.pdf_enrich import enrich as _enrich

            # rename=False: Die Pipeline darf die Eingabedatei nie mutieren. Das
            # Enrichment-Ergebnis wird direkt in pdf_meta gemergt (statt über den
            # früheren Rename-Umweg), sodass korrekte Quellen weiterhin in die Note
            # fließen — aber ein (mit dem Title-Match-Gate verworfener) Fehltreffer
            # die Quelldatei nicht mehr umbenennt und damit die Quelle verfälscht.
            _enrich_meta = _enrich(
                source_path, dry_run=args.dry_run, llm_fallback=getattr(args, "llm_fallback", False), rename=False
            )
            if _enrich_meta:
                if _enrich_meta.get("title") and not pdf_meta.get("Title"):
                    pdf_meta["Title"] = _enrich_meta["title"]
                if _enrich_meta.get("author") and not pdf_meta.get("Author"):
                    pdf_meta["Author"] = _enrich_meta["author"]
                if _enrich_meta.get("year") and not pdf_meta.get("Year"):
                    pdf_meta["Year"] = str(_enrich_meta["year"])
        except Exception as _e:
            print(f"  [warn] PDF-Enrichment fehlgeschlagen: {_e}", file=sys.stderr)

    # --- Schritt 2+3: Context-Builder + Quality-Agent ---
    print("[2/7] Context-Builder: Vault scannen…")
    relevance_profile = context_builder.build_relevance_profile()
    existing_concepts = relevance_profile["existing_concepts"]
    print(f"      {len(existing_concepts)} existierende Konzepte gefunden")
    concept_links = context_builder.build_concept_links(existing_concepts)

    print("[3/7] Quality-Agent: Quellen-Qualität prüfen…")
    fb = vault_writer.parse_filename_fallback(source_path.name)
    q_title = pdf_meta.get("Title")
    if not q_title or vault_writer._TITLE_LOOKS_BAD.match(q_title or ""):
        q_title = fb.get("Title") or q_title
    # Zitier-Autor/-Jahr aus dem Dateiname befüllen (Info-Dict liefert keinen
    # zitierfähigen Autor/CreationDate-Jahr mehr — pdf_metadata). Muss vor dem
    # Extractor laufen, sonst stünde der Platzhalter "Autor" im Body.
    vault_writer.apply_filename_citation_metadata(pdf_meta, fb)
    quality_report = quality.check_quality(
        doi=args.doi,
        title=q_title,
        author=pdf_meta.get("Author") or fb.get("Author"),
        year=pdf_meta.get("Year") or fb.get("Year"),
    )
    if quality_report.flags:
        print(f"      Flags: {', '.join(quality_report.flags)}")
    else:
        print("      Keine Qualitäts-Warnungen")

    # E3a (#96): CitationMeta EINMAL konstruieren — VOR den Planner-/Extractor-
    # Aufrufen, damit beide (statt wie bisher nur der Vault-Writer) die CrossRef-
    # korrigierten Werte sehen. quality_report + q_title liegen hier bereits vor
    # (Quality ist Stage 3, Planner Stage 4).
    # #95: physical_pages-Signal aus demselben PDF, das text/pdf_meta oben schon
    # gelesen hat — Quellen ohne /PageLabels rendern Seitenangaben gekennzeichnet.
    citation = _build_citation(
        pdf_meta,
        quality_report,
        q_title,
        source_path.name,
        physical_pages=pdf_chunker.pdf_uses_physical_pages(source_path),
    )

    tag_whitelist = relevance_profile.get("tag_whitelist", [])
    background_map: dict = {}
    related_mentions: list[str] = []

    if getattr(args, "by_chapter", False) and len(chunks) > 1:
        # --- Schritt 4+5: Planner + Extractor kapitelweise ---
        print("[4-5/7] Planner + Extractor: Kapitel einzeln verarbeiten")
        # #102: Background-Extractor läuft hier bewusst NICHT — Sichtbarkeit
        # statt Verdrahtung, siehe _background_extractor_by_chapter_skip_line().
        _skip_line = _background_extractor_by_chapter_skip_line(ENABLE_BACKGROUND_EXTRACTOR)
        if _skip_line:
            print(_skip_line)
        all_drafts: list[AtomicNoteDraft] = []
        all_concept_map: dict = {}
        dropped_total = 0
        extractor_failures: list[tuple[str, str]] = []  # #210
        remaining_concepts = runtime_config.max_concepts if runtime_config is not None else None

        for i, chunk in enumerate(chunks, 1):
            title_preview = chunk.title[:60]
            suffix = "..." if len(chunk.title) > 60 else ""
            print(f"\n[4-5/7] Kapitel {i}/{len(chunks)}: {title_preview}{suffix}")

            if not chunk.text.strip():
                print("      Leerer Chunk, uebersprungen")
                continue

            # #152: gemeinsame Planner→Extractor-Kette. by-chapter reicht dreimal
            # chunk.text herein (plan/hall/extract), cappt gegen das kumulative
            # remaining_concepts und fährt background_map={} (#102, Skip-Zeile oben).
            _pe = _plan_and_extract(
                plan_text=chunk.text,
                hall_text=chunk.text,
                extract_text=chunk.text,
                relevance_profile=relevance_profile,
                existing_concepts=existing_concepts,
                citation=citation,
                tag_whitelist=tag_whitelist,
                runtime_config=runtime_config,
                cap_budget=remaining_concepts,
                cap_label="remaining_concepts",
                hall_ellipsis="...",
                by_chapter=True,
                source_name=source_path.name,
                n_chunks=len(chunks),
            )
            if runtime_config is not None and remaining_concepts is not None:
                remaining_concepts = max(0, remaining_concepts - _pe.kept_actionable)
            for t in _pe.related:
                if t not in related_mentions:
                    related_mentions.append(t)
            dropped_total += _pe.dropped
            extractor_failures.extend(_pe.failures)  # #210
            all_drafts.extend(_pe.drafts)
            for draft_title, concept_context in _pe.concept_map.items():
                all_concept_map.setdefault(draft_title, concept_context)

        drafts, concept_map = all_drafts, all_concept_map
        print(f"\n      {len(drafts)} Draft-Notes aus {len(chunks)} Kapiteln extrahiert")
    else:
        # --- Schritt 4+4.5+5: Planner + Background + Extractor (Einzeldokument) ---
        print("[4/7] Planner: Konzept-Plan erstellen…")
        # #152: dieselbe gemeinsame Kette; Normalpfad plant auf `overview`, filtert
        # gegen den Volltext `text`, cappt einmalig gegen max_concepts und fährt den
        # (gated) Background-Extractor.
        _pe = _plan_and_extract(
            plan_text=overview,
            hall_text=text,
            extract_text=text,
            relevance_profile=relevance_profile,
            existing_concepts=existing_concepts,
            citation=citation,
            tag_whitelist=tag_whitelist,
            runtime_config=runtime_config,
            cap_budget=(runtime_config.max_concepts if runtime_config is not None else None),
            cap_label="max_concepts",
            hall_ellipsis="…",
            by_chapter=False,
            source_name=source_path.name,
            n_chunks=len(chunks),
        )
        drafts = _pe.drafts
        concept_map = _pe.concept_map
        dropped_total = _pe.dropped
        extractor_failures = _pe.failures
        related_mentions = _pe.related
        background_map = _pe.background_map

    return RunContext(
        drafts=drafts,
        concept_map=concept_map,
        existing_concepts=existing_concepts,
        concept_links=concept_links,
        text=text,
        chunks=chunks,
        acronym_dict=acronym_dict,
        quality_report=quality_report,
        pdf_meta=pdf_meta,
        source_path=source_path,
        tag_whitelist=tag_whitelist,
        background_map=background_map,
        fb_year=fb.get("Year"),
        dropped_total=dropped_total,
        word_count=word_count,
        related_mentions=related_mentions,
        q_title=q_title,
        citation=citation,
        extractor_failures=extractor_failures,
    )


def _save_draft_state(
    path: str,
    *,
    drafts: list,
    concept_map: dict,
    existing_concepts: dict,
    concept_links: dict,
    text: str,
    chunks: list,
    acronym_dict: dict,
    quality_report,
    pdf_meta: dict,
    source_name: str,
    tag_whitelist: list,
    background_map: dict,
    filename_year: str | None,
    related_mentions: list[str] | None = None,
) -> None:
    state = {
        "drafts": [dataclasses.asdict(d) for d in drafts],
        "concept_map": {k: [dataclasses.asdict(v[0]), v[1]] for k, v in concept_map.items()},
        "existing_concepts": existing_concepts,
        "concept_links": {k: list(v) for k, v in concept_links.items()},
        "text": text,
        "chunks": [dataclasses.asdict(c) for c in chunks],
        "acronym_dict": acronym_dict,
        "quality_report": dataclasses.asdict(quality_report),
        "pdf_meta": pdf_meta,
        "source_name": source_name,
        "tag_whitelist": tag_whitelist,
        "background_map": background_map or {},
        "filename_year": filename_year,
        "related_mentions": related_mentions or [],
    }
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [save-drafts] {len(drafts)} Drafts → {path}")


def _load_draft_state(path: str) -> RunContext:
    from generative.schemas.atomic_note import AtomicNoteDraft, TextAnchor, QualityReport, ConceptItem
    from generative.pipeline.pdf_chunker import Chunk

    state = json.loads(Path(path).read_text(encoding="utf-8"))

    def _to_draft(d: dict) -> AtomicNoteDraft:
        d["source_anchors"] = [TextAnchor(**a) for a in d["source_anchors"]]
        return AtomicNoteDraft(**d)

    drafts = [_to_draft(d) for d in state["drafts"]]
    concept_map = {k: (ConceptItem(**v[0]), v[1]) for k, v in state["concept_map"].items()}
    concept_links = {k: set(v) for k, v in state["concept_links"].items()}
    quality_report = QualityReport(**state["quality_report"])
    chunks = [Chunk(**c) for c in state["chunks"]]
    pdf_meta = state["pdf_meta"]
    source_path = Path(state["source_name"])
    text = state["text"]

    # --- Rekonstruktion der in _run_extraction_stages berechneten, aber NICHT
    # persistierten Felder (Stage 1–5 sind hier übersprungen), damit der
    # --load-drafts-Pfad dieselbe RunContext-Struktur liefert wie der Normalpfad
    # (#152). Früher lag diese Rekonstruktion inline in main(). ---
    # q_title wird im Normalpfad von _run_extraction_stages durchgereicht; hier aus
    # dem geladenen pdf_meta abgeleitet.
    q_title = (pdf_meta or {}).get("Title")
    # citation (CitationMeta, #96 E3a): Stage 1–5 übersprungen, daher aus dem
    # geladenen pdf_meta/quality_report neu konstruiert — dieselbe deterministische
    # Factory wie im Normalpfad (_build_citation).
    # #95: physical_pages hier per Zweit-Check auf dieselbe source_path neu ermittelt
    # (analog zur Edition-Verifikation in main(), die _pdf_page_labels(source_path)
    # ebenfalls unabhängig neu aufruft) statt über den State persistiert —
    # deterministisch, solange die PDF-Datei am gespeicherten Pfad noch existiert.
    citation = _build_citation(
        pdf_meta,
        quality_report,
        q_title,
        source_path.name,
        physical_pages=pdf_chunker.pdf_uses_physical_pages(source_path),
    )

    return RunContext(
        drafts=drafts,
        concept_map=concept_map,
        existing_concepts=state["existing_concepts"],
        concept_links=concept_links,
        text=text,
        chunks=chunks,
        acronym_dict=state["acronym_dict"],
        quality_report=quality_report,
        pdf_meta=pdf_meta,
        source_path=source_path,
        tag_whitelist=state["tag_whitelist"],
        background_map=state.get("background_map") or {},
        fb_year=state.get("filename_year"),
        dropped_total=0,
        word_count=len(text.split()),
        related_mentions=state.get("related_mentions") or [],
        q_title=q_title,
        citation=citation,
        extractor_failures=[],  # #210: Stage 1-5 übersprungen → keine Extractor-Calls
    )


def main(argv: list[str] | None = None):
    from generative.ui_strings import msg

    ap = argparse.ArgumentParser(description=msg("orch.description"))
    ap.add_argument("--source", default=None, help=msg("orch.arg.source"))
    ap.add_argument("--doi", default=None, help=msg("orch.arg.doi"))
    ap.add_argument("--dry-run", action="store_true", help=msg("orch.arg.dry_run"))
    ap.add_argument("--by-chapter", action="store_true", help=msg("orch.arg.by_chapter"))
    ap.add_argument("--no-llm", action="store_true", help=msg("orch.arg.no_llm"))
    ap.add_argument("--target-tag", default=None, help=msg("orch.arg.target_tag"))
    ap.add_argument("--llm-fallback", action="store_true", help=msg("orch.arg.llm_fallback"))
    ap.add_argument("--fresh-run", action="store_true", help=msg("orch.arg.fresh_run"))
    ap.add_argument("--save-drafts", default=None, metavar="PATH", help=msg("orch.arg.save_drafts"))
    ap.add_argument("--load-drafts", default=None, metavar="PATH", help=msg("orch.arg.load_drafts"))
    ap.add_argument("--inbox-dir", default=None, metavar="PATH", help=msg("orch.arg.inbox_dir"))
    ap.add_argument(
        "--export-format",
        default=None,
        metavar="FMT[,FMT...]",
        help=msg("orch.arg.export_format", future=", ".join(export_runner.FUTURE_FORMATS)),
    )
    ap.add_argument("--export-dir", default=None, metavar="PATH", help=msg("orch.arg.export_dir"))
    args = ap.parse_args(argv)
    if not args.source and not args.load_drafts:
        ap.error(msg("orch.error.source_required"))

    # Fail-fast (F4): Format-Fehler VOR jeder teuren/mutierenden Pipeline-Stufe
    # melden, nicht erst nach Stunden LLM-Arbeit im [7/7]-Block. Ergebnis wird
    # unten am eigentlichen Verwendungsort wiederverwendet (kein zweites Parsen).
    export_formats: tuple[str, ...] = ()
    if args.export_format:
        try:
            export_formats = export_runner.parse_export_formats(args.export_format)
        except ValueError as e:
            sys.exit(str(e))
        # Auch die Export-Deps fail-fast prüfen (Review-Fund PR #136): ein
        # Binärformat (docx/pdf/…) ohne installierte pandoc/typst-Deps würde
        # sonst erst NACH dem kompletten Lauf im Export-Block crashen. Lazy
        # Import — reine Formate (json/portable-md/obsidian-md) berühren
        # export_convert weiterhin nie.
        if export_runner.requires_export_deps(export_formats):
            from generative.pipeline import export_convert as _export_convert

            _deps_ok, _deps_detail = _export_convert.export_available()
            if not _deps_ok:
                sys.exit(
                    f"--export-format {args.export_format}: pandoc/typst nicht verfügbar: {_deps_detail} — "
                    'installiere sie mit: pip install "atomic-notes[export]"'
                )

    _setup_phoenix_tracing()
    # #156: Die schreibenden Erst-Lauf-Nebeneffekte laufen NUR für den Maintainer
    # (ATOMIC_AGENT_MAINTAINER=1) — Default aus. _auto_version_bump() mutiert den
    # getrackten Quellcode (config.py), _auto_start_dashboard() spawnt ein Dashboard
    # auf :8051; für Fremd-Nutzer bräche das den „Vorschau schreibt nichts"-Vertrag
    # (dirty checkout + undokumentierter Server nach dem ersten Dry-Run).
    # ATOMIC_AGENT_GUI=1 (gesetzt vom GUI-Subprocess-Runner) unterdrückt zusätzlich,
    # unabhängig vom Maintainer-Flag: der GUI-Pfad spawnt sonst ein zweites Dashboard.
    if is_maintainer() and not os.getenv("ATOMIC_AGENT_GUI"):
        _auto_start_dashboard()
        _auto_version_bump()

    runtime_config = load_runtime_config()
    from generative.agents.base import set_llm_runtime_config

    set_llm_runtime_config(runtime_config)
    refine_budget = RunBudget(max_refines_per_run=runtime_config.refine.max_refines_per_run)
    print(
        "[runtime-config] "
        f"profile={runtime_config.profile} "
        f"inline_eval={runtime_config.inline_eval} "
        f"max_concepts={runtime_config.max_concepts} "
        f"max_refines_per_run={runtime_config.refine.max_refines_per_run} "
        f"timeout_retries={runtime_config.timeout_retries}"
    )

    if getattr(args, "fresh_run", False):
        from generative.agents.base import set_cache_namespace
        from generative.agents.tracing import _RUN_ID

        set_cache_namespace(_RUN_ID)
        print(f"  [cache] --fresh-run: Namespace={_RUN_ID} (kein Hit aus alten Runs)")

    if getattr(args, "no_llm", False):
        from generative import config as _cfg

        _cfg.ENABLE_LLM = False  # Modul-Attribut mutieren — sichtbar für alle Agents
        print("[no-llm] Stage-6-Agents im FOSS-Modus (Verifier/CrossRef/Critic ohne LLM)")

    import time as _time

    _run_start = _time.time()
    from generative.agents.base import trace_event as _trace_event

    if args.load_drafts:
        # #152: _load_draft_state liefert dieselbe RunContext-Struktur wie
        # _run_extraction_stages (Rekonstruktion von q_title/citation/word_count etc.
        # liegt jetzt dort, inkl. #95/#96/#210-Begründungen).
        ctx = _load_draft_state(args.load_drafts)
        source_path = ctx.source_path
        print(f"\n=== Atomic Agent (load-drafts): {source_path.name} ===\n")
        print(f"  [load-drafts] {len(ctx.drafts)} Drafts geladen · Stage 1–5 übersprungen")
    else:
        # #186-Nachbesserung: derselbe Apostroph-/Anfuehrungszeichen-Glob-Fallback
        # wie extractive/orchestrator.py und eval_chunk_recall.py -- vorher brach
        # dieser Haupt-CLI-Pfad mit einem nackten sys.exit bei reinen Apostroph-
        # Varianten ab.
        try:
            source_path = resolve_source_path(args.source)
        except FileNotFoundError as exc:
            sys.exit(f"Datei nicht gefunden: {exc}")
        print(f"\n=== Atomic Agent: {source_path.name} ===\n")
        ctx = _run_extraction_stages(args, source_path, runtime_config)
        if args.save_drafts:
            _save_draft_state(
                args.save_drafts,
                drafts=ctx.drafts,
                concept_map=ctx.concept_map,
                existing_concepts=ctx.existing_concepts,
                concept_links=ctx.concept_links,
                text=ctx.text,
                chunks=ctx.chunks,
                acronym_dict=ctx.acronym_dict,
                quality_report=ctx.quality_report,
                pdf_meta=ctx.pdf_meta,
                source_name=str(source_path),
                tag_whitelist=ctx.tag_whitelist,
                background_map=ctx.background_map,
                filename_year=ctx.fb_year,
                related_mentions=ctx.related_mentions,
            )

    # `drafts` wird ab hier durch Dedup-/Stage-6-Stufen ersetzt → lokale (mutierbare)
    # Bindung; alle übrigen Stage-Ergebnisse werden per Attribut aus `ctx` gelesen.
    drafts = ctx.drafts

    # #210: Extractor-Ausfälle (Timeout/CLI-Fehler nach Retries) sichtbar machen.
    # n_attempted = erfolgreiche Extraktionen (drafts vor Dedup) + dropped (Fehler+Leer);
    # exit_code wird an ALLEN Rückgabepunkten zurückgegeben, damit ein Teilverlust den
    # Prozess mit 3 beendet (unterscheidbar von hartem Abbruch=1) statt still mit 0.
    n_extract_attempted = len(drafts) + ctx.dropped_total
    exit_code = extractor_failure_exit_code(ctx.extractor_failures)

    if not drafts:
        print("\nKeine Konzepte extrahiert. Fertig.")
        for _line in format_extractor_failure_report(ctx.extractor_failures, n_extract_attempted):
            print(_line, file=sys.stderr)
        return exit_code

    # #197 Schritt 2: Funnel-Top "nach Planner/Extractor generiert" festhalten,
    # BEVOR Artifact-/Dedup-/Stage-6-Drops die Liste stutzen. Wird als neues Feld
    # n_extracted persistiert; n_generated (= geschriebene Notes) bleibt für
    # Alt-Daten-Vergleichbarkeit unverändert.
    n_extracted = len(drafts)

    # --- Artifact-Detector: Abwesenheits-Noten früh verwerfen (kein LLM-Call) ---
    drafts = _drop_artifacts(drafts)
    if not drafts:
        print("\nAlle Drafts als Artefakte verworfen. Fertig.")
        for _line in format_extractor_failure_report(ctx.extractor_failures, n_extract_attempted):
            print(_line, file=sys.stderr)
        return exit_code

    # Qualitäts-Flags aus QualityReport auf alle Notes übertragen
    for d in drafts:
        d.quality_flags.extend(ctx.quality_report.flags)

    # --- Dedup Stage A: Exact-Match (deterministisch, keine LLM-Calls) ---
    drafts = dedup_exact(drafts, ctx.existing_concepts)
    print(f"      {len(drafts)} nach Exact-Dedup")

    # --- Dedup Stage B: Entity-Resolution (Embedding-Cluster + LLM-Merge) ---
    # Christen-2012-Pipeline: Blocking → Embedding-Cosine → Clustering → Canonicalization.
    # Verhindert dass Title-Varianten desselben Konzepts (z.B. 'HIB' + 'HIB (Bates)')
    # als getrennte Notes überleben — Bodies werden semantisch gemergt, kein Inhaltsverlust.
    pre_er_count = len(drafts)
    drafts = asyncio.run(
        entity_resolution(
            drafts,
            max_concurrent_calls=(runtime_config.max_concurrent_calls if runtime_config is not None else None),
        )
    )
    if len(drafts) < pre_er_count:
        print(f"      {len(drafts)} nach Entity-Resolution ({pre_er_count - len(drafts)} Cluster gemergt)")

    # --- Cross-Draft-Hub-Resolution (v29) ---
    # Erkennt MoC-Drafts anhand parallel erzeugter Stage-Drafts. Critic kann das nicht,
    # weil sein existing_concepts der Vault-Index VOR dem Run ist — Stage-Notes sind
    # dort nicht. Modell-Übersichten (z.B. ADKAR-Modell mit Mentions zu seinen 5 Stages)
    # bleiben sonst fälschlich als atomic. Siehe pipeline/cross_draft_hub.py.
    from generative.pipeline import cross_draft_hub

    hub_resolved = cross_draft_hub.resolve(drafts)
    if hub_resolved:
        print(f"      [hub-resolution] {hub_resolved} Draft(s) als MoC erkannt (Cross-Mentions)")
    # #4: marker-lose thematische Cluster, die resolve() nicht fängt — nur vorschlagen,
    # nicht auto-anlegen (Fabrikations-Risiko, separate User-Entscheidung).
    for _token, _members in cross_draft_hub.suggest_unmarked_clusters(drafts):
        _preview = ", ".join(_members[:4]) + ("…" if len(_members) > 4 else "")
        print(
            f"      [moc-suggestion] {len(_members)} marker-lose Drafts teilen "
            f"'{_token}' → MoC-{_token.capitalize()}? ({_preview})"
        )

    # --- Schritte 6a-c: Verifier + Cross-Reference + Critic pro Note (parallel) ---
    print(f"\n[6/7] Verifier + Cross-Reference + Critic für {len(drafts)} Notes…")

    chunk_map = {c.title: c.text for c in ctx.chunks}

    with _span("Stage6-Verifier-CrossRef-Critic", pdf=source_path.name, n_drafts=len(drafts)):
        drafts = asyncio.run(
            process_all_notes_async(
                drafts,
                ctx.existing_concepts,
                ctx.concept_links,
                chunk_map,
                full_text=ctx.text,
                acronym_dict=ctx.acronym_dict,
                concept_map=ctx.concept_map,
                quality_report=ctx.quality_report,
                citation=ctx.citation,
                source_path=source_path,
                tag_whitelist=ctx.tag_whitelist,
                background_map=ctx.background_map,
                related_mentions=ctx.related_mentions,
                runtime_config=runtime_config,
                refine_budget=refine_budget,
            )
        )

    # --- Dedup Stage C: Intra-Run-Sibling-Dedup (Befund D) ---
    # cross_reference setzt bei dup_risk=high action=extend + extend_path=<Sibling-Titel>.
    # Zeigt das auf einen Draft DESSELBEN Laufs (keine Vault-Datei), verpufft es beim
    # Writer und beide Notes würden geschrieben. Hier auf das vorhandene Signal reagieren
    # und Geschwister eines Laufs deterministisch zu EINER Note kollabieren — nach den
    # per-Draft-Calls (Signal steht erst jetzt fest), vor boilerplate_dedup und Writer.
    drafts, n_sib = resolve_sibling_dups(drafts, ctx.existing_concepts)
    if n_sib:
        print(f"      [sibling-dedup] {n_sib} Intra-Run-Near-Dup(s) in Geschwister-Note(s) gemergt")

    # --- Hebel #5: Boilerplate-Dedup zwischen Hub-Drafts und Sub-Konzept-Drafts ---
    drafts, stripped = boilerplate_dedup.dedup_hub_subconcepts(drafts)
    if stripped:
        print(f"\n[boilerplate-dedup] {stripped} geteilte Sätze aus Sub-Notes in Hubs zentralisiert")

    # --- #8: Body-Redundanz-Flag zwischen DISTINKTEN Geschwister-Notes ---
    # Nach den Dedup-Stages (echte Dups/Hub→Sub schon behandelt): distinkte create-Notes mit
    # hoher Body-Cosine sind weder mergebar noch satz-strippbar (2 empirische Gates,
    # Ebner-Audit) → seiteneffekt-freier Flag für den menschlichen Reviewer, kein Eingriff.
    drafts, n_redund = flag_redundant_siblings(drafts)
    if n_redund:
        print(
            f"[redundanz-flag] {n_redund} Note-Paar(e) mit hoher Body-Überlappung markiert (Review-Hinweis, kein Merge)"
        )

    # --- E3b (#96): Zitations-Attributions-Validierung gegen CitationMeta ---
    # Analog zu #8 (flag_redundant_siblings): seiteneffekt-freier Flag, wenn eine
    # LLM-generierte Autor-/Jahr-Attribution im Body von der kanonischen
    # CitationMeta abweicht (Regressionsfall: "Landry 2019" statt Knowles).
    # Kein Body-Edit, kein Routing-Eingriff — nur ein Review-Hinweis.
    n_citation_flags = citation_check.apply_citation_check(drafts, ctx.citation)
    if n_citation_flags:
        print(f"[citation-check] {n_citation_flags} Attribution(s) ohne Quellendeckung geflaggt")

    # --- Issue #95: Quelle ohne /PageLabels -> Seitenangaben sind PDF-Position ---
    # Seiteneffekt-freier Review-Hinweis (analog #8/E3b): render_note/render_moc
    # kennzeichnen Seitenangaben bereits als "PDF-S." (citation.physical_pages),
    # dieses Flag macht die Einschränkung zusätzlich im Frontmatter sichtbar.
    n_physical_flags = citation_check.apply_physical_pages_flag(drafts, ctx.citation)
    if n_physical_flags:
        print(f"[physical-pages] {n_physical_flags} Note(s) ohne /PageLabels — Seiten als PDF-Position geflaggt")

    # --- Schritt 7: Vault-Writer ---
    # E3a (#96): citation (CitationMeta) wurde bereits vor dem Planner konstruiert
    # (siehe _build_citation) — die frühere enriched_meta/pdf_meta-Zweiteilung
    # entfällt, Extractor/Planner/Writer lesen dieselbe CrossRef-korrigierte Quelle.

    # #45: fail-closed sichtbar machen — wenn die Quelle nicht zuverlässig
    # aufgelöst werden konnte (CrossRef-Override verworfen ODER Autor/Jahr nach
    # Enrichment weiter unbekannt), die create-Notes mit source-status: unresolved
    # markieren und eine ehrliche NL-Zeile drucken. Friction nur auf diesem Pfad —
    # aufgelöste Quellen bleiben frictionless.
    # Resolved-Check via pure Helper (testbar). fb wird hier in main-Scope neu
    # geparst (deterministisch, idempotent — dieselbe Funktion wie in der
    # Extraction-Stage). Nur create-Notes werden markiert (extend/hub out-of-scope).
    _fb = vault_writer.parse_filename_fallback(source_path.name)
    _source_unresolved = routing_report.is_source_unresolved(
        ctx.citation.as_meta_dict(), _fb, crossref_override_blocked(ctx.quality_report, ctx.q_title)
    )
    if _source_unresolved:
        _marked = 0
        for draft in drafts:
            if draft.action == "create":
                draft.source_status = "unresolved"
                _marked += 1
        if _marked:
            _framing = routing_report.source_status_framing("unresolved", source_path.name)
            if _framing:
                print(_framing)

    # Edition-Verifikation (Layer 1): Auszug aus einem größeren Werk OHNE DOI-Beleg
    # → Auflage/Jahr/Seiten sind nur dateiname-geraten (Impressum fehlt im Extrakt).
    # first_print_page = erste numerische Druckseite aus /PageLabels (>1 ⟺ Auszug);
    # doi_verified, wenn eine harte DOI via CrossRef auflöste (--doi ODER hartes
    # Enrichment, kein Title-Match-Raten). Nur create-Notes, unresolved bleibt stärker.
    _ed_labels = pdf_chunker._pdf_page_labels(source_path)
    _first_print_page = int(str(_ed_labels[0]).strip()) if _ed_labels else None
    # doi_verified NUR wenn CrossRef die DOI tatsächlich auflöste (crossref_year
    # gesetzt) und sie nicht per Title-Match geraten wurde. Ein gepinntes --doi, das
    # nicht auflöst (falsch/CrossRef down), zählt NICHT als verifiziert → fail-closed,
    # die Note wird geflaggt statt still vertraut. (Codex-Review, fail-open-Lücke.)
    _doi_verified = bool(ctx.quality_report.crossref_year) and not ctx.quality_report.doi_from_title_match
    if routing_report.is_edition_unverified(_doi_verified, _first_print_page):
        _ed_marked = 0
        for draft in drafts:
            if draft.action == "create" and draft.source_status != "unresolved":
                draft.source_status = "edition-unverified"
                draft.quality_flags.append(
                    f"⚠️ Edition unverifiziert — Auszug ab Druckseite {_first_print_page} "
                    f"ohne DOI; Jahr+Seiten nur aus Dateiname. Auflage manuell prüfen "
                    f"oder via --doi pinnen."
                )
                _ed_marked += 1
        if _ed_marked:
            _ed_framing = routing_report.source_status_framing("edition-unverified", source_path.name)
            if _ed_framing:
                print(_ed_framing)

    # v23: Tag-Hint via --target-tag wird allen Drafts angehängt → Auto-Note-Mover
    # routet beim Öffnen aus 00-inbox/ in den Zielordner (siehe CLAUDE.md-Mapping).
    if args.target_tag:
        target_tag = args.target_tag.strip().lstrip("#")
        for draft in drafts:
            if target_tag not in draft.tags:
                draft.tags.append(target_tag)
        print(f"\n[target-tag] '{target_tag}' an {len(drafts)} Notes angehängt (Auto-Note-Mover-Routing)")

    _inbox_dir = Path(args.inbox_dir) if args.inbox_dir else None
    if _inbox_dir and not args.dry_run:
        _inbox_dir.mkdir(parents=True, exist_ok=True)

    # Issue #21: Sibling-related-Links auf Merge-Targets umschreiben, bevor
    # geschrieben wird — sonst zeigen sie auf nie-erzeugte Draft-Titel-Dateien.
    n_rewritten = vault_writer.rewrite_merged_related_links(drafts, ctx.existing_concepts)
    if n_rewritten:
        print(f"[merge-links] {n_rewritten} related-Link(s) auf Merge-Target umgeschrieben")

    # Figur-Alt-Text aus PDF-UA-getaggten PDFs einbetten (Pfad C). Mutiert create-Draft-
    # Bodies VOR dem Render. No-op auf untagged PDFs (Gate). Precision-first: nur exakte
    # 1:1-Bindung Figur→Note via source_anchor-Seite, sonst skip. Siehe figure_alt.py.
    fig_report = figure_alt.embed_alt_figures(source_path, drafts)
    if fig_report.bound or fig_report.skipped:
        print(
            f"[figures] {len(fig_report.bound)} Alt-Text-Figur(en) eingebettet, "
            f"{len(fig_report.skipped)} ohne eindeutige Bindung übersprungen"
        )
    elif fig_report.untagged:
        # #50/M11: untagged-PDF einmal melden statt stumm zu überspringen.
        print(
            "[figures] PDF nicht PDF-UA-getaggt — Abbildungen (falls vorhanden) "
            "werden übersprungen (nur getaggte PDFs liefern Alt-Text)."
        )

    print("\n[7/7] Vault-Writer…")
    written = 0
    written_targets: list[tuple[Path, bool]] = []
    # #241: derselbe run_id wird unten (Stage 8, Reader-Seite) für cache_note_dir
    # wiederverwendet — Writer und Reader müssen denselben Namespace treffen,
    # sonst findet dry_run_eval_targets die gerade geschriebenen Notes nicht.
    from generative.agents.base import _RUN_ID as _run_id_for_eval_cache

    with _span("VaultWriter", pdf=source_path.name, n_drafts=len(drafts), dry_run=args.dry_run):
        for draft in drafts:
            target = vault_writer.write_note(
                draft,
                source_file=source_path.name,
                dry_run=args.dry_run,
                citation=ctx.citation,
                existing_concepts=ctx.existing_concepts,
                inbox_dir=_inbox_dir,
                run_id=_run_id_for_eval_cache,
            )
            will_vault, _ = vault_writer.auto_write_decision(draft)
            written_targets.append((target, will_vault))
            _trace_event(
                "orchestrator",
                "note_outcome",
                {
                    "title": draft.title,
                    "destination": "vault" if will_vault else "inbox",
                    "critic_score": draft.critic_score,
                    "hard_gates_pass": draft.hard_gates_pass,
                },
            )
            written += 1

    print(f"\n=== Fertig: {written} Notes {'(dry-run)' if args.dry_run else 'geschrieben'} ===")
    # #210: verlorene Konzepte (Timeout/CLI-Fehler) direkt im Summary ausweisen —
    # kein stilles Exit 0. Der Prozess endet unten mit exit_code (3), wenn befüllt.
    for _line in format_extractor_failure_report(ctx.extractor_failures, n_extract_attempted):
        print(_line, file=sys.stderr)
    # #45: Final-Report um Gründe-Aggregat erweitern (Routing-Verteilung +
    # "0 PDFs verändert"-Zusicherung sichtbar machen).
    _summary = routing_report.summarize_routing(drafts)
    for _line in routing_report.final_report_lines(drafts):
        print(_line)
    vault_count = _summary["vault"]
    inbox_count = _summary["inbox"]

    _trace_event(
        "orchestrator",
        "plan_stats",
        {
            "written": written,
            "vault": vault_count,
            "inbox": inbox_count,
            "vault_rate": round(vault_count / written, 3) if written > 0 else 0.0,
        },
    )
    from generative.agents.tracing import flush_tracing as _flush_tracing

    _flush_tracing()

    # Cache-Rotation (#151, Punkt 6): .cache/llm und .cache/runs anzahl-basiert stutzen,
    # einmal pro Lauf. Nie fatal — Disk-Hygiene darf keinen Lauf abbrechen.
    try:
        from generative import cache_rotation as _cache_rotation

        _n_llm, _n_runs = _cache_rotation.rotate_run_caches()
        if _n_llm or _n_runs:
            print(f"   [cache-rotation] {_n_llm} llm + {_n_runs} runs Dateien entfernt (Cap)")
    except Exception as _rot_err:
        print(f"   [cache-rotation] uebersprungen: {_rot_err}", file=sys.stderr)

    # F4 (Output-Projekt): zusätzliche Export-Formate — läuft bewusst UNABHÄNGIG
    # von args.dry_run (json/portable-md/docx/… brauchen keinen Vault-Schreib-
    # Lauf, nur die bereits vorhandenen Drafts). export_formats ist oben bereits
    # fail-fast validiert; hier nur noch die Ausführung + sichtbares Reporting.
    if export_formats:
        export_root = (
            Path(args.export_dir)
            if args.export_dir
            else Path(__file__).resolve().parent / ".cache" / "exports" / source_path.stem
        )
        exported_files, export_messages = export_runner.run_export(
            drafts,
            ctx.citation,
            export_formats,
            export_root,
            written_files=[_t for _t, _ in written_targets],
            dry_run=args.dry_run,
        )
        print(f"\n[export] {len(exported_files)} Datei(en) → {export_root}")
        for _f in exported_files:
            print(f"  [export] {_f.name}")
        for _msg in export_messages:
            print(f"  [export] {_msg}")

    # Token + Laufzeit-Summary (Pipeline Stages 1–7) — immer gedruckt (auch dry-run).
    # Stage-8-Eval läuft erst danach; deren Tokens/Zeit kommen in der finalen
    # Re-Aggregation am Run-Ende dazu (sonst unsichtbar — siehe Reporting-Quirk).
    _wall_s_early = round(_time.time() - _run_start, 1)
    try:
        from generative.agents.base import _RUN_ID, _RUN_DIR
        from generative import eval_agent_stats as _eas

        _trace_path = _RUN_DIR / f"{_RUN_ID}.jsonl"
        _pipe = _eas.run_totals(_trace_path)
        print(f"   -> Zeit:   {_wall_s_early}s")
        print(
            f"   -> Tokens: {_pipe['total']:,} (In:{_pipe['input']:,} Out:{_pipe['output']:,} Cache-R:{_pipe['cache_read']:,} Cache-C:{_pipe['cache_create']:,})"
        )
        print(f"   -> Quelle: {source_path.name}")
    except Exception:
        print(f"   -> Zeit:   {_wall_s_early}s  |  Tokens: n/a  |  Quelle: {source_path.name}")

    # --- pipeline_runs-Insert: entkoppelt vom Inline-Eval (#198 P1) ---
    # Dieser Insert lag früher IN Stage 8. Bei deaktiviertem Inline-Eval (Profil
    # fast/balanced oder ATOMIC_AGENT_INLINE_EVAL=0) kehrte main() VOR dem Insert
    # zurück — der Lauf hatte einen vollständigen Trace, aber keine DB-Zeile, war
    # also keiner Pipeline-Version zuordenbar und fiel aus allen versions-gefilterten
    # Ansichten. Der Insert hängt an nichts Eval-spezifischem (nur an Trace-Tokens +
    # Run-Zählern), darum läuft er jetzt unbedingt. Die eval-abhängigen note_evals
    # werden weiterhin ausschließlich bei aktivem Stage-8-Eval geschrieben.
    from generative import eval_agent_stats as _eas
    from generative.agents.base import _RUN_DIR as _run_dir_for_meta
    from generative.agents.base import _RUN_ID as _run_id_for_meta

    _trace_path = _run_dir_for_meta / f"{_run_id_for_meta}.jsonl"
    _wall_s = round(_time.time() - _run_start, 1)
    _pre = _eas.run_totals(_trace_path)  # tolerant bei fehlendem/kaputtem Trace → Nullen
    _tok_in, _tok_out = _pre["input"], _pre["output"]
    _tok_cache_r, _tok_cache_c = _pre["cache_read"], _pre["cache_create"]
    _tok_total = _pre["total"]
    _cost_usd = _pre["cost_usd"]

    run_meta = {
        "wall_time_s": _wall_s,
        "tokens_input": _tok_in,
        "tokens_output": _tok_out,
        "tokens_cache_read": _tok_cache_r,
        "tokens_cache_create": _tok_cache_c,
        "tokens_total": _tok_total,
    }

    # DB: pipeline_run persistieren. Eigener try/except — ein DB-Fehler darf den
    # Lauf nie abbrechen. get_db(DB_PATH) liest den Modul-Pfad zur Laufzeit (wie
    # calibration.collect), damit Tests ihn auf eine tmp-DB umbiegen statt die
    # produktive DB zu treffen.
    try:
        from generative import config as _db_cfg
        from generative import db as _db
        from generative.agents.base import _RUN_ID as _db_run_id

        with _db.get_db(_db.DB_PATH) as _conn:
            _db.insert_run(
                _conn,
                {
                    "run_id": _db_run_id,
                    "pipeline_version": AGENT_VERSION,
                    "pdf_source": source_path.name,
                    "pdf_key": source_path.stem.split(" - ")[0].strip().lower(),
                    "pdf_label": source_path.stem.split(" - ")[0].strip(),
                    "n_generated": written,
                    "n_extracted": n_extracted,
                    "n_vault": vault_count,
                    "n_inbox": inbox_count,
                    "n_merge": sum(1 for d in drafts if getattr(d, "action", "") == "extend"),
                    "n_dropped": ctx.dropped_total,
                    "n_words": ctx.word_count,
                    "model": getattr(_db_cfg, "MODEL_PLANNER", ""),
                    "cost_usd": _cost_usd,
                    "tokens_total": _tok_total,
                    "tokens_input": _tok_in,
                    "tokens_output": _tok_out,
                    "tokens_cache_read": _tok_cache_r,
                    "duration_s": _wall_s,
                },
            )
    except Exception as _db_err:
        print(f"   [warn] DB-Write fehlgeschlagen: {_db_err}")

    # --- Stage 8: Qualitäts-Eval (deterministisch, immer gespeichert) ---
    # Läuft nach jedem Run automatisch — PyMuPDF + Fuzzy + Semantic gegen Quell-PDF.
    # Ergebnisse in .cache/quality_history.jsonl für Longitudinal-Vergleiche.
    # Abschaltbar via ATOMIC_AGENT_INLINE_EVAL=0 oder Profil (fast/balanced); retroaktive Eval via reeval_baseline.py.
    if not inline_eval_enabled(runtime_config):
        print(
            f"\n[8/8] Qualitäts-Eval übersprungen (Profil: {runtime_config.profile}, inline_eval deaktiviert) — retro via reeval_baseline.py."
        )
        return exit_code
    print("\n[8/8] Qualitäts-Eval…")
    try:
        from generative.config import CACHE_DIR as _CACHE_DIR

        # Dry-Run: Notes im Cache-Verzeichnis; Live: im Vault (00-inbox oder 04-wissen)
        if args.dry_run:
            # #241: derselbe run_id wie beim Schreiben oben (_run_id_for_eval_cache) —
            # sonst sucht der Reader im falschen (leeren) Namespace.
            cache_note_dir = _CACHE_DIR / "eval" / "baseline" / source_path.stem / _run_id_for_eval_cache
            note_files = dry_run_eval_targets(written_targets, cache_note_dir)
        else:
            from generative.config import INBOX, WISSEN

            note_files = []
            _eval_search_dirs = ([_inbox_dir] if _inbox_dir else []) + [INBOX, WISSEN]
            for d in drafts:
                slug = vault_writer.slugify(d.title)
                for search_dir in _eval_search_dirs:
                    candidates = list(search_dir.glob(f"{slug}*.md")) + list(search_dir.glob(f"*{slug}*.md"))
                    if candidates:
                        note_files.append(candidates[0])
                        break

        # run_meta + Pipeline-Tokens/Kosten (_pre, _wall_s, _trace_path) sind oben
        # bereits berechnet und der pipeline_run bereits persistiert (#198 P1 — der
        # Insert läuft jetzt unbedingt, auch ohne Inline-Eval). Der Stage-8-Eval-
        # Overhead bleibt bewusst außerhalb von run_meta/DB, nur am Run-Ende geprintet.
        eval_results, _evaluated_count, _reused_count = run_stage8_eval(
            note_files, source_path, run_meta, fresh_run=bool(getattr(args, "fresh_run", False))
        )

        if eval_results:
            hall_rates = [
                r["hallucination_rate"]
                for r in eval_results
                if "hallucination_rate" in r and r["hallucination_rate"] >= 0
            ]
            cov_rates = [
                r.get("coverage_factual", r.get("coverage_rate", 0.0))
                for r in eval_results
                if r.get("coverage_factual", r.get("coverage_rate", -1.0)) >= 0
            ]
            if hall_rates:
                avg_hall = sum(hall_rates) / len(hall_rates)
                avg_cov = sum(cov_rates) / len(cov_rates) if cov_rates else 0.0
                print(f"      Ø Halluzinationsrate: {avg_hall:.1%}  |  Ø Coverage (faktisch): {avg_cov:.1%}")
                reused_note = f"  (+{_reused_count} wiederverwendet, Hash-Guard)" if _reused_count else ""
                print(f"      {_evaluated_count} Notes → .cache/quality_history.jsonl{reused_note}")

        # Re-Aggregation NACH der Eval-Schleife: die eval_quality-Calls stehen jetzt
        # im Trace. Macht den sonst unsichtbaren Stage-8-Tail (~33 % Out-Tok, Eval-
        # Wandzeit) im Run-Ende-Print sichtbar (Reporting-Quirk-Fix).
        _final_wall = round(_time.time() - _run_start, 1)
        _grand = _eas.run_totals(_trace_path)
        _eval_out = _grand["output"] - _pre["output"]
        _eval_wall = round(_final_wall - _wall_s, 1)
        _eval_pct = (_eval_out / _grand["output"]) if _grand["output"] else 0.0
        print("\n   === Run-Gesamt (inkl. Stage-8-Eval) ===")
        print(f"   -> Zeit:   {_final_wall}s  (davon Stage-8: +{_eval_wall}s)")
        print(
            f"   -> Tokens: {_grand['total']:,} (In:{_grand['input']:,} Out:{_grand['output']:,} Cache-R:{_grand['cache_read']:,} Cache-C:{_grand['cache_create']:,})"
        )
        print(f"   -> Stage-8-Eval: +{_eval_out:,} Out-Tok ({_eval_pct:.0%} des Out-Totals)")
    except Exception as e:
        print(f"      [eval-warn] Qualitäts-Eval übersprungen: {e}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
