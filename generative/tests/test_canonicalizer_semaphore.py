# -*- coding: utf-8 -*-
"""Canonicalizer-Merges unter der max_concurrent_calls-Semaphore (#151, Punkt 5a).

Stage 4 der Entity-Resolution gatherte die Opus-Merge-Calls bisher ungebremst am
Concurrency-Limit vorbei. Jetzt laufen sie unter derselben Semaphore wie der
Extraktor — die beobachtete Parallelitaet darf max_concurrent_calls nicht ueberschreiten.
"""

from __future__ import annotations

import asyncio

import numpy as np

from generative import orchestrator
from generative.schemas.atomic_note import AtomicNoteDraft


def _draft(title: str, body: str) -> AtomicNoteDraft:
    return AtomicNoteDraft(
        title=title,
        body=body,
        source_anchors=[],
        related=[],
        tags=[],
        synthesis_confidence="high",
    )


class _MergeTracker:
    def __init__(self):
        self.current = 0
        self.max_seen = 0
        self.total = 0

    async def merge_cluster(self, members):
        self.total += 1
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)
        await asyncio.sleep(0.02)
        self.current -= 1
        return members[0]


def _fake_embed_body(body: str):
    h = abs(hash(body)) % 997
    v = np.array([h % 7 + 1.0, h % 5 + 1.0, h % 3 + 1.0, 1.0])
    return v / np.linalg.norm(v)  # identischer Body → identischer Vektor → cosine 1.0


def _fake_embed_title(title: str):
    return np.zeros(4)  # Semantic-Title-Fallback neutralisieren


def test_merges_respect_semaphore_limit(monkeypatch):
    # Fuenf 2er-Cluster: innerhalb eines Paars identischer Titel+Body (blockt+clustert),
    # ueber Paare hinweg kein Token-Subset (kein Cluster).
    drafts = []
    for k in range(5):
        title = f"themafoo{k} bausteinbar"
        body = f"Cluster {k}: identischer Body fuer beide Mitglieder dieses Paars."
        drafts.append(_draft(title, body))
        drafts.append(_draft(title, body))

    tracker = _MergeTracker()
    monkeypatch.setattr(orchestrator.embeddings, "embed_body", _fake_embed_body)
    monkeypatch.setattr(orchestrator.embeddings, "embed_title", _fake_embed_title)
    monkeypatch.setattr(orchestrator.canonicalizer, "merge_cluster", tracker.merge_cluster)

    result = asyncio.run(orchestrator.entity_resolution(drafts, max_concurrent_calls=2))

    assert tracker.total == 5  # alle fuenf Cluster gemergt
    assert tracker.max_seen <= 2  # Semaphore hat die Parallelitaet gedeckelt
    assert tracker.max_seen == 2  # ... und sie wurde auch ausgeschoepft
    assert len(result) == 5  # 10 Drafts → 5 gemergte Repraesentanten
