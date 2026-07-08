# -*- coding: utf-8 -*-
"""embed_body-LRU (#151, Punkt 4).

ER, flag_redundant_siblings und das CrossRef-Sibling-Ranking embedden im selben Lauf
teils dieselben Bodies mehrfach. Ein prozess-lokaler LRU (Key = Body-Text) senkt die
model.encode-Aufrufe bei wiederholtem gleichen Body — deterministisch, also identische
Vektoren, nur weniger Rechenlast.
"""

from __future__ import annotations

import numpy as np

from generative.pipeline import embeddings


class _FakeModel:
    def __init__(self, counter: dict):
        self._counter = counter

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        self._counter["encode"] = self._counter.get("encode", 0) + 1
        out = []
        for t in texts:
            h = abs(hash(t)) % 997
            v = np.array([h % 7 + 1.0, h % 5 + 1.0, h % 3 + 1.0, 1.0])
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def get_sentence_embedding_dimension(self):
        return 4


def _clear():
    getattr(embeddings.embed_body, "cache_clear", lambda: None)()


class TestEmbedBodyLru:
    def test_repeated_body_encoded_once(self, monkeypatch):
        _clear()
        counter: dict = {}
        monkeypatch.setattr(embeddings, "_model", lambda: _FakeModel(counter))

        body = "Ein Body mit mehreren Saetzen. Er wird mehrfach embeddet. Und nochmal."
        embeddings.embed_body(body)
        embeddings.embed_body(body)
        embeddings.embed_body(body)

        assert counter["encode"] == 1  # RED (ohne LRU): 3

    def test_distinct_bodies_each_encoded(self, monkeypatch):
        _clear()
        counter: dict = {}
        monkeypatch.setattr(embeddings, "_model", lambda: _FakeModel(counter))

        embeddings.embed_body("Erster Body mit genug Textlaenge fuer Saetze.")
        embeddings.embed_body("Zweiter, klar verschiedener Body mit Textlaenge.")

        assert counter["encode"] == 2

    def test_cached_result_value_identical(self, monkeypatch):
        _clear()
        counter: dict = {}
        monkeypatch.setattr(embeddings, "_model", lambda: _FakeModel(counter))

        body = "Deterministischer Body. Zweiter Satz hier. Dritter Satz."
        first = embeddings.embed_body(body)

        # Ohne Cache neu berechnet (nach cache_clear) — muss wertgleich sein.
        _clear()
        recomputed = embeddings.embed_body(body)

        assert np.array_equal(first, recomputed)
