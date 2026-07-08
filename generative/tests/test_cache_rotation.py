# -*- coding: utf-8 -*-
"""Anzahl-basierte Cache-Rotation (#151, Punkt 6).

.cache/llm und .cache/runs wachsen monoton; rotate_run_caches stutzt sie am
Run-Ende auf ihre Obergrenzen, aelteste zuerst. quality_history.jsonl und
.bak-Dateien bleiben unberuehrt.
"""

from __future__ import annotations

import os
import time

from generative import cache_rotation


def _touch(path, mtime):
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


class TestPruneCacheDir:
    def test_deletes_oldest_beyond_keep(self, tmp_path):
        base = time.time()
        for i in range(5):
            _touch(tmp_path / f"f{i}.json", base + i)  # f0 aeltest, f4 neuest

        deleted = cache_rotation.prune_cache_dir(tmp_path, keep=2)

        assert deleted == 3
        survivors = sorted(p.name for p in tmp_path.iterdir())
        assert survivors == ["f3.json", "f4.json"]  # die zwei neuesten

    def test_noop_when_under_keep(self, tmp_path):
        for i in range(3):
            _touch(tmp_path / f"f{i}.json", time.time() + i)

        assert cache_rotation.prune_cache_dir(tmp_path, keep=10) == 0
        assert len(list(tmp_path.iterdir())) == 3

    def test_bak_files_never_deleted(self, tmp_path):
        base = time.time()
        _touch(tmp_path / "stale.bak", base)  # aeltest, aber .bak
        for i in range(4):
            _touch(tmp_path / f"f{i}.json", base + 1 + i)

        cache_rotation.prune_cache_dir(tmp_path, keep=1)

        assert (tmp_path / "stale.bak").exists()  # .bak bleibt trotz Alter

    def test_missing_dir_returns_zero(self, tmp_path):
        assert cache_rotation.prune_cache_dir(tmp_path / "nope", keep=5) == 0


class TestRotateRunCaches:
    def test_rotates_llm_and_runs_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_rotation, "CACHE_LLM_MAX_FILES", 2)
        monkeypatch.setattr(cache_rotation, "CACHE_RUNS_MAX_FILES", 1)

        llm = tmp_path / "llm"
        runs = tmp_path / "runs"
        llm.mkdir()
        runs.mkdir()
        base = time.time()
        for i in range(4):
            _touch(llm / f"c{i}.json", base + i)
        for i in range(3):
            _touch(runs / f"r{i}.jsonl", base + i)

        # quality_history.jsonl im Cache-Root darf nie angefasst werden.
        _touch(tmp_path / "quality_history.jsonl", base)

        n_llm, n_runs = cache_rotation.rotate_run_caches(cache_dir=tmp_path)

        assert n_llm == 2  # 4 → keep 2
        assert n_runs == 2  # 3 → keep 1
        assert (tmp_path / "quality_history.jsonl").exists()
