"""Unit tests for chap_gis.io.cache."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from chap_gis.io.cache import cache_dir, cache_key, cached_download


def test_cache_dir_respects_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAP_GIS_CACHE", str(tmp_path))
    assert cache_dir() == tmp_path.resolve()


def test_cache_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "cache"
    monkeypatch.setenv("CHAP_GIS_CACHE", str(target))
    cache_dir()
    assert target.is_dir()


def test_cache_dir_rereads_env_var(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("CHAP_GIS_CACHE", str(first))
    assert cache_dir() == first.resolve()
    monkeypatch.setenv("CHAP_GIS_CACHE", str(second))
    assert cache_dir() == second.resolve()


def test_cache_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAP_GIS_CACHE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert cache_dir() == (tmp_path / "data" / "cache").resolve()


def test_cache_key_basic():
    assert cache_key("chelsa_temperature_monthly", "RWA") == "rwa_chelsa_temperature_monthly"


def test_cache_key_lowercase_normalization():
    assert cache_key("foo", "rwa") == cache_key("foo", "Rwa") == cache_key("foo", "RWA")


def test_cache_key_strips_whitespace():
    assert cache_key("foo", " rwa ") == "rwa_foo"


def test_cache_key_no_country():
    assert cache_key("nasa_crops", None) == "nasa_crops"
    assert cache_key("nasa_crops", "") == "nasa_crops"
    assert cache_key("nasa_crops", "   ") == "nasa_crops"


def test_cache_key_with_parts():
    assert cache_key("chelsa_temperature_monthly", "RWA", 2021, 1) == "rwa_chelsa_temperature_monthly_2021_1"


def test_cache_key_parts_without_country():
    assert cache_key("worldpop_population", None, 2021) == "worldpop_population_2021"


def _writer(item, path):
    path.write_text(str(item))


def test_cached_download_sequential_happy_path(tmp_path):
    calls = []

    def fetch(item, path):
        calls.append(item)
        path.write_text(str(item))

    paths = cached_download(
        [1, 2, 3], fetch,
        dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
    )
    assert calls == [1, 2, 3]
    assert paths == [tmp_path / "item_1.txt", tmp_path / "item_2.txt", tmp_path / "item_3.txt"]
    assert all(p.exists() for p in paths)


def test_cached_download_cache_hit_short_circuits(tmp_path, caplog):
    (tmp_path / "item_2.txt").write_text("preexisting")
    calls = []

    def fetch(item, path):
        calls.append(item)
        path.write_text(str(item))

    with caplog.at_level(logging.INFO, logger="chap_gis.io.cache"):
        paths = cached_download(
            [1, 2, 3], fetch,
            dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
        )
    assert calls == [1, 3]
    assert (tmp_path / "item_2.txt").read_text() == "preexisting"
    assert any("item_2.txt" in r.message for r in caplog.records)
    assert len(paths) == 3


def test_cached_download_overwrite_bypasses_cache(tmp_path):
    (tmp_path / "item_1.txt").write_text("old")
    cached_download(
        [1], _writer,
        dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
        overwrite=True,
    )
    assert (tmp_path / "item_1.txt").read_text() == "1"


def test_cached_download_parallel_runs_concurrently(tmp_path):
    def slow_fetch(item, path):
        time.sleep(0.05)
        path.write_text(str(item))

    start = time.monotonic()
    cached_download(
        list(range(4)), slow_fetch,
        dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
        parallel=True, max_workers=4,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 0.18, f"parallel run took {elapsed:.3f}s, expected concurrency"
    assert all((tmp_path / f"item_{i}.txt").exists() for i in range(4))


def test_cached_download_propagates_exceptions_sequential(tmp_path):
    def fetch(item, path):
        if item == 2:
            raise RuntimeError("boom")
        path.write_text(str(item))

    with pytest.raises(RuntimeError, match="boom"):
        cached_download(
            [1, 2, 3], fetch,
            dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
        )


def test_cached_download_propagates_exceptions_parallel(tmp_path):
    def fetch(item, path):
        if item == 2:
            raise RuntimeError("boom")
        path.write_text(str(item))

    with pytest.raises(RuntimeError, match="boom"):
        cached_download(
            [1, 2, 3], fetch,
            dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
            parallel=True,
        )


def test_cached_download_returns_input_order_with_cache_hits(tmp_path):
    (tmp_path / "item_1.txt").write_text("hit")
    (tmp_path / "item_3.txt").write_text("hit")
    paths = cached_download(
        [1, 2, 3], _writer,
        dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
    )
    assert [p.name for p in paths] == ["item_1.txt", "item_2.txt", "item_3.txt"]


def test_cached_download_creates_dirname(tmp_path):
    nested = tmp_path / "a" / "b"
    cached_download(
        [1], _writer,
        dirname=nested, name_fn=lambda i: f"item_{i}.txt",
    )
    assert (nested / "item_1.txt").exists()


def test_cached_download_uses_provided_logger(tmp_path):
    custom = logging.getLogger("test.custom.cache")
    (tmp_path / "item_1.txt").write_text("hit")
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    custom.addHandler(Handler())
    custom.setLevel(logging.INFO)
    cached_download(
        [1], _writer,
        dirname=tmp_path, name_fn=lambda i: f"item_{i}.txt",
        log=custom,
    )
    assert any("item_1.txt" in r.getMessage() for r in records)
