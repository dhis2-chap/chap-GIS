"""On-disk download cache."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

_default_log = logging.getLogger(__name__)


def cache_dir() -> Path:
    d = Path(os.environ.get("CHAP_GIS_CACHE", "data/cache")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_key(source: str, country: str | None, *parts: object) -> str:
    """Build the canonical cache filename stem for a (source, country, parts) tuple.

    Country is lowercased and stripped; missing/blank country drops the prefix.
    Extension is the caller's responsibility.
    """
    head = f"{country.strip().lower()}_{source}" if country and country.strip() else source
    if not parts:
        return head
    return "_".join((head, *(str(p) for p in parts)))


def cached_download(
    items: Iterable[T],
    fetch_fn: Callable[[T, Path], None],
    *,
    dirname: Path,
    name_fn: Callable[[T], str],
    overwrite: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
    log: logging.Logger | None = None,
) -> list[Path]:
    """Run ``fetch_fn`` for each item that's not already cached at ``dirname/name_fn(item)``.

    ``fetch_fn(item, save_path)`` is responsible for writing ``save_path``.
    Returns paths in the input order, including cache hits. Exceptions from
    ``fetch_fn`` propagate (in the parallel path via ``future.result()``).
    """
    log = log or _default_log
    dirname = Path(dirname)
    dirname.mkdir(parents=True, exist_ok=True)
    items = list(items)
    paths = [dirname / name_fn(item) for item in items]

    to_fetch = [(item, path) for item, path in zip(items, paths) if overwrite or not path.exists()]
    cached = [path for path in paths if not overwrite and path.exists()]
    for path in cached:
        log.info("File already downloaded: %s", path)

    if parallel and to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(fetch_fn, item, path) for item, path in to_fetch]
            for future in futures:
                future.result()
    else:
        for item, path in to_fetch:
            fetch_fn(item, path)

    return paths
