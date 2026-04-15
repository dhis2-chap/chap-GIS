"""On-disk download cache."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_CACHE_DIR = Path(os.environ.get("CHAP_GIS_CACHE", "data/cache")).resolve()


def cache_dir() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR


def download_file(
    url: str,
    local: Path | None = None,
    *,
    label: str = "",
    chunk_size: int = 1024 * 1024,
    timeout: int = 300,
) -> Path:
    """Download `url` to `local`, skipping if already cached.

    If `local` is None, the file is placed in the cache directory under the
    URL's basename.
    """
    if local is None:
        local = cache_dir() / url.rsplit("/", 1)[-1]
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists():
        return local

    print(f"  Downloading {label or local.name}...")
    resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dl = 0
    with open(local, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            dl += len(chunk)
            if total:
                print(
                    f"\r    {dl // (1024 * 1024)}/{total // (1024 * 1024)} MB",
                    end="",
                    flush=True,
                )
    if total:
        print()
    return local
