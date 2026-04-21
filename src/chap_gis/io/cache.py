"""On-disk download cache."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_CACHE_DIR = Path(os.environ.get("CHAP_GIS_CACHE", "data/cache")).resolve()


def cache_dir() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR

