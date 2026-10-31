"""Portable dataset-cache path resolution.

Benchmark/dataset storage location, in priority order:

1. ``$SURGE_DATA_DIR`` — explicit override, always honoured first.
2. ``<repo>/data/datasets`` — when running from a SURGE git checkout (a
   directory containing ``pyproject.toml`` two levels above this file).
3. ``$XDG_CACHE_HOME/surge/datasets`` (or ``~/.cache/surge/datasets`` when
   ``XDG_CACHE_HOME`` is unset) — the installed-wheel fallback, so an
   installed SURGE never writes inside ``site-packages``.

Dependency-free (``os``/``pathlib`` only) so any module can import it without
risking an import cycle.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path | None:
    """Return the SURGE checkout root, or ``None`` outside of one."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def datasets_root() -> Path:
    """Return the root directory for all SURGE dataset storage (no mkdir).

    See the module docstring for the ``SURGE_DATA_DIR`` → checkout → XDG
    cache resolution order.
    """
    override = os.environ.get("SURGE_DATA_DIR")
    if override:
        return Path(override)
    root = repo_root()
    if root is not None:
        return root / "data" / "datasets"
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "surge" / "datasets"


def bench_data_root() -> Path:
    """Return the unified benchmark dataset cache, creating it if needed."""
    root = datasets_root() / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    return root
