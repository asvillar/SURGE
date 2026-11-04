"""Export a benchmark's dataset to CSV / Parquet through the registry loaders.

``surge data export <benchmark> out.csv`` lets a pip-installed user get the
exact arrays a benchmark trains on (with named columns from the benchmark's
metadata card) and feed them to ``surge run`` / ``surge init``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "test", "all")


def _column_names(prefix: str, count: int, card_entries, single: str | None = None):
    names = [e.get("name") if isinstance(e, dict) else e for e in (card_entries or [])]
    if len(names) == count and all(names):
        return [str(n) for n in names]
    if count == 1 and single:
        return [single]
    return [f"{prefix}{i}" for i in range(count)]


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"cannot export {arr.ndim}-D arrays (shape {arr.shape}) to a table; "
            "only tabular benchmarks are exportable")
    return arr


def export_benchmark(key: str, path, split: str = "all") -> pd.DataFrame:
    """Write benchmark ``key`` (full key or alias) to ``path`` (.csv or .parquet).

    Fixed-split benchmarks gain a ``split`` column (``train``/``test``) and
    honour ``split``; for all others only ``split="all"`` is valid. Returns
    the DataFrame that was written.
    """
    from . import leaderboard
    from .registry import resolve_benchmark_key

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".parquet"):
        raise ValueError("output path must end in .csv or .parquet")
    key = resolve_benchmark_key(key)

    fixed_name = leaderboard._FIXED_SPLIT_LOADERS.get(key)
    if fixed_name is not None:
        X_tr, y_tr, X_te, y_te = getattr(leaderboard, fixed_name)()
        parts = []
        if split in ("train", "all"):
            parts.append(("train", X_tr, y_tr))
        if split in ("test", "all"):
            parts.append(("test", X_te, y_te))
        X = np.vstack([_as_2d(p[1]) for p in parts])
        y = np.vstack([_as_2d(p[2]) for p in parts])
        labels = np.concatenate([[p[0]] * len(_as_2d(p[1])) for p in parts])
    else:
        if split != "all":
            raise ValueError(
                f"{key} has no fixed train/test split; only --split all is available")
        X, y = leaderboard._load_dataset(key)
        X, y = _as_2d(X), _as_2d(y)
        labels = None

    try:
        from surge.report.leaderboard import load_metadata
        card = load_metadata().get(key, {})
    except Exception:  # noqa: BLE001 - metadata only names columns
        card = {}
    xcols = _column_names("x", X.shape[1], card.get("inputs"))
    ycols = _column_names("y", y.shape[1], card.get("outputs"), single="y")
    if set(xcols) & set(ycols):
        ycols = [f"y_{c}" if c in xcols else c for c in ycols]

    df = pd.DataFrame(X, columns=xcols)
    for i, c in enumerate(ycols):
        df[c] = y[:, i]
    if labels is not None:
        df["split"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return df
