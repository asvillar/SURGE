"""QLKNN10D loader: filtering, subsampling, caching, and error paths.

Uses a synthetic mini-HDF with the real file's group layout so the test
runs without the 13.3 GB download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

INPUTS = ["Zeff", "Ati", "Ate", "An", "q", "smag", "x", "Ti_Te", "Nustar"]


def _write_mini_h5(path: Path, n: int = 4000, seed: int = 0) -> dict:
    """Mimic gen5_9D_nions0_flat_filter10.h5 structure at tiny scale."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.1, 5.0, (n, len(INPUTS))).astype("float32")
    itg = (rng.random(n) < 0.3).astype("uint8")          # ~30% unstable
    efi = np.where(itg == 1, rng.uniform(0.1, 50, n), 0.0).astype("float32")
    efi[:20] = np.nan                                     # NaNs must be dropped
    with h5py.File(path, "w") as f:
        g = f.create_group("input")
        g.create_dataset("block0_values", data=X)
        g.create_dataset("block0_items",
                         data=np.array([c.encode() for c in INPUTS]))
        f.create_dataset("output/ITG/block0_values", data=itg[:, None])
        f.create_dataset("output/efiITG_GB/block0_values", data=efi[:, None])
    valid = (itg == 1) & np.isfinite(efi) & (efi > 0)
    return {"n_valid": int(valid.sum())}


def test_qlknn10d_loader_filters_and_caches(tmp_path: Path):
    from surge.benchmarks.leaderboard import _load_qlknn10d

    h5 = tmp_path / "mini.h5"
    meta = _write_mini_h5(h5)
    cache = tmp_path / "cache.npz"

    X, y = _load_qlknn10d(n_rows=500, h5_path=h5, cache_path=cache)

    assert X.shape[1] == 9, "9-D QuaLiKiz input grid"
    assert len(X) == len(y) <= 500
    assert len(y) <= meta["n_valid"], "cannot exceed ITG-unstable rows"
    assert np.isfinite(y).all() and (y > 0).all(), "NaN/zero fluxes dropped"
    assert np.isfinite(X).all()
    assert cache.exists(), "loader must cache its subsample"

    # second call hits the cache and is byte-identical
    X2, y2 = _load_qlknn10d(n_rows=500, h5_path=h5, cache_path=cache)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_qlknn10d_loader_missing_file(tmp_path: Path):
    from surge.benchmarks.leaderboard import _load_qlknn10d

    with pytest.raises(FileNotFoundError, match="zenodo"):
        _load_qlknn10d(h5_path=tmp_path / "absent.h5",
                       cache_path=tmp_path / "c.npz")


def test_qlknn10d_registered():
    from surge.benchmarks.registry import benchmark_info
    from surge.report.leaderboard import load_metadata

    info = benchmark_info("plasma.qlknn10d")
    assert info["task_type"] == "regression"
    meta = load_metadata()["plasma.qlknn10d"]
    assert "0.95" in meta["threshold"]
    assert [i["name"] for i in meta["inputs"]] == INPUTS
