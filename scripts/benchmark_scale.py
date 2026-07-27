#!/usr/bin/env python3
"""Measure SURGE scaling levers and cache the numbers for the scale figure.

Two measurements, written to examples/viz_gallery_output/scale_bench.json:

1. device — fit wall-time for FNO-2D and U-Net on a fixed synthetic
   64x64 workload, cpu vs mps (skipped when MPS is unavailable).
2. parallel — wall time of the same benchmark job set run through
   ``surge bench --parallel {1, 2, 4}`` (subprocess fan-out).

Run occasionally (numbers are hardware-specific); the gallery's
``scale`` figure reads the JSON instead of re-measuring.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "examples" / "viz_gallery_output" / "scale_bench.json"

PARALLEL_JOBS = ("sklearn.random_forest,sklearn.mlp,pytorch.mlp,"
                 "pytorch.residual_mlp,lgbm.regressor,catboost.regressor")


def measure_device() -> list[dict]:
    import torch

    from surge.model.backends.fno2d import FNO2dModel
    from surge.model.backends.unet import UNetModel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 64 * 64)).astype("float32")
    y = (X * 0.5 + 0.1 * rng.standard_normal(X.shape)).astype("float32")

    devices = ["cpu"]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    rows = []
    for label, cls, kw in (("FNO-2D", FNO2dModel,
                            {"n_epochs": 3, "n_modes": 24}),
                           ("U-Net", UNetModel,
                            {"n_epochs": 3, "patience": 0})):
        for dev in devices:
            model = cls(device=dev, **kw)
            t0 = time.perf_counter()
            model.fit(X, y)
            dt = time.perf_counter() - t0
            rows.append({"model": label, "device": dev,
                         "fit_seconds": round(dt, 2)})
            print(f"[device] {label} {dev}: {dt:.1f}s")
    return rows


def measure_parallel(workers=(1, 2, 4)) -> list[dict]:
    import tempfile

    rows = []
    for n in workers:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, "-m", "surge.benchmarks.run",
                   "--benchmark", "plasma.qlknn_transport",
                   "--compare-models", PARALLEL_JOBS,
                   "--seeds", "3", "--save-dir", tmp]
            if n > 1:
                cmd += ["--parallel", str(n)]
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True, text=True)
            dt = time.perf_counter() - t0
        rows.append({"workers": n, "wall_seconds": round(dt, 2),
                     "jobs": PARALLEL_JOBS.count(",") + 1})
        print(f"[parallel] {n} worker(s): {dt:.1f}s")
    return rows


def main() -> None:
    payload = {
        "device": measure_device(),
        "parallel": measure_parallel(),
        "workload": {
            "device": "200 x 64x64 synthetic fields, 3 epochs",
            "parallel": ("plasma.qlknn_transport x 6 models x 3 seeds "
                         "(surge bench --parallel N)"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
