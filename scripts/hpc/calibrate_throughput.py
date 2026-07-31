#!/usr/bin/env python3
"""Measure real training throughput before spending the allocation.

Run this FIRST on the target machine (~10-15 min, one GPU). It times
forward+backward+step on the true protocol shapes (16 -> 4 channels at
128x384) across model sizes, batch sizes and AMP settings, then projects
epoch time and GPU-hours for every point of ``scripts/hpc/grid.sh``.

The sweep's cost estimate in ``scripts/hpc/README.md`` is extrapolated from
an Apple M2 Pro with an assumed 20-50x speedup. That is a guess across two
vendors' hardware. This script replaces it with a number.

Usage:
    python scripts/hpc/calibrate_throughput.py --out calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

N_HIST, N_FIELDS = 4, 4
H, W = 128, 384

# (label, base_channels, depth) — the two capacity rungs of the sweep
CAPACITIES = [("small", 40, 3), ("published", 64, 4)]

# (tag, n_train, capacity_label, epochs) — must match grid.sh
GRID = [
    ("A_control", 1400, "small", 60),
    ("B_data", 6984, "small", 60),
    ("C_capacity", 1400, "published", 60),
    ("D_epochs", 1400, "small", 500),
    ("E_data_capacity", 6984, "published", 60),
    ("F_data_epochs", 6984, "small", 500),
    ("G_capacity_epochs", 1400, "published", 500),
    ("H_full", 6984, "published", 500),
]


def time_step(net, device, batch, amp_dtype, channels_last, n_iter=5):
    """Median seconds per optimizer step, warmup excluded."""
    import torch

    x = torch.randn(batch, N_HIST * N_FIELDS, H, W, device=device)
    y = torch.randn(batch, N_FIELDS, H, W, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scaler = (torch.amp.GradScaler(device.type)
              if amp_dtype is torch.float16 else None)

    def one():
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                loss = ((net(x) - y) ** 2).mean()
        else:
            loss = ((net(x) - y) ** 2).mean()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    for _ in range(3):
        one()
    _sync(device)
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        one()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from surge.model.backends.convnext_unet import _CNextUNet
    from surge.utils import resolve_device

    device = (torch.device(args.device) if args.device
              else resolve_device("auto"))
    modes = [("off", None, False)]
    if device.type == "cuda":
        modes = [("bf16+cl", torch.bfloat16, True),
                 ("bf16", torch.bfloat16, False),
                 ("off", None, True)]

    print(f"[calibrate] device {device}", flush=True)
    if device.type == "cuda":
        print(f"[calibrate] {torch.cuda.get_device_name(0)}", flush=True)

    rows, best_per_capacity = [], {}
    for label, base, depth in CAPACITIES:
        net = _CNextUNet(N_HIST * N_FIELDS, N_FIELDS, base, depth, 2)
        net = net.to(device)
        n_par = sum(p.numel() for p in net.parameters())
        for mode, dtype, cl in modes:
            net_m = (net.to(memory_format=torch.channels_last) if cl else net)
            for batch in args.batches:
                try:
                    dt = time_step(net_m, device, batch, dtype, cl)
                except RuntimeError as exc:  # OOM at large batch is a result
                    print(f"  {label} {mode} b={batch}: {exc}"[:120],
                          flush=True)
                    continue
                per_sample = dt / batch
                rows.append({"capacity": label, "params": n_par,
                             "mode": mode, "batch": batch,
                             "s_per_step": round(dt, 4),
                             "ms_per_sample": round(per_sample * 1e3, 2)})
                print(f"  {label:9s} {mode:8s} b={batch:<3d} "
                      f"{dt*1e3:7.1f} ms/step  "
                      f"{per_sample*1e3:6.2f} ms/sample", flush=True)
                prev = best_per_capacity.get(label)
                if prev is None or per_sample < prev:
                    best_per_capacity[label] = per_sample

    # ---- project the grid -------------------------------------------------
    projection, total_gpu_h = [], 0.0
    for tag, n_train, cap, epochs in GRID:
        per_sample = best_per_capacity.get(cap)
        if per_sample is None:
            continue
        gpu_h = per_sample * n_train * epochs / 3600
        total_gpu_h += gpu_h
        projection.append({"tag": tag, "n_train": n_train, "capacity": cap,
                           "epochs": epochs,
                           "epoch_s": round(per_sample * n_train, 1),
                           "gpu_hours": round(gpu_h, 2)})

    result = {"device": str(device),
              "gpu_name": (torch.cuda.get_device_name(0)
                           if device.type == "cuda" else None),
              "measurements": rows,
              "best_ms_per_sample": {k: round(v * 1e3, 2)
                                     for k, v in best_per_capacity.items()},
              "grid_projection": projection,
              "total_gpu_hours": round(total_gpu_h, 1)}
    Path(args.out).write_text(json.dumps(result, indent=2))

    print("\n[projection] per-config cost at best measured throughput")
    for p in projection:
        print(f"  {p['tag']:20s} epoch {p['epoch_s']:7.1f}s  "
              f"{p['gpu_hours']:6.2f} GPU-h")
    longest = max((p["gpu_hours"] for p in projection), default=0)
    print(f"\n  TOTAL {total_gpu_h:.1f} GPU-hours; "
          f"longest single config {longest:.1f} h")
    if longest > 11:
        print("  NOTE: longest config exceeds a 12 h queue slot — "
              "--time-limit-s + --resume chaining is required.")
    print(f"\n[calibrate] wrote {args.out}")


if __name__ == "__main__":
    main()
