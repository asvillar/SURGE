#!/usr/bin/env python3
"""TRL-2D under TheWell's OWN benchmark protocol — external comparison.

Protocol (Ohana et al. 2024, §benchmarks): predict the next stored
snapshot of ALL four fields (density, pressure, vx, vy) from a 4-frame
history, at full 128x384 resolution, scored with VRMSE

    VRMSE(u, v) = sqrt( mean((u-v)^2) / mean((v - v̄)^2) )

(v̄ = per-sample-per-field spatial mean of the truth; averaged over
fields then samples). Published validation baselines for this dataset:
FNO 0.4906, TFNO 0.4938, U-Net 0.2394, ConvNeXt-U-Net 0.1247 — models
at 15-20M parameters trained 12 h on an H100. This study trains
laptop-scale models (minutes-hours on Apple GPU) under the same
protocol so the numbers are directly comparable, with the compute gap
stated honestly.

Also saves the trained ConvNeXt model + a held-out trajectory for the
rollout/physics-statistics study.

Usage:
    SURGE_DEVICE=auto python examples/thewell_wellprotocol_study.py \
        [--n-train 1400] [--n-test 200] [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "examples" / "viz_gallery_output"
FIELDS = ["density", "pressure", "velocity_x", "velocity_y"]
N_HIST = 4

PUBLISHED = {"FNO": 0.4906, "TFNO": 0.4938, "U-Net": 0.2394,
             "CNextU-Net": 0.1247}


def vrmse(pred, truth):
    """TheWell metric: variance-scaled RMSE, per sample+field, averaged."""
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    t = truth.reshape(truth.shape[0], truth.shape[1], -1)
    num = ((p - t) ** 2).mean(axis=-1)
    den = ((t - t.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1) + 1e-12
    return float(np.sqrt(num / den).mean())


def load_split(split: str, n: int, seed: int = 0):
    """X: (n, 16, 128, 384) 4 frames x 4 fields; y: (n, 4, 128, 384)."""
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = (_well_base_path(None) / "turbulent_radiative_layer_2D"
            / "data" / split)
    ds = WellDataset(path=str(root), n_steps_input=N_HIST, n_steps_output=1,
                     use_normalization=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    X, Y = [], []
    for i in idx:
        item = ds[int(i)]
        xin = np.asarray(item["input_fields"])       # (4, 128, 384, 4)
        yout = np.asarray(item["output_fields"])[0]  # (128, 384, 4)
        X.append(np.moveaxis(xin, -1, 1).reshape(-1, *xin.shape[1:3]))
        Y.append(np.moveaxis(yout, -1, 0))
    return (np.stack(X).astype("float32"), np.stack(Y).astype("float32"))


class _StreamingWellPairs:
    """Lazy torch Dataset over WellDataset: 4-frame input -> 4-field target.

    HDF5 reads happen per item, so the pair count is bounded by disk and
    epoch time — not RAM (the in-memory path caps at ~1.5k pairs on a
    16 GB machine).
    """

    def __init__(self, split: str, n: int, mu, sd, seed: int = 0):
        import torch  # noqa: F401  (torch Dataset protocol, duck-typed)
        from the_well.data import WellDataset

        from surge.benchmarks.loaders.thewell import _well_base_path

        root = (_well_base_path(None) / "turbulent_radiative_layer_2D"
                / "data" / split)
        self.ds = WellDataset(path=str(root), n_steps_input=N_HIST,
                              n_steps_output=1, use_normalization=False)
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(len(self.ds), size=min(n, len(self.ds)),
                              replace=False)
        self.x_mu = np.tile(mu, (1, N_HIST, 1, 1))[0]
        self.x_sd = np.tile(sd, (1, N_HIST, 1, 1))[0]
        self.y_mu, self.y_sd = mu[0], sd[0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch

        item = self.ds[int(self.idx[i])]
        xin = np.asarray(item["input_fields"])       # (4, H, W, 4)
        yout = np.asarray(item["output_fields"])[0]  # (H, W, 4)
        x = np.moveaxis(xin, -1, 1).reshape(-1, *xin.shape[1:3])
        y = np.moveaxis(yout, -1, 0)
        x = (x.astype("float32") - self.x_mu) / self.x_sd
        y = (y.astype("float32") - self.y_mu) / self.y_sd
        return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stream", action="store_true",
                    help="lazy HDF5 streaming: n-train bounded by disk, "
                         "not RAM; trains ConvNeXt U-Net only")
    args = ap.parse_args()

    from surge.model import MODEL_REGISTRY

    if args.stream:
        return _main_stream(args)

    Xtr, ytr = load_split("train", args.n_train)
    Xte, yte = load_split("valid", args.n_test)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}")

    # per-field z-score normalization from training stats (Well practice);
    # frame channels of the same field share that field's stats
    f_mu = ytr.mean(axis=(0, 2, 3), keepdims=True)
    f_sd = ytr.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    x_mu = np.tile(f_mu, (1, N_HIST, 1, 1))
    x_sd = np.tile(f_sd, (1, N_HIST, 1, 1))
    Xtr_n, Xte_n = (Xtr - x_mu) / x_sd, (Xte - x_mu) / x_sd
    ytr_n = (ytr - f_mu) / f_sd

    # persistence = last input frame of each field; channel layout is
    # FRAME-major after the reshape: [t0f0, t0f1, t0f2, t0f3, t1f0, ...]
    last = np.stack([Xte[:, (N_HIST - 1) * len(FIELDS) + f]
                     for f in range(len(FIELDS))], axis=1)
    results = {"Persistence": vrmse(last, yte)}
    print(f"[done] Persistence      VRMSE {results['Persistence']:.4f}")

    preds_store: dict[str, np.ndarray] = {}
    for key, label, params in (
        ("pytorch.unet", "U-Net (ours)",
         {"base_channels": 48, "n_epochs": args.epochs, "batch_size": 8,
          "patience": 0}),
        ("pytorch.convnext_unet", "CNextU-Net (ours)",
         {"base_channels": 48, "depth": 3, "blocks_per_stage": 2,
          "n_epochs": args.epochs, "batch_size": 8, "patience": 0}),
    ):
        model = MODEL_REGISTRY.create(key, **params)
        t0 = time.perf_counter()
        model.fit(Xtr_n, ytr_n)
        pred = np.asarray(model.predict(Xte_n)) * f_sd + f_mu
        dt = time.perf_counter() - t0
        results[label] = vrmse(pred, yte)
        preds_store[label] = pred
        print(f"[done] {label:16s} VRMSE {results[label]:.4f} ({dt:.0f}s)")
        if key == "pytorch.convnext_unet":
            model.save(str(OUT / "cunet_wellprotocol.joblib"))

    np.savez_compressed(
        OUT / "wellprotocol_preds.npz", yte=yte,
        norm_mu=f_mu, norm_sd=f_sd,
        **{f"pred__{k}": v for k, v in preds_store.items()})
    summary = {"ours": results, "published_12h_H100": PUBLISHED,
               "protocol": "next-step, 4-frame history, 4 fields, "
                           "128x384, VRMSE"}
    (OUT / "wellprotocol_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))




def _main_stream(args) -> None:
    """Streaming ConvNeXt run: order-of-magnitude more pairs than RAM allows."""
    import torch

    from surge.model.backends.convnext_unet import ConvNeXtUNetModel

    # normalization stats from a small in-memory sample (RAM-safe)
    _, y_stat = load_split("train", 400)
    f_mu = y_stat.mean(axis=(0, 2, 3), keepdims=True)
    f_sd = y_stat.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    del y_stat

    train_ds = _StreamingWellPairs("train", args.n_train, f_mu, f_sd)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=2,
        persistent_workers=True)
    print(f"[stream] {len(train_ds)} training pairs (lazy), "
          f"{args.epochs} epochs", flush=True)

    model = ConvNeXtUNetModel(
        base_channels=40, depth=3, blocks_per_stage=2,
        n_epochs=args.epochs, batch_size=8, patience=0,
        log_file="/tmp/cunet_stream_epochs.jsonl")
    t0 = time.perf_counter()
    model.fit_from_loader(loader, in_channels=N_HIST * len(FIELDS),
                          out_channels=len(FIELDS))
    dt = time.perf_counter() - t0

    Xte, yte = load_split("valid", args.n_test)
    x_mu = np.tile(f_mu, (1, N_HIST, 1, 1))
    x_sd = np.tile(f_sd, (1, N_HIST, 1, 1))
    pred = np.asarray(model.predict((Xte - x_mu) / x_sd)) * f_sd + f_mu
    v = vrmse(pred, yte)
    print(f"[done] CNextU-Net (stream, n={len(train_ds)}) "
          f"VRMSE {v:.4f} ({dt:.0f}s)", flush=True)
    model.save(str(OUT / "cunet_wellprotocol_stream.joblib"))

    summary = json.loads((OUT / "wellprotocol_summary.json").read_text())
    summary["ours"][f"CNextU-Net (stream n={len(train_ds)})"] = v
    (OUT / "wellprotocol_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary["ours"], indent=2))


if __name__ == "__main__":
    main()
