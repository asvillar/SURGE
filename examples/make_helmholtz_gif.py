#!/usr/bin/env python3
"""Animated Helmholtz rollout for the README.

Trains a 2-in/2-out FNO-2D to advance the (Re, Im) acoustic pressure
quadratures by 2/50 of the harmonic cycle, then rolls it out
autoregressively over a full cycle on one held-out trajectory. Each GIF
frame shows truth | prediction | error side by side with the rolling
rel-L2 — error accumulation is visible honestly, not hidden.

Output: docs/assets/readme/helmholtz_rollout.gif (~25 frames)

Usage:
    SURGE_DEVICE=auto python examples/make_helmholtz_gif.py [--epochs 40]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from PIL import Image

from surge.viz.theme import diverging_cmap, surge_theme

STRIDE = 8
STEP = 2                     # model advances 2/50 of the cycle per call


def load_pairs_and_traj(n_train: int, seed: int = 0):
    """Training pairs (both quadratures in AND out) + one test trajectory."""
    import glob
    import os

    import h5py

    root = Path.home() / ".surge/data/thewell/datasets/helmholtz_staircase/data"

    def field(f):
        with h5py.File(f) as h:
            re = h["t0_fields/pressure_re"][:, :, ::STRIDE, ::STRIDE]
            im = h["t0_fields/pressure_im"][:, :, ::STRIDE, ::STRIDE]
        return np.stack([re, im], axis=2).astype("float32")  # (S, T, 2, H, W)

    train_files = sorted(glob.glob(str(root / "train" / "*.hdf5")),
                         key=os.path.getmtime)[:4]
    Xs, ys = [], []
    for f in train_files:
        d = field(f)
        Xs.append(d[:, :-STEP].reshape(-1, 2, *d.shape[-2:]))
        ys.append(d[:, STEP:].reshape(-1, 2, *d.shape[-2:]))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n_train]

    valid_file = sorted(glob.glob(str(root / "valid" / "*.hdf5")))[0]
    traj = field(valid_file)[0]          # (T, 2, H, W) one trajectory
    return X[idx], y[idx], traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default=str(
        _REPO / "docs" / "assets" / "readme" / "helmholtz_rollout.gif"))
    args = ap.parse_args()

    from surge.model import MODEL_REGISTRY

    X, y, traj = load_pairs_and_traj(args.n_train)
    print(f"[data] pairs {X.shape}  trajectory {traj.shape}")
    model = MODEL_REGISTRY.create("pytorch.fno2d", n_modes=16,
                                  n_epochs=args.epochs)
    model.fit(X, y)

    # autoregressive rollout over one full harmonic cycle
    n_steps = (traj.shape[0] - 1) // STEP
    state = traj[0:1]                     # (1, 2, H, W)
    preds = [state[0]]
    for _ in range(n_steps):
        state = np.asarray(model.predict(state)).reshape(1, 2, *traj.shape[-2:])
        preds.append(state[0])
    preds = np.stack(preds)               # (n_steps+1, 2, H, W)
    truth = traj[::STEP][:len(preds)]

    frames = []
    v = float(np.abs(truth[:, 0]).max())
    with surge_theme("dark") as p:
        for k in range(len(preds)):
            fig, axes = plt.subplots(1, 3, figsize=(5.6, 3.1))
            rel = (np.linalg.norm(preds[k, 0] - truth[k, 0])
                   / max(np.linalg.norm(truth[k, 0]), 1e-12))
            for ax, img, title in (
                (axes[0], truth[k, 0], "truth"),
                (axes[1], preds[k, 0], "FNO-2D rollout"),
                (axes[2], preds[k, 0] - truth[k, 0], "error"),
            ):
                ax.imshow(img.T, cmap=diverging_cmap("dark"),
                          norm=CenteredNorm(halfrange=v), origin="lower",
                          aspect="auto")
                ax.set_title(title, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            fig.suptitle(
                f"Helmholtz staircase — Re $p$, phase {2 * k}/50 of cycle · "
                f"rolling rel-L2 {rel:.3f}", fontsize=9.5)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=96)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("P",
                                                  palette=Image.ADAPTIVE))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=140, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(frames)} frames)")


if __name__ == "__main__":
    main()
