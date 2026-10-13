#!/usr/bin/env python3
"""Turbulence (TRL-2D) improvement battery — the untried levers.

Tested against the SAME h=8 test set as the published study (verified
vs the saved artifacts):

  A. temporal context — 4 input frames (the Well's own baseline setup)
  B. full resolution — 128x384 (no downsampling)
  C. stacking — blend POD+ridge with the saved U-Net prediction
  D. composition — h=2 model rolled out 4x vs direct h=8

Usage:  SURGE_DEVICE=auto python examples/thewell_turbulence_improve.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from surge.preprocessing import pod_fit, pod_inverse, pod_transform

OUT = _REPO / "examples" / "viz_gallery_output"
FIELD = 0


def rel_l2(pred, truth):
    n = len(truth)
    p = pred.reshape(n, -1)
    t = truth.reshape(n, -1)
    return np.median(np.linalg.norm(p - t, axis=1)
                     / np.maximum(np.linalg.norm(t, axis=1), 1e-12))


def load_frames(split, n, seed, horizon, n_in=1, stride=2):
    """(X: (n, n_in, H, W) log-density frames, y: (n, H, W) at t+horizon)."""
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = (_well_base_path(None) / "turbulent_radiative_layer_2D"
            / "data" / split)
    ds = WellDataset(path=str(root), n_steps_input=n_in, n_steps_output=1,
                     use_normalization=False, min_dt_stride=horizon,
                     max_dt_stride=horizon)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    X, Y = [], []
    for i in idx:
        item = ds[int(i)]
        xin = np.asarray(item["input_fields"])[:, ::stride, ::stride, FIELD]
        yout = np.asarray(item["output_fields"])[0, ::stride, ::stride, FIELD]
        X.append(np.log10(xin))
        Y.append(np.log10(yout))
    return np.stack(X).astype("float32"), np.stack(Y).astype("float32")


def train_unet(X, y, Xq, epochs=60, **kw):
    from surge.model import MODEL_REGISTRY

    m = MODEL_REGISTRY.create("pytorch.unet", n_epochs=epochs, **kw)
    t0 = time.perf_counter()
    m.fit(X, y)
    pred = np.asarray(m.predict(Xq))
    return m, pred, time.perf_counter() - t0


def main() -> None:
    # reference test set (h=8, 64x192, seed 0) — must match saved artifacts
    from examples.thewell_turbulence_study import load_trl2d
    (Xtr, ytr), (Xte, yte) = load_trl2d(800, 100, seed=0, horizon=8)
    npz = np.load(OUT / "thewell_turbulence_preds.npz")
    assert np.allclose(npz["yte"], yte, atol=1e-5)
    unet_pred = npz["pred__U-Net"].reshape(yte.shape)
    print(f"[ref] persistence {rel_l2(Xte, yte):.4f}  "
          f"U-Net {rel_l2(unet_pred, yte):.4f}  "
          f"(published floor ~0.25)")

    # ── A. temporal context: 4 input frames ──────────────────────────
    # NOTE: 4-frame windows shift the valid (t, t+8) pair set, so this
    # uses its own test draw at the same horizon — matched task, not
    # bitwise-identical samples. Persistence is recomputed accordingly.
    Xtr4, ytr4 = load_frames("train", 800, 0, horizon=8, n_in=4)
    Xte4, yte4 = load_frames("valid", 100, 0, horizon=8, n_in=4)
    pers4 = rel_l2(Xte4[:, -1], yte4)
    _, pred4, dt = train_unet(Xtr4, ytr4, Xte4)
    print(f"[A] 4-frame U-Net: rel-L2 {rel_l2(pred4, yte4):.4f} "
          f"(persistence on this draw {pers4:.4f}, {dt:.0f}s)")

    # ── C. stacking: POD+ridge x saved U-Net (honest split-half) ─────
    from surge.model import MODEL_REGISTRY
    bx = pod_fit(Xtr.reshape(len(Xtr), -1), 64)
    by = pod_fit(ytr.reshape(len(ytr), -1), 64)
    ridge = MODEL_REGISTRY.create("sklearn.ridge")
    ridge.fit(pod_transform(bx, Xtr.reshape(len(Xtr), -1)).astype("float32"),
              pod_transform(by, ytr.reshape(len(ytr), -1)).astype("float32"))
    coeffs = np.asarray(ridge.predict(
        pod_transform(bx, Xte.reshape(len(Xte), -1)).astype("float32")))
    pod_pred = pod_inverse(by, coeffs).reshape(yte.shape)
    print(f"[C] POD+ridge alone: {rel_l2(pod_pred, yte):.4f}")
    alphas = np.linspace(0, 1, 21)
    tune, hold = slice(0, 50), slice(50, 100)
    best_a = min(alphas, key=lambda a: rel_l2(
        a * pod_pred[tune] + (1 - a) * unet_pred[tune], yte[tune]))
    blend_hold = rel_l2(best_a * pod_pred[hold]
                        + (1 - best_a) * unet_pred[hold], yte[hold])
    print(f"[C] blend alpha={best_a:.2f} (tuned on half): held-out rel-L2 "
          f"{blend_hold:.4f}  vs U-Net {rel_l2(unet_pred[hold], yte[hold]):.4f} "
          f"POD {rel_l2(pod_pred[hold], yte[hold]):.4f}")

    # ── D. composition: h=2 model rolled 4x on the h=8 test inputs ───
    Xtr2, ytr2 = load_frames("train", 1600, 0, horizon=2)
    m2, _, dt = train_unet(Xtr2[:, 0], ytr2, Xte[:1])  # warm predict
    state = Xte.copy()
    for _ in range(4):
        state = np.asarray(m2.predict(state)).reshape(state.shape)
    print(f"[D] h=2 rolled x4: rel-L2 {rel_l2(state, yte):.4f} ({dt:.0f}s train)")

    # ── B. full resolution 128x384 (own draw, same protocol) ─────────
    Xtr_hr, ytr_hr = load_frames("train", 800, 0, horizon=8, stride=1)
    Xte_hr, yte_hr = load_frames("valid", 100, 0, horizon=8, stride=1)
    pers_hr = rel_l2(Xte_hr[:, 0], yte_hr)
    _, pred_hr, dt = train_unet(Xtr_hr[:, 0], ytr_hr, Xte_hr[:, 0])
    print(f"[B] full-res U-Net: rel-L2 {rel_l2(pred_hr, yte_hr):.4f} "
          f"(persistence {pers_hr:.4f}, {dt:.0f}s)")

    # ── A+B combo if A helped: 4 frames at full res ──────────────────
    Xtr4h, ytr4h = load_frames("train", 800, 0, horizon=8, n_in=4, stride=1)
    Xte4h, yte4h = load_frames("valid", 100, 0, horizon=8, n_in=4, stride=1)
    _, pred4h, dt = train_unet(Xtr4h, ytr4h, Xte4h)
    print(f"[A+B] 4-frame full-res U-Net: rel-L2 {rel_l2(pred4h, yte4h):.4f} "
          f"(persistence {rel_l2(Xte4h[:, -1], yte4h):.4f}, {dt:.0f}s)")


if __name__ == "__main__":
    main()
