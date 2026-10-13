#!/usr/bin/env python3
"""TheWell turbulent radiative layer (2D) — forecast surrogate study.

Task: forecast the density field of the turbulent radiative layer
(Fielding et al. setup; TheWell ``turbulent_radiative_layer_2D``,
Ohana et al. NeurIPS 2024) ``horizon`` stored steps ahead, on the
non-square 64×192 grid (downsampled from 128×384). A persistence
baseline anchors the leaderboard exactly as in the Gray-Scott study.

Unlike Gray-Scott, the grid is non-square: operator models (FNO-2D,
U-Net) consume ``(B, H, W)`` grids directly; flat models (DeepONet,
ridge) get the flattened vector.

Requires ``download_thewell("turbulence_2d")`` (~6 GB).

Usage:
    python examples/thewell_turbulence_study.py \
        [--n-train 800] [--n-test 100] [--epochs 60] [--horizon 8]
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, LogNorm

from surge.viz.theme import (
    diverging_cmap,
    fmt_metric,
    save_figure,
    sequential_cmap,
    surge_theme,
)

FIELD = 0            # TRL-2D channels: 0 = density, 1 = pressure, 2/3 = velocity
STRIDE = 2           # 128x384 -> 64x192


def load_trl2d(n_train: int, n_test: int, seed: int = 0, horizon: int = 8,
               channels: str = "density"):
    """(X, y): density (or all 4 fields) at t -> density at t + horizon.

    channels="density": X is (B, 64, 192) log-density.
    channels="all":     X is (B, 4, 64, 192) — log density, log pressure,
    z-scored vx, vy — the physics that actually advects the interface.
    y is always (B, 64, 192) log-density.
    """
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "turbulent_radiative_layer_2D" / "data"
    out = {}
    for split, needed in (("train", n_train), ("valid", n_test)):
        ds = WellDataset(
            path=str(root / split),
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=False,
            min_dt_stride=horizon,
            max_dt_stride=horizon,
        )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(needed, len(ds)), replace=False)
        X_list, y_list = [], []
        for i in idx:
            item = ds[int(i)]
            xin = np.asarray(item["input_fields"])[0, ::STRIDE, ::STRIDE, :]
            yout = np.asarray(item["output_fields"])[0, ::STRIDE, ::STRIDE,
                                                     FIELD]
            X_list.append(xin)
            y_list.append(yout)
        Xf = np.stack(X_list).astype(np.float32)      # (B, H, W, 4)
        # density is strictly positive and spans decades -> log-space targets
        y = np.log10(np.stack(y_list).astype(np.float32))
        if channels == "density":
            X = np.log10(Xf[..., FIELD])
        else:  # all four fields as channels
            dens = np.log10(Xf[..., 0])
            pres = np.log10(np.maximum(Xf[..., 1], 1e-12))
            vx = Xf[..., 2] / (np.abs(Xf[..., 2]).max() + 1e-9)
            vy = Xf[..., 3] / (np.abs(Xf[..., 3]).max() + 1e-9)
            X = np.stack([dens, pres, vx, vy], axis=1)  # (B, 4, H, W)
        out[split] = (X, y)
    return out["train"], out["valid"]


def run_study(n_train: int, n_test: int, epochs: int, seed: int = 0,
              horizon: int = 8, channels: str = "density"):
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_trl2d(n_train, n_test, seed, horizon,
                                        channels)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}  horizon {horizon}"
          f"  channels {channels}")
    nte = len(Xte)
    multi = channels != "density"
    dens_te = Xte[:, 0] if multi else Xte

    def rel_l2(pred_flat):
        yf = yte.reshape(nte, -1)
        return (np.linalg.norm(pred_flat - yf, axis=1)
                / np.maximum(np.linalg.norm(yf, axis=1), 1e-12))

    rel0 = rel_l2(dens_te.reshape(nte, -1))
    results = [{
        "key": "baseline.persistence", "label": "Persistence",
        "residual_target": False, "runtime_s": 0.0,
        "rel_l2_median": float(np.median(rel0)),
        "pred": dens_te.reshape(nte, -1).copy(), "rel": rel0,
    }]
    print(f"[done] {'Persistence':18s} median rel-L2 {np.median(rel0):.4f} (0s)")

    # (key, label, params, residual_target, input_layout)
    candidates = [
        ("pytorch.fno2d", "FNO-2D", {"n_modes": 24, "n_epochs": epochs},
         False, "grid"),
        ("pytorch.fno2d", "FNO-2D (residual)",
         {"n_modes": 24, "n_epochs": epochs}, True, "grid"),
        ("pytorch.unet", "U-Net", {"n_epochs": epochs}, False, "grid"),
        ("pytorch.unet", "U-Net (residual)", {"n_epochs": epochs},
         True, "grid"),
        ("pytorch.deeponet", "DeepONet (residual)", {"n_epochs": epochs},
         True, "flat"),
        ("sklearn.ridge", "Ridge (linear)", {}, False, "flat"),
    ]
    dens_tr = Xtr[:, 0] if multi else Xtr
    for key, label, params, residual, layout in candidates:
        if multi and layout == "flat":
            print(f"[skip] {label}: flat models use --channels density")
            continue
        try:
            model = MODEL_REGISTRY.create(key, **params)
        except KeyError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        if layout == "grid":
            xin, xq = Xtr, Xte
            yin = ytr - dens_tr if residual else ytr
        else:
            xin = Xtr.reshape(len(Xtr), -1)
            yin = (ytr - dens_tr if residual else ytr).reshape(len(ytr), -1)
            xq = Xte.reshape(nte, -1)
        t0 = time.perf_counter()
        try:
            model.fit(xin, yin)
            pred = np.asarray(model.predict(xq)).reshape(nte, -1)
            if residual:
                pred = pred + dens_te.reshape(nte, -1)
        except Exception as exc:  # noqa: BLE001 - study reports, never dies
            print(f"[fail] {label}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        rel = rel_l2(pred)
        results.append({
            "key": key, "label": label, "residual_target": residual,
            "runtime_s": dt,
            "rel_l2_median": float(np.median(rel)),
            "pred": pred, "rel": rel,
        })
        print(f"[done] {label:18s} median rel-L2 {np.median(rel):.4f} "
              f"({fmt_metric(dt, 'runtime')})")
    return (Xte, yte), results


def study_figure(data, results, mode: str = "light", horizon: int = 8):
    Xte, yte = data
    if Xte.ndim == 4:            # multichannel input: show the density channel
        Xte = Xte[:, 0]
    nte, H, W = yte.shape
    results = sorted(results, key=lambda r: r["rel_l2_median"])
    best = results[0]
    i = int(np.argsort(best["rel"])[len(best["rel"]) // 2])
    truth_img = yte[i]

    panels = [r for r in results if r["key"] != "baseline.persistence"][:3]

    with surge_theme(mode) as p:
        n = len(panels)
        # tall skinny fields (64 wide x 192 tall) sit side by side
        fig = plt.figure(figsize=(1.55 * (n + 3) + 4.6, 5.4))
        gs = fig.add_gridspec(1, n + 3, width_ratios=[1] * (n + 2) + [2.6],
                              wspace=0.15)

        vmin, vmax = float(truth_img.min()), float(truth_img.max())
        kw = dict(cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax,
                  origin="lower", aspect="auto")

        field_axes = []
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(Xte[i].T, **kw)
        ax.set_title(r"input $\rho(t)$", fontsize=9)
        field_axes.append(ax)
        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(truth_img.T, **kw)
        ax.set_title(r"truth $\rho(t+\Delta t)$", fontsize=9)
        field_axes.append(ax)
        for k, r in enumerate(panels):
            ax = fig.add_subplot(gs[0, k + 2])
            ax.imshow(r["pred"][i].reshape(H, W).T, **kw)
            ax.set_title(f"{r['label']}\nrel-L2 {r['rel'][i]:.3f}", fontsize=8)
            field_axes.append(ax)
        for ax in field_axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cbar = fig.colorbar(im, ax=field_axes, fraction=0.025, pad=0.012)
        cbar.outline.set_visible(False)
        cbar.set_label(r"$\log_{10}\rho$ (shared)", fontsize=8)

        ax = fig.add_subplot(gs[0, -1])
        ypos = np.arange(len(results))[::-1]
        colors = [p["series"][2] if r["key"] == "baseline.persistence"
                  else p["series"][0] for r in results]
        ax.barh(ypos, [r["rel_l2_median"] for r in results], height=0.6,
                color=colors, alpha=0.9)
        for y, r in zip(ypos, results):
            label = f"  {r['rel_l2_median']:.4f}"
            if r["key"] != "baseline.persistence":
                label += f" · {fmt_metric(r['runtime_s'], 'runtime')}"
            ax.text(r["rel_l2_median"], y, label, va="center", fontsize=7.5,
                    color=p["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in results], fontsize=8.5)
        ax.set_xlabel("median rel-L2 (lower is better)")
        ax.set_title("model comparison", fontsize=9)

        fig.suptitle(
            "TheWell · turbulent radiative layer 2D — forecast surrogates"
            r"  $\rho(t) \mapsto \rho(t+\Delta t)$"
            f", $\\Delta t$ = {horizon} stored steps  (64×192, log density)",
            fontsize=11, fontweight="bold")
        return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--channels", choices=["density", "all"],
                    default="density",
                    help="operator-model input: density only, or all 4 fields")
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    args = ap.parse_args()

    data, results = run_study(args.n_train, args.n_test, args.epochs,
                              horizon=args.horizon, channels=args.channels)
    if not results:
        raise SystemExit("no model produced results")
    out = Path(args.out)
    for mode in ("light", "dark"):
        fig = study_figure(data, results, mode, horizon=args.horizon)
        for path in save_figure(fig, out / f"thewell_turbulence_{mode}"):
            print("wrote", path)
        plt.close(fig)
    Xte, yte = data
    np.savez_compressed(
        out / "thewell_turbulence_preds.npz", Xte=Xte, yte=yte,
        **{f"pred__{r['label']}": r["pred"] for r in results})
    summary = [{k: v for k, v in r.items() if k not in ("pred", "rel")}
               for r in results]
    (out / "thewell_turbulence_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
