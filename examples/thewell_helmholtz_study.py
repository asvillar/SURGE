#!/usr/bin/env python3
"""TheWell Helmholtz staircase — phase-advance surrogate study.

Task: advance the time-harmonic acoustic pressure field over a staircase
boundary (TheWell ``helmholtz_staircase``; Ohana et al. NeurIPS 2024) by
``horizon`` steps of its 50-step harmonic cycle. Operator models see
both quadratures (Re, Im) as channels — together they determine the
phase — and predict the future real part on the 128×32 grid
(downsampled 8× from 1024×256). Persistence fails badly here (a
quarter-period shift decorrelates a standing wave), which makes the
baseline gate especially meaningful.

Requires ``download_thewell("helmholtz")`` (~80 GB).

Usage:
    SURGE_DEVICE=auto python examples/thewell_helmholtz_study.py \
        [--n-train 800] [--n-test 100] [--epochs 60] [--horizon 12]
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
from matplotlib.colors import CenteredNorm

from surge.viz.theme import (
    diverging_cmap,
    fmt_metric,
    save_figure,
    surge_theme,
)

STRIDE = 8           # 1024x256 -> 128x32


def load_helmholtz(n_train: int, n_test: int, seed: int = 0,
                   horizon: int = 12):
    """X = (B, 2, 128, 32) [Re, Im] at t; y = (B, 128, 32) Re at t+horizon."""
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "helmholtz_staircase" / "data"
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
            yout = np.asarray(item["output_fields"])[0, ::STRIDE, ::STRIDE, 0]
            X_list.append(np.moveaxis(xin, -1, 0))   # (C, H, W)
            y_list.append(yout)
        out[split] = (np.stack(X_list).astype(np.float32),
                      np.stack(y_list).astype(np.float32))
    return out["train"], out["valid"]


def run_study(n_train: int, n_test: int, epochs: int, seed: int = 0,
              horizon: int = 12):
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_helmholtz(n_train, n_test, seed, horizon)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}  horizon {horizon}")
    nte = len(Xte)
    re_te = Xte[:, 0]

    def rel_l2(pred_flat):
        yf = yte.reshape(nte, -1)
        return (np.linalg.norm(pred_flat - yf, axis=1)
                / np.maximum(np.linalg.norm(yf, axis=1), 1e-12))

    rel0 = rel_l2(re_te.reshape(nte, -1))
    results = [{
        "key": "baseline.persistence", "label": "Persistence",
        "residual_target": False, "runtime_s": 0.0,
        "rel_l2_median": float(np.median(rel0)),
        "pred": re_te.reshape(nte, -1).copy(), "rel": rel0,
    }]
    print(f"[done] {'Persistence':16s} median rel-L2 {np.median(rel0):.4f} (0s)")

    candidates = [
        ("pytorch.fno2d", "FNO-2D", {"n_modes": 16, "n_epochs": epochs}),
        ("pytorch.unet", "U-Net", {"n_epochs": epochs}),
    ]
    for key, label, params in candidates:
        try:
            model = MODEL_REGISTRY.create(key, **params)
        except KeyError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xte)).reshape(nte, -1)
        except Exception as exc:  # noqa: BLE001 - study reports, never dies
            print(f"[fail] {label}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        rel = rel_l2(pred)
        results.append({
            "key": key, "label": label, "residual_target": False,
            "runtime_s": dt,
            "rel_l2_median": float(np.median(rel)),
            "pred": pred, "rel": rel,
        })
        print(f"[done] {label:16s} median rel-L2 {np.median(rel):.4f} "
              f"({fmt_metric(dt, 'runtime')})")
    return (Xte, yte), results


def study_figure(data, results, mode: str = "light", horizon: int = 12):
    Xte, yte = data
    nte, H, W = yte.shape
    results = sorted(results, key=lambda r: r["rel_l2_median"])
    best = results[0]
    i = int(np.argsort(best["rel"])[len(best["rel"]) // 2])
    truth_img = yte[i]

    panels = [r for r in results if r["key"] != "baseline.persistence"][:2]

    with surge_theme(mode) as p:
        n = len(panels)
        fig = plt.figure(figsize=(1.35 * (n + 3) + 4.8, 4.6))
        gs = fig.add_gridspec(1, n + 3, width_ratios=[1] * (n + 2) + [2.4],
                              wspace=0.15)
        v = float(np.abs(truth_img).max())
        kw = dict(cmap=diverging_cmap(mode), norm=CenteredNorm(halfrange=v),
                  origin="lower", aspect="auto")

        field_axes = []
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(Xte[i, 0].T, **kw)
        ax.set_title(r"input Re $p(t)$", fontsize=8.5)
        field_axes.append(ax)
        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(truth_img.T, **kw)
        ax.set_title(r"truth Re $p(t+\Delta t)$", fontsize=8.5)
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
        cbar.set_label(r"Re $p$ (shared)", fontsize=8)

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
            "TheWell · Helmholtz staircase — harmonic phase advance"
            r"  $p(t) \mapsto p(t+\Delta t)$"
            f", $\\Delta t$ = {horizon}/50 cycle  (128×32, Re+Im input)",
            fontsize=11, fontweight="bold")
        return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    args = ap.parse_args()

    data, results = run_study(args.n_train, args.n_test, args.epochs,
                              horizon=args.horizon)
    if not results:
        raise SystemExit("no model produced results")
    out = Path(args.out)
    for mode in ("light", "dark"):
        fig = study_figure(data, results, mode, horizon=args.horizon)
        for path in save_figure(fig, out / f"thewell_helmholtz_{mode}"):
            print("wrote", path)
        plt.close(fig)
    Xte, yte = data
    np.savez_compressed(
        out / "thewell_helmholtz_preds.npz", Xte=Xte, yte=yte,
        **{f"pred__{r['label']}": r["pred"] for r in results})
    summary = [{k: v for k, v in r.items() if k not in ("pred", "rel")}
               for r in results]
    (out / "thewell_helmholtz_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
