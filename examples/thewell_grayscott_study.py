#!/usr/bin/env python3
"""TheWell Gray-Scott surrogate study: FNO-2D vs U-Net vs DeepONet vs ridge.

Task: operator forecasting on the Gray-Scott reaction–diffusion system
from TheWell (Ohana et al., NeurIPS 2024) — predict the species-B field
160 stored timesteps ahead. A persistence baseline anchors the
leaderboard: consecutive frames are so similar that the single-step task
is trivial (persistence rel-L2 ~0.002), so the horizon is chosen where
"predict no change" visibly fails (~0.28).

Requires the dataset cache created by
``surge.benchmarks.loaders.thewell.download_thewell("gray_scott")``
(~132 GB on disk: 117 GB train + 15 GB valid).
Fields are downsampled 128 -> 64 and a single species channel is used so
every registered operator model consumes the same flat (B, 64*64) task.

Usage (from repo root):
    python examples/thewell_grayscott_study.py \
        [--n-train 400] [--n-test 80] [--side 64] [--epochs 60] \
        [--out examples/viz_gallery_output]
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
    sequential_cmap,
    surge_theme,
)

SPECIES = 1          # Gray-Scott channel: 0 = A (feed), 1 = B (pattern-forming)


def load_grayscott(n_train: int, n_test: int, side: int, seed: int = 0,
                   horizon: int = 160):
    """(X, y) pairs: species-B field at t -> t + horizon stored steps.

    Consecutive stored frames are nearly identical (persistence scores
    rel-L2 ~0.002 at horizon 1), so a single-step task is trivial and every
    model loses to "predict no change". At horizon 160 persistence degrades
    to ~0.34 and the surrogate must learn actual pattern dynamics.
    """
    from the_well.data import WellDataset

    from surge.benchmarks.loaders.thewell import _well_base_path

    root = _well_base_path(None) / "gray_scott_reaction_diffusion" / "data"
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
        stride = None
        for i in idx:
            item = ds[int(i)]
            xin = np.asarray(item["input_fields"])[0, ..., SPECIES]
            yout = np.asarray(item["output_fields"])[0, ..., SPECIES]
            if stride is None:
                stride = max(1, xin.shape[0] // side)
            X_list.append(xin[::stride, ::stride][:side, :side])
            y_list.append(yout[::stride, ::stride][:side, :side])
        X = np.stack(X_list).astype(np.float32)
        y = np.stack(y_list).astype(np.float32)
        out[split] = (X.reshape(len(X), -1), y.reshape(len(y), -1))
    return out["train"], out["valid"]


def run_study(n_train: int, n_test: int, side: int, epochs: int, seed: int = 0,
              horizon: int = 160):
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_grayscott(n_train, n_test, side, seed,
                                            horizon)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}  horizon {horizon}")

    # honesty anchor: every model must beat "predict no change"
    rel0 = (np.linalg.norm(Xte - yte, axis=1)
            / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
    results = [{
        "key": "baseline.persistence", "label": "Persistence B(t)",
        "residual_target": False, "runtime_s": 0.0,
        "rel_l2_median": float(np.median(rel0)),
        "pred": Xte.copy(), "rel": rel0,
    }]
    print(f"[done] {'Persistence B(t)':16s} median rel-L2 "
          f"{np.median(rel0):.4f} (0s)")

    # residual=True trains on the *increment* B(t+1) − B(t) and adds the
    # input back at prediction time — the standard fix for architectures
    # without local spatial bias (DeepONet's low-rank global basis cannot
    # represent "sharpen each filament in place", but the small residual
    # field is far easier to expand)
    candidates = [
        ("pytorch.fno2d", "FNO-2D", {"n_modes": 24, "n_epochs": epochs}, False),
        ("pytorch.fno2d", "FNO-2D (residual)",
         {"n_modes": 24, "n_epochs": epochs}, True),
        ("pytorch.unet", "U-Net", {"n_epochs": epochs}, False),
        ("pytorch.unet", "U-Net (residual)", {"n_epochs": epochs}, True),
        ("pytorch.deeponet", "DeepONet", {"n_epochs": epochs}, False),
        ("pytorch.deeponet", "DeepONet (residual)", {"n_epochs": epochs}, True),
        ("sklearn.ridge", "Ridge (linear)", {}, False),
    ]
    for key, label, params, residual in candidates:
        try:
            model = MODEL_REGISTRY.create(key, **params)
        except KeyError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr - Xtr if residual else ytr)
            pred = np.asarray(model.predict(Xte)).reshape(len(Xte), -1)
            if residual:
                pred = pred + Xte
        except Exception as exc:  # noqa: BLE001 - study reports, never dies
            print(f"[fail] {label}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        rel = (np.linalg.norm(pred - yte, axis=1)
               / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
        # median only: early-time Gray-Scott frames have near-zero species-B
        # norm, so the per-sample relative error (and its mean) diverges
        results.append({
            "key": key, "label": label, "residual_target": residual,
            "runtime_s": dt,
            "rel_l2_median": float(np.median(rel)),
            "pred": pred, "rel": rel,
        })
        print(f"[done] {label:16s} median rel-L2 {np.median(rel):.4f} "
              f"({fmt_metric(dt, 'runtime')})")
    return (Xte, yte), results


def study_figure(data, results, side: int, mode: str = "light"):
    Xte, yte = data
    results = sorted(results, key=lambda r: r["rel_l2_median"])
    # median-difficulty sample judged by the best model
    best = results[0]
    i = int(np.argsort(best["rel"])[len(best["rel"]) // 2])
    truth_img = yte[i].reshape(side, side)

    # the persistence "prediction" is the input field itself — keep it in
    # the comparison bars but don't show the same image twice; cap the image
    # row at the four best models so the figure stays readable
    panels = [r for r in results if r["key"] != "baseline.persistence"][:4]

    with surge_theme(mode) as p:
        n_models = len(panels)
        fig = plt.figure(figsize=(2.9 * (n_models + 2), 5.6))
        gs = fig.add_gridspec(2, n_models + 2, height_ratios=[1.15, 1.0])

        # one shared color scale anchored to the truth: models that produce
        # out-of-range noise must LOOK wrong, not get their own flattering
        # auto-scaled colorbar
        vmin = float(min(truth_img.min(), Xte[i].min()))
        vmax = float(max(truth_img.max(), Xte[i].max()))
        field_kw = dict(cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax)

        field_axes = []
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(Xte[i].reshape(side, side), **field_kw)
        ax.set_title(r"input  $B(t)$", fontsize=9.5)
        field_axes.append(ax)

        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(truth_img, **field_kw)
        ax.set_title(r"truth  $B(t+\Delta t)$", fontsize=9.5)
        field_axes.append(ax)

        for k, r in enumerate(panels):
            ax = fig.add_subplot(gs[0, k + 2])
            ax.imshow(r["pred"][i].reshape(side, side), **field_kw)
            ax.set_title(f"{r['label']}\nrel-L2 {r['rel'][i]:.3f}", fontsize=9)
            field_axes.append(ax)

        for ax in field_axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cbar = fig.colorbar(im, ax=field_axes, fraction=0.012, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.set_label(r"$B$  (shared scale)", fontsize=8.5)

        # bottom: error map of best model + rel-L2 comparison bars
        ax = fig.add_subplot(gs[1, 0:2])
        err = best["pred"][i].reshape(side, side) - truth_img
        vmax = float(np.abs(truth_img).max()) * 0.1 + 1e-9
        im = ax.imshow(err, cmap=diverging_cmap(mode),
                       norm=CenteredNorm(halfrange=vmax))
        ax.set_title(f"error map — {best['label']}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)

        ax = fig.add_subplot(gs[1, 2:])
        ypos = np.arange(len(results))[::-1]
        meds = [r["rel_l2_median"] for r in results]
        colors = [p["series"][2] if r["key"] == "baseline.persistence"
                  else p["series"][0] for r in results]
        ax.barh(ypos, meds, height=0.6, color=colors, alpha=0.9)
        for y, r in zip(ypos, results):
            label = f"  {r['rel_l2_median']:.4f}"
            if r["key"] != "baseline.persistence":
                label += f" · {fmt_metric(r['runtime_s'], 'runtime')}"
            ax.text(r["rel_l2_median"], y, label,
                    va="center", fontsize=8, color=p["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in results], fontsize=9)
        ax.set_xlabel("median rel-L2 (lower is better)")
        ax.set_title("model comparison", fontsize=9.5)

        fig.suptitle(
            "TheWell · Gray-Scott reaction–diffusion — forecast surrogates"
            r"  $B(t) \mapsto B(t+\Delta t)$, $\Delta t$ = 160 stored steps"
            f"  ({side}×{side}, species B)",
            fontsize=11.5, fontweight="bold")
        return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=160,
                    help="prediction horizon in stored timesteps")
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    args = ap.parse_args()

    data, results = run_study(args.n_train, args.n_test, args.side,
                              args.epochs, horizon=args.horizon)
    if not results:
        raise SystemExit("no model produced results")
    out = Path(args.out)
    for mode in ("light", "dark"):
        fig = study_figure(data, results, args.side, mode)
        for path in save_figure(fig, out / f"thewell_grayscott_{mode}"):
            print("wrote", path)
        plt.close(fig)
    (Xte, yte) = data
    np.savez_compressed(
        out / "thewell_grayscott_preds.npz",
        Xte=Xte, yte=yte,
        **{f"pred__{r['label']}": r["pred"] for r in results})
    summary = [{k: v for k, v in r.items() if k not in ("pred", "rel")}
               for r in results]
    (out / "thewell_grayscott_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
