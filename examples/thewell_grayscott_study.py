#!/usr/bin/env python3
"""TheWell Gray-Scott surrogate study: FNO-2D vs U-Net vs DeepONet vs ridge.

Task: next-step operator learning on the Gray-Scott reaction–diffusion
system from TheWell (Ohana et al., NeurIPS 2024) — predict species-B
concentration at the next stored timestep from the current field.

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


def load_grayscott(n_train: int, n_test: int, side: int, seed: int = 0):
    """(X, y) pairs: species-B field at t -> at t+1, downsampled to side²."""
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


def run_study(n_train: int, n_test: int, side: int, epochs: int, seed: int = 0):
    from surge.model import MODEL_REGISTRY

    (Xtr, ytr), (Xte, yte) = load_grayscott(n_train, n_test, side, seed)
    print(f"[data] train {Xtr.shape}  test {Xte.shape}")

    candidates = [
        ("pytorch.fno2d", "FNO-2D", {"n_modes": 12, "n_epochs": epochs}),
        ("pytorch.unet", "U-Net", {"n_epochs": epochs}),
        ("pytorch.deeponet", "DeepONet", {"n_epochs": epochs}),
        ("sklearn.ridge", "Ridge (linear)", {}),
    ]
    results = []
    for key, label, params in candidates:
        try:
            model = MODEL_REGISTRY.create(key, **params)
        except KeyError as exc:
            print(f"[skip] {label}: {exc}")
            continue
        t0 = time.perf_counter()
        try:
            model.fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xte)).reshape(len(Xte), -1)
        except Exception as exc:  # noqa: BLE001 - study reports, never dies
            print(f"[fail] {label}: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        rel = (np.linalg.norm(pred - yte, axis=1)
               / np.maximum(np.linalg.norm(yte, axis=1), 1e-12))
        # median only: early-time Gray-Scott frames have near-zero species-B
        # norm, so the per-sample relative error (and its mean) diverges
        results.append({
            "key": key, "label": label, "runtime_s": dt,
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

    with surge_theme(mode) as p:
        n_models = len(results)
        fig = plt.figure(figsize=(2.9 * (n_models + 2), 5.6))
        gs = fig.add_gridspec(2, n_models + 2, height_ratios=[1.15, 1.0])

        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(Xte[i].reshape(side, side), cmap=sequential_cmap(mode))
        ax.set_title("input  B(t)", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)

        ax = fig.add_subplot(gs[0, 1])
        im = ax.imshow(truth_img, cmap=sequential_cmap(mode))
        ax.set_title("truth  B(t+1)", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)

        for k, r in enumerate(results):
            ax = fig.add_subplot(gs[0, k + 2])
            im = ax.imshow(r["pred"][i].reshape(side, side),
                           cmap=sequential_cmap(mode))
            ax.set_title(f"{r['label']}\nrel-L2 {r['rel'][i]:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046,
                         pad=0.02).outline.set_visible(False)

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
        ypos = np.arange(n_models)[::-1]
        meds = [r["rel_l2_median"] for r in results]
        ax.barh(ypos, meds, height=0.6, color=p["series"][0], alpha=0.9)
        for y, r in zip(ypos, results):
            ax.text(r["rel_l2_median"], y,
                    f"  {r['rel_l2_median']:.4f} · {fmt_metric(r['runtime_s'], 'runtime')}",
                    va="center", fontsize=8, color=p["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([r["label"] for r in results], fontsize=9)
        ax.set_xlabel("median rel-L2 (lower is better)")
        ax.set_title("model comparison", fontsize=9.5)

        fig.suptitle(
            "TheWell · Gray-Scott reaction–diffusion — next-step surrogates "
            f"({side}×{side}, species B)",
            fontsize=11.5, fontweight="bold")
        return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--side", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    args = ap.parse_args()

    data, results = run_study(args.n_train, args.n_test, args.side, args.epochs)
    if not results:
        raise SystemExit("no model produced results")
    out = Path(args.out)
    for mode in ("light", "dark"):
        fig = study_figure(data, results, args.side, mode)
        for path in save_figure(fig, out / f"thewell_grayscott_{mode}"):
            print("wrote", path)
        plt.close(fig)
    summary = [{k: v for k, v in r.items() if k not in ("pred", "rel")}
               for r in results]
    (out / "thewell_grayscott_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
