#!/usr/bin/env python3
"""POD reduced-order surrogates vs neural operators on TheWell tasks.

The classic reduced-order-model recipe: fit a POD basis on the training
fields, learn the coefficient map (k inputs -> k outputs) with CHEAP
tabular models, reconstruct, and compare against the neural operators —
whose predictions are reused from the saved study artifacts
(``thewell_*_preds.npz``), so nothing expensive is retrained.

Tasks (identical splits to the published studies — verified against the
stored test fields):
* turbulent radiative layer 2D, horizon 8 (log density, 64x192)
* Helmholtz staircase, horizon 12 (Re pressure from Re+Im, 128x32)

Usage:
    SURGE_DEVICE=auto python examples/thewell_pod_study.py
"""
from __future__ import annotations

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

from surge.preprocessing import pod_fit, pod_inverse, pod_transform
from matplotlib.colors import CenteredNorm

from surge.viz.theme import diverging_cmap, save_figure, surge_theme

K_MODES = (16, 64, 256)
OUT = _REPO / "examples" / "viz_gallery_output"


def _rel_l2(pred, truth):
    n = len(truth)
    p = pred.reshape(n, -1)
    t = truth.reshape(n, -1)
    return (np.linalg.norm(p - t, axis=1)
            / np.maximum(np.linalg.norm(t, axis=1), 1e-12))


def load_task(name: str):
    """(Xtr, ytr, Xte, yte, baselines{label: median rel-L2}, npz yte check)."""
    if name == "turbulence":
        from examples.thewell_turbulence_study import load_trl2d
        (Xtr, ytr), (Xte, yte) = load_trl2d(800, 100, seed=0, horizon=8)
        npz = np.load(OUT / "thewell_turbulence_preds.npz")
    elif name == "helmholtz":
        from examples.thewell_helmholtz_study import load_helmholtz
        (Xtr, ytr), (Xte, yte) = load_helmholtz(800, 100, seed=0, horizon=12)
        npz = np.load(OUT / "thewell_helmholtz_preds.npz")
    else:
        raise ValueError(name)

    if not np.allclose(npz["yte"], yte, atol=1e-5):
        raise RuntimeError(f"{name}: reloaded test set differs from the "
                           "saved study artifacts — splits out of sync")
    baselines = {}
    for key in npz.files:
        if key.startswith("pred__"):
            label = key.removeprefix("pred__")
            baselines[label] = float(np.median(_rel_l2(npz[key], yte)))
    return Xtr, ytr, Xte, yte, baselines


def run_task(name: str):
    from surge.model import MODEL_REGISTRY

    Xtr, ytr, Xte, yte, baselines = load_task(name)
    nte = len(yte)
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(nte, -1)
    ytr_f = ytr.reshape(len(ytr), -1)
    print(f"[{name}] train {Xtr_f.shape} test {Xte_f.shape}  "
          f"baselines: { {k: round(v, 4) for k, v in baselines.items()} }")

    rows = []
    example = None
    for k in K_MODES:
        bx = pod_fit(Xtr_f, k)
        by = pod_fit(ytr_f, k)
        cx_tr = pod_transform(bx, Xtr_f).astype("float32")
        cy_tr = pod_transform(by, ytr_f).astype("float32")
        cx_te = pod_transform(bx, Xte_f).astype("float32")
        # how much of the target lives in k modes at all (recon ceiling)
        ceiling = float(np.median(_rel_l2(
            pod_inverse(by, pod_transform(by, yte.reshape(nte, -1))), yte)))

        for key, label, params in (
            ("sklearn.ridge", "POD+ridge", {}),
            ("pytorch.residual_mlp", "POD+resMLP",
             {"hidden_layers": [256, 256], "n_epochs": 200,
              "patience": 30, "patience_window": 5, "random_state": 0}),
        ):
            model = MODEL_REGISTRY.create(key, **params)
            mu, sd = cy_tr.mean(0), cy_tr.std(0) + 1e-8
            t0 = time.perf_counter()
            model.fit(cx_tr, (cy_tr - mu) / sd)
            coeffs = np.asarray(model.predict(cx_te)).reshape(nte, k) * sd + mu
            dt = time.perf_counter() - t0
            pred = pod_inverse(by, coeffs)
            med = float(np.median(_rel_l2(pred, yte)))
            rows.append({"k": k, "model": label, "rel_l2_median": med,
                         "recon_ceiling": ceiling, "runtime_s": dt})
            print(f"[{name}] k={k:3d} {label:11s} rel-L2 {med:.4f} "
                  f"(ceiling {ceiling:.4f}, {dt:.1f}s)")
            if example is None or med < example["rel"]:
                i = int(np.argsort(_rel_l2(pred, yte))[nte // 2])
                example = {"rel": med, "truth": yte[i],
                           "pred": pred[i].reshape(yte.shape[1:]),
                           "label": f"{label}, k={k}"}
    return rows, baselines, example


def figure(results, mode: str):
    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4),
                                 width_ratios=[1.15, 1.15, 1.0])
        for ax, name, title in ((axes[0], "turbulence",
                                 "(a) turbulent layer — Δt = 8"),
                                (axes[1], "helmholtz",
                                 "(b) Helmholtz — Δt = 12/50 cycle")):
            rows, baselines, _ = results[name]
            for j, label in enumerate(("POD+ridge", "POD+resMLP")):
                ks = [r["k"] for r in rows if r["model"] == label]
                vs = [r["rel_l2_median"] for r in rows if r["model"] == label]
                ax.plot(ks, vs, marker="o", ms=4, lw=1.7,
                        color=p["series"][j], label=label)
            ceil = {r["k"]: r["recon_ceiling"] for r in rows}
            ax.plot(sorted(ceil), [ceil[k] for k in sorted(ceil)],
                    ls=(0, (2, 2)), lw=1.1, color=p["muted"],
                    label="POD recon ceiling")
            styles = {"Persistence": p["series"][2], "FNO-2D": p["ink2"],
                      "U-Net": p["axis"]}
            for lbl, col in styles.items():
                if lbl in baselines:
                    ax.axhline(baselines[lbl], color=col, lw=1.1,
                               ls=(0, (5, 3)))
                    ax.annotate(f"{lbl} {baselines[lbl]:.3f}",
                                xy=(0.02, baselines[lbl]),
                                xycoords=("axes fraction", "data"),
                                xytext=(0, 3), textcoords="offset points",
                                fontsize=6.5, color=col)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(K_MODES))
            ax.set_xticklabels([str(k) for k in K_MODES])
            ax.set_xlabel("POD modes k")
            ax.set_ylabel("median rel-L2")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")

        ax = axes[2]
        _, _, ex = results["helmholtz"]
        img = np.concatenate([ex["truth"].T, ex["pred"].T], axis=0)
        v = float(np.abs(ex["truth"]).max())
        ax.imshow(img, cmap=diverging_cmap(mode),
                  norm=CenteredNorm(halfrange=v), origin="lower",
                  aspect="auto")
        ax.axhline(ex["truth"].T.shape[0] - 0.5, color=p["ink"], lw=1.0)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(f"(c) truth (bottom) vs {ex['label']}", fontsize=8.5)

        fig.suptitle(
            "POD reduced-order surrogates — tabular models through k modes "
            "vs neural operators (same splits, operators not retrained)",
            fontsize=10.5, fontweight="bold")
        return fig


def main() -> None:
    results = {}
    for name in ("turbulence", "helmholtz"):
        rows, baselines, example = run_task(name)
        results[name] = (rows, baselines, example)

    for mode in ("light", "dark"):
        fig = figure(results, mode)
        for path in save_figure(fig, OUT / f"thewell_pod_{mode}"):
            print("wrote", path)
        plt.close(fig)

    summary = {name: {"rows": rows, "baselines": baselines}
               for name, (rows, baselines, _) in results.items()}
    (OUT / "thewell_pod_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
