#!/usr/bin/env python3
"""Gallery of the SURGE visual system (surge.viz.theme).

Every figure is driven by machine-readable SURGE artifacts — run
directories and benchmark result.json files — never hand-encoded numbers.
Missing artifacts skip that figure with a note instead of failing.

Usage (from repo root):
    python examples/viz_theme_gallery.py \
        [--run runs/diabetes_rf] [--hpo-run runs/qlknn_multi_hpo] \
        [--benchmark tabular.california_housing] [--modes light dark] \
        [--out examples/viz_gallery_output]

Produces (per mode): parity, training_curves, hpo_convergence,
leaderboard, classification — as deterministic PNG/SVG/PDF.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surge.viz.theme import fmt_metric, save_figure, surge_theme

# ------------------------------------------------------------------ parity

def parity_figure(run_dir: Path, mode: str):
    import pandas as pd

    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_key = next(iter(metrics))
    df = pd.read_parquet(run_dir / "predictions" / f"{model_key}_test.parquet")
    y_true = df[next(c for c in df.columns if c.startswith("y_true"))].to_numpy()
    y_pred = df[next(c for c in df.columns if c.startswith("y_pred"))].to_numpy()
    m = metrics[model_key]["test"]

    with surge_theme(mode) as p:
        fig, (ax, axr) = plt.subplots(
            1, 2, figsize=(6.6, 3.0), width_ratios=[1.0, 0.85])
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, color=p["axis"], lw=0.8, zorder=1)
        ax.scatter(y_true, y_pred, s=14, color=p["series"][0],
                   alpha=0.55, linewidths=0, zorder=2)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        ax.set_xlabel("observed"); ax.set_ylabel("predicted")
        ax.set_title(f"Parity — {model_key} (test)")
        ax.text(0.03, 0.94,
                f"R² {fmt_metric(m['r2'])}   RMSE {fmt_metric(m['rmse'], 'rmse')}"
                f"   MAE {fmt_metric(m['mae'], 'rmse')}",
                transform=ax.transAxes, fontsize=8, color=p["ink2"], va="top")

        resid = y_pred - y_true
        axr.hist(resid, bins=21, color=p["series"][0], alpha=0.85)
        axr.axvline(0.0, color=p["axis"], lw=0.8)
        axr.set_xlabel("residual (pred − obs)"); axr.set_ylabel("count")
        axr.set_title("Residuals")
        axr.text(0.03, 0.94, f"mean {resid.mean():+.2f}   σ {resid.std():.2f}",
                 transform=axr.transAxes, fontsize=8, color=p["ink2"], va="top")
        return fig


# --------------------------------------------------------- training curves

def training_figure(hpo_run: Path, mode: str):
    logs = sorted(hpo_run.glob("training_log_*.jsonl"))
    if not logs:
        return None
    rows = [json.loads(line) for line in logs[0].read_text().splitlines() if line]
    name = logs[0].stem.replace("training_log_", "")
    epochs = [r["epoch"] for r in rows]
    train = [r.get("train_loss") for r in rows]
    val = [r.get("val_loss") for r in rows]

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.plot(epochs, train, label="train", color=p["series"][0])
        if any(v is not None for v in val):
            ax.plot(epochs, val, label="validation", color=p["series"][1])
            best = int(np.nanargmin(np.array(val, dtype=float)))
            ax.scatter([epochs[best]], [val[best]], s=26, zorder=3,
                       color=p["series"][1])
            ax.annotate(f"best {fmt_metric(val[best], 'loss')}",
                        (epochs[best], val[best]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color=p["ink2"])
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
        ax.set_title(f"Training — {name}")
        ax.legend(loc="upper right")
        return fig


# --------------------------------------------------------- HPO convergence

def hpo_figure(hpo_run: Path, mode: str):
    files = sorted((hpo_run / "hpo").glob("*_hpo.json"))
    if not files:
        return None
    d = json.loads(files[0].read_text())
    trials = d.get("trials", [])
    if not trials:
        return None
    values = [t["value"] for t in trials]
    numbers = [t["number"] for t in trials]
    maximize = d.get("direction", "maximize") == "maximize"
    running = np.maximum.accumulate(values) if maximize \
        else np.minimum.accumulate(values)
    metric = d.get("metric", "objective")
    best = d.get("best_trial", {})

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.scatter(numbers, values, s=22, color=p["series"][0], alpha=0.6,
                   linewidths=0, label="trial")
        ax.plot(numbers, running, color=p["series"][1], drawstyle="steps-post",
                label="best so far")
        if best:
            ax.annotate(f"best {fmt_metric(best.get('value'))} "
                        f"(trial {best.get('number')})",
                        (best.get("number"), best.get("value")),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=8, color=p["ink2"])
        ax.set_xlabel("trial"); ax.set_ylabel(metric)
        ax.set_title(f"HPO convergence — {files[0].stem.replace('_hpo', '')}")
        ax.legend(loc="lower right")
        return fig


# ------------------------------------------------------------- leaderboard

def leaderboard_figure(reports_dir: Path, benchmark_key: str, mode: str,
                       threshold: float | None = None):
    rows = defaultdict(lambda: {"scores": [], "runtimes": []})
    for rj in sorted((reports_dir / benchmark_key).glob("*/result.json")):
        d = json.loads(rj.read_text())
        model = d.get("model_key")
        met = d.get("metrics", {})
        score = met.get("test_r2", met.get("test_accuracy"))
        if model is None or score is None:
            continue
        rows[model]["scores"].append(float(score))
        if met.get("runtime_s") is not None:
            rows[model]["runtimes"].append(float(met["runtime_s"]))
    stats = sorted(
        ({"model": k,
          "mean": statistics.fmean(v["scores"]),
          "std": statistics.stdev(v["scores"]) if len(v["scores"]) > 1 else 0.0,
          "n": len(v["scores"]),
          "runtime": statistics.fmean(v["runtimes"]) if v["runtimes"] else None}
         for k, v in rows.items()),
        key=lambda r: r["mean"], reverse=True)[:10]
    if not stats:
        return None

    with surge_theme(mode) as p:
        fig, (ax, axt) = plt.subplots(
            1, 2, figsize=(7.2, 0.42 * len(stats) + 1.3),
            width_ratios=[1.0, 0.38], sharey=True)
        ypos = np.arange(len(stats))[::-1]
        for y, r in zip(ypos, stats):
            ax.barh(y, r["mean"], height=0.62, color=p["series"][0], alpha=0.9)
            if r["std"] > 0:
                ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none",
                            ecolor=p["ink2"], elinewidth=1.1, capsize=2.5)
            label = fmt_metric(r["mean"])
            if r["std"] > 0:
                label += f' ± {r["std"]:.3f}'
            label += f'  (n={r["n"]})'
            ax.text(0.005, y, " " + r["model"], va="center", fontsize=8,
                    color=p["ink"], zorder=3)
            ax.text(min(r["mean"] + r["std"] + 0.015, 1.0), y, label,
                    va="center", fontsize=7, color=p["ink2"])
            if r["runtime"] is not None:
                axt.barh(y, r["runtime"], height=0.62,
                         color=p["series"][1], alpha=0.9)
                axt.text(r["runtime"], y,
                         " " + fmt_metric(r["runtime"], "runtime"),
                         va="center", fontsize=7, color=p["ink2"])
        if threshold is not None:
            ax.axvline(threshold, color=p["serious"], lw=1.0,
                       ls=(0, (4, 3)), zorder=1)
            ax.text(threshold, -0.62, f"threshold {threshold} ",
                    color=p["serious"], fontsize=7, va="top", ha="right")
        ax.set_yticks([]); ax.set_xlim(0, 1.02)
        ax.set_xlabel("test R²  (mean ± std over runs)")
        ax.set_title(f"Leaderboard — {benchmark_key}")
        axt.set_title("runtime", fontsize=9); axt.set_xlabel("seconds")
        return fig


# ---------------------------------------------------------- classification

def classification_figure(mode: str, seed: int = 42):
    """ROC / PR / confusion / reliability, themed.

    Demo trains a small classifier through the SURGE registry on a public
    sklearn dataset (no cached artifacts carry per-sample probabilities yet;
    once runs do, this reads them instead).
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
    from sklearn.model_selection import train_test_split

    from surge.model import MODEL_REGISTRY

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    adapter = MODEL_REGISTRY.create("sklearn.random_forest_classifier",
                                    random_state=seed)
    adapter.fit(X_tr, y_tr)
    proba = adapter.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    cm = confusion_matrix(y_te, pred)
    bins = np.linspace(0, 1, 11)
    binned = np.digitize(proba, bins) - 1
    conf, acc = [], []
    for b in range(10):
        mask = binned == b
        if mask.sum() >= 5:
            conf.append(proba[mask].mean())
            acc.append(y_te[mask].mean())

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
        ax = axes[0]
        ax.plot(fpr, tpr, color=p["series"][0])
        ax.plot([0, 1], [0, 1], color=p["axis"], lw=0.8)
        ax.set_title(f"ROC  (AUC {auc(fpr, tpr):.3f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")

        ax = axes[1]
        ax.plot(rec, prec, color=p["series"][0])
        ax.set_title("Precision–Recall")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")

        ax = axes[2]
        im = ax.imshow(cm, cmap=None, interpolation="nearest")
        from surge.viz.theme import sequential_cmap
        im.set_cmap(sequential_cmap(mode))
        for (i, j), v in np.ndenumerate(cm):
            frac = cm[i, j] / cm.max() if cm.max() else 0
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color=p["surface"] if frac > 0.55 else p["ink"])
        ax.set_title("Confusion"); ax.grid(False)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")

        ax = axes[3]
        ax.plot([0, 1], [0, 1], color=p["axis"], lw=0.8)
        ax.plot(conf, acc, marker="o", color=p["series"][0])
        ax.set_title("Reliability")
        ax.set_xlabel("confidence"); ax.set_ylabel("observed accuracy")
        return fig


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(_REPO / "runs" / "diabetes_rf"))
    ap.add_argument("--hpo-run", default=str(_REPO / "runs" / "qlknn_multi_hpo"))
    ap.add_argument("--benchmark", default="tabular.california_housing")
    ap.add_argument("--reports", default=str(_REPO / "benchmark_reports"))
    ap.add_argument("--modes", nargs="+", default=["light", "dark"])
    ap.add_argument("--out", default=str(_REPO / "examples" / "viz_gallery_output"))
    args = ap.parse_args()
    out = Path(args.out)

    builders = {
        "parity": lambda m: parity_figure(Path(args.run), m),
        "training_curves": lambda m: training_figure(Path(args.hpo_run), m),
        "hpo_convergence": lambda m: hpo_figure(Path(args.hpo_run), m),
        "leaderboard": lambda m: leaderboard_figure(
            Path(args.reports), args.benchmark, m, threshold=0.75),
        "classification": classification_figure,
    }
    for mode in args.modes:
        for name, build in builders.items():
            try:
                fig = build(mode)
            except FileNotFoundError as exc:
                print(f"[skip] {name}_{mode}: missing artifact ({exc})")
                continue
            if fig is None:
                print(f"[skip] {name}_{mode}: no source data")
                continue
            for path in save_figure(fig, out / f"{name}_{mode}"):
                print("wrote", path)
            plt.close(fig)


if __name__ == "__main__":
    main()
