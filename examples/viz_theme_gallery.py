#!/usr/bin/env python3
"""Gallery of the SURGE visual system (surge.viz.theme) — publication grade.

Every figure is driven by machine-readable SURGE artifacts — run
directories, benchmark result.json files, cached benchmark datasets —
never hand-encoded numbers. Missing artifacts skip that figure with a
note instead of failing.

Usage (from repo root):
    python examples/viz_theme_gallery.py \
        [--run runs/diabetes_rf] [--hpo-run runs/qlknn_multi_hpo] \
        [--benchmark tabular.california_housing] [--modes light dark] \
        [--only parity field_operator ...] [--out examples/viz_gallery_output]

Figures (per mode, deterministic PNG/SVG/PDF):
    parity            density parity + residual distribution
    training_curves   train/val loss with generalisation gap + best epoch
    hpo_convergence   trials coloured by learning rate + running best
    leaderboard       score ± std vs published threshold + runtime panel
    classification    ROC / PR / confusion / reliability (with ECE)
    field_operator    Burgers' operator learning: truth · prediction · error
    uncertainty       GP surrogate with 95% credible band and coverage
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
from matplotlib.colors import CenteredNorm, LogNorm

from surge.viz.theme import (
    diverging_cmap,
    fmt_metric,
    save_figure,
    sequential_cmap,
    surge_theme,
)


def _stat_chip(ax, text, p, loc=(0.03, 0.95)):
    """Metrics annotation in a subtle rounded chip (consistent across figs)."""
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, fontsize=8,
            color=p["ink2"], va="top",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": p["page"],
                  "edgecolor": p["grid"], "linewidth": 0.7})


# ------------------------------------------------------------------ parity

def parity_figure(run_dir: Path, mode: str):
    import pandas as pd
    from scipy.stats import gaussian_kde

    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_key = next(iter(metrics))
    df = pd.read_parquet(run_dir / "predictions" / f"{model_key}_test.parquet")
    y_true = df[next(c for c in df.columns if c.startswith("y_true"))].to_numpy()
    y_pred = df[next(c for c in df.columns if c.startswith("y_pred"))].to_numpy()
    m = metrics[model_key]["test"]

    with surge_theme(mode) as p:
        fig, (ax, axr) = plt.subplots(
            1, 2, figsize=(7.0, 3.2), width_ratios=[1.0, 0.8])

        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.06 * (hi - lo)
        lims = (lo - pad, hi + pad)

        # point density via KDE -> sequential ramp (magnitude = crowding)
        xy = np.vstack([y_true, y_pred])
        dens = gaussian_kde(xy)(xy)
        order = dens.argsort()  # draw densest points on top
        ax.plot(lims, lims, color=p["axis"], lw=0.9, zorder=1)
        sc = ax.scatter(y_true[order], y_pred[order], c=dens[order],
                        cmap=sequential_cmap(mode), s=22, alpha=0.9,
                        linewidths=0, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_ticks([]); cb.set_label("density", fontsize=7, color=p["muted"])
        cb.outline.set_visible(False)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        ax.set_xlabel("observed"); ax.set_ylabel("predicted")
        ax.set_title(f"Parity — {model_key} (test)")
        _stat_chip(ax, f"R² {fmt_metric(m['r2'])}    "
                       f"RMSE {fmt_metric(m['rmse'], 'rmse')}    "
                       f"MAE {fmt_metric(m['mae'], 'rmse')}", p)

        resid = y_pred - y_true
        axr.hist(resid, bins=21, color=p["series"][0], alpha=0.55,
                 density=True, label="residuals")
        kde_x = np.linspace(resid.min(), resid.max(), 200)
        axr.plot(kde_x, gaussian_kde(resid)(kde_x), color=p["series"][0],
                 lw=1.8, label="KDE")
        axr.axvline(0.0, color=p["axis"], lw=0.9)
        axr.axvline(float(resid.mean()), color=p["series"][1], lw=1.2,
                    ls=(0, (4, 3)), label="mean")
        axr.set_xlabel("residual (pred − obs)"); axr.set_ylabel("density")
        axr.set_title("Residual distribution")
        _stat_chip(axr, f"mean {resid.mean():+.2f}    σ {resid.std():.2f}", p)
        axr.legend(loc="upper right", fontsize=7)
        return fig


# --------------------------------------------------------- training curves

def training_figure(hpo_run: Path, mode: str):
    logs = sorted(hpo_run.glob("training_log_*.jsonl"))
    if not logs:
        return None
    rows = [json.loads(line) for line in logs[0].read_text().splitlines() if line]
    name = logs[0].stem.replace("training_log_", "")
    epochs = np.array([r["epoch"] for r in rows])
    train = np.array([r.get("train_loss", np.nan) for r in rows], dtype=float)
    val = np.array([r.get("val_loss", np.nan) for r in rows], dtype=float)

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(epochs, train, label="train", color=p["series"][0], lw=1.8)
        if np.isfinite(val).any():
            ax.plot(epochs, val, label="validation", color=p["series"][1], lw=1.8)
            # generalisation gap, shaded
            ax.fill_between(epochs, train, val, where=val >= train,
                            color=p["series"][1], alpha=0.10, linewidth=0,
                            label="generalisation gap")
            best = int(np.nanargmin(val))
            ax.axvline(epochs[best], color=p["muted"], lw=0.9, ls=(0, (2, 3)))
            ax.scatter([epochs[best]], [val[best]], s=42, zorder=4,
                       color=p["series"][1], edgecolor=p["surface"],
                       linewidth=1.2)
            ax.annotate(f"best epoch {epochs[best]}\nval {fmt_metric(val[best], 'loss')}",
                        (epochs[best], val[best]),
                        textcoords="offset points", xytext=(-10, -22),
                        ha="right", fontsize=7.5, color=p["ink2"])
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(f"Training — {name}")
        ax.legend(loc="upper right", fontsize=7.5)
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
    numbers = np.array([t["number"] for t in trials])
    values = np.array([t["value"] for t in trials], dtype=float)
    lrs = np.array([t.get("params", {}).get("learning_rate", np.nan)
                    for t in trials], dtype=float)
    maximize = d.get("direction", "maximize") == "maximize"
    running = np.maximum.accumulate(values) if maximize \
        else np.minimum.accumulate(values)
    metric = d.get("metric", "objective")
    best = d.get("best_trial", {})

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(numbers, running, color=p["ink2"], lw=1.4,
                drawstyle="steps-post", label="best so far", zorder=2)
        if np.isfinite(lrs).all() and (lrs > 0).all():
            sc = ax.scatter(numbers, values, c=lrs, norm=LogNorm(),
                            cmap=sequential_cmap(mode), s=48, zorder=3,
                            edgecolor=p["surface"], linewidth=0.8,
                            label="trial")
            cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label("learning rate", fontsize=7.5, color=p["muted"])
            cb.ax.tick_params(labelsize=7)
            cb.outline.set_visible(False)
        else:
            ax.scatter(numbers, values, color=p["series"][0], s=42,
                       zorder=3, label="trial")
        if best:
            ax.annotate(f"best {fmt_metric(best.get('value'))} (trial {best.get('number')})",
                        (best.get("number"), best.get("value")),
                        textcoords="offset points", xytext=(-8, -16),
                        ha="right", fontsize=7.5, color=p["ink2"])
        ax.set_xlabel("trial"); ax.set_ylabel(metric)
        ax.set_title(f"HPO — {files[0].stem.replace('_hpo', '')}")
        ax.legend(loc="lower right", fontsize=7.5)
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
            1, 2, figsize=(7.6, 0.44 * len(stats) + 1.4),
            width_ratios=[1.0, 0.36], sharey=True)
        ypos = np.arange(len(stats))[::-1]
        if threshold is not None:  # sub-threshold zone, tinted
            ax.axvspan(0, threshold, color=p["critical"], alpha=0.05, zorder=0)
            ax.axvline(threshold, color=p["serious"], lw=1.0,
                       ls=(0, (4, 3)), zorder=1)
            ax.text(threshold, -0.66, f"threshold {threshold} ",
                    color=p["serious"], fontsize=7, va="top", ha="right")
        for y, r in zip(ypos, stats):
            ax.barh(y, r["mean"], height=0.62, color=p["series"][0], alpha=0.92)
            if r["std"] > 0:
                ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none",
                            ecolor=p["ink2"], elinewidth=1.1, capsize=2.5)
            label = fmt_metric(r["mean"])
            if r["std"] > 0:
                label += f' ± {r["std"]:.3f}'
            label += f'  (n={r["n"]})'
            ax.text(0.006, y, " " + r["model"], va="center", fontsize=8,
                    color=p["ink"], zorder=3)
            ax.text(min(r["mean"] + r["std"] + 0.015, 1.0), y, label,
                    va="center", fontsize=7, color=p["ink2"])
            if r["runtime"] is not None:
                axt.barh(y, max(r["runtime"], 1e-2), height=0.62,
                         color=p["series"][1], alpha=0.92)
                axt.text(max(r["runtime"], 1e-2), y,
                         " " + fmt_metric(r["runtime"], "runtime"),
                         va="center", fontsize=7, color=p["ink2"])
        ax.set_yticks([]); ax.set_xlim(0, 1.02)
        ax.set_xlabel("test R²  (mean ± std over runs)")
        ax.set_title(f"Leaderboard — {benchmark_key}")
        axt.set_xscale("log")
        axt.set_title("runtime", fontsize=9)
        axt.set_xlabel("seconds (log)")
        return fig


# ---------------------------------------------------------- classification

def classification_figure(mode: str, seed: int = 42):
    """ROC / PR / confusion / reliability with ECE, themed."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import (
        auc,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
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
    ap = average_precision_score(y_te, proba)
    cm = confusion_matrix(y_te, pred)

    edges = np.linspace(0, 1, 11)
    binned = np.digitize(proba, edges) - 1
    conf, acc, weight = [], [], []
    for b in range(10):
        mask = binned == b
        if mask.sum() >= 5:
            conf.append(proba[mask].mean())
            acc.append(y_te[mask].mean())
            weight.append(mask.mean())
    ece = float(np.sum(np.array(weight) * np.abs(np.array(conf) - np.array(acc))))

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.75))

        ax = axes[0]
        ax.plot(fpr, tpr, color=p["series"][0], lw=1.8)
        ax.fill_between(fpr, 0, tpr, color=p["series"][0], alpha=0.12)
        ax.plot([0, 1], [0, 1], color=p["axis"], lw=0.9)
        ax.set_title(f"ROC · AUC {auc(fpr, tpr):.3f}")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")

        ax = axes[1]
        ax.plot(rec, prec, color=p["series"][0], lw=1.8)
        ax.fill_between(rec, 0, prec, color=p["series"][0], alpha=0.12)
        ax.axhline(float(y_te.mean()), color=p["axis"], lw=0.9, ls=(0, (3, 3)))
        ax.set_ylim(0, 1.03)
        ax.set_title(f"Precision–Recall · AP {ap:.3f}")
        ax.set_xlabel("recall"); ax.set_ylabel("precision")

        ax = axes[2]
        im = ax.imshow(cm, cmap=sequential_cmap(mode), interpolation="nearest")
        for (i, j), v in np.ndenumerate(cm):
            frac = cm[i, j] / cm.max() if cm.max() else 0
            share = v / cm.sum()
            ax.text(j, i, f"{v}\n{share:.0%}", ha="center", va="center",
                    fontsize=8.5, linespacing=1.4,
                    color=p["surface"] if frac > 0.55 else p["ink"])
        ax.set_title("Confusion"); ax.grid(False)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        im.colorbar = None

        ax = axes[3]
        ax.plot([0, 1], [0, 1], color=p["axis"], lw=0.9)
        ax.vlines(conf, acc, conf, color=p["critical"], lw=2.2, alpha=0.65,
                  label="gap")
        ax.plot(conf, acc, marker="o", color=p["series"][0], lw=1.6,
                markersize=5, label="model")
        ax.set_title(f"Reliability · ECE {ece:.3f}")
        ax.set_xlabel("confidence"); ax.set_ylabel("observed accuracy")
        ax.legend(loc="upper left", fontsize=7)
        return fig


# --------------------------------------------------- field / operator demo

def field_operator_figure(mode: str, seed: int = 0):
    """Operator learning diagnostic on cached Burgers' data.

    Trains a residual MLP (SURGE registry) on u(x,0) -> u(x,T), then shows
    the canonical field triptych: truth · prediction · signed error, plus a
    worst/median sample overlay and the per-sample error distribution.
    """
    from surge.model import MODEL_REGISTRY

    npz = _REPO / "data" / "datasets" / "benchmarks" / "pde" / "burgers_1d.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    X, Y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = len(X) // 5
    tr, te = idx[n_test:], idx[:n_test]

    try:
        model = MODEL_REGISTRY.create("pytorch.residual_mlp", n_epochs=60,
                                      random_state=seed)
        model.fit(X[tr], Y[tr])
        pred = np.asarray(model.predict(X[te]))
        model_name = "pytorch.residual_mlp"
    except (KeyError, ImportError):  # torch adapter absent -> tree fallback
        model = MODEL_REGISTRY.create("sklearn.random_forest", n_estimators=60,
                                      random_state=seed)
        model.fit(X[tr], Y[tr])
        pred = np.asarray(model.predict(X[te]))
        model_name = "sklearn.random_forest"

    truth = Y[te]
    err = pred - truth
    rel_l2 = np.linalg.norm(err, axis=1) / np.linalg.norm(truth, axis=1)
    rank = np.argsort(rel_l2)
    picks = [(rank[0], "best"), (rank[len(rank) // 2], "median"),
             (rank[-1], "worst")]
    xgrid = np.arange(truth.shape[1])

    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(9.8, 5.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

        # row 1 — best / median / worst sample overlays
        for k, (i, tag) in enumerate(picks):
            ax = fig.add_subplot(gs[0, k])
            ax.plot(xgrid, X[te][i], color=p["muted"], lw=1.1,
                    ls=(0, (3, 3)), label="u(x, 0)")
            ax.plot(xgrid, truth[i], color=p["series"][0], lw=2.0,
                    label="truth u(x, T)")
            ax.plot(xgrid, pred[i], color=p["series"][1], lw=1.7,
                    ls=(0, (5, 2)), label="prediction")
            ax.fill_between(xgrid, truth[i], pred[i], color=p["series"][7],
                            alpha=0.20, linewidth=0)
            ax.set_title(f"{tag} · rel-L2 {rel_l2[i]:.3f}", fontsize=9)
            ax.set_xlabel("x index")
            if k == 0:
                ax.set_ylabel("u")
                ax.legend(loc="upper right", fontsize=6.5)

        # row 2 — signed error field over all test samples (sorted by error)
        ax = fig.add_subplot(gs[1, :2])
        vmax = float(np.quantile(np.abs(err), 0.98))
        im = ax.imshow(err[rank], aspect="auto", cmap=diverging_cmap(mode),
                       norm=CenteredNorm(halfrange=vmax),
                       interpolation="nearest")
        ax.set_xlabel("x index")
        ax.set_ylabel("test samples · sorted best → worst")
        ax.set_title("signed error  (prediction − truth)", fontsize=9)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)

        axh = fig.add_subplot(gs[1, 2])
        axh.hist(rel_l2, bins=30, color=p["series"][0], alpha=0.88)
        axh.axvline(float(np.median(rel_l2)), color=p["series"][1], lw=1.4,
                    ls=(0, (4, 3)))
        axh.set_xlabel("per-sample rel-L2"); axh.set_ylabel("count")
        axh.set_title(f"error distribution · median {np.median(rel_l2):.3f}",
                      fontsize=9)

        fig.suptitle(
            f"Burgers' 1D operator learning · {model_name} · "
            f"{len(tr):,} train / {len(te):,} test",
            fontsize=11, fontweight="bold")
        return fig


# ------------------------------------------------------------- uncertainty

def uncertainty_figure(mode: str, seed: int = 7):
    """GP surrogate with 95% credible band — canonical UQ diagnostic."""
    from surge.model import MODEL_REGISTRY

    rng = np.random.default_rng(seed)
    def f(x):
        return np.sin(6.0 * x) + 0.35 * np.cos(14.0 * x)
    X_tr = np.sort(rng.uniform(0.02, 0.98, 26))[:, None]
    # leave a gap so the band visibly widens where data is missing
    X_tr = X_tr[(X_tr[:, 0] < 0.45) | (X_tr[:, 0] > 0.72)]
    y_tr = f(X_tr[:, 0]) + 0.07 * rng.standard_normal(len(X_tr))
    Xg = np.linspace(0, 1, 400)[:, None]

    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    gp = MODEL_REGISTRY.create(
        "sklearn.gpr",
        kernel=(ConstantKernel(1.0, (0.05, 20.0))
                * RBF(0.12, (0.03, 0.6))
                + WhiteKernel(5e-3, (1e-5, 5e-2))),
        normalize_y=True, n_restarts_optimizer=6, random_state=seed)
    gp.fit(X_tr, y_tr)
    mean, std = gp.predict_with_uncertainty(Xg)
    mean, std = np.asarray(mean).ravel(), np.asarray(std).ravel()
    lo, hi = mean - 1.96 * std, mean + 1.96 * std
    inside = float(np.mean((f(Xg[:, 0]) >= lo) & (f(Xg[:, 0]) <= hi)))

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.fill_between(Xg[:, 0], lo, hi, color=p["series"][0], alpha=0.16,
                        linewidth=0, label="95% credible band")
        ax.fill_between(Xg[:, 0], mean - std, mean + std, color=p["series"][0],
                        alpha=0.16, linewidth=0)
        ax.plot(Xg[:, 0], mean, color=p["series"][0], lw=2.0,
                label="posterior mean")
        ax.plot(Xg[:, 0], f(Xg[:, 0]), color=p["series"][1], lw=1.4,
                ls=(0, (4, 3)), label="true function")
        ax.scatter(X_tr[:, 0], y_tr, s=26, color=p["ink"], zorder=4,
                   edgecolor=p["surface"], linewidth=1.0,
                   label="training data")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("GP surrogate — sklearn.gpr")
        _stat_chip(ax, f"band coverage of truth  {inside:.0%}", p,
                   loc=(0.03, 0.13))
        ax.legend(loc="upper right", fontsize=7.5)
        return fig


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(_REPO / "runs" / "diabetes_rf"))
    ap.add_argument("--hpo-run", default=str(_REPO / "runs" / "qlknn_multi_hpo"))
    ap.add_argument("--benchmark", default="tabular.california_housing")
    ap.add_argument("--reports", default=str(_REPO / "benchmark_reports"))
    ap.add_argument("--modes", nargs="+", default=["light", "dark"])
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of figure names to build")
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
        "field_operator": field_operator_figure,
        "uncertainty": uncertainty_figure,
    }
    if args.only:
        builders = {k: v for k, v in builders.items() if k in args.only}
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
