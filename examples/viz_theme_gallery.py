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
    density_cmap,
    diverging_cmap,
    fmt_metric,
    save_figure,
    sequential_cmap,
    surge_theme,
)


def _is_loss_like(metric: str) -> bool:
    return any(tag in metric.lower()
               for tag in ("loss", "mse", "rmse", "nrmse", "mae", "error"))


def _stat_chip(ax, text, p, loc=(0.03, 0.95)):
    """Metrics annotation in a subtle rounded chip (consistent across figs)."""
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, fontsize=8,
            color=p["ink2"], va="top",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": p["page"],
                  "edgecolor": p["grid"], "linewidth": 0.7})


# ------------------------------------------------------------------ parity

def _load_split(run_dir: Path, model_key: str, split: str):
    import pandas as pd
    df = pd.read_parquet(run_dir / "predictions" / f"{model_key}_{split}.parquet")
    y_true = df[next(c for c in df.columns if c.startswith("y_true"))].to_numpy()
    y_pred = df[next(c for c in df.columns if c.startswith("y_pred"))].to_numpy()
    return y_true, y_pred


def parity_figure(run_dir: Path, mode: str, units: str = "a.u."):
    """Publication parity: 2D histogram density, reversed plasma colormap,
    log-scaled counts — the SURGE signature style (Sánchez-Villar et al.,
    Nucl. Fusion): (a) training / (b) test panels, dashed identity, R² box,
    plus (c) test residual distribution.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score

    metrics = json.loads((run_dir / "metrics.json").read_text())
    model_key = next(iter(metrics))
    splits = {}
    for split in ("train", "test"):
        try:
            splits[split] = _load_split(run_dir, model_key, split)
        except FileNotFoundError:
            pass
    if "test" not in splits:
        return None

    all_vals = np.concatenate([v for pair in splits.values() for v in pair])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)
    bins = np.linspace(*lims, 56)
    peak = max(
        np.histogram2d(t, q, bins=[bins, bins])[0].max()
        for t, q in splits.values())

    with surge_theme(mode) as p:
        cmap = density_cmap(mode)
        under = cmap.get_under()
        fig, axes = plt.subplots(
            1, len(splits) + 1, figsize=(3.35 * (len(splits) + 1) + 0.7, 3.5),
            width_ratios=[1.0] * len(splits) + [0.78])

        letters = "abc"
        norm = LogNorm(vmin=1, vmax=peak)
        im = None
        for k, (split, (y_t, y_q)) in enumerate(splits.items()):
            ax = axes[k]
            ax.set_facecolor(under)
            im = ax.hist2d(y_t, y_q, bins=[bins, bins], cmap=cmap,
                           norm=norm, cmin=1)[3]
            ax.plot(lims, lims, color=p["ink2"], lw=1.2, ls=(0, (5, 3)),
                    zorder=3)
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
            ax.grid(color=p["grid"], linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)
            ax.set_xlabel(f"Ground truth [{units}]")
            if k == 0:
                ax.set_ylabel(f"Prediction [{units}]")
            ax.set_title(f"({letters[k]}) {model_key} — {split}", fontsize=9.5)
            r2 = r2_score(y_t, y_q)
            ax.text(0.05, 0.94, f"R² = {r2:.3f}", transform=ax.transAxes,
                    fontsize=9, va="top", color=p["ink"],
                    bbox={"boxstyle": "round,pad=0.35",
                          "facecolor": p["surface"],
                          "edgecolor": p["ink2"], "linewidth": 0.8})
        cb = fig.colorbar(im, ax=list(axes[:len(splits)]), fraction=0.04,
                          pad=0.015)
        cb.set_label("counts (log)", fontsize=8, color=p["ink2"])
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)

        # (c) residual distribution — test split
        y_t, y_q = splits["test"]
        resid = y_q - y_t
        axr = axes[-1]
        axr.hist(resid, bins=41, color=p["series"][0], alpha=0.55,
                 density=True, label="residuals")
        kde_x = np.linspace(resid.min(), resid.max(), 240)
        axr.plot(kde_x, gaussian_kde(resid)(kde_x), color=p["series"][0],
                 lw=1.8, label="KDE")
        axr.axvline(0.0, color=p["axis"], lw=0.9)
        axr.axvline(float(resid.mean()), color=p["series"][1], lw=1.2,
                    ls=(0, (4, 3)), label="mean")
        axr.set_xlabel(f"Residual [{units}]"); axr.set_ylabel("density")
        axr.set_title(f"({letters[len(splits)]}) residuals — test",
                      fontsize=9.5)
        _stat_chip(axr, f"mean {resid.mean():+.3g}   σ {resid.std():.3g}", p)
        axr.legend(loc="upper right", fontsize=6.5)
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

def hpo_figure(hpo_run: Path, mode: str, reference: float | None = None):
    """HPO history, publication style (Sánchez-Villar et al.): one series
    per optimizer/model — thin solid per-trial trace, dashed running best,
    gold-edged star at the best trial with a labelled score box, and an
    optional dashed black reference line.
    """
    files = sorted((hpo_run / "hpo").glob("*_hpo.json"))
    series = []
    for f in files[:4]:
        d = json.loads(f.read_text())
        trials = d.get("trials", [])
        if not trials:
            continue
        values = np.array([t["value"] for t in trials], dtype=float)
        maximize = d.get("direction", "maximize") == "maximize"
        running = (np.maximum if maximize else np.minimum).accumulate(values)
        series.append({
            "name": f.stem.replace("_hpo", ""),
            "numbers": np.array([t["number"] for t in trials]),
            "values": values,
            "running": running,
            "metric": d.get("metric", "objective"),
            "best": d.get("best_trial", {}),
        })
    if not series:
        return None
    metric = series[0]["metric"]
    star_face = "#FFD700"  # gold star fill, edge in series color

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        for i, s in enumerate(series):
            c = p["series"][i]
            ax.plot(s["numbers"], s["values"], color=c, lw=1.1, alpha=0.85,
                    label=s["name"], zorder=2)
            ax.plot(s["numbers"], s["running"], color=c, lw=1.9,
                    ls=(0, (5, 2)), drawstyle="steps-post", zorder=3)
            best = s["best"]
            if best:
                bi, bv = best.get("number"), best.get("value")
                ax.scatter([bi], [bv], marker="*", s=430, zorder=5,
                           facecolor=star_face, edgecolor=c, linewidth=1.6)
                ax.annotate(f"{metric} = {fmt_metric(bv)}",
                            (bi, bv), textcoords="offset points",
                            xytext=(-8, 12), ha="right", fontsize=7.5,
                            color=p["ink"],
                            bbox={"boxstyle": "round,pad=0.3",
                                  "facecolor": p["surface"],
                                  "edgecolor": c, "linewidth": 1.0})
        if reference is not None:
            ax.axhline(reference, color=p["ink"], lw=1.4, ls=(0, (6, 3)),
                       zorder=1, label=f"reference ({reference})")
        ax.plot([], [], color=p["ink2"], lw=1.9, ls=(0, (5, 2)),
                label="best so far")
        if _is_loss_like(metric) and all((s["values"] > 0).all() for s in series):
            ax.set_yscale("log")
        ax.set_xlabel("iteration"); ax.set_ylabel(metric)
        ax.set_title("Hyperparameter optimization")
        ax.legend(loc="lower right", fontsize=7)
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


# ------------------------------------------ dataset characterization (EDA)

_QLKNN_FEATURES = ["Ati", "Ate", "Ane", "Ani", "q", "smag", "x",
                   "Ti/Te", "logNuStar", "normni"]


def characterization_figure(mode: str):
    """Pre-training dataset characterization (style of the RF-heating
    input–output analysis): input violins, target distribution, SNR bars,
    input–target correlations, strongest relationship.
    """
    npz = _REPO / "data" / "datasets" / "benchmarks" / "plasma" / "qlknn_transport.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    X, y = d["X"].astype(float), d["y"].astype(float).ravel()
    names = _QLKNN_FEATURES[: X.shape[1]]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    corr = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
    jbest = int(np.argmax(np.abs(corr)))
    snr = np.abs(X.mean(0)) / (X.std(0) + 1e-12)

    from surge.preprocessing import pca_summary
    pca = pca_summary(X, feature_names=names)

    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(9.8, 5.4))
        gs = fig.add_gridspec(2, 4)

        ax = fig.add_subplot(gs[0, 0:2])        # (a) input distributions
        parts = ax.violinplot([Xs[:, j] for j in range(len(names))],
                              showmedians=True, widths=0.8)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(p["series"][i % 8]); body.set_alpha(0.55)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[part].set_color(p["ink2"]); parts[part].set_linewidth(0.8)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel("standardized value")
        ax.set_title("(a) input distributions", fontsize=9)

        ax = fig.add_subplot(gs[0, 2])          # (b) target distribution
        ax.hist(y, bins=60, color=p["series"][0], alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("efeITG [gB]"); ax.set_ylabel("count (log)")
        ax.set_title("(b) target distribution", fontsize=9)

        ax = fig.add_subplot(gs[0, 3])          # (c) signal-to-noise
        ax.bar(range(len(names)), snr, color=p["series"][2], alpha=0.9)
        for j, v in enumerate(snr):
            ax.text(j, v, f"{v:.1f}", ha="center", va="bottom", fontsize=6,
                    color=p["ink2"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel("|μ|/σ")
        ax.set_title("(c) input signal-to-noise", fontsize=9)

        ax = fig.add_subplot(gs[1, 0:2])        # (d) input–target correlation
        order = np.argsort(corr)
        colors = [p["series"][0] if c < 0 else p["series"][7]
                  for c in corr[order]]
        ax.barh(range(len(names)), corr[order], color=colors, alpha=0.9,
                height=0.62)
        ax.axvline(0, color=p["axis"], lw=0.9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[j] for j in order], fontsize=7)
        ax.set_xlabel("Pearson r with target")
        ax.set_title("(d) input–target correlations", fontsize=9)

        ax = fig.add_subplot(gs[1, 2])          # (e) strongest relationship
        cmap = density_cmap(mode)
        ax.set_facecolor(cmap.get_under())
        ax.hist2d(X[:, jbest], y, bins=48, cmap=cmap,
                  norm=LogNorm(vmin=1), cmin=1)
        z = np.polyfit(X[:, jbest], y, 1)
        xs = np.linspace(X[:, jbest].min(), X[:, jbest].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=p["series"][7], lw=1.6)
        ax.set_xlabel(names[jbest]); ax.set_ylabel("efeITG [gB]")
        ax.set_title(f"(e) strongest input · r = {corr[jbest]:.2f}", fontsize=9)
        ax.grid(alpha=0.5)

        ax = fig.add_subplot(gs[1, 3])          # (f) PCA effective dimension
        ncomp = len(pca["explained_variance_ratio"])
        xs = np.arange(1, ncomp + 1)
        ax.bar(xs, pca["explained_variance_ratio"], color=p["series"][0],
               alpha=0.85, label="per PC")
        ax.plot(xs, pca["cumulative_variance"], color=p["series"][1],
                marker="o", markersize=3.5, lw=1.6, label="cumulative")
        n90 = pca["n_components_90"]
        ax.axhline(0.90, color=p["muted"], lw=0.8, ls=(0, (3, 3)))
        ax.axvline(n90, color=p["muted"], lw=0.8, ls=(0, (3, 3)))
        ax.annotate(f"{n90} PCs → 90%", (n90, 0.90),
                    textcoords="offset points", xytext=(4, -12),
                    fontsize=6.5, color=p["ink2"])
        ax.set_xlabel("principal component"); ax.set_ylabel("variance share")
        ax.set_ylim(0, 1.02)
        ax.set_title("(f) PCA spectrum", fontsize=9)
        ax.legend(fontsize=6, loc="center right")

        fig.suptitle("QLKNN transport — dataset characterization",
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
    ap.add_argument("--run", default=str(_REPO / "runs" / "qlknn_multi_hpo"))
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
        "characterization": characterization_figure,
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
