#!/usr/bin/env python3
"""Gallery of the SURGE visual system (surge.viz.theme) — publication grade.

Every figure is driven by machine-readable SURGE artifacts — run
directories, benchmark result.json files, cached benchmark datasets —
never hand-encoded numbers. Missing artifacts skip that figure with a
note instead of failing.

Usage (from repo root):
    python examples/viz_theme_gallery.py \
        [--run runs/diabetes_rf] [--hpo-run runs/qlknn_multi_hpo] \
        [--benchmark plasma.qlknn_transport] [--modes light dark] \
        [--only parity field_operator ...] [--out examples/viz_gallery_output]

Figures (per mode, deterministic PNG/SVG/PDF):
    parity            density parity + residual distribution
    training_curves   train/val loss with generalisation gap + best epoch
    hpo_convergence   trials coloured by learning rate + running best
    leaderboard       score ± std vs published threshold + runtime panel
    classification    ROC / PR / confusion / reliability (with ECE)
    field_operator    Burgers' operator learning: truth · prediction · error
    field2d           FNO-2D on the periodic Poisson problem (2D fields)
    uncertainty       GP surrogate with 95% credible band and coverage
    ensemble          deep-ensemble UQ with raw vs calibrated coverage
    trio              RF + PyTorch MLP + GP on identical splits
    characterization  dataset EDA panel (distributions, SNR, PCA)
    mission_control   HPO campaign dashboard from run artifacts
    constellaration   stellarator boundary → 12 equilibrium metrics
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


def _benchmark_threshold(key: str):
    """Published pass threshold from the benchmark metadata (or None)."""
    import re

    try:
        from surge.report.leaderboard import load_metadata
        spec = load_metadata()[key]["threshold"]  # e.g. "R² ≥ 0.90"
        if any(t in spec.lower() for t in ("rmse", "mse", "mae", "error")):
            return None  # lower-is-better gates don't fit the R² axis
        m = re.search(r"(\d+(?:\.\d+)?)", spec)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 - decorative line only
        return None


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


def parity_figure(run_dir: Path, mode: str, units: str = "a.u.",
                  singles: bool = False):
    """Publication parity: 2D histogram density, reversed plasma colormap,
    log-scaled counts — the SURGE signature style (Sánchez-Villar et al.,
    Nucl. Fusion): (a) training / (b) test panels, dashed identity, R² box,
    plus (c) test residual distribution.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score

    metrics = json.loads((run_dir / "metrics.json").read_text())
    # showcase the strongest model in the run (highest test R²)
    model_key = max(
        metrics,
        key=lambda k: (metrics[k].get("test") or {}).get("r2", float("-inf")))
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

    from scipy.stats import gaussian_kde as _kde
    from sklearn.metrics import r2_score as _r2

    def _draw_parity(fig, ax, p, cmap, split, y_t, y_q, norm,
                     show_cbar=False, letter=None):
        ax.set_facecolor(cmap.get_under())
        im = ax.hist2d(y_t, y_q, bins=[bins, bins], cmap=cmap,
                       norm=norm, cmin=1)[3]
        ax.plot(lims, lims, color=p["ink2"], lw=1.2, ls=(0, (5, 3)), zorder=3)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        ax.grid(color=p["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Ground truth [{units}]")
        ax.set_ylabel(f"Prediction [{units}]")
        head = f"({letter}) " if letter else ""
        ax.set_title(f"{head}{model_key} — {split}", fontsize=9.5)
        ax.text(0.05, 0.94, f"R² = {_r2(y_t, y_q):.3f}", transform=ax.transAxes,
                fontsize=9, va="top", color=p["ink"],
                bbox={"boxstyle": "round,pad=0.35", "facecolor": p["surface"],
                      "edgecolor": p["ink2"], "linewidth": 0.8})
        if show_cbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("counts (log)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            cb.outline.set_visible(False)
        return im

    def _draw_resid(ax, p):
        y_t, y_q = splits["test"]
        resid = y_q - y_t
        ax.hist(resid, bins=41, color=p["series"][0], alpha=0.55,
                density=True, label="residuals")
        kde_x = np.linspace(resid.min(), resid.max(), 240)
        ax.plot(kde_x, _kde(resid)(kde_x), color=p["series"][0], lw=1.8,
                label="KDE")
        ax.axvline(0.0, color=p["axis"], lw=0.9)
        ax.axvline(float(resid.mean()), color=p["series"][1], lw=1.2,
                   ls=(0, (4, 3)), label="mean")
        ax.set_xlabel(f"Residual [{units}]"); ax.set_ylabel("density")
        ax.set_title("residuals — test", fontsize=9.5)
        _stat_chip(ax, f"mean {resid.mean():+.3g}   σ {resid.std():.3g}", p)
        ax.legend(loc="upper right", fontsize=6.5)

    if singles:
        out = {}
        with surge_theme(mode) as p:
            cmap = density_cmap(mode)
            norm = LogNorm(vmin=1, vmax=peak)
            for split, (y_t, y_q) in splits.items():
                fig, ax = plt.subplots(figsize=(4.3, 3.9))
                _draw_parity(fig, ax, p, cmap, split, y_t, y_q, norm,
                             show_cbar=True)
                out[f"parity_{split}"] = fig
            fig, ax = plt.subplots(figsize=(4.3, 3.9))
            _draw_resid(ax, p)
            out["parity_residuals"] = fig
        return out

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
            # power-law convergence fit on the log-log val curve
            ok = np.isfinite(val) & (val > 0) & (epochs > 0)
            if ok.sum() > 5:
                b, a = np.polyfit(np.log(epochs[ok]), np.log(val[ok]), 1)
                ax.plot(epochs[ok], np.exp(a) * epochs[ok]**b,
                        color=p["muted"], lw=1.2, ls=(0, (1, 2)),
                        label=f"power-law fit (slope {b:.2f})")
        if rows and rows[-1].get("early_stop"):
            ax.axvline(epochs[-1], color=p["critical"], lw=1.0,
                       ls=(0, (2, 2)))
            mid_y = float(np.exp((np.log(np.nanmax(val))
                                  + np.log(np.nanmin(train))) / 2))
            ax.annotate("early stop (smoothed patience)",
                        (epochs[-1], mid_y),
                        textcoords="offset points", xytext=(-7, 0),
                        rotation=90, ha="right", va="center",
                        fontsize=7.5, color=p["critical"])
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
    rows = defaultdict(lambda: {"scores": [], "runtimes": [], "rmses": [],
                                "mems": []})
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
        if met.get("test_rmse") is not None:
            rows[model]["rmses"].append(float(met["test_rmse"]))
        if met.get("peak_memory_mb") is not None:
            rows[model]["mems"].append(float(met["peak_memory_mb"]))
    stats = sorted(
        ({"model": k,
          "mean": statistics.fmean(v["scores"]),
          "std": statistics.stdev(v["scores"]) if len(v["scores"]) > 1 else 0.0,
          "n": len(v["scores"]),
          "runtime": statistics.fmean(v["runtimes"]) if v["runtimes"] else None,
          "rmse": statistics.fmean(v["rmses"]) if v["rmses"] else None,
          "mem": statistics.fmean(v["mems"]) if v["mems"] else None}
         for k, v in rows.items()),
        key=lambda r: r["mean"], reverse=True)[:10]
    if not stats:
        return None

    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    from surge.report.leaderboard import load_metadata

    meta = load_metadata().get(benchmark_key, {})

    def _blend(c1, c2, t):
        a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
        return tuple((1 - t) * a + t * b)

    # pretty math symbols for the flagship benchmarks; fall back to \mathrm
    _SYM = {
        "Ati": r"A_{T_i}", "Ate": r"A_{T_e}", "Ane": r"A_{n_e}",
        "Ani": r"A_{n_i}", "q": r"q", "smag": r"\hat{s}", "x": r"\rho",
        "Ti_Te": r"T_i/T_e", "LogNuStar": r"\log\nu^{*}",
        "normni": r"\bar{n}_i", "efeITG": r"q_e^{\mathrm{ITG}}",
    }

    def _sym(name):
        return _SYM.get(name, r"\mathrm{%s}" % name.replace("_", r"\_"))

    mono = {"family": "monospace"}

    with surge_theme(mode) as p:
        n = len(stats)
        fig = plt.figure(figsize=(11.6, 0.52 * n + 3.2))
        # manual header band + gridspec margins — constrained layout would
        # reclaim the reserved space and collide the header with the bars
        fig.set_layout_engine("none")
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.72, 1.0], height_ratios=[1.0, 1.12],
            left=0.045, right=0.975, top=0.795, bottom=0.09,
            wspace=0.16, hspace=0.42)
        ax = fig.add_subplot(gs[:, 0])

        # ── identity strip ────────────────────────────────────────────
        fig.text(0.045, 0.955, benchmark_key.upper(), fontsize=15,
                 fontweight="bold", color=p["ink"], va="top", **mono)
        if meta.get("name"):
            fig.text(0.045, 0.905, meta["name"], fontsize=10,
                     color=p["ink2"], va="top")
        inputs = [i["name"] for i in meta.get("inputs", [])]
        outputs = [o["name"] for o in meta.get("outputs", [])]
        if inputs and outputs:
            shown = inputs[:4] + ([r"\ldots"] if len(inputs) > 4 else [])
            task = (r"$(" + r",\; ".join(_sym(i) if i != r"\ldots" else i
                                        for i in shown)
                    + r") \;\mapsto\; " + _sym(outputs[0]) + r"$")
            fig.text(0.045, 0.868, task, fontsize=11.5, color=p["ink"],
                     va="top")
        chips = []
        if meta.get("n"):
            chips.append(f"n = {meta['n']}")
        if meta.get("shape"):
            chips.append(f"shape {meta['shape']}")
        if threshold is not None:
            chips.append(f"gate {meta.get('threshold', f'≥ {threshold}')}")
        if meta.get("citation"):
            chips.append(meta["citation"])
        if chips:
            fig.text(0.045, 0.828, "  ·  ".join(chips), fontsize=8,
                     color=p["muted"], va="top", **mono)

        # ── score bars: gradient fill, glowing gate, pass/fail caps ──
        base = p["series"][0]
        bar_cmap = LinearSegmentedColormap.from_list(
            "surge_bar",
            [_blend(p["surface"], base, 0.18), base,
             _blend(base, "#ffffff", 0.22)])
        ypos = np.arange(n)[::-1]
        h = 0.335
        if threshold is not None:
            ax.axvspan(0, threshold, color=p["critical"], alpha=0.04, zorder=0)
            for lw, al in ((7.0, 0.07), (3.4, 0.18), (1.4, 0.95)):
                ax.axvline(threshold, color=p["serious"], lw=lw, alpha=al,
                           zorder=1)
            ax.text(threshold, n - 0.22, f" gate {threshold:.2f}", fontsize=7.5,
                    color=p["serious"], ha="left", va="bottom", **mono)
        grad = np.linspace(0, 1, 256)[None, :]
        for y, r in zip(ypos, stats):
            passed = threshold is None or r["mean"] >= threshold
            cap = p["good"] if passed else p["critical"]
            ax.imshow(grad, extent=(0, r["mean"], y - h, y + h),
                      cmap=bar_cmap, aspect="auto", zorder=2,
                      interpolation="bilinear")
            ax.plot([r["mean"]] * 2, [y - h, y + h], color=cap, lw=2.4,
                    solid_capstyle="butt", zorder=4)
            if r["std"] > 0:
                ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none",
                            ecolor=p["ink2"], elinewidth=1.0, capsize=2.4,
                            zorder=5)
            ax.text(0.012, y, r["model"], va="center", fontsize=8.5,
                    color=p["ink"], zorder=6, **mono)
            label = fmt_metric(r["mean"])
            if r["std"] > 0:
                label += f'±{r["std"]:.3f}'
            label += f' · n={r["n"]}'
            if r["runtime"] is not None:
                label += f' · {fmt_metric(r["runtime"], "runtime")}'
            ax.text(min(r["mean"] + r["std"] + 0.014, 1.005), y, label,
                    va="center", fontsize={True: 7.5, False: 7}[n <= 8],
                    color=cap if not passed else p["ink2"], zorder=6, **mono)
        ax.set_yticks([])
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(0, 1.06)
        ax.set_xlabel("test R²  (mean ± std over runs)")

        # ── spider: top-3 models, 6 normalised axes, circular rings ──
        axs = fig.add_subplot(gs[0, 1], projection="polar")

        def _lognorm(vals, value):
            """1 = best (smallest), 0 = worst, on a log scale."""
            if value is None or not vals:
                return 0.0
            logs = [np.log10(max(v, 1e-2)) for v in vals]
            lo, hi = min(logs), max(logs)
            if hi - lo < 1e-12:
                return 1.0
            return 1.0 - (np.log10(max(value, 1e-2)) - lo) / (hi - lo)

        all_rt = [r["runtime"] for r in stats if r["runtime"] is not None]
        all_rmse = [r["rmse"] for r in stats if r["rmse"] is not None]
        all_mem = [r["mem"] for r in stats if r["mem"] is not None]

        # only draw axes whose metric is actually reported for the top 3 —
        # a missing measurement must not render as "worst in field"
        top3 = stats[:3]

        def _spec():
            axes_spec = [("accuracy", lambda r: max(r["mean"], 0.0))]
            if all(r["rmse"] is not None for r in top3):
                axes_spec.append(("precision\n(RMSE)",
                                  lambda r: _lognorm(all_rmse, r["rmse"])))
            axes_spec.append(("stability",
                              lambda r: 1.0 - min(r["std"] / 0.05, 1.0)))
            if all(r["runtime"] is not None for r in top3):
                axes_spec.append(("speed",
                                  lambda r: _lognorm(all_rt, r["runtime"])))
            if all(r["mem"] is not None for r in top3):
                axes_spec.append(("memory",
                                  lambda r: _lognorm(all_mem, r["mem"])))
            if threshold is not None:
                axes_spec.append(("gate\nmargin", lambda r: float(
                    np.clip((r["mean"] - threshold) / (1 - threshold), 0, 1))))
            return axes_spec

        axes_spec = _spec()

        def _axes6(r):
            return [f(r) for _, f in axes_spec]

        names6 = [n for n, _ in axes_spec]
        theta = np.linspace(0, 2 * np.pi, len(axes_spec), endpoint=False)
        axs.set_theta_offset(np.pi / 2)          # first axis points up
        handles = []
        for k, r in enumerate(top3):
            vals = _axes6(r)
            t = np.concatenate([theta, theta[:1]])
            v = np.array(vals + vals[:1])
            line, = axs.plot(t, v, color=p["series"][k], lw=1.5,
                             marker="o", ms=2.6, zorder=4 - k * 0.1)
            axs.fill(t, v, color=p["series"][k],
                     alpha=0.16 if k == 0 else 0.08)
            handles.append(line)
        axs.set_xticks(theta)
        axs.set_xticklabels(names6, fontsize=7, color=p["ink2"], **mono)
        axs.tick_params(pad=1)
        axs.set_ylim(0, 1.0)
        axs.set_yticks([0.25, 0.5, 0.75, 1.0])
        axs.set_yticklabels([])
        # faint rings + one bright spoke per metric radiating from a
        # bright center dot, ticked at 0.25/0.5/0.75/1.0
        axs.grid(color=p["grid"], lw=0.6, alpha=0.55)
        axs.spines["polar"].set_visible(False)
        spoke = p["ink"]
        axs.plot(0, 0, "o", ms=4.5, color=spoke, zorder=6, clip_on=False)
        levels = (0.25, 0.5, 0.75, 1.0)
        for t_ax in theta:
            axs.plot([t_ax, t_ax], [0, 1.0], color=spoke, lw=0.9,
                     alpha=0.55, zorder=1)
            axs.plot([t_ax] * len(levels), levels, ls="none", marker="o",
                     ms=1.8, color=spoke, alpha=0.8, zorder=2)
            for lev in levels:
                axs.annotate(f"{lev:g}", (t_ax, lev),
                             textcoords="offset points", xytext=(5, 1),
                             fontsize=4.4, color=p["muted"], **mono)
        axs.set_title("top 3  ·  1 = best in field", fontsize=8.5,
                      pad=13, **mono)
        axs.legend(handles, [r["model"] for r in top3],
                   loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1,
                   frameon=False, labelcolor=p["ink2"],
                   prop={"family": "monospace", "size": 6.6})

        # ── dataset preview: density of strongest input vs target ────
        axd = fig.add_subplot(gs[1, 1])
        d = None
        if "." in benchmark_key:
            grp, name = benchmark_key.split(".", 1)
            from surge.report.dataset_previews import _npz
            d = _npz(f"{grp}/{name}.npz")
        if d is not None and "X" in d and "y" in d:
            X, yv = np.asarray(d["X"], float), np.asarray(d["y"], float).ravel()
            corr = [abs(np.corrcoef(X[:, j], yv)[0, 1]) for j in range(X.shape[1])]
            j = int(np.argmax(corr))
            axd.hist2d(X[:, j], yv, bins=90, cmap=density_cmap(mode),
                       norm=LogNorm(vmin=1), cmin=1)
            xin = inputs[j] if j < len(inputs) else f"x_{j}"
            axd.set_xlabel(f"${_sym(xin)}$" + "  (strongest input)",
                           fontsize=8.5)
            yout = outputs[0] if outputs else "y"
            axd.set_ylabel(f"${_sym(yout)}$", fontsize=8.5)
            axd.set_title(f"the data — {len(yv):,} samples", fontsize=8.5,
                          loc="left", **mono)
        else:
            axd.text(0.5, 0.5, "dataset preview unavailable", fontsize=8,
                     ha="center", va="center", color=p["muted"], **mono)
            axd.set_xticks([]); axd.set_yticks([])
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


# ---------------------------------------------------------------- ensemble

def ensemble_figure(mode: str, seed: int = 0):
    """Deep-ensemble UQ (pytorch.mlp_ensemble): mean parity, uncertainty
    vs error, and empirical coverage — ensemble spread as error bars.
    """
    npz = _REPO / "data" / "datasets" / "benchmarks" / "plasma" / "qlknn_transport.npz"
    if not npz.exists():
        return None
    from surge.model import MODEL_REGISTRY
    try:
        adapter = MODEL_REGISTRY.create(
            "pytorch.mlp_ensemble", n_ensembles=6, hidden_dim=128,
            n_layers=2, n_epochs=80, patience=15)
    except KeyError:
        return None

    d = np.load(npz)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d["X"]))[:3000]
    X, y = d["X"][idx].astype(np.float32), d["y"][idx].astype(np.float32)
    n_te = 700
    Xtr, ytr, Xte, yte = X[n_te:], y[n_te:], X[:n_te], y[:n_te]
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    adapter.fit((Xtr - mu_x) / sd_x, ytr)
    mean, std = adapter.predict_with_uncertainty((Xte - mu_x) / sd_x)
    mean, std = np.asarray(mean).ravel(), np.asarray(std).ravel()
    # per-member predictions (n_members, n_test) — the raw ingredient the
    # concept panel needs to show WHY the spread measures uncertainty
    members = np.asarray(
        adapter._model._predict_raw((Xte - mu_x) / sd_x)).reshape(
        adapter._model.n_ensembles, len(yte))
    err = np.abs(mean - yte)

    from sklearn.metrics import r2_score
    r2 = r2_score(yte, mean)
    # raw deep-ensemble spread is typically overconfident; rescale sigma on
    # a held-out calibration half so that 1-sigma coverage matches 68.3%
    n_cal = len(yte) // 2
    lam = float(np.quantile(err[:n_cal] / np.maximum(std[:n_cal], 1e-9),
                            0.6827))
    err_t, std_t = err[n_cal:], std[n_cal:]
    ks = np.linspace(0.2, 3.0, 15)
    coverage = [(err_t <= k * std_t).mean() for k in ks]
    coverage_cal = [(err_t <= k * lam * std_t).mean() for k in ks]
    from scipy.stats import norm
    expected = [norm.cdf(k) - norm.cdf(-k) for k in ks]

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))

        # (a) the concept: 6 independently-seeded MLPs vote; their mean is
        # the prediction and their disagreement is the uncertainty
        ax = axes[0]
        sel = rng.choice(len(yte), 130, replace=False)
        for k in range(members.shape[0]):
            ax.plot(yte[sel], members[k, sel], "o", ms=2.2,
                    color=p["series"][1], alpha=0.35, zorder=2,
                    label="members (6 seeds)" if k == 0 else None)
        ax.errorbar(yte[sel], mean[sel], yerr=2 * lam * std[sel], fmt="o",
                    ms=3.4, color=p["series"][0], ecolor=p["series"][0],
                    elinewidth=0.8, alpha=0.8, capsize=0, zorder=3,
                    label="mean ± 2σ (calibrated)")
        lims = (min(yte.min(), mean.min()), max(yte.max(), mean.max()))
        ax.plot(lims, lims, color=p["axis"], lw=0.9, zorder=1)
        ax.set_xlabel("ground truth [gB]")
        ax.set_ylabel("member / ensemble prediction [gB]")
        ax.set_title("(a) 6 MLPs vote — spread = uncertainty", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left")
        _stat_chip(ax, f"R² = {r2:.3f}", p, loc=(0.66, 0.14))

        # (b) does the spread track the error? (density over the visible
        # range only, with both the raw and the calibrated diagonal)
        ax = axes[1]
        cmap = density_cmap(mode)
        ax.set_facecolor(cmap.get_under())
        # σ and |error| live on very different scales — that IS the story,
        # so give each axis its own range instead of a mostly-empty square
        x_hi = float(np.quantile(std, 0.995))
        y_hi = float(np.quantile(err, 0.995))
        ax.hist2d(std, err, bins=46, range=[[0, x_hi], [0, y_hi]], cmap=cmap,
                  norm=LogNorm(vmin=1), cmin=1)
        ax.plot([0, x_hi], [0, x_hi], color=p["ink2"], lw=1.0, ls=(0, (4, 3)),
                label="|error| = σ (raw)")
        ax.plot([0, x_hi], [0, lam * x_hi], color=p["series"][2], lw=1.4,
                label=f"|error| = {lam:.1f}·σ (calibrated)")
        ax.set_xlim(0, x_hi); ax.set_ylim(0, y_hi)
        ax.set_xlabel("predicted σ [gB]"); ax.set_ylabel("|error| [gB]")
        ax.set_title("(b) errors exceed the raw spread", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.5)

        # (c) the honesty check: fraction of truths inside k·σ
        ax = axes[2]
        ax.plot(ks, expected, color=p["axis"], lw=1.2, ls=(0, (4, 3)),
                label="Gaussian ideal")
        ax.plot(ks, coverage, color=p["series"][0], lw=1.6, marker="o",
                ms=3.2, label="raw σ")
        ax.plot(ks, coverage_cal, color=p["series"][2], lw=1.8, marker="s",
                ms=3.2, label=f"calibrated σ (×{lam:.1f})")
        for k_ref, cov_ref in ((1.0, 0.683), (2.0, 0.954)):
            ax.axvline(k_ref, color=p["grid"], lw=0.8, zorder=1)
            ax.annotate(f"{cov_ref:.0%}", (k_ref, cov_ref),
                        textcoords="offset points", xytext=(4, -10),
                        fontsize=6.5, color=p["muted"])
        ax.set_xlabel("k (σ multiples)"); ax.set_ylabel("coverage")
        ax.set_title("(c) calibration restores honest coverage", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        return fig


# --------------------------------------------------- three-backend success

def trio_figure(mode: str, seed: int = 0, singles: bool = False):
    """RF + PyTorch MLP + Gaussian process succeeding side by side on the
    QLKNN transport task (2,400-sample subsample so exact GP is exact).
    """
    npz = _REPO / "data" / "datasets" / "benchmarks" / "plasma" / "qlknn_transport.npz"
    if not npz.exists():
        return None
    import time

    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.metrics import r2_score

    from surge.model import MODEL_REGISTRY

    d = np.load(npz)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d["X"]))[:2400]
    X, y = d["X"][idx].astype(np.float64), d["y"][idx].astype(np.float64)
    n_te = 600
    Xtr, ytr, Xte, yte = X[n_te:], y[n_te:], X[:n_te], y[:n_te]
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_s, Xte_s = (Xtr - mu_x) / sd_x, (Xte - mu_x) / sd_x

    models = [
        ("sklearn.random_forest", "Random forest", {}),
        ("pytorch.mlp", "PyTorch MLP", {}),
        ("sklearn.gpr", "Gaussian process",
         {"kernel": ConstantKernel(1.0) * RBF(np.ones(X.shape[1]))
                    + WhiteKernel(1e-2, (1e-6, 1e1)),
          "normalize_y": True, "n_restarts_optimizer": 1}),
    ]
    results = []
    for key, label, params in models:
        try:
            adapter = MODEL_REGISTRY.create(key, **params)
        except KeyError:
            continue
        t0 = time.perf_counter()
        adapter.fit(Xtr_s, ytr)
        dt = time.perf_counter() - t0
        pred = np.asarray(adapter.predict(Xte_s)).ravel()
        results.append((label, pred, r2_score(yte, pred), dt))
    if len(results) < 2:
        return None

    def _draw_model(ax, p, cmap, label, pred, r2, dt, lims, letter=None):
        ax.set_facecolor(cmap.get_under())
        ax.hist2d(yte, pred, bins=44, range=[lims, lims], cmap=cmap,
                  norm=LogNorm(vmin=1), cmin=1)
        ax.plot(lims, lims, color=p["ink2"], lw=1.1, ls=(0, (5, 3)))
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        head = f"({letter}) " if letter else ""
        ax.set_title(f"{head}{label}", fontsize=9.5)
        ax.set_xlabel("ground truth [gB]")
        ax.set_ylabel("prediction [gB]")
        _stat_chip(ax, f"R² {r2:.3f} · {fmt_metric(dt, 'runtime')}", p)
        ax.grid(alpha=0.5)

    if singles:
        out = {}
        with surge_theme(mode) as p:
            cmap = density_cmap(mode)
            lims = (float(yte.min()) - 1, float(yte.max()) + 1)
            for label, pred, r2, dt in results:
                fig, ax = plt.subplots(figsize=(4.1, 3.9))
                _draw_model(ax, p, cmap, label, pred, r2, dt, lims)
                key = label.lower().replace(" ", "_")
                out[f"trio_{key}"] = fig
        return out

    with surge_theme(mode) as p:
        cmap = density_cmap(mode)
        fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.3))
        lims = (float(yte.min()) - 1, float(yte.max()) + 1)
        letters = "abc"
        for k, (ax, (label, pred, r2, dt)) in enumerate(zip(axes, results)):
            ax.set_facecolor(cmap.get_under())
            ax.hist2d(yte, pred, bins=44, range=[lims, lims], cmap=cmap,
                      norm=LogNorm(vmin=1), cmin=1)
            ax.plot(lims, lims, color=p["ink2"], lw=1.1, ls=(0, (5, 3)))
            ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
            ax.set_title(f"({letters[k]}) {label}", fontsize=9.5)
            ax.set_xlabel("ground truth [gB]")
            if k == 0:
                ax.set_ylabel("prediction [gB]")
            _stat_chip(ax, f"R² {r2:.3f} · {fmt_metric(dt, 'runtime')}", p)
            ax.grid(alpha=0.5)
        fig.suptitle("Three backends, one registry — QLKNN ITG heat flux "
                     f"({len(Xtr):,} train / {len(Xte):,} test)",
                     fontsize=11, fontweight="bold")
        return fig


# ------------------------------------------------------ 2D operator (FNO)

def field2d_figure(mode: str, seed: int = 0, n: int = 600, side: int = 32,
                   singles: bool = False):
    """2D operator learning: solve the periodic Poisson problem
    ∇²u = −f with an FNO-2D surrogate trained source→solution.
    Dataset generated on the fly (FFT spectral solve = exact reference).
    """
    from surge.model import MODEL_REGISTRY
    try:
        model = MODEL_REGISTRY.create("pytorch.fno2d", n_modes=10, n_epochs=60)
    except KeyError:
        return None

    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(side) * side
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    # smooth Gaussian-random-field sources (low-pass filtered white noise)
    noise = rng.standard_normal((n, side, side))
    filt = np.exp(-(K2) / (2 * 6.0**2))
    f_hat = np.fft.fft2(noise) * filt
    f = np.real(np.fft.ifft2(f_hat))
    f -= f.mean(axis=(1, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_hat = np.where(K2 > 0, np.fft.fft2(f) / K2, 0.0)
    u = np.real(np.fft.ifft2(u_hat))
    scale = u.std()
    u /= scale
    f_in = f / f.std()

    n_te = n // 5
    Xtr = f_in[n_te:].reshape(n - n_te, -1).astype(np.float32)
    ytr = u[n_te:].reshape(n - n_te, -1).astype(np.float32)
    Xte = f_in[:n_te].reshape(n_te, -1).astype(np.float32)
    yte = u[:n_te].reshape(n_te, -1).astype(np.float32)
    model.fit(Xtr, ytr)
    pred = np.asarray(model.predict(Xte)).reshape(n_te, side, side)
    truth = yte.reshape(n_te, side, side)
    rel = (np.linalg.norm((pred - truth).reshape(n_te, -1), axis=1)
           / np.linalg.norm(truth.reshape(n_te, -1), axis=1))
    i = int(np.argsort(rel)[len(rel) // 2])  # median sample

    if singles:
        out = {}
        with surge_theme(mode) as p:
            single_panels = [
                ("field2d_truth", truth[i], "truth  u(x, y)",
                 sequential_cmap(mode), None),
                ("field2d_prediction", pred[i],
                 f"FNO-2D prediction · rel-L2 {rel[i]:.3f}",
                 sequential_cmap(mode), None),
                ("field2d_error", pred[i] - truth[i], "error (pred − truth)",
                 diverging_cmap(mode),
                 CenteredNorm(halfrange=float(np.abs(truth[i]).max()) * 0.1)),
            ]
            for key, img, title, cmap, nrm in single_panels:
                fig, ax = plt.subplots(figsize=(4.1, 3.6))
                im = ax.imshow(img, cmap=cmap, norm=nrm,
                               interpolation="nearest")
                ax.set_title(title, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_visible(False)
                out[key] = fig
        return out

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 5, figsize=(11.8, 2.75),
                                 width_ratios=[1, 1, 1, 1, 0.9])
        panels = [
            (f_in[i], "source  f(x, y)", sequential_cmap(mode), None),
            (truth[i], "truth  u(x, y)", sequential_cmap(mode), None),
            (pred[i], "FNO-2D prediction", sequential_cmap(mode), None),
            (pred[i] - truth[i], "error", diverging_cmap(mode),
             CenteredNorm(halfrange=float(np.abs(truth[i]).max()) * 0.1)),
        ]
        for ax, (img, title, cmap, nrm) in zip(axes[:4], panels):
            im = ax.imshow(img, cmap=cmap, norm=nrm, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_visible(False)
        axh = axes[4]
        axh.hist(rel, bins=24, color=p["series"][0], alpha=0.9)
        axh.axvline(float(np.median(rel)), color=p["series"][1], lw=1.3,
                    ls=(0, (4, 3)))
        axh.set_xlabel("rel-L2"); axh.set_ylabel("count")
        axh.set_title(f"median {np.median(rel):.3f}", fontsize=9)
        fig.suptitle(
            f"2D operator learning — periodic Poisson ∇²u = −f · FNO-2D · "
            f"{n - n_te} train / {n_te} test fields ({side}×{side})",
            fontsize=10.5, fontweight="bold")
        return fig


# --------------------------------------------------- HPO mission control

def mission_control_figure(run_dir: Path, mode: str,
                           model_name: str = "qlknn_residual_mlp"):
    """One-look HPO campaign dashboard for a run: per-trial loss curves,
    convergence, parameter sensitivity, best-trial detail, outcome parity,
    and a run-summary card — all from the run's own artifacts.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    manifest_f = run_dir / "hpo_trials_manifest.jsonl"
    if not manifest_f.exists():
        return None
    trials = [json.loads(ln) for ln in manifest_f.read_text().splitlines()]
    hists = {}
    for t in range(len(trials)):
        f = run_dir / f"hpo_trial_{t:04d}_training_history.json"
        if f.exists():
            hists[t] = json.loads(f.read_text())
    if not hists:
        return None
    values = [t["value"] for t in trials]
    best_i = int(np.argmax(values))
    metrics = json.loads((run_dir / "metrics.json").read_text()).get(
        model_name, {})

    mono = {"family": "monospace"}

    def _card(ax, p, title):
        ax.set_facecolor(p["surface"])
        for s in ax.spines.values():
            s.set_color(p["grid"]); s.set_linewidth(0.8); s.set_visible(True)
        ax.set_title(title, fontsize=8.5, loc="left", color=p["ink2"],
                     pad=5, **mono)

    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(12.6, 7.0))
        fig.set_layout_engine("none")
        gs = fig.add_gridspec(2, 3, left=0.05, right=0.975, top=0.845,
                              bottom=0.075, wspace=0.24, hspace=0.36)

        fig.text(0.05, 0.965, "SURGE · MISSION CONTROL", fontsize=15,
                 fontweight="bold", color=p["ink"], va="top", **mono)
        fig.text(0.05, 0.915,
                 f"HPO campaign — {model_name}  ·  Optuna TPE  ·  "
                 f"{len(trials)} trials × {max(len(h) for h in hists.values())}"
                 " epochs", fontsize=9.5, color=p["ink2"], va="top", **mono)

        # (1) all trials: validation loss per epoch, best highlighted
        ax = fig.add_subplot(gs[0, 0])
        _card(ax, p, "VAL LOSS — ALL TRIALS")
        for t, h in hists.items():
            ep = [r["epoch"] for r in h]
            vl = [r["val_loss"] for r in h]
            if t == best_i:
                ax.plot(ep, vl, color=p["series"][0], lw=2.0, zorder=5,
                        label=f"best (trial {t})")
            else:
                ax.plot(ep, vl, color=p["muted"], lw=0.9, alpha=0.55, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("val loss")
        ax.legend(fontsize=7, loc="upper right")

        # (2) HPO convergence: per-trial score + running best + gold star
        ax = fig.add_subplot(gs[0, 1])
        _card(ax, p, "SEARCH CONVERGENCE")
        xs = np.arange(len(values))
        run_best = np.maximum.accumulate(values)
        ax.plot(xs, values, "o", ms=4, color=p["series"][0], alpha=0.75,
                label="trial")
        ax.plot(xs, run_best, color=p["series"][0], lw=1.4, ls=(0, (4, 3)),
                label="running best")
        ax.plot(best_i, values[best_i], marker="*", ms=15,
                color=p["warning"], markeredgecolor=p["ink"],
                markeredgewidth=0.7, zorder=6)
        ax.annotate(f"{values[best_i]:.3f}", (best_i, values[best_i]),
                    textcoords="offset points", xytext=(-8, -13),
                    ha="right", fontsize=7.5, color=p["ink"], **mono)
        ax.set_xlabel("trial"); ax.set_ylabel(trials[0].get("metric", "value"))
        ax.legend(fontsize=7, loc="lower right")

        # (3) run summary card
        ax = fig.add_subplot(gs[0, 2])
        _card(ax, p, "RUN SUMMARY")
        ax.set_xticks([]); ax.set_yticks([])
        bp = trials[best_i].get("params", {})
        te = metrics.get("test", {})
        tm = metrics.get("timings", {})
        lines = [
            ("MODEL", model_name),
            ("TRIALS", f"{len(trials)} (TPE, max val_r2)"),
            ("BEST VAL R²", f"{values[best_i]:.4f}"),
            ("TEST R²", f"{te.get('r2', float('nan')):.4f}"),
            ("TEST RMSE", f"{te.get('rmse', float('nan')):.3f} gB"),
            ("FIT TIME", fmt_metric(tm.get("train_seconds"), "runtime")
             if tm.get("train_seconds") else "—"),
        ] + [(k.upper().replace("_", " "),
              f"{v:.3g}" if isinstance(v, float) else str(v))
             for k, v in bp.items()]
        for j, (k, v) in enumerate(lines):
            y = 0.93 - j * 0.095
            ax.text(0.06, y, k, transform=ax.transAxes, fontsize=7.5,
                    color=p["muted"], va="top", **mono)
            ax.text(0.52, y, v, transform=ax.transAxes, fontsize=8,
                    color=p["ink"], va="top", **mono)

        # (4) best trial: train vs val loss with generalisation gap
        ax = fig.add_subplot(gs[1, 0])
        _card(ax, p, f"BEST TRIAL {best_i} — TRAIN VS VAL")
        h = hists[best_i]
        ep = np.array([r["epoch"] for r in h])
        tl = np.array([r["train_loss"] for r in h])
        vl = np.array([r["val_loss"] for r in h])
        ax.plot(ep, tl, color=p["series"][1], lw=1.4, label="train")
        ax.plot(ep, vl, color=p["series"][0], lw=1.6, label="val")
        ax.fill_between(ep, tl, vl, color=p["series"][0], alpha=0.10)
        be = int(ep[np.argmin(vl)])
        ax.axvline(be, color=p["good"], lw=1.0, ls=(0, (4, 3)))
        ax.text(be, float(vl.min()), f" best ep {be}", fontsize=7,
                color=p["good_text"], **mono)
        ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.legend(fontsize=7, loc="upper right")

        # (5) which hyperparameters mattered (rank correlation with score)
        ax = fig.add_subplot(gs[1, 1])
        _card(ax, p, "PARAMETER SENSITIVITY")
        pkeys = [k for k in trials[0].get("params", {})
                 if isinstance(trials[0]["params"][k], (int, float))]
        rho = []
        for k in pkeys:
            vals = [t["params"][k] for t in trials]
            r = spearmanr(vals, values).statistic
            rho.append(0.0 if np.isnan(r) else float(r))
        order = np.argsort(np.abs(rho))
        ypos = np.arange(len(pkeys))
        for y, o in zip(ypos, order):
            col = p["series"][0] if rho[o] >= 0 else p["series"][1]
            ax.barh(y, rho[o], height=0.55, color=col, alpha=0.85)
            ax.text(0.02 if rho[o] < 0 else -0.02, y, pkeys[order[y]] if False
                    else pkeys[o],
                    ha="left" if rho[o] < 0 else "right", va="center",
                    fontsize=7.5, color=p["ink"], **mono)
        ax.axvline(0, color=p["axis"], lw=0.9)
        ax.set_yticks([])
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman ρ (param vs score)")

        # (6) outcome: test parity density of the tuned model
        ax = fig.add_subplot(gs[1, 2])
        _card(ax, p, "OUTCOME — TEST PARITY")
        try:
            y_t, y_q = _load_split(run_dir, model_name, "test")
            cmap = density_cmap(mode)
            ax.set_facecolor(cmap.get_under())
            lims = (float(min(y_t.min(), y_q.min())),
                    float(max(y_t.max(), y_q.max())))
            bins = np.linspace(*lims, 48)
            ax.hist2d(y_t, y_q, bins=[bins, bins], cmap=cmap,
                      norm=LogNorm(vmin=1), cmin=1)
            ax.plot(lims, lims, color=p["ink2"], lw=1.0, ls=(0, (5, 3)))
            ax.set_aspect("equal")
            ax.set_xlabel("ground truth [gB]"); ax.set_ylabel("prediction")
            _stat_chip(ax, f"R² = {r2_score(y_t, y_q):.3f}", p,
                       loc=(0.05, 0.93))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "predictions unavailable", fontsize=8,
                    ha="center", va="center", color=p["muted"], **mono)
            ax.set_xticks([]); ax.set_yticks([])
        return fig


# ---------------------------------------------- ConStellaration stellarators

def constellaration_figure(mode: str, seed: int = 0, n_epochs: int = 400):
    """Stellarator design surrogate on the ConStellaration dataset
    (Goodman et al. 2025, arXiv:2506.19583, proxima-fusion/constellaration):
    boundary Fourier coefficients (5×9 r_cos + 5×9 z_sin, n_fp = 3) → 12
    equilibrium figures of merit. Shows real plasma boundary shapes, the
    log₁₀(qi) parity, and which metrics are learnable.
    """
    npz = _REPO / "data" / "datasets" / "constellaration" / "paper_nfp3_clip0.05.npz"
    if not npz.exists():
        return None
    from sklearn.metrics import r2_score

    from surge.model import MODEL_REGISTRY

    d = np.load(npz, allow_pickle=True)
    X, Y = d["X"], d["Y"]
    names = [str(s) for s in d["metric_names"]]
    split = np.load(_REPO / "data" / "datasets" / "benchmarks" / "plasma"
                    / "constellaration" / "split_n26897_seed42_test0.2.npz")
    tr, te = split["train_idx"], split["test_idx"]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-9
    mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-9

    try:
        # trained to saturation: 60-epoch caps left ~0.015 R2 on the table
        adapter = MODEL_REGISTRY.create(
            "pytorch.residual_mlp", hidden_layers=[512, 512, 256],
            n_epochs=n_epochs, patience=60, patience_window=10,
            random_state=seed)
    except KeyError:
        return None
    adapter.fit(((Xtr - mu_x) / sd_x).astype(np.float32),
                ((Ytr - mu_y) / sd_y).astype(np.float32))
    pred = np.asarray(adapter.predict(
        ((Xte - mu_x) / sd_x).astype(np.float32))) * sd_y + mu_y

    r2s = [r2_score(Yte[:, j], pred[:, j]) for j in range(Y.shape[1])]
    qi_j = names.index("log_10_qi")

    NFP = 3
    theta = np.linspace(0, 2 * np.pi, 181)

    def boundary_rz(x_row, phi):
        r_mn = x_row[:45].reshape(5, 9)   # m = 0..4, n = -4..4
        z_mn = x_row[45:].reshape(5, 9)
        R = np.zeros_like(theta); Z = np.zeros_like(theta)
        for m in range(5):
            for jn, n in enumerate(range(-4, 5)):
                ang = m * theta - NFP * n * phi
                R += r_mn[m, jn] * np.cos(ang)
                Z += z_mn[m, jn] * np.sin(ang)
        return R, Z

    # three test-set stellarators spanning the elongation range
    elong = Yte[:, names.index("max_elongation")]
    picks = [int(np.argsort(elong)[k]) for k in
             (len(elong) // 20, len(elong) // 2, int(len(elong) * 0.95))]
    phis = np.array([0.0, 0.25, 0.5]) * (2 * np.pi / NFP) / 2

    _PRETTY = {
        "aspect_ratio": "aspect ratio",
        "aspect_ratio_over_edge_rotational_transform": r"$A/\iota_{edge}$",
        "max_elongation": "max elongation",
        "axis_rotational_transform_over_n_field_periods": r"$\iota_{axis}/n_{fp}$",
        "edge_rotational_transform_over_n_field_periods": r"$\iota_{edge}/n_{fp}$",
        "axis_magnetic_mirror_ratio": r"mirror ratio (axis)",
        "edge_magnetic_mirror_ratio": r"mirror ratio (edge)",
        "average_triangularity": "avg triangularity",
        "vacuum_well": "vacuum well",
        "minimum_normalized_magnetic_gradient_scale_length":
            r"min $L_{\nabla B}$",
        "flux_compression_in_regions_of_bad_curvature": "flux compression",
        "log_10_qi": r"$\log_{10}$ QI residual",
    }

    def boundary_rz_grid(x_row, theta_g, phi_g):
        """Vectorised R, Z over (theta, phi) grids for the 3D surface."""
        r_mn = x_row[:45].reshape(5, 9)
        z_mn = x_row[45:].reshape(5, 9)
        R = np.zeros(theta_g.shape); Z = np.zeros(theta_g.shape)
        for m in range(5):
            for jn, n in enumerate(range(-4, 5)):
                ang = m * theta_g - NFP * n * phi_g
                R += r_mn[m, jn] * np.cos(ang)
                Z += z_mn[m, jn] * np.sin(ang)
        return R, Z

    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(12.2, 3.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.15],
                              wspace=0.24)
        axes = [None,
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2])]

        # (a) the LCFS in 3D: full torus, cross-section rings highlighted
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        i_mid = picks[1]
        th = np.linspace(0, 2 * np.pi, 90)
        ph = np.linspace(0, 2 * np.pi, 240)
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        R, Z = boundary_rz_grid(Xte[i_mid], TH, PH)
        Xs, Ys = R * np.cos(PH), R * np.sin(PH)
        ax.plot_surface(Xs, Ys, Z, color=p["series"][0], alpha=0.22,
                        linewidth=0, antialiased=True, shade=True,
                        rcount=45, ccount=120)
        # section rings: half a field period, then repeated by symmetry
        ring_phis = np.array([0, 1 / 6, 2 / 6]) * (2 * np.pi / NFP)
        for jp, phi in enumerate(ring_phis):
            for rep in range(NFP):
                phv = phi + rep * 2 * np.pi / NFP
                Rr, Zr = boundary_rz(Xte[i_mid], phv)
                ax.plot(Rr * np.cos(phv), Rr * np.sin(phv), Zr,
                        color=p["series"][jp], lw=1.7,
                        label=(r"$\varphi$ = 0", r"1/6 period",
                               r"1/3 period")[jp] if rep == 0 else None)
        ax.set_box_aspect((1, 1, 0.5))
        # mpl3d does not clip to limits — shrinking them zooms the torus in
        lim = float(np.abs(R).max()) * 0.72
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim * 0.5, lim * 0.5)
        ax.view_init(elev=26, azim=-58)
        ax.set_axis_off()
        ax.set_facecolor("none")
        ax.set_title(f"(a) LCFS in 3D — $n_{{fp}}$ = {NFP}, "
                     "sections repeat each period", fontsize=8.5, pad=0,
                     x=0.62)
        ax.legend(fontsize=6, loc="upper left", frameon=False)

        # (b) parity for the headline metric: QI quality
        ax = axes[1]
        cmap = density_cmap(mode)
        ax.set_facecolor(cmap.get_under())
        yt, yq = Yte[:, qi_j], pred[:, qi_j]
        lims = (float(min(yt.min(), yq.min())), float(max(yt.max(), yq.max())))
        bins = np.linspace(*lims, 52)
        ax.hist2d(yt, yq, bins=[bins, bins], cmap=cmap,
                  norm=LogNorm(vmin=1), cmin=1)
        ax.plot(lims, lims, color=p["ink2"], lw=1.1, ls=(0, (5, 3)))
        ax.set_aspect("equal")
        ax.set_xlabel(r"true $\log_{10}$ QI residual")
        ax.set_ylabel("predicted")
        ax.set_title("(b) quasi-isodynamic quality", fontsize=9)
        _stat_chip(ax, f"R² = {r2s[qi_j]:.3f}", p, loc=(0.05, 0.94))

        # (c) which figures of merit are learnable from shape alone
        ax = axes[2]
        order = np.argsort(r2s)
        ypos = np.arange(len(names))
        for y, o in zip(ypos, order):
            ax.barh(y, max(r2s[o], 0.0), height=0.62,
                    color=p["series"][0] if o != qi_j else p["series"][2],
                    alpha=0.9)
            ax.text(max(r2s[o], 0.0) + 0.012, y, f"{r2s[o]:.2f}",
                    va="center", fontsize=6.5, color=p["ink2"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([_PRETTY.get(names[o], names[o]) for o in order],
                           fontsize=7)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("test R²  (one 90 → 12 surrogate)")
        ax.set_title("(c) learnability by metric", fontsize=9)

        fig.suptitle(
            "ConStellaration — stellarator boundary "
            r"$(R_{mn}, Z_{mn}) \mapsto$ 12 equilibrium metrics · "
            f"{len(tr):,} train / {len(te):,} test QI configurations",
            fontsize=10.5, fontweight="bold")
        return fig


# ------------------------------------------------------------- scale demo

def scale_figure(mode: str):
    """Training-at-scale levers, from measured numbers
    (scripts/benchmark_scale.py -> scale_bench.json): device speedups for
    the 2D operator models and the parallel benchmark fan-out.
    """
    bench = _REPO / "examples" / "viz_gallery_output" / "scale_bench.json"
    if not bench.exists():
        return None
    d = json.loads(bench.read_text())
    dev_rows = d.get("device", [])
    par_rows = d.get("parallel", [])
    if not dev_rows or not par_rows:
        return None
    mono = {"family": "monospace"}

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))

        # (a) cpu vs accelerator fit time per model
        ax = axes[0]
        models = sorted({r["model"] for r in dev_rows})
        devs = [dv for dv in ("cpu", "mps", "cuda")
                if any(r["device"] == dv for r in dev_rows)]
        width = 0.8 / len(devs)
        for j, dv in enumerate(devs):
            xs, ts = [], []
            for i, mdl in enumerate(models):
                row = next((r for r in dev_rows
                            if r["model"] == mdl and r["device"] == dv), None)
                if row:
                    xs.append(i + (j - (len(devs) - 1) / 2) * width)
                    ts.append(row["fit_seconds"])
            ax.bar(xs, ts, width=width * 0.9, color=p["series"][j],
                   alpha=0.9, label=dv)
            for x, t in zip(xs, ts):
                ax.text(x, t, f" {t:.1f}s", ha="center", va="bottom",
                        fontsize=7, color=p["ink2"], **mono)
        for i, mdl in enumerate(models):
            cpu = next((r["fit_seconds"] for r in dev_rows
                        if r["model"] == mdl and r["device"] == "cpu"), None)
            acc = min((r["fit_seconds"] for r in dev_rows
                       if r["model"] == mdl and r["device"] != "cpu"),
                      default=None)
            if cpu and acc:
                ax.text(i, cpu * 1.16, f"{cpu / acc:.1f}x", ha="center",
                        fontsize=9, fontweight="bold", color=p["good_text"],
                        **mono)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel("fit time [s]")
        ax.set_ylim(0, max(r["fit_seconds"] for r in dev_rows) * 1.35)
        ax.set_title("(a) SURGE_DEVICE=auto — Apple-GPU speedup",
                     fontsize=9)
        ax.legend(fontsize=7.5)

        # (b) parallel benchmark fan-out
        ax = axes[1]
        ws = [r["workers"] for r in par_rows]
        ts = [r["wall_seconds"] for r in par_rows]
        ax.plot(ws, ts, marker="o", ms=5, lw=1.8, color=p["series"][0],
                label="measured")
        ax.plot(ws, [ts[0] / w for w in ws], ls=(0, (4, 3)), lw=1.2,
                color=p["axis"], label="ideal 1/N")
        for w, t in zip(ws, ts):
            ax.annotate(f"{t:.0f}s", (w, t), textcoords="offset points",
                        xytext=(6, 5), fontsize=7.5, color=p["ink2"], **mono)
        ax.set_xticks(ws)
        ax.set_xlabel("workers  (surge bench --parallel N)")
        ax.set_ylabel("wall time [s]")
        ax.set_ylim(0, ts[0] * 1.15)
        ax.set_title(f"(b) {par_rows[0]['jobs']} models x 3 seeds — "
                     "QLKNN benchmark", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right")
        return fig


# -------------------------------------------------------------------- main\n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(_REPO / "runs" / "qlknn_multi_hpo"))
    ap.add_argument("--hpo-run", default=str(_REPO / "runs" / "qlknn_multi_hpo"))
    ap.add_argument("--benchmark", default="plasma.qlknn_transport")
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
            Path(args.reports), args.benchmark, m,
            threshold=_benchmark_threshold(args.benchmark)),
        "classification": classification_figure,
        "field_operator": field_operator_figure,
        "uncertainty": uncertainty_figure,
        "characterization": characterization_figure,
        "ensemble": ensemble_figure,
        "trio": trio_figure,
        "field2d": field2d_figure,
        "mission_control": lambda m: mission_control_figure(
            Path(args.hpo_run), m),
        "constellaration": constellaration_figure,
        "scale": scale_figure,
        "parity_singles": lambda m: parity_figure(Path(args.run), m, singles=True),
        "trio_singles": lambda m: trio_figure(m, singles=True),
        "field2d_singles": lambda m: field2d_figure(m, singles=True),
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
            figs = fig if isinstance(fig, dict) else {name: fig}
            for fname, f in figs.items():
                for path in save_figure(f, out / f"{fname}_{mode}"):
                    print("wrote", path)
                plt.close(f)


if __name__ == "__main__":
    main()


