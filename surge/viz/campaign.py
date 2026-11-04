"""Run-directory figures: parity, training curves, HPO convergence, mission control.

Everything here is driven by the artifacts a ``surge run`` writes
(``metrics.json``, ``hpo_trials_manifest.jsonl``,
``hpo_trial_XXXX_training_history.json``, ``predictions/<model>_<split>.parquet``,
``hpo/<model>_hpo.json``) -- no hand-encoded numbers. The figures are the
README / video panels; ``render_run`` builds all of them for one run and is
what ``surge viz`` calls.

Public API: :func:`detect_models`, :func:`detect_hpo_model`,
:func:`parity_figure`, :func:`training_figure`, :func:`hpo_figure`,
:func:`mission_control_figure`, :func:`render_run`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from surge.viz.hpo import robust_hpo_ylim  # noqa: E402
from surge.viz.theme import (  # noqa: E402
    density_cmap,
    fmt_metric,
    save_figure,
    sequential_cmap,
    surge_theme,
)

FIGURE_NAMES = ("mission_control", "hpo_convergence", "training_curves", "parity")


# ------------------------------------------------------- run introspection

def _read_json(path: Path):
    return json.loads(path.read_text())


def detect_models(run_dir: Path) -> list[str]:
    """Model names in ``metrics.json`` (insertion order)."""
    f = Path(run_dir) / "metrics.json"
    if not f.exists():
        return []
    return [k for k, v in _read_json(f).items() if isinstance(v, dict)]


def _manifest_rows(run_dir: Path) -> list[dict]:
    f = Path(run_dir) / "hpo_trials_manifest.jsonl"
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


def detect_hpo_model(run_dir: Path) -> str | None:
    """The model whose HPO trials appear in the run (manifest or hpo/*_hpo.json)."""
    run_dir = Path(run_dir)
    models = detect_models(run_dir)
    rows = _manifest_rows(run_dir)
    for r in rows:
        if r.get("model"):
            return str(r["model"])
    hpo_files = sorted((run_dir / "hpo").glob("*_hpo.json"))
    if hpo_files:
        return hpo_files[0].stem[: -len("_hpo")]
    if rows:  # manifest lines carry no model name: single-HPO-model runs
        return models[0] if models else None
    return None


def _resolve_model(run_dir: Path, model_name: str | None) -> str | None:
    if model_name:
        return model_name
    return detect_hpo_model(run_dir) or next(iter(detect_models(run_dir)), None)


def _maximize(metric: str) -> bool:
    """Objective direction inferred from the metric name (loss-like = minimise)."""
    return not _is_loss_like(metric)


def _best_index(values, maximize: bool) -> int:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, -np.inf if maximize else np.inf)
    return int(np.argmax(arr) if maximize else np.argmin(arr))


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


def _parity(run_dir: Path, mode: str, model_key: str, units: str = "a.u.",
            singles: bool = False):
    """Publication parity: 2D histogram density, reversed plasma colormap,
    log-scaled counts — the SURGE signature style (Sánchez-Villar et al.,
    Nucl. Fusion): (a) training / (b) test panels, dashed identity, R² box,
    plus (c) test residual distribution.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score

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
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
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
        ax.set_xlabel(f"Residual [{units}]")
        ax.set_ylabel("density")
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
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
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
        axr.set_xlabel(f"Residual [{units}]")
        axr.set_ylabel("density")
        axr.set_title(f"({letters[len(splits)]}) residuals — test",
                      fontsize=9.5)
        _stat_chip(axr, f"mean {resid.mean():+.3g}   σ {resid.std():.3g}", p)
        axr.legend(loc="upper right", fontsize=6.5)
        return fig


# --------------------------------------------------------- training curves

def _training_rows(run_dir: Path, model: str) -> list[dict]:
    """Per-epoch history of the final model, falling back to the best HPO trial."""
    log = run_dir / f"training_log_{model}.jsonl"
    if log.exists():
        return [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    hist = run_dir / f"training_history_{model}.json"
    if hist.exists():
        return _read_json(hist)
    trials = _manifest_rows(run_dir)
    if trials:
        best = _best_index([t["value"] for t in trials],
                           _maximize(trials[0].get("metric", "")))
        f = run_dir / f"hpo_trial_{trials[best].get('trial', best):04d}_training_history.json"
        if f.exists():
            return _read_json(f)
    return []


def _training(run_dir: Path, mode: str, name: str):
    rows = _training_rows(run_dir, name)
    if not rows:
        return None
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
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Training — {name}")
        ax.legend(loc="upper right", fontsize=7.5)
        return fig


# --------------------------------------------------------- HPO convergence

def _hpo_series(run_dir: Path, model_name: str | None) -> list[dict]:
    """HPO series from ``hpo/*_hpo.json`` (preferred) or the trial manifest."""
    files = sorted((run_dir / "hpo").glob("*_hpo.json"))
    if model_name and (run_dir / "hpo" / f"{model_name}_hpo.json") in files:
        files = [run_dir / "hpo" / f"{model_name}_hpo.json"]
    series = []
    for f in files[:4]:
        d = json.loads(f.read_text())
        trials = d.get("trials", [])
        if not trials:
            continue
        values = np.array([t["value"] for t in trials], dtype=float)
        maximize = d.get("direction", "maximize") == "maximize"
        running = (np.maximum if maximize else np.minimum).accumulate(values)
        lrs = np.array([t.get("params", {}).get("learning_rate", np.nan)
                        for t in trials], dtype=float)
        series.append({
            "name": f.stem.replace("_hpo", ""),
            "numbers": np.array([t["number"] for t in trials]),
            "values": values,
            "running": running,
            "lr": lrs,
            "metric": d.get("metric", "objective"),
            "best": d.get("best_trial", {}),
            "maximize": maximize,
        })
    if series:
        return series
    rows = _manifest_rows(run_dir)
    if not rows:
        return []
    values = np.array([t["value"] for t in rows], dtype=float)
    metric = rows[0].get("metric", "objective")
    maximize = _maximize(metric)
    running = (np.maximum if maximize else np.minimum).accumulate(values)
    bi = _best_index(values, maximize)
    return [{
        "name": model_name or "hpo",
        "numbers": np.array([t.get("trial", i) for i, t in enumerate(rows)]),
        "values": values,
        "running": running,
        "lr": np.array([t.get("params", {}).get("learning_rate", np.nan)
                        for t in rows], dtype=float),
        "metric": metric,
        "best": {"number": rows[bi].get("trial", bi), "value": float(values[bi])},
        "maximize": maximize,
    }]


def _hpo(run_dir: Path, mode: str, reference: float | None = None,
         model_name: str | None = None):
    """HPO history, publication style (Sánchez-Villar et al.): one series
    per optimizer/model — per-trial markers (colored by log10 learning
    rate for a single tuned model), dashed running-best step line,
    gold-edged star at the best trial with an in-axes score chip, and an
    optional dashed reference line. Y-limits are the 5th-100th (maximize)
    or 0th-95th (minimize) percentile of finite values with small padding,
    so a handful of diverged trials among many can't collapse the axis;
    those trials are drawn instead as small muted triangles clamped to the
    axis edge. Campaigns of <=10 trials keep plain autoscaling.
    """
    series = _hpo_series(run_dir, model_name)
    if not series:
        return None
    metric = series[0]["metric"]
    maximize = series[0]["maximize"]
    star_face = "#FFD700"  # gold star fill, edge in series color

    # Robust y-limits + per-series diverged-trial masks (shared helper —
    # also used by mission_control_figure's SEARCH CONVERGENCE inset).
    ylim_lo, ylim_hi, ylim_pad, diverged_masks = robust_hpo_ylim(
        [s["values"] for s in series], maximize)

    # Color scatter by log10(learning_rate) only for a single tuned series
    # that actually records it; a multi-model overlay (e.g. rf + mlp) keeps
    # each model's own categorical series color instead.
    color_by_lr = (len(series) == 1 and series[0]["lr"].size
                   and np.isfinite(series[0]["lr"]).all())

    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        cmap = sequential_cmap(mode)
        mappable = None
        n_diverged = 0
        for i, s in enumerate(series):
            c = p["series"][1] if color_by_lr else p["series"][i]
            diverged = diverged_masks[i]
            n_diverged += int(diverged.sum())
            keep = ~diverged
            if color_by_lr:
                mappable = ax.scatter(
                    s["numbers"][keep], s["values"][keep],
                    c=np.log10(s["lr"][keep]), cmap=cmap, s=20, alpha=0.85,
                    zorder=2, label=s["name"])
            else:
                ax.scatter(s["numbers"][keep], s["values"][keep], color=c,
                           s=20, alpha=0.8, zorder=2, label=s["name"])
            if diverged.any():
                clamp_y = ylim_lo if maximize else ylim_hi
                ax.scatter(s["numbers"][diverged],
                           np.full(int(diverged.sum()), clamp_y),
                           marker="v" if maximize else "^", s=26,
                           color=p["muted"], zorder=2)
            ax.plot(s["numbers"], s["running"], color=c, lw=1.9,
                    ls=(0, (5, 2)), drawstyle="steps-post", zorder=3)
            best = s["best"]
            if best:
                bi, bv = best.get("number"), best.get("value")
                ax.scatter([bi], [bv], marker="*", s=430, zorder=5,
                           facecolor=star_face, edgecolor=c, linewidth=1.6)
                _stat_chip(ax, f"{s['name']}: {metric} = {fmt_metric(bv)}", p,
                           loc=(0.03, 0.95 - i * 0.10))
        if reference is not None:
            ax.axhline(reference, color=p["ink"], lw=1.4, ls=(0, (6, 3)),
                       zorder=1, label=f"reference ({reference})")
        ax.plot([], [], color=p["ink2"], lw=1.9, ls=(0, (5, 2)),
                label="best so far")
        if n_diverged:
            # legend proxy (not ax.text) so the muted label sits in the
            # reserved legend box instead of overlapping the clamped
            # triangles it describes
            ax.scatter([], [], marker="v" if maximize else "^", s=26,
                       color=p["muted"], label=f"{n_diverged} diverged")
        if _is_loss_like(metric) and all((s["values"] > 0).all() for s in series):
            ax.set_yscale("log")
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo - ylim_pad, ylim_hi + ylim_pad)
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric)
        ax.set_title("Hyperparameter optimization")
        ax.legend(loc="lower right" if maximize else "upper right", fontsize=7)
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, pad=0.02)
            cb.set_label(r"$\log_{10}$ learning rate", fontsize=7.5,
                         color=p["ink2"])
            cb.ax.tick_params(labelsize=7, colors=p["muted"])
        return fig


# --------------------------------------------------- HPO mission control

def _param_label(key: str) -> str:
    """Human-readable label for a raw Optuna HPO search-space key."""
    fixed = {
        "n_blocks": "depth (blocks)",
        "learning_rate": "learning rate",
        "batch_size": "batch size",
        "dropout_rate": "dropout",
    }
    if key in fixed:
        return fixed[key]
    m = re.match(r"hidden_layers_(\d+)_log$", key)
    if m:
        return f"width · block {m.group(1)}"
    return key.replace("_", " ")


def _mission_control(run_dir: Path, mode: str, model_name: str,
                     units: str = "a.u."):
    """One-look HPO campaign dashboard for a run: per-trial loss curves,
    convergence, parameter sensitivity, best-trial detail, outcome parity,
    and a run-summary card — all from the run's own artifacts.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    trials = _manifest_rows(run_dir)
    if not trials:
        return None
    hists = {}
    for t in range(len(trials)):
        f = run_dir / f"hpo_trial_{t:04d}_training_history.json"
        if f.exists():
            hists[t] = json.loads(f.read_text())
    if not hists:
        return None
    values = [t["value"] for t in trials]
    metric_name = trials[0].get("metric", "value")
    maximize = _maximize(metric_name)
    best_i = _best_index(values, maximize)
    mfile = run_dir / "metrics.json"
    metrics = (_read_json(mfile) if mfile.exists() else {}).get(model_name, {})

    mono = {"family": "monospace"}

    def _card(ax, p, title):
        ax.set_facecolor(p["surface"])
        for s in ax.spines.values():
            s.set_color(p["grid"])
            s.set_linewidth(0.8)
            s.set_visible(True)
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
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.legend(fontsize=7, loc="upper right")

        # (2) HPO convergence: per-trial score + running best + gold star
        ax = fig.add_subplot(gs[0, 1])
        _card(ax, p, "SEARCH CONVERGENCE")
        xs = np.arange(len(values))
        values_arr = np.asarray(values, dtype=float)
        run_best = (np.maximum if maximize else np.minimum).accumulate(values_arr)
        # robust y-limits + diverged-trial mask (shared with hpo_figure) —
        # a handful of diverged trials can't collapse this inset either
        sc_lo, sc_hi, sc_pad, sc_masks = robust_hpo_ylim([values_arr], maximize)
        sc_diverged = sc_masks[0]
        ax.plot(xs[~sc_diverged], values_arr[~sc_diverged], "o", ms=4,
                color=p["series"][0], alpha=0.75, label="trial")
        if sc_diverged.any():
            ax.scatter(xs[sc_diverged],
                       np.full(int(sc_diverged.sum()), sc_lo),
                       marker="v" if maximize else "^", s=22, color=p["muted"],
                       zorder=3,
                       label=f"{int(sc_diverged.sum())} diverged")
        ax.plot(xs, run_best, color=p["series"][0], lw=1.4, ls=(0, (4, 3)),
                label="running best")
        ax.plot(best_i, values[best_i], marker="*", ms=15,
                color=p["warning"], markeredgecolor=p["ink"],
                markeredgewidth=0.7, zorder=6)
        ax.annotate(f"{values[best_i]:.3f}", (best_i, values[best_i]),
                    textcoords="offset points", xytext=(-8, -13),
                    ha="right", fontsize=7.5, color=p["ink"], **mono)
        if sc_lo is not None:
            ax.set_ylim(sc_lo - sc_pad, sc_hi + sc_pad)
        ax.set_xlabel("trial")
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=7, loc="lower right")

        # (3) run summary card
        ax = fig.add_subplot(gs[0, 2])
        _card(ax, p, "RUN SUMMARY")
        ax.set_xticks([])
        ax.set_yticks([])
        bp = trials[best_i].get("params", {})
        te = metrics.get("test", {})
        tm = metrics.get("timings", {})

        # collapse the raw Optuna depth/width keys (n_blocks +
        # hidden_layers_<i>_log, which vary in count trial to trial) into
        # one readable row; prefer an explicit widths list from the run's
        # own artifacts if one is present.
        hidden_widths = (metrics.get("hidden_layers")
                          or metrics.get("architecture", {}).get("hidden_layers"))
        if not hidden_widths:
            log_keys = sorted(
                (k for k in bp if re.match(r"hidden_layers_\d+_log$", k)),
                key=lambda k: int(k.split("_")[2]))
            if log_keys:
                hidden_widths = [round(float(np.exp(bp[k]))) for k in log_keys]
        collapsed = {"n_blocks"} | {k for k in bp
                                     if re.match(r"hidden_layers_\d+_log$", k)}
        extra_params = {k: v for k, v in bp.items() if k not in collapsed}

        lines = [
            ("MODEL", model_name),
            ("TRIALS", f"{len(trials)} (TPE, {'max' if maximize else 'min'} "
                        f"{metric_name})"),
            (f"BEST {metric_name.upper().replace('_R2', ' R²')}",
             f"{values[best_i]:.4f}"),
            ("TEST R²", f"{te.get('r2', float('nan')):.4f}"),
            ("TEST RMSE", f"{te.get('rmse', float('nan')):.3f} {units}"),
            ("FIT TIME", fmt_metric(tm.get("train_seconds"), "runtime")
             if tm.get("train_seconds") else "—"),
        ]
        if hidden_widths:
            lines.append(("HIDDEN LAYERS", str(list(hidden_widths))))
        lines += [(_param_label(k).upper(),
                    f"{v:.3g}" if isinstance(v, float) else str(v))
                   for k, v in extra_params.items()]
        lines = lines[:10]  # cap card rows so it never overflows into (6)
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
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7, loc="upper right")

        # (5) which hyperparameters mattered (rank correlation with score)
        ax = fig.add_subplot(gs[1, 1])
        _card(ax, p, "PARAMETER SENSITIVITY")
        # keys common to every trial only — variable_list spaces (e.g. a
        # tunable number of hidden-layer blocks) give trials different
        # param counts, so anything not shared by all of them is dropped
        # rather than KeyError-ing on the Spearman pass below.
        common_keys = set(trials[0].get("params", {}))
        for t in trials[1:]:
            common_keys &= set(t.get("params", {}))
        pkeys = [k for k in trials[0].get("params", {})
                 if k in common_keys
                 and isinstance(trials[0]["params"][k], (int, float))]
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
            ax.text(0.02 if rho[o] < 0 else -0.02, y, _param_label(pkeys[o]),
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
            ax.set_xlabel(f"ground truth [{units}]")
            ax.set_ylabel("prediction")
            _stat_chip(ax, f"R² = {r2_score(y_t, y_q):.3f}", p,
                       loc=(0.05, 0.93))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "predictions unavailable", fontsize=8,
                    ha="center", va="center", color=p["muted"], **mono)
            ax.set_xticks([])
            ax.set_yticks([])
        return fig




# ------------------------------------------------------------- public API

def _finish(fig, out_stem):
    if fig is not None and out_stem is not None:
        save_figure(fig, Path(out_stem))
    return fig


def parity_figure(run_dir, mode: str = "light", *, model_name: str | None = None,
                  units: str = "a.u.", out_stem=None, singles: bool = False):
    """Density parity (train/test) + test residual distribution.

    Returns the Figure (a dict of Figures when ``singles``), or None when the
    run has no test predictions. Saved via ``save_figure`` when ``out_stem``
    is given (not supported with ``singles``).
    """
    run_dir = Path(run_dir)
    name = _resolve_model(run_dir, model_name)
    if name is None:
        return None
    fig = _parity(run_dir, mode, name, units, singles)
    return fig if singles else _finish(fig, out_stem)


def training_figure(run_dir, mode: str = "light", *, model_name: str | None = None,
                    out_stem=None):
    """Train/val loss of the final model (or the best HPO trial) with gap + fit."""
    run_dir = Path(run_dir)
    name = _resolve_model(run_dir, model_name)
    if name is None:
        return None
    return _finish(_training(run_dir, mode, name), out_stem)


def hpo_figure(run_dir, mode: str = "light", *, reference: float | None = None,
               model_name: str | None = None, out_stem=None):
    """HPO convergence; None when the run has no HPO artifacts."""
    run_dir = Path(run_dir)
    return _finish(_hpo(run_dir, mode, reference, model_name or detect_hpo_model(run_dir)),
                   out_stem)


def mission_control_figure(run_dir, mode: str = "light", *,
                           model_name: str | None = None, units: str = "a.u.",
                           out_stem=None):
    """One-look HPO campaign dashboard; None when the run has no HPO artifacts."""
    run_dir = Path(run_dir)
    name = _resolve_model(run_dir, model_name)
    if name is None:
        return None
    return _finish(_mission_control(run_dir, mode, name, units), out_stem)


def render_run(run_dir, out_dir=None, modes=("light",), only=None,
               model_name: str | None = None, units: str = "a.u.") -> dict[str, Path]:
    """Render every available run figure to ``out_dir`` (default ``<run>/figures``).

    Writes ``<name>_<mode>.{png,svg,pdf}`` for mission_control, hpo_convergence,
    training_curves and parity, skipping figures whose artifacts are missing.
    Returns ``{"<name>_<mode>": <png path>}`` for what was written.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "figures"
    names = [n for n in FIGURE_NAMES if only is None or n in only]
    builders = {
        "mission_control": lambda m: mission_control_figure(
            run_dir, m, model_name=model_name, units=units),
        "hpo_convergence": lambda m: hpo_figure(run_dir, m, model_name=model_name),
        "training_curves": lambda m: training_figure(run_dir, m, model_name=model_name),
        "parity": lambda m: parity_figure(run_dir, m, model_name=model_name,
                                          units=units),
    }
    written: dict[str, Path] = {}
    for mode in modes:
        for name in names:
            try:
                fig = builders[name](mode)
            except FileNotFoundError:
                fig = None
            if fig is None:
                continue
            stem = out_dir / f"{name}_{mode}"
            save_figure(fig, stem)
            plt.close(fig)
            written[f"{name}_{mode}"] = stem.with_suffix(".png")
    return written
