"""Animated renderings of real SURGE run artifacts.

Every ``*_frames`` function writes ``f_%04d.png`` frames with an IDENTICAL
layout (fixed figure size, axes limits computed from the full data up front)
so they play as a smooth animation, and returns the list of paths.
``frames_to_video`` / ``frames_to_gif`` encode a frame directory with ffmpeg.

CLI::

    python -m surge.viz.animate --run runs/qlknn_hpo_200_best \
        --reports benchmark_reports --out audit/presentation/assets/anim --mode dark
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

from surge.viz.hpo import robust_hpo_ylim  # noqa: E402
from surge.viz.theme import (  # noqa: E402
    density_cmap,
    fmt_metric,
    sequential_cmap,
    surge_theme,
)

FIG_W, FIG_H, DPI = 12.8, 7.2, 100  # 1280 x 720, even
_MONO = {"family": "monospace"}
_STAR = "#FFD700"


# ----------------------------------------------------------------- helpers

def _write_frames(n: int, out_dir: Path, mode: str,
                  draw: Callable[[int, dict], plt.Figure]) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.png"):
        old.unlink()
    paths = []
    for k in range(n):
        with surge_theme(mode) as p:
            fig = draw(k, p)
            path = out_dir / f"f_{k:04d}.png"
            fig.savefig(path, dpi=DPI, facecolor=p["page"], bbox_inches=None)
            plt.close(fig)
        paths.append(path)
    return paths


def _new_fig(p):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.set_layout_engine("none")
    fig.set_facecolor(p["page"])
    return fig


def _card(ax, p, title):
    ax.set_facecolor(p["surface"])
    for s in ax.spines.values():
        s.set_color(p["grid"])
        s.set_linewidth(0.8)
        s.set_visible(True)
    ax.set_title(title, fontsize=8.5, loc="left", color=p["ink2"], pad=5, **_MONO)


def _n_shown(k: int, n_frames: int, n_trials: int) -> int:
    """Trials revealed at frame k (0..n_frames-1): ceil(N*k/(F-1)), at least 1."""
    frac = k / max(n_frames - 1, 1)
    return max(1, min(n_trials, math.ceil(n_trials * frac)))


def _widths(params: dict) -> list[int]:
    keys = sorted((k for k in params if re.match(r"hidden_layers_\d+_log$", k)),
                  key=lambda k: int(k.split("_")[2]))
    return [round(float(np.exp(params[k]))) for k in keys]


def _param_label(key: str) -> str:
    fixed = {"n_blocks": "depth (blocks)", "learning_rate": "learning rate",
             "batch_size": "batch size", "dropout_rate": "dropout"}
    if key in fixed:
        return fixed[key]
    m = re.match(r"hidden_layers_(\d+)_log$", key)
    return f"width · block {m.group(1)}" if m else key.replace("_", " ")


class _Campaign:
    """HPO campaign loaded from a run directory (manifest + histories)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        lines = (self.run_dir / "hpo_trials_manifest.jsonl").read_text().splitlines()
        self.trials = [json.loads(ln) for ln in lines if ln.strip()]
        self.n = len(self.trials)
        self.values = np.array([t["value"] for t in self.trials], dtype=float)
        self.metric = self.trials[0].get("metric", "val_r2")
        self.lr = np.array([t["params"].get("learning_rate", np.nan)
                            for t in self.trials], dtype=float)
        self.hists: dict[int, dict] = {}
        for i in range(self.n):
            f = self.run_dir / f"hpo_trial_{i:04d}_training_history.json"
            if f.exists():
                h = json.loads(f.read_text())
                self.hists[i] = {
                    "ep": np.array([r["epoch"] for r in h]),
                    "tl": np.array([r["train_loss"] for r in h]),
                    "vl": np.array([r["val_loss"] for r in h])}
        self.best_final = int(np.argmax(self.values))
        self.run_best = np.maximum.accumulate(self.values)
        lo, hi, pad, masks = robust_hpo_ylim([self.values], True)
        if lo is None:
            lo, hi, pad = float(self.values.min()), float(self.values.max()), 0.02
            masks = [np.zeros(self.n, dtype=bool)]
        self.ylim = (lo - pad, hi + pad)
        self.ylo = lo
        self.diverged = masks[0]
        lr = np.log10(self.lr[np.isfinite(self.lr)]) if np.isfinite(self.lr).any() else np.array([-4., -2.])
        self.lr_norm = Normalize(float(lr.min()), float(lr.max()))

    def best_upto(self, m: int) -> int:
        return int(np.argmax(self.values[:m]))


def _draw_convergence(ax, p, c: _Campaign, m: int, compact: bool = False):
    """Search-convergence scatter (first m trials) with fixed limits."""
    xs = np.arange(m)
    v = c.values[:m]
    keep = ~c.diverged[:m]
    cmap = sequential_cmap(p["_mode"])
    lr = np.log10(np.where(np.isfinite(c.lr[:m]), c.lr[:m], 1e-3))
    ax.scatter(xs[keep], v[keep], c=lr[keep], cmap=cmap, norm=c.lr_norm,
               s=14 if compact else 22, alpha=0.88, zorder=2, label="trial")
    dv = c.diverged[:m]
    if dv.any():
        ax.scatter(xs[dv], np.full(int(dv.sum()), c.ylo), marker="v", s=24,
                   color=p["muted"], zorder=2, label=f"{int(dv.sum())} diverged")
    ax.plot(xs, c.run_best[:m], color=p["series"][0], lw=1.9, ls=(0, (5, 2)),
            drawstyle="steps-post", zorder=3, label="best so far")
    if m > c.best_final:  # the final best trial has landed
        bv = c.values[c.best_final]
        ax.scatter([c.best_final], [bv], marker="*", s=300 if compact else 430,
                   zorder=5, facecolor=_STAR, edgecolor=p["series"][0], linewidth=1.5)
        ax.annotate(f"{bv:.4f}", (c.best_final, bv), textcoords="offset points",
                    xytext=(-14, -26), ha="right", fontsize=8, color=p["ink"],
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": p["page"],
                          "edgecolor": p["grid"], "linewidth": 0.7}, **_MONO)
    ax.set_xlim(-c.n * 0.02, c.n * 1.02)
    ax.set_ylim(*c.ylim)
    ax.set_xlabel("trial")
    ax.set_ylabel(c.metric)
    ax.legend(loc="lower right", fontsize=7)


# ----------------------------------------------------------- 1. HPO evolution

def hpo_evolution_frames(run_dir: Path, out_dir: Path, mode: str,
                         n_frames: int = 60) -> list[Path]:
    """Convergence scatter revealing trials progressively (fixed robust ylim)."""
    c = _Campaign(run_dir)

    def draw(k, p):
        p["_mode"] = mode
        m = _n_shown(k, n_frames, c.n)
        fig = _new_fig(p)
        ax = fig.add_axes([0.07, 0.11, 0.80, 0.74])
        cax = fig.add_axes([0.89, 0.11, 0.015, 0.74])
        _draw_convergence(ax, p, c, m)
        cb = fig.colorbar(ScalarMappable(norm=c.lr_norm, cmap=sequential_cmap(mode)),
                          cax=cax)
        cb.set_label(r"$\log_{10}$ learning rate", fontsize=8, color=p["ink2"])
        cb.ax.tick_params(labelsize=7, colors=p["muted"])
        best = float(c.run_best[m - 1])
        fig.text(0.07, 0.945, f"HPO · trial {m} / {c.n} · best {c.metric.replace('_r2', ' R²')} "
                 f"{best:.4f}", fontsize=17, fontweight="bold", color=p["ink"], va="top")
        return fig

    return _write_frames(n_frames, out_dir, mode, draw)


# --------------------------------------------------------- 2. mission control

def _find_model(run_dir: Path) -> str | None:
    try:
        met = json.loads((Path(run_dir) / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    return next((k for k, v in met.items() if isinstance(v, dict) and "test" in v), None)


def _load_parity(run_dir: Path, model: str | None):
    import pandas as pd
    files = sorted((Path(run_dir) / "predictions").glob(f"{model or '*'}_test.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0])
    yt = df[next(col for col in df.columns if col.startswith("y_true"))].to_numpy().ravel()
    yp = df[next(col for col in df.columns if col.startswith("y_pred"))].to_numpy().ravel()
    return yt, yp


def mission_control_frames(run_dir: Path, out_dir: Path, mode: str,
                           n_frames: int = 60) -> list[Path]:
    """Mission-control dashboard building up trial by trial; the parity panel
    fades in over the last ~10 % of frames."""
    from scipy.stats import spearmanr

    c = _Campaign(run_dir)
    model = _find_model(run_dir) or "qlknn_residual_mlp"
    try:
        metrics = json.loads((Path(run_dir) / "metrics.json").read_text()).get(model, {})
    except (OSError, ValueError):
        metrics = {}
    test = metrics.get("test", {})
    parity = _load_parity(run_dir, model)
    fade_n = max(1, round(0.10 * n_frames))

    # fixed limits and static parity histogram, computed once
    all_ep = max(len(h["ep"]) for h in c.hists.values())
    all_vl = np.concatenate([h["vl"] for h in c.hists.values()])
    all_tl = np.concatenate([h["tl"] for h in c.hists.values()])
    loss_all = np.concatenate([all_vl, all_tl])
    loss_all = loss_all[np.isfinite(loss_all) & (loss_all > 0)]
    vl_lim = (float(np.percentile(loss_all, 0.5)) * 0.8, float(np.percentile(all_vl[all_vl > 0], 99)) * 1.2)
    common = set(c.trials[0]["params"])
    for t in c.trials[1:]:
        common &= set(t["params"])
    pkeys = [k for k in c.trials[0]["params"]
             if k in common and isinstance(c.trials[0]["params"][k], (int, float))]
    r2 = lims = counts = None
    if parity is not None:
        yt, yp = parity
        from sklearn.metrics import r2_score
        r2 = float(r2_score(yt, yp))
        lims = (float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max())))
        bins = np.linspace(*lims, 48)
        counts, _, _ = np.histogram2d(yt, yp, bins=[bins, bins])
        counts = np.ma.masked_less(counts.T, 1)

    def draw(k, p):
        p["_mode"] = mode
        m = _n_shown(k, n_frames, c.n)
        bi = c.best_upto(m)
        fig = _new_fig(p)
        gs = fig.add_gridspec(2, 3, left=0.05, right=0.975, top=0.845, bottom=0.075,
                              wspace=0.24, hspace=0.36)
        fig.text(0.05, 0.965, "SURGE · MISSION CONTROL", fontsize=15, fontweight="bold",
                 color=p["ink"], va="top", **_MONO)
        fig.text(0.05, 0.915, f"HPO campaign — {model}  ·  Optuna TPE  ·  "
                 f"trial {m} / {c.n}", fontsize=9.5, color=p["ink2"], va="top", **_MONO)

        # (1) val loss, trials accumulate
        ax = fig.add_subplot(gs[0, 0])
        _card(ax, p, "VAL LOSS — ALL TRIALS")
        for t in range(m):
            if t in c.hists and t != bi:
                h = c.hists[t]
                ax.plot(h["ep"], h["vl"], color=p["muted"], lw=0.9, alpha=0.5, zorder=2)
        if bi in c.hists:
            ax.plot(c.hists[bi]["ep"], c.hists[bi]["vl"], color=p["series"][0], lw=2.2,
                    zorder=5, label=f"best so far (trial {bi})")
        ax.set_yscale("log")
        ax.set_xlim(0, all_ep * 1.02)
        ax.set_ylim(*vl_lim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val loss")
        ax.legend(fontsize=7, loc="upper right")

        # (2) search convergence
        ax = fig.add_subplot(gs[0, 1])
        _card(ax, p, "SEARCH CONVERGENCE")
        _draw_convergence(ax, p, c, m, compact=True)

        # (3) run summary
        ax = fig.add_subplot(gs[0, 2])
        _card(ax, p, "RUN SUMMARY")
        ax.set_xticks([])
        ax.set_yticks([])
        bp = c.trials[bi]["params"]
        rows = [("MODEL", model), ("TRIALS", f"{m} / {c.n}"),
                ("BEST VAL R²", f"{c.values[bi]:.4f}"), ("BEST TRIAL", str(bi)),
                ("DEPTH", f"{bp.get('n_blocks', '—')} blocks"),
                ("WIDTHS", str(_widths(bp) or "—")),
                ("LEARNING RATE", f"{bp.get('learning_rate', float('nan')):.3g}"),
                ("BATCH SIZE", str(bp.get("batch_size", "—"))),
                ("DROPOUT", f"{bp.get('dropout_rate', float('nan')):.3f}")]
        if test and k >= n_frames - fade_n:
            rows.append(("TEST R²", f"{test.get('r2', float('nan')):.4f}"))
        for j, (a, b) in enumerate(rows):
            y = 0.93 - j * 0.092
            ax.text(0.06, y, a, transform=ax.transAxes, fontsize=7.5, color=p["muted"],
                    va="top", **_MONO)
            ax.text(0.52, y, b, transform=ax.transAxes, fontsize=8, color=p["ink"],
                    va="top", **_MONO)

        # (4) best-so-far train vs val
        ax = fig.add_subplot(gs[1, 0])
        _card(ax, p, f"BEST TRIAL {bi} — TRAIN VS VAL")
        if bi in c.hists:
            h = c.hists[bi]
            ax.plot(h["ep"], h["tl"], color=p["series"][1], lw=1.4, label="train")
            ax.plot(h["ep"], h["vl"], color=p["series"][0], lw=1.6, label="val")
            ax.fill_between(h["ep"], h["tl"], h["vl"], color=p["series"][0], alpha=0.10)
            be = int(h["ep"][np.argmin(h["vl"])])
            ax.axvline(be, color=p["good"], lw=1.0, ls=(0, (4, 3)))
            right = be > 0.8 * all_ep
            ax.text(be, float(h["vl"].min()) * 0.8, f"best ep {be} " if right else f" best ep {be}",
                    fontsize=7, color=p["good_text"], ha="right" if right else "left",
                    va="top", **_MONO)
        ax.set_yscale("log")
        ax.set_xlim(0, all_ep * 1.02)
        ax.set_ylim(*vl_lim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=7, loc="upper right")

        # (5) sensitivity on trials so far
        ax = fig.add_subplot(gs[1, 1])
        _card(ax, p, "PARAMETER SENSITIVITY")
        ax.set_xlim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel("Spearman ρ (param vs score)")
        if m >= 10:
            rho = []
            for key in pkeys:
                vals = [t["params"][key] for t in c.trials[:m]]
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = spearmanr(vals, c.values[:m]).statistic
                rho.append(0.0 if np.isnan(r) else float(r))
            for y, o in enumerate(np.argsort(np.abs(rho))):
                col = p["series"][0] if rho[o] >= 0 else p["series"][1]
                ax.barh(y, rho[o], height=0.55, color=col, alpha=0.85)
                ax.text(0.02 if rho[o] < 0 else -0.02, y, _param_label(pkeys[o]),
                        ha="left" if rho[o] < 0 else "right", va="center",
                        fontsize=7.5, color=p["ink"], **_MONO)
            ax.set_ylim(-0.6, len(pkeys) - 0.4)
            ax.axvline(0, color=p["axis"], lw=0.9)
        else:
            ax.text(0.5, 0.5, "needs ≥ 10 trials", transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color=p["muted"], **_MONO)

        # (6) parity fades in at the end
        ax = fig.add_subplot(gs[1, 2])
        _card(ax, p, "OUTCOME — TEST PARITY")
        if lims is not None:
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_aspect("equal")
            ax.set_xlabel("ground truth [gB]")
            ax.set_ylabel("prediction")
            alpha = max(0.0, min(1.0, (k - (n_frames - fade_n - 1)) / fade_n))
            if alpha > 0:
                cmap = density_cmap(mode)
                edges = np.linspace(*lims, 48)
                ax.pcolormesh(edges, edges, counts, cmap=cmap, norm=LogNorm(vmin=1),
                              alpha=alpha, rasterized=True)
                ax.plot(lims, lims, color=p["ink2"], lw=1.0, ls=(0, (5, 3)), alpha=alpha)
                ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=8,
                        color=p["ink2"], va="top", alpha=alpha,
                        bbox={"boxstyle": "round,pad=0.45", "facecolor": p["page"],
                              "edgecolor": p["grid"], "linewidth": 0.7, "alpha": alpha})
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return fig

    return _write_frames(n_frames, out_dir, mode, draw)


# ------------------------------------------------------------- 3. leaderboard

def _threshold(key: str) -> float | None:
    try:
        from surge.report.leaderboard import load_metadata
        spec = load_metadata()[key]["threshold"]
        if any(t in spec.lower() for t in ("rmse", "mse", "mae", "error")):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", spec)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 - decorative line only
        return None


def leaderboard_frames(reports_dir: Path, benchmark_key: str, out_dir: Path,
                       mode: str, order: str = "by_date", n_hold: int = 8,
                       threshold: float | None = None) -> list[Path]:
    """One frame per model added (sorted by first-result date); bars re-sort by
    mean test R² each frame; the gate line is fixed. ``n_hold`` copies of the
    final frame are appended."""
    rows: dict[str, dict] = defaultdict(lambda: {"scores": [], "first": "~"})
    for rj in sorted((Path(reports_dir) / benchmark_key).glob("*/result.json")):
        d = json.loads(rj.read_text())
        met = d.get("metrics", {})
        score = met.get("test_r2", met.get("test_accuracy"))
        if d.get("model_key") is None or score is None:
            continue
        r = rows[d["model_key"]]
        r["scores"].append(float(score))
        r["first"] = min(r["first"], str(d.get("timestamp") or rj.parent.name))
    if not rows:
        return []
    models = [{"model": k, "mean": statistics.fmean(v["scores"]), "first": v["first"]}
              for k, v in rows.items()]
    if order == "by_date":
        models.sort(key=lambda r: r["first"])
    else:
        models.sort(key=lambda r: -r["mean"])
    n_models = len(models)
    thr = threshold if threshold is not None else _threshold(benchmark_key)
    lo = min(r["mean"] for r in models)
    xlo = max(0.0, math.floor((lo - 0.05) * 10) / 10)
    xhi = 1.0

    def draw(k, p):
        shown = sorted(models[:k + 1], key=lambda r: -r["mean"])
        newest = models[k]["model"]
        fig = _new_fig(p)
        ax = fig.add_axes([0.27, 0.11, 0.69, 0.74])
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(n_models - 0.4, -0.8)  # fixed slots, rank 0 at top
        for i, r in enumerate(shown):
            passed = thr is not None and r["mean"] >= thr
            col = p["good"] if passed else p["series"][0]
            ax.hlines(i, xlo, r["mean"], color=col, lw=2.2,
                      alpha=1.0 if r["model"] == newest else 0.6)
            ax.scatter([r["mean"]], [i], s=90, color=col, zorder=4,
                       edgecolor=_STAR if r["model"] == newest else col, linewidth=2.0)
            ax.text(r["mean"] + 0.006, i, f"{r['mean']:.3f}", va="center", fontsize=9,
                    color=p["ink"], zorder=6,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": p["surface"],
                          "edgecolor": "none"}, **_MONO)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([r["model"] for r in shown], fontsize=10, **_MONO)
        for tl, r in zip(ax.get_yticklabels(), shown):
            tl.set_color(p["ink"] if r["model"] == newest else p["ink2"])
        if thr is not None:
            ax.axvline(thr, color=p["good"], lw=1.4, ls=(0, (5, 3)), zorder=1)
            ax.text(thr, -0.62, f" gate R² ≥ {thr:.2f}", fontsize=8, color=p["good_text"],
                    va="center", **_MONO)
        ax.set_xlabel("test R² (mean over runs)")
        ax.grid(axis="x", color=p["grid"], lw=0.6)
        ax.grid(axis="y", visible=False)
        fig.text(0.07, 0.945, f"benchmark · {benchmark_key} · {k + 1} models",
                 fontsize=17, fontweight="bold", color=p["ink"], va="top")
        return fig

    paths = _write_frames(n_models, out_dir, mode, draw)
    for j in range(n_hold):
        dst = Path(out_dir) / f"f_{n_models + j:04d}.png"
        shutil.copyfile(paths[-1], dst)
        paths.append(dst)
    return paths


# -------------------------------------------------------------- 4. training

def training_frames(history_json: Path, out_dir: Path, mode: str,
                    n_frames: int = 40) -> list[Path]:
    """Train/val loss curves drawing in; early-stopping marker lands at the end."""
    h = json.loads(Path(history_json).read_text())
    ep = np.array([r["epoch"] for r in h])
    tl = np.array([r["train_loss"] for r in h])
    vl = np.array([r["val_loss"] for r in h])
    bi = int(np.argmin(vl))
    allv = np.concatenate([tl, vl])
    ylim = (float(allv.min()) * 0.8, float(allv.max()) * 1.25)
    tail = max(1, round(0.10 * n_frames))

    def draw(k, p):
        m = max(2, min(len(ep), math.ceil(len(ep) * (k + 1) / max(n_frames - 2, 1)))
                if k < n_frames - 1 else len(ep))
        fig = _new_fig(p)
        ax = fig.add_axes([0.07, 0.11, 0.88, 0.74])
        ax.plot(ep[:m], tl[:m], color=p["series"][1], lw=2.0, label="train")
        ax.plot(ep[:m], vl[:m], color=p["series"][0], lw=2.4, label="val")
        ax.fill_between(ep[:m], tl[:m], vl[:m], color=p["series"][0], alpha=0.10)
        if k >= n_frames - tail:
            ax.axvline(ep[bi], color=p["good"], lw=1.4, ls=(0, (4, 3)))
            ax.scatter([ep[bi]], [vl[bi]], marker="*", s=420, zorder=5,
                       facecolor=_STAR, edgecolor=p["series"][0], linewidth=1.5)
            ax.text(ep[bi], ylim[1] * 0.85, f"  early stop · best epoch {ep[bi]}\n"
                    f"  val loss {fmt_metric(vl[bi])}", fontsize=9, color=p["good_text"],
                    va="top", **_MONO)
        ax.set_yscale("log")
        ax.set_xlim(ep.min() - 1, ep.max() + 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(loc="upper right", fontsize=9)
        fig.text(0.07, 0.945, f"Training · epoch {int(ep[m - 1])} / {int(ep[-1])}",
                 fontsize=17, fontweight="bold", color=p["ink"], va="top")
        return fig

    return _write_frames(n_frames, out_dir, mode, draw)


# ----------------------------------------------------------------- encoding

def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg") or (
        "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None)


def _run_ffmpeg(args: list[str]) -> None:
    exe = _ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg not found")
    subprocess.run([exe, "-y", "-loglevel", "error", *args], check=True)


def frames_to_video(frame_dir: Path, out_path: Path, fps: int = 12,
                    hold_last: float = 0.0) -> list[Path]:
    """Encode ``f_%04d.png`` frames to ``<out_path>.mp4`` (H.264 yuv420p) and
    ``<out_path>.webm`` (VP9). ``out_path`` may carry any suffix (replaced)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = ["-framerate", str(fps), "-i", str(Path(frame_dir) / "f_%04d.png")]
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if hold_last > 0:
        vf += f",tpad=stop_mode=clone:stop_duration={hold_last}"
    mp4, webm = out_path.with_suffix(".mp4"), out_path.with_suffix(".webm")
    _run_ffmpeg([*src, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", "-movflags", "+faststart", str(mp4)])
    _run_ffmpeg([*src, "-vf", vf, "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p",
                 "-crf", "32", "-b:v", "0", str(webm)])
    return [mp4, webm]


def frames_to_gif(frame_dir: Path, out_path: Path, fps: int = 12,
                  width: int = 960, max_mb: float = 6.0) -> Path:
    """Palette GIF scaled to ``width`` px; shrinks until <= ``max_mb``."""
    out_path = Path(out_path).with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = ["-framerate", str(fps), "-i", str(Path(frame_dir) / "f_%04d.png")]
    for w, colors in ((width, 128), (int(width * 0.83), 96), (int(width * 0.67), 64)):
        vf = (f"scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}[pal];"
              "[b][pal]paletteuse=dither=bayer:bayer_scale=4")
        _run_ffmpeg([*src, "-filter_complex", vf, "-loop", "0", str(out_path)])
        if out_path.stat().st_size <= max_mb * 1024 * 1024:
            break
    return out_path


# ---------------------------------------------------------------------- CLI

def _best_history(run: Path) -> Path | None:
    c = _Campaign(run)
    f = run / f"hpo_trial_{c.best_final:04d}_training_history.json"
    return f if f.exists() else None


def render_all(run: Path, reports: Path, out: Path, mode: str,
               benchmark_key: str = "plasma.qlknn_transport", fps: int = 12,
               keep_frames: bool = True) -> dict[str, list[Path]]:
    out = Path(out)
    frames_root = out / "frames" / mode
    jobs = {
        "hpo_evolution": lambda d: hpo_evolution_frames(run, d, mode),
        "mission_control": lambda d: mission_control_frames(run, d, mode),
        "leaderboard": lambda d: leaderboard_frames(reports, benchmark_key, d, mode),
        "training": lambda d: training_frames(_best_history(run), d, mode),
    }
    result = {}
    for name, fn in jobs.items():
        fdir = frames_root / name
        frames = fn(fdir)
        if not frames:
            continue
        final = out / f"{name}_{mode}_final.png"
        shutil.copyfile(frames[-1], final)
        outs = [final]
        if _ffmpeg():
            outs += frames_to_video(fdir, out / f"{name}_{mode}", fps=fps, hold_last=1.5)
        result[name] = outs
        print(f"{name}: {len(frames)} frames -> {[o.name for o in outs]}")
    if not keep_frames:
        shutil.rmtree(frames_root, ignore_errors=True)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reports", type=Path, default=Path("benchmark_reports"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["light", "dark"], default="dark")
    ap.add_argument("--benchmark", default="plasma.qlknn_transport")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--no-frames", action="store_true", help="delete frame dirs afterwards")
    a = ap.parse_args(argv)
    render_all(a.run, a.reports, a.out, a.mode, a.benchmark, a.fps, not a.no_frames)


if __name__ == "__main__":
    main()
