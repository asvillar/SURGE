"""Inline SVG thumbnails of a benchmark run's held-out predictions.

Reads the ``predictions.npz`` written next to ``result.json`` by
``surge bench`` (see :func:`surge.benchmarks.base.write_predictions`) and
renders, with the SURGE theme:

* regression -> parity density (log counts, dashed identity, R^2 chip);
  multi-output targets are z-scored per column and pooled;
* classification -> row-normalised confusion matrix with an accuracy chip;
* field tasks (``field_shape`` stored) -> truth | prediction | error triptych
  of the median-error sample.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

PREDICTIONS_FILE = "predictions.npz"
_MAX_CLASSES = 16


def load_predictions(result_dir: Path | str) -> dict | None:
    """Return the arrays of ``<result_dir>/predictions.npz`` or ``None``."""
    path = Path(result_dir) / PREDICTIONS_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _chip(ax, text: str, p: dict) -> None:
    ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=p["ink"],
            bbox=dict(boxstyle="round,pad=0.3", fc=p["surface"], ec=p["muted"], lw=0.6))


def _parity(ax, d: dict, p: dict, mode: str) -> None:
    from matplotlib.colors import LogNorm
    from sklearn.metrics import r2_score

    from surge.viz.theme import density_cmap

    yt, yp = d["y_true"].astype(float), d["y_pred"].astype(float)
    if yt.ndim == 1:
        yt = yt[:, None]
    yp = yp.reshape(len(yt), -1)
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
    if yt.shape[1] > 1:  # pool outputs on a common (z-scored) scale
        mu, sd = yt.mean(0), yt.std(0) + 1e-12
        yt, yp = (yt - mu) / sd, (yp - mu) / sd
        ax.set_xlabel("truth (z-scored, pooled)")
        ax.set_ylabel("prediction (z-scored)")
    else:
        ax.set_xlabel("truth")
        ax.set_ylabel("prediction")
    t, q = yt.ravel(), yp.ravel()
    lo, hi = float(min(t.min(), q.min())), float(max(t.max(), q.max()))
    pad = 0.05 * (hi - lo or 1.0)
    bins = np.linspace(lo - pad, hi + pad, 56)
    cmap = density_cmap(mode)
    ax.set_facecolor(cmap.get_under())
    ax.hist2d(t, q, bins=[bins, bins], cmap=cmap, norm=LogNorm(vmin=1), cmin=1,
              rasterized=True)
    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], ls="--", lw=1, color=p["ink2"])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    ax.set_aspect("equal")
    _chip(ax, f"R² = {r2:.3f}", p)


def _confusion(ax, d: dict, p: dict, mode: str) -> None:
    from surge.viz.theme import sequential_cmap

    yt = d["y_true"].astype(int).ravel()
    yp = d["y_pred"].astype(int).ravel()
    classes = np.unique(np.concatenate([yt, yp]))
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)))
    for a, b in zip(yt, yp):
        m[idx[a], idx[b]] += 1
    rows = m.sum(1, keepdims=True)
    norm = m / np.where(rows == 0, 1, rows)
    ax.imshow(norm, cmap=sequential_cmap(mode), vmin=0, vmax=1, aspect="equal")
    n = len(classes)
    if n <= _MAX_CLASSES:
        ax.set_xticks(range(n), [str(c) for c in classes], fontsize=6)
        ax.set_yticks(range(n), [str(c) for c in classes], fontsize=6)
        if n <= 8:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color=p["page"] if norm[i, j] > 0.55 else p["ink"])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    _chip(ax, f"accuracy = {float((yt == yp).mean()):.3f}", p)


def _as_image(v: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    a = v.reshape(shape)
    if a.ndim == 3 and a.shape[0] <= 8:
        a = a[0]
    elif a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    return a


def _triptych(fig, d: dict, p: dict, mode: str) -> None:
    from matplotlib.colors import CenteredNorm

    from surge.viz.theme import diverging_cmap, sequential_cmap

    shape = tuple(int(s) for s in d["field_shape"])
    yt, yp = d["y_true"].astype(float), d["y_pred"].astype(float)
    err = np.linalg.norm(yp - yt, axis=1) / (np.linalg.norm(yt, axis=1) + 1e-12)
    i = int(np.argsort(err)[len(err) // 2])  # median-error sample
    t, q = _as_image(yt[i], shape), _as_image(yp[i], shape)
    e = q - t
    axes = fig.subplots(1, 3)
    if t.ndim == 1:
        for ax, a, name in zip(axes, (t, q, e), ("truth", "prediction", "error")):
            ax.plot(a, color=p["series"][0], lw=1.2)
            ax.set_title(name, fontsize=8)
    else:
        vmin, vmax = float(min(t.min(), q.min())), float(max(t.max(), q.max()))
        for ax, a, name in zip(axes[:2], (t, q), ("truth", "prediction")):
            ax.imshow(a, cmap=sequential_cmap(mode), vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=8)
        axes[2].imshow(e, cmap=diverging_cmap(mode), norm=CenteredNorm(0.0))
        axes[2].set_title("error", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.text(0.5, 0.02, f"median-error sample · rel. L2 = {err[i]:.3f}",
             ha="center", fontsize=7, color=p["ink2"])


def prediction_thumbnail(result_dir: Path | str, task_type: str,
                         mode: str = "light") -> str | None:
    """Inline ``<svg>`` string for the run in *result_dir*, or ``None`` when
    the run has no (readable) ``predictions.npz``."""
    d = load_predictions(result_dir)
    if d is None or "y_true" not in d or "y_pred" not in d:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        from surge.viz.theme import surge_theme

        is_field = "field_shape" in d
        with surge_theme(mode) as p, matplotlib.rc_context(
                {"svg.fonttype": "none", "svg.hashsalt": "surge"}):
            if is_field:
                fig = plt.figure(figsize=(4.4, 1.9), dpi=100)
                _triptych(fig, d, p, mode)
            else:
                fig, ax = plt.subplots(figsize=(4.0, 3.1), dpi=100)
                if task_type == "classification":
                    _confusion(ax, d, p, mode)
                else:
                    _parity(ax, d, p, mode)
                fig.tight_layout()
            buf = io.StringIO()
            fig.savefig(buf, format="svg", transparent=True)
            plt.close(fig)
    except Exception:
        return None
    svg = buf.getvalue()
    svg = svg[svg.find("<svg"):]
    return re.sub(r'\s(width|height)="[\d.]+pt"', "", svg, count=2)


# Set by ``build_report`` so ``_bench_card`` can find result dirs / the theme.
CONTEXT: dict = {"base": Path("."), "mode": "dark"}


def prediction_figure_html(bench_results: dict, pm: str | None, task_type: str) -> str:
    """``<figure>`` with the best model's prediction thumbnail (newest sweep),
    or ``""`` when that run has no ``predictions.npz`` (older results)."""
    from surge.report.leaderboard import _best_result_path

    rel = _best_result_path(bench_results, pm)
    if not rel:
        return ""
    result_dir = Path(CONTEXT["base"]) / Path(rel).parent
    import html
    import json

    try:
        res = json.loads((Path(CONTEXT["base"]) / rel).read_text())
    except Exception:
        res = {}
    model = res.get("model_key", "?")
    svg = prediction_thumbnail(result_dir, res.get("task_type", task_type),
                               CONTEXT["mode"])
    if svg is None:
        return ""
    return ("<figure class='preview'>" + svg + "<figcaption>best model predictions · "
            f"{html.escape(str(model))} · test split</figcaption></figure>")
