"""Publication ROC figure for disruption-style binary classifiers.

``roc_figure`` draws the full ROC (a) and the low-false-alarm zoom (b, log x),
the regime in which a disruption predictor is judged, with AUROC in the legend
and the TPR at fixed false-alarm rates in a compact table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence  # noqa: UP035 — py3.9 support

import numpy as np

__all__ = ["roc_figure", "tpr_at_fpr"]


def _finish(fig, out_stem):
    if out_stem is not None:
        from surge.viz.theme import save_figure
        save_figure(fig, Path(out_stem))
    return fig


def _box(ax, text, p, loc, va="bottom", ha="right"):
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, fontsize=7,
            va=va, ha=ha, color=p["ink"], linespacing=1.4,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": p["surface"],
                  "edgecolor": p["ink2"], "linewidth": 0.8})


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, mark: float) -> float:
    """TPR of the last ROC point whose FPR does not exceed *mark*."""
    idx = np.searchsorted(fpr, mark, side="right") - 1
    return float(tpr[max(idx, 0)])


def _roc(y_true, y_score):
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(y_score, float).ravel()
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def roc_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], *,
               baseline: tuple[str, np.ndarray, np.ndarray] | None = None,
               fpr_marks: Sequence[float] = (0.01, 0.05),
               mode: str = "light", out_stem=None):
    """ROC (a) and low-false-alarm zoom (b) for ``{label: (y_true, y_score)}``.

    ``baseline`` = (label, y_true, y_score) is drawn dashed in ``ink2`` (e.g.
    a Greenwald-fraction physics rule). Returns the ``Figure``; pass
    ``out_stem`` to also export png/svg/pdf.
    """
    import matplotlib.pyplot as plt

    from surge.viz.theme import surge_theme

    if len(curves) > 8:
        raise ValueError("roc_figure supports at most 8 models")
    marks = list(fpr_marks)
    mlabels = [f"{100 * m:g} %" for m in marks]
    with surge_theme(mode) as p:
        entries = []  # (label, fpr, tpr, auc, style kwargs)
        for color, (name, (yt, ys)) in zip(p["series"], curves.items()):
            entries.append((name, *_roc(yt, ys), {"color": color, "lw": 1.7}))
        if baseline is not None:
            name, yt, ys = baseline
            entries.append((name, *_roc(yt, ys),
                            {"color": p["ink2"], "lw": 1.4,
                             "ls": (0, (5, 3))}))
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 4.2))
        for name, fpr, tpr, auc, kw in entries:
            for ax in (a, b):
                ax.plot(fpr, tpr, drawstyle="steps-post", zorder=3,
                        label=f"{name} — AUROC {auc:.4f}" if ax is a else None,
                        **kw)
        a.plot([0, 1], [0, 1], color=p["muted"], lw=0.8, ls=":", zorder=1)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_title("(a) ROC", fontsize=9.5)
        b.set_xscale("log")
        b.set_xlim(1e-4, 1)
        b.set_ylim(0, 1.02)
        b.set_title("(b) low false-alarm regime", fontsize=9.5)
        for m in marks:
            b.axvline(m, color=p["muted"], lw=0.8, zorder=1)
            b.text(m, 0.02, f" {100 * m:g} % FPR", rotation=90, fontsize=6.5,
                   color=p["muted"], va="bottom", ha="right")
        for ax in (a, b):
            ax.set_xlabel("false positive rate (false alarms)")
            ax.set_ylabel("true positive rate (precursor slices caught)")
        a.legend(loc="lower right", fontsize=7)
        head = "TPR at " + " / ".join(mlabels) + " FPR"
        width = max(len(e[0]) for e in entries)
        rows = [head] + [
            f"{e[0]:<{width}}  " + " / ".join(
                f"{tpr_at_fpr(e[1], e[2], m):.3f}" for m in marks)
            for e in entries]
        _box(a, "\n".join(rows), p, (0.97, 0.20))
        a.texts[-1].set_family("monospace")
    return _finish(fig, out_stem)
