"""Publication figures for the SOLPS-NN exhaust-surrogate benchmark.

SOLPS-ITER scan: 8 scalar inputs -> 3 exhaust scalars (P_SOL, peak heat flux,
particle flux) and a 2-D electron-temperature field on a 104x50 B2 grid.
All functions take plain numpy arrays in physical units (Te in eV, P_SOL in W,
heat flux in W/m^2, particle flux in atoms/s), draw with the SURGE theme and
return the ``Figure``; pass ``out_stem`` to also export png/svg/pdf.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence  # noqa: UP035 — py3.9 support

import numpy as np

__all__ = [
    "dataset_overview_figure",
    "error_cdf_figure",
    "field_figure",
    "metrics_from_predictions",
    "scalar_parity_figure",
    "target_profile_figure",
]

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def metrics_from_predictions(y_true, y_pred) -> dict:
    """Scalar-target metrics shared by figures and reports (positive data).

    ``median_rel_err`` and ``mape_linear`` are fractions (0.05 = 5 %);
    ``r2_log10`` and ``rel_l2_log10`` operate on log10 values.
    """
    t = np.asarray(y_true, dtype=float).ravel()
    q = np.asarray(y_pred, dtype=float).ravel()
    rel = np.abs(q - t) / np.abs(t)
    lt, lq = np.log10(t), np.log10(q)
    ss_res = float(np.sum((lt - lq) ** 2))
    ss_tot = float(np.sum((lt - lt.mean()) ** 2))
    return {
        "median_rel_err": float(np.median(rel)),
        "r2_log10": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rel_l2_log10": float(np.sqrt(ss_res / np.sum(lt ** 2))),
        "mape_linear": float(np.mean(rel)),
    }


def _finish(fig, out_stem):
    if out_stem is not None:
        from surge.viz.theme import save_figure
        save_figure(fig, Path(out_stem))
    return fig


def _box(ax, text, p, loc=(0.05, 0.95)):
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", color=p["ink"], linespacing=1.4,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": p["surface"],
                  "edgecolor": p["ink2"], "linewidth": 0.8})


def _log_ticks(ax, axis):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    fmt = FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(MaxNLocator(integer=True, nbins=5))
    a.set_major_formatter(fmt)


def _plain_log_ticks(axis):
    """Decimal tick labels (5, 10, 50) instead of 5×10^0 on a log axis."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if axis.get_scale() != "log":
        axis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------ scalar parity

def scalar_parity_figure(preds: dict[str, tuple[np.ndarray, np.ndarray]], *,
                         mode: str = "light",
                         labels: dict[str, str] | None = None,
                         units: dict[str, str] | None = None,
                         out_stem=None):
    """Log-log density parity, one panel per scalar target."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from surge.viz.theme import density_cmap, surge_theme

    labels = labels or {}
    units = units or {}
    keys = list(preds)
    logs = {k: (np.log10(np.asarray(preds[k][0], float).ravel()),
                np.log10(np.asarray(preds[k][1], float).ravel()))
            for k in keys}
    with surge_theme(mode) as p:
        cmap = density_cmap(mode)
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(3.3 * n + 0.8, 3.5),
                                 squeeze=False)
        axes = axes[0]
        peak = 1
        hists = {}
        for k in keys:
            lt, lq = logs[k]
            lo = min(lt.min(), lq.min())
            hi = max(lt.max(), lq.max())
            pad = 0.04 * (hi - lo)
            lims = (lo - pad, hi + pad)
            bins = np.linspace(*lims, 60)
            hists[k] = (lims, bins)
            peak = max(peak, np.histogram2d(lt, lq, bins=[bins, bins])[0].max())
        norm = LogNorm(vmin=1, vmax=max(peak, 2))
        im = None
        for i, (ax, k) in enumerate(zip(axes, keys)):
            lt, lq = logs[k]
            lims, bins = hists[k]
            ax.set_facecolor(cmap.get_under())
            im = ax.hist2d(lt, lq, bins=[bins, bins], cmap=cmap, norm=norm,
                           cmin=1)[3]
            ax.plot(lims, lims, color=p["ink2"], lw=1.2, ls=(0, (5, 3)),
                    zorder=3)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
            ax.grid(color=p["grid"], linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)
            _log_ticks(ax, "x")
            _log_ticks(ax, "y")
            lab = labels.get(k, k)
            unit = units.get(k, "")
            u = f" [{unit}]" if unit else ""
            ax.set_xlabel(f"SOLPS-ITER {lab}{u}")
            ax.set_ylabel(f"surrogate {lab}{u}")
            ax.set_title(f"({_LETTERS[i]}) {lab}", fontsize=9.5)
            m = metrics_from_predictions(preds[k][0], preds[k][1])
            _box(ax, f"median rel. err. {100 * m['median_rel_err']:.2f} %\n"
                     f"R² (log10) {m['r2_log10']:.3f}", p)
        cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.015)
        cb.set_label("counts (log)", fontsize=8, color=p["ink2"])
        _plain_log_ticks(cb.ax.yaxis)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    return _finish(fig, out_stem)


# ---------------------------------------------------------------- error CDF

def error_cdf_figure(rel_errors: dict[str, np.ndarray], *,
                     mode: str = "light",
                     reference: dict[str, float] | None = None,
                     out_stem=None):
    """Empirical CDF of relative error per entry, medians marked."""
    import matplotlib.pyplot as plt

    from surge.viz.theme import surge_theme

    if len(rel_errors) > 8:
        raise ValueError("error_cdf_figure supports at most 8 series")
    with surge_theme(mode) as p:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for color, (name, e) in zip(p["series"], rel_errors.items()):
            e = np.sort(np.asarray(e, float).ravel())
            e = e[np.isfinite(e)]
            y = np.arange(1, e.size + 1) / e.size
            ax.step(np.clip(e, 1e-12, None), y, where="post", color=color,
                    lw=1.6, label=name)
            med = float(np.median(e))
            ax.plot([med], [0.5], "o", color=color, ms=5,
                    markeredgecolor=p["surface"], markeredgewidth=0.8,
                    zorder=4)
        ax.axhline(0.5, color=p["axis"], lw=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1e1)
        ax.set_ylim(0, 1.02)
        for j, (name, v) in enumerate((reference or {}).items()):
            ax.axvline(v, color=p["muted"], lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.text(v, 0.04 + 0.09 * (j % 3), f" {name}", rotation=0,
                    fontsize=6.5, color=p["muted"], va="bottom", ha="left")
        ax.set_xlabel("relative error |pred − true| / |true|")
        ax.set_ylabel("cumulative fraction of test cases")
        ax.set_title("Relative-error distribution", fontsize=10)
        ax.legend(loc="lower right", fontsize=7.5)
    return _finish(fig, out_stem)


# ------------------------------------------------------------ target profile

def target_profile_figure(te_true: np.ndarray, te_pred: np.ndarray, *,
                          radial: np.ndarray | None = None,
                          cases: Sequence[int] | None = None,
                          mode: str = "light", out_stem=None):
    """Outer-target Te profiles (truth solid, surrogate dashed), <= 4 cases."""
    import matplotlib.pyplot as plt

    from surge.viz.theme import surge_theme

    te_true = np.asarray(te_true, float)
    te_pred = np.asarray(te_pred, float)
    ids = list(cases) if cases is not None else list(range(min(4, len(te_true))))
    ids = ids[:4]
    x = (np.arange(1, te_true.shape[1] + 1) if radial is None
         else np.asarray(radial, float))
    xlabel = ("radial cell index at the outer target" if radial is None
              else "radial coordinate [m]")
    with surge_theme(mode) as p:
        n = len(ids)
        fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.4, 3.1),
                                 squeeze=False, sharey=True)
        for i, (ax, c) in enumerate(zip(axes[0], ids)):
            t, q = te_true[c], te_pred[c]
            ax.plot(x, t, color=p["series"][0], lw=1.8, label="SOLPS-ITER")
            ax.plot(x, q, color=p["series"][1], lw=1.6, ls=(0, (5, 3)),
                    label="surrogate")
            ax.set_yscale("log")
            med = float(np.median(np.abs(q - t) / np.abs(t)))
            ax.set_title(f"({_LETTERS[i]}) test case {c}\n"
                         f"median rel. err. {100 * med:.1f} %", fontsize=9)
            ax.set_xlabel(xlabel)
            _plain_log_ticks(ax.yaxis)
            if i == 0:
                ax.set_ylabel("outer-target $T_e$ [eV]")
        axes[0][-1].legend(loc="best", fontsize=7.5)
    return _finish(fig, out_stem)


# ------------------------------------------------------------------- fields

def field_figure(te_true: np.ndarray, te_pred: np.ndarray, crx: np.ndarray,
                 cry: np.ndarray, *, case_ids: Sequence[int] | None = None,
                 mode: str = "light", out_stem=None):
    """Per case: log10 Te truth | surrogate | signed log10(pred/true)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import CenteredNorm

    from surge.viz.theme import diverging_cmap, field_cmap, surge_theme

    te_true = np.asarray(te_true, float)
    te_pred = np.asarray(te_pred, float)
    crx = np.asarray(crx, float)
    cry = np.asarray(cry, float)
    ids = list(case_ids) if case_ids is not None else list(range(len(te_true)))
    ids = ids[:len(te_true)]  # labels only; rows follow array order
    order = [0, 1, 3, 2]  # (BL, BR, TL, TR) -> non-self-intersecting quad
    sl = (slice(1, -1), slice(1, -1))  # drop guard cells
    verts = np.stack([crx[sl][..., order], cry[sl][..., order]], axis=-1)
    verts = verts.reshape(-1, 4, 2)
    xmin, xmax = verts[..., 0].min(), verts[..., 0].max()
    ymin, ymax = verts[..., 1].min(), verts[..., 1].max()
    aspect = (ymax - ymin) / (xmax - xmin)

    with surge_theme(mode):
        n = len(ids)
        row_h = float(np.clip(3.0 * aspect, 2.8, 4.2))
        width = float(np.clip(3 * (row_h / aspect + 1.1), 7.0, 12.0))
        fig, axes = plt.subplots(n, 3, figsize=(width, row_h * n + 0.4),
                                 squeeze=False)
        for r, c in enumerate(ids):
            lt = np.log10(te_true[r][sl]).ravel()
            lq = np.log10(te_pred[r][sl]).ravel()
            err = lq - lt
            vmin = float(min(lt.min(), lq.min()))
            vmax = float(max(lt.max(), lq.max()))
            half = max(float(np.percentile(np.abs(err), 99)), 1e-3)
            panels = [
                (lt, field_cmap(mode), None, vmin, vmax,
                 "SOLPS-ITER"),
                (lq, field_cmap(mode), None, vmin, vmax, "surrogate"),
                (err, diverging_cmap(mode), CenteredNorm(halfrange=half),
                 None, None, "log$_{10}$(surrogate / SOLPS-ITER)"),
            ]
            pcs = []
            for k, (val, cmap, norm, lo, hi, title) in enumerate(panels):
                ax = axes[r, k]
                pc = PolyCollection(verts, array=val, cmap=cmap, norm=norm,
                                    edgecolors="face", linewidths=0.2)
                if norm is None:
                    pc.set_clim(lo, hi)
                ax.add_collection(pc)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal")
                ax.grid(False)
                ax.set_title(f"({_LETTERS[r * 3 + k]}) {title}"
                             + (f" · case {c}" if k == 0 else ""),
                             fontsize=8.5)
                ax.set_xlabel("R [m]")
                if k == 0:
                    ax.set_ylabel("Z [m]")
                pcs.append(pc)
            cb1 = fig.colorbar(pcs[0], ax=list(axes[r, :2]), fraction=0.03,
                               pad=0.01)
            cb1.set_label("log$_{10}$ $T_e$ [eV]", fontsize=8)
            cb2 = fig.colorbar(pcs[2], ax=axes[r, 2], fraction=0.05, pad=0.02)
            cb2.set_label("signed log$_{10}$ ratio", fontsize=8)
            for cb in (cb1, cb2):
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_visible(False)
    return _finish(fig, out_stem)


# ---------------------------------------------------------- dataset overview

def dataset_overview_figure(X_train: np.ndarray, X_test: np.ndarray,
                            scalars_train: dict[str, np.ndarray],
                            crx: np.ndarray, cry: np.ndarray, *,
                            param_names: Sequence[str],
                            param_units: Sequence[str],
                            log_columns: Sequence[int] = (3, 4, 5),
                            mode: str = "light", out_stem=None):
    """Input histograms (train vs test), target histograms and the B2 grid."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from surge.viz.theme import surge_theme

    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)
    crx = np.asarray(crx, float)
    cry = np.asarray(cry, float)
    nparam = X_train.shape[1]
    ncol = 4
    nrow = int(np.ceil(nparam / ncol)) + 1
    logc = set(log_columns)

    def _hist(ax, data, color, log, bins, label=None):
        ax.hist(data, bins=bins, histtype="step", color=color, lw=1.5,
                label=label)
        if log:
            ax.set_xscale("log")

    def _bins(arrs, log, n=30):
        lo = min(a.min() for a in arrs)
        hi = max(a.max() for a in arrs)
        return np.logspace(np.log10(lo), np.log10(hi), n) if log \
            else np.linspace(lo, hi, n)

    with surge_theme(mode) as p:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow))
        for ax in axes.ravel():
            ax.set_visible(False)
        for j in range(nparam):
            ax = axes[j // ncol, j % ncol]
            ax.set_visible(True)
            log = j in logc
            bins = _bins([X_train[:, j], X_test[:, j]], log)
            _hist(ax, X_train[:, j], p["series"][0], log, bins, "train")
            _hist(ax, X_test[:, j], p["series"][1], log, bins, "test")
            unit = param_units[j]
            ax.set_title(f"{param_names[j]}" + (f" [{unit}]" if unit else ""),
                         fontsize=9)
            ax.tick_params(labelsize=7)
            if j % ncol == 0:
                ax.set_ylabel("count")
        axes[0, 0].legend(loc="best", fontsize=7.5)
        last = nrow - 1
        for k, (key, v) in enumerate(list(scalars_train.items())[:3]):
            ax = axes[last, k]
            ax.set_visible(True)
            v = np.asarray(v, float).ravel()
            _hist(ax, v, p["series"][0], True, _bins([v], True))
            ax.set_title(key, fontsize=9)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.set_ylabel("count")
        axg = axes[last, 3]
        axg.set_visible(True)
        sl = (slice(1, -1), slice(1, -1))
        verts = np.stack([crx[sl][..., [0, 1, 3, 2]],
                          cry[sl][..., [0, 1, 3, 2]]], axis=-1).reshape(-1, 4, 2)
        axg.add_collection(PolyCollection(
            verts, facecolors="none", edgecolors=p["muted"], linewidths=0.15))
        axg.set_xlim(verts[..., 0].min(), verts[..., 0].max())
        axg.set_ylim(verts[..., 1].min(), verts[..., 1].max())
        axg.set_aspect("equal")
        axg.grid(False)
        axg.set_xlabel("R [m]")
        axg.set_ylabel("Z [m]")
        axg.set_title("B2 grid (104×50)", fontsize=9)
        axg.tick_params(labelsize=7)
    return _finish(fig, out_stem)


# ------------------------------------------------------------- field gallery

def te_cases_figure(te: np.ndarray, crx: np.ndarray, cry: np.ndarray, *,
                    titles: Sequence[str] | None = None,
                    separatrix: int | None = None,
                    label: str = r"$\log_{10} T_e$ [eV]",
                    mode: str = "light", out_stem=None):
    """Row of up to six SOLPS-ITER fields on the B2 grid (``log10`` of ``te``),
    one shared colour scale, optional separatrix ring outline (radial index of
    the first SOL ring). Pass a normalised field (e.g. ``te / te_upstream``)
    with a matching ``label`` to show within-case structure instead of level."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    from surge.viz.theme import field_cmap, surge_theme

    te = np.asarray(te, float)[:6]
    crx = np.asarray(crx, float)
    cry = np.asarray(cry, float)
    order = [0, 1, 3, 2]
    sl = (slice(1, -1), slice(1, -1))
    verts = np.stack([crx[sl][..., order], cry[sl][..., order]], axis=-1)
    verts = verts.reshape(-1, 4, 2)
    lt = np.log10(np.clip(te[:, sl[0], sl[1]], 1e-6, None))
    vmin, vmax = np.percentile(lt, 0.5), np.percentile(lt, 99.5)
    norm = Normalize(vmin=vmin, vmax=vmax)
    xmin, xmax = verts[..., 0].min(), verts[..., 0].max()
    ymin, ymax = verts[..., 1].min(), verts[..., 1].max()
    aspect = (ymax - ymin) / (xmax - xmin)
    titles = list(titles) if titles is not None else [f"case {i}" for i in range(len(te))]

    with surge_theme(mode) as p:
        n = len(te)
        h = 3.4
        fig, axes = plt.subplots(1, n, figsize=(n * (h / aspect + 0.35) + 1.2, h + 0.9),
                                 squeeze=False, sharey=True)
        cmap = field_cmap(mode)
        for i, ax in enumerate(axes[0]):
            pc = PolyCollection(verts, array=lt[i].ravel(), cmap=cmap, norm=norm,
                                edgecolors="none", antialiaseds=False)
            ax.add_collection(pc)
            if separatrix is not None:
                # lower edge (corners BL, BR) of the first SOL ring = separatrix
                xs = np.concatenate([crx[1:-1, separatrix, 0], crx[-2:-1, separatrix, 1]])
                ys = np.concatenate([cry[1:-1, separatrix, 0], cry[-2:-1, separatrix, 1]])
                ax.plot(xs, ys, color=p["ink2"], lw=0.7, alpha=0.8)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.set_title(f"({_LETTERS[i]}) {titles[i]}", fontsize=9)
            ax.set_xlabel("R [m]")
            if i == 0:
                ax.set_ylabel("Z [m]")
        cb = fig.colorbar(pc, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
        cb.set_label(label)
    return _finish(fig, out_stem)


# -------------------------------------------------------------------- atlas

def te_atlas_figure(rows: "dict[str, np.ndarray]", crx: np.ndarray, cry: np.ndarray, *,
                    titles: "dict[str, Sequence[str]] | None" = None,
                    separatrix: int | None = None,
                    label: str = r"$\log_{10}(T_e / T_e^{\mathrm{omp,sep}})$",
                    vlim: tuple[float, float] | None = None,
                    mode: str = "light", out_stem=None):
    """Atlas of SOLPS-NN fields: one row per scenario type, up to eight small
    panels per row on the B2 grid, a single shared colour scale (``log10`` of
    the given field; pass a normalised field for structure, raw Te for level).
    ``rows`` maps the row label to an array (n_cases, nx, ny); ``titles`` the
    row label to one short title per panel."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    from surge.viz.theme import field_cmap, surge_theme

    crx = np.asarray(crx, float)
    cry = np.asarray(cry, float)
    order = [0, 1, 3, 2]
    sl = (slice(1, -1), slice(1, -1))
    verts = np.stack([crx[sl][..., order], cry[sl][..., order]], axis=-1).reshape(-1, 4, 2)
    xmin, xmax = verts[..., 0].min(), verts[..., 0].max()
    ymin, ymax = verts[..., 1].min(), verts[..., 1].max()
    aspect = (ymax - ymin) / (xmax - xmin)
    keys = list(rows)
    fields = {k: np.log10(np.clip(np.asarray(rows[k], float)[:8][:, sl[0], sl[1]], 1e-6, None))
              for k in keys}
    ncol = max(f.shape[0] for f in fields.values())
    allv = np.concatenate([f.ravel() for f in fields.values()])
    if vlim is None:
        vlim = (float(np.percentile(allv, 0.5)), float(np.percentile(allv, 99.5)))
    norm = Normalize(*vlim)
    sep_xy = None
    if separatrix is not None:
        sep_xy = (np.concatenate([crx[1:-1, separatrix, 0], crx[-2:-1, separatrix, 1]]),
                  np.concatenate([cry[1:-1, separatrix, 0], cry[-2:-1, separatrix, 1]]))

    with surge_theme(mode) as p:
        h = 2.3
        w = max(h / aspect, 1.55)  # wide enough for a two-line title
        fig, axes = plt.subplots(len(keys), ncol, squeeze=False,
                                 figsize=(ncol * (w + 0.12) + 1.3, len(keys) * (h + 0.55)))
        cmap = field_cmap(mode)
        pc = None
        for r, k in enumerate(keys):
            f = fields[k]
            for c in range(ncol):
                ax = axes[r, c]
                if c >= f.shape[0]:
                    ax.set_axis_off()
                    continue
                pc = PolyCollection(verts, array=f[c].ravel(), cmap=cmap, norm=norm,
                                    edgecolors="none", antialiaseds=False)
                ax.add_collection(pc)
                if sep_xy is not None:
                    ax.plot(*sep_xy, color=p["ink2"], lw=0.5, alpha=0.8)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.grid(False)
                if titles and k in titles and c < len(titles[k]):
                    ax.set_title(titles[k][c], fontsize=6.3, color=p["ink2"], pad=2)
            axes[r, 0].set_ylabel(k, fontsize=9, color=p["ink"], labelpad=4)
        # the theme enables constrained layout; let it place the shared colourbar
        cb = fig.colorbar(pc, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01, aspect=45)
        cb.set_label(label)
    return _finish(fig, out_stem)
