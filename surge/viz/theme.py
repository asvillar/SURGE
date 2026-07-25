"""SURGE visual system: one theme for every figure, table, and report.

Every SURGE plot imports *roles* from here ("series 1", "good", "grid",
"sequential ramp") instead of raw colors, so the whole visual identity can
be rebranded in one place. The default palette is brand-neutral and
colorblind-validated (adjacent-pair CVD deltaE >= 8 and normal-vision
deltaE >= 15 in both light and dark modes).

Rules encoded by this module — do not work around them in plot code:

* Categorical hues are assigned in fixed slot order and never cycled past
  eight; a ninth series folds into "Other" or a small-multiple panel.
* Sequential data (density, magnitude) uses the one-hue blue ramp;
  diverging data uses blue<->red with a neutral gray midpoint.
* PASS/FAIL and other status colors are reserved (`good`, `warning`,
  `serious`, `critical`) and never used as series colors.
* Text always wears ink colors (`ink`, `ink2`, `muted`), never a series
  color; a colored mark next to the text carries identity.
* One axis per chart. Two measures of different scale get two panels.
* Exports are deterministic: fixed dpi, stripped timestamps, fixed SVG
  hashsalt — so figure files are byte-stable and diffable in CI.

Usage::

    from surge.viz.theme import surge_theme, save_figure, fmt_metric

    with surge_theme() as palette:          # or surge_theme("dark")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, y)                        # series colors applied
        ax.set_title("Loss")
    save_figure(fig, out_dir / "loss")       # loss.png/.svg/.pdf
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Sequence  # noqa: UP035 — py3.9 support

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "PALETTES",
    "fmt_metric",
    "rc_params",
    "save_figure",
    "sequential_cmap",
    "series_color",
    "surge_theme",
]

# ---------------------------------------------------------------------------
# Palette — light and dark are both hand-stepped, not automatic inversions.
# ---------------------------------------------------------------------------

PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                   "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#184f95", "#0d366b"],
        "good": "#0ca30c", "warning": "#fab219",
        "serious": "#ec835a", "critical": "#d03b3b",
        "good_text": "#006300",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500",
                   "#d55181", "#008300", "#9085e9", "#e66767"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5",
                "#6da7ec", "#9ec5f4", "#cde2fb"],
        "good": "#0ca30c", "warning": "#fab219",
        "serious": "#ec835a", "critical": "#d03b3b",
        "good_text": "#0ca30c",
    },
}

_FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
_BASE = 9.0        # pt; figures are sized for single-column publication use
_DPI_SCREEN = 150
_DPI_PRINT = 300


def rc_params(mode: str = "light") -> dict:
    """Matplotlib rcParams implementing the SURGE theme for *mode*."""
    p = PALETTES[mode]
    return {
        "figure.facecolor": p["surface"],
        "figure.dpi": _DPI_SCREEN,
        "savefig.dpi": _DPI_PRINT,
        "savefig.facecolor": p["surface"],
        "figure.constrained_layout.use": True,
        "axes.facecolor": p["surface"],
        "axes.edgecolor": p["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": _BASE + 2,
        "axes.titleweight": "bold",
        "axes.titlecolor": p["ink"],
        "axes.labelsize": _BASE,
        "axes.labelcolor": p["ink2"],
        "axes.prop_cycle": mpl.cycler(color=p["series"]),
        "grid.color": p["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": p["muted"], "ytick.color": p["muted"],
        "xtick.labelsize": _BASE - 1, "ytick.labelsize": _BASE - 1,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": _BASE - 1,
        "font.family": "sans-serif",
        "font.sans-serif": _FONT_STACK,
        "font.size": _BASE,
        "text.color": p["ink"],
        "svg.hashsalt": "surge",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


@contextmanager
def surge_theme(mode: str = "light"):
    """Apply the SURGE theme to all figures created inside the block.

    Yields the palette dict for the chosen mode so plot code can reference
    roles (``p["good"]``, ``p["seq"]``) without importing PALETTES.
    """
    with mpl.rc_context(rc_params(mode)):
        yield PALETTES[mode]


def series_color(index: int, mode: str = "light") -> str:
    """Fixed-order categorical slot. Raises past slot 8 by design."""
    series = PALETTES[mode]["series"]
    if index >= len(series):
        raise ValueError(
            f"series slot {index} exceeds the {len(series)}-color palette; "
            "fold extra series into 'Other' or use small multiples")
    return series[index]


def sequential_cmap(mode: str = "light") -> LinearSegmentedColormap:
    """One-hue sequential colormap (light -> dark blue) for magnitude."""
    return LinearSegmentedColormap.from_list(
        f"surge_seq_{mode}", PALETTES[mode]["seq"])


def save_figure(fig, out_stem: Path | str,
                formats: Sequence[str] = ("png", "svg", "pdf")) -> list[Path]:
    """Export *fig* deterministically to every requested format.

    Timestamps are stripped from metadata and the SVG hashsalt is pinned,
    so re-rendering unchanged inputs yields byte-identical files.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "png": {"Software": "SURGE", "Creation Time": ""},
        "svg": {"Date": None},
        "pdf": {"CreationDate": None, "ModDate": None, "Creator": "SURGE"},
    }
    written: list[Path] = []
    for fmt in formats:
        path = out_stem.with_suffix("." + fmt)
        fig.savefig(path, format=fmt, metadata=meta.get(fmt))
        written.append(path)
    return written


def fmt_metric(value, kind: str = "r2") -> str:
    """One metric-formatting convention for CLI, plots, and reports."""
    if value is None:
        return "—"
    if kind == "pm":
        mean, std = value
        return f"{mean:.3f} ± {std:.3f}"
    if kind in ("r2", "accuracy", "auroc", "f1"):
        return f"{value:.3f}"
    if kind in ("rmse", "mae", "nrmse", "rel_l2", "loss"):
        return f"{value:.3g}"
    if kind == "runtime":
        return f"{value:.1f}s" if value >= 1 else f"{value * 1000:.0f}ms"
    return f"{value:.4g}"
