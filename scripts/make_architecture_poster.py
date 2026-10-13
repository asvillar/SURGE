#!/usr/bin/env python3
"""Render the SURGE architecture poster (README hero figure).

A single panel that explains the workflow BY FUNCTIONALITY, with real
result thumbnails embedded per stage — regenerated, never hand-drawn:

    INGEST & CHARACTERIZE -> MODEL & OPTIMIZE -> EVALUATE & QUANTIFY
    -> REPORT & DEPLOY, over a provenance strip.

Usage (from repo root):
    python scripts/make_architecture_poster.py \
        [--assets docs/assets/gallery] [--out docs/assets/readme/architecture]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from surge.viz.theme import save_figure, surge_theme

COLUMNS = [
    {
        "title": "1 · INGEST &\nCHARACTERIZE",
        "chips": ["CSV · Parquet · HDF5", "NetCDF · Pickle · sim dirs",
                  "schema inference", "PCA · SNR · correlations"],
        "thumb": "characterization.png",
        "caption": "dataset characterization",
    },
    {
        "title": "2 · MODEL &\nOPTIMIZE",
        "chips": ["40+ adapters, one registry", "sklearn · PyTorch · Keras/TF",
                  "GPflow · XGB/LGBM/CatBoost", "Optuna HPO (TPE · BoTorch)"],
        "thumb": "hpo_convergence.png",
        "caption": "HPO with starred bests",
    },
    {
        "title": "3 · EVALUATE &\nQUANTIFY",
        "chips": ["held-out metrics per split", "parity · residuals · fields",
                  "GP credible bands (UQ)", "classification + calibration"],
        "thumb": "parity.png",
        "caption": "train/test parity density",
    },
    {
        "title": "4 · REPORT &\nDEPLOY",
        "chips": ["benchmark leaderboards", "self-contained HTML report",
                  "MLflow tracking (opt-in)", "ONNX / .keras / joblib export"],
        "thumb": "leaderboard.png",
        "caption": "leaderboard vs thresholds",
    },
]

PROVENANCE = ("every run reproducible from runs/<tag>/ :   spec.yaml   ·   "
              "git revision   ·   environment   ·   metrics.json   ·   "
              "model card   ·   parquet predictions   ·   training logs")



def _resolve_thumb(assets: Path, name: str, mode: str) -> Path:
    """Pick the thumbnail for the requested theme.

    Search order (first hit wins):
      dark  -> <assets>/<name>_dark.png, <assets>/dark/<name>.png,
               docs/assets/readme/dark/<name>.png,
               examples/viz_gallery_output/<name>_dark.png
      light -> <assets>/<name>.png, docs/assets/readme/<name>.png,
               examples/viz_gallery_output/<name>_light.png
    Falls back to the light file so a missing dark variant never blanks a card.
    """
    stem = name[:-4] if name.endswith(".png") else name
    readme = _REPO / "docs/assets/readme"
    gallery_out = _REPO / "examples/viz_gallery_output"
    if mode == "dark":
        candidates = [assets / f"{stem}_dark.png", assets / "dark" / f"{stem}.png",
                      readme / "dark" / f"{stem}.png", gallery_out / f"{stem}_dark.png"]
    else:
        candidates = [assets / f"{stem}.png", readme / f"{stem}.png",
                      gallery_out / f"{stem}_light.png"]
    for c in candidates:
        if c.exists():
            return c
    return assets / f"{stem}.png"


def build(assets: Path, mode: str = "light"):
    with surge_theme(mode) as p:
        fig = plt.figure(figsize=(12.4, 6.4))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        fig.patch.set_facecolor(p["page"])

        ax.text(3, 94.5, "SURGE", fontsize=26, fontweight="bold",
                color=p["ink"], va="center")
        ax.text(14.5, 94.5,
                "one declarative workflow for scientific surrogate models "
                "— YAML in, reproducible science out",
                fontsize=12.5, color=p["ink2"], va="center")

        col_w, gap, x0, y0, col_h = 22.0, 2.0, 3.0, 14.5, 72.0
        for i, col in enumerate(COLUMNS):
            x = x0 + i * (col_w + gap)
            ax.add_patch(FancyBboxPatch(
                (x, y0), col_w, col_h,
                boxstyle="round,pad=0.6,rounding_size=1.4",
                facecolor=p["surface"], edgecolor=p["grid"], linewidth=1.2))
            ax.add_patch(FancyBboxPatch(
                (x, y0 + col_h - 1.2), col_w, 1.2,
                boxstyle="round,pad=0.0,rounding_size=0.5",
                facecolor=p["series"][i % 4], edgecolor="none", alpha=0.9))
            ax.text(x + col_w / 2, y0 + col_h - 6.5, col["title"],
                    ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color=p["ink"], linespacing=1.3)

            for j, chip in enumerate(col["chips"]):
                cy = y0 + col_h - 13.5 - j * 4.6
                ax.text(x + col_w / 2, cy, chip, ha="center", va="center",
                        fontsize=8.4, color=p["ink2"],
                        bbox={"boxstyle": "round,pad=0.32",
                              "facecolor": p["page"],
                              "edgecolor": p["grid"], "linewidth": 0.7})

            thumb_path = _resolve_thumb(assets, col["thumb"], mode)
            if thumb_path.exists():
                img = mpimg.imread(thumb_path)
                ih, iw = img.shape[:2]
                band_lo, band_hi = y0 + 6.0, y0 + 33.0   # fixed image band
                tw = col_w - 3.0
                th = tw * ih / iw * (12.4 / 6.4)
                if th > band_hi - band_lo:               # too tall: shrink
                    tw *= (band_hi - band_lo) / th
                    th = band_hi - band_lo
                cy = (band_lo + band_hi - th) / 2        # center in band
                cx = x + (col_w - tw) / 2
                iax = fig.add_axes([cx / 100, cy / 100, tw / 100, th / 100])
                iax.imshow(img)
                iax.axis("off")
            ax.text(x + col_w / 2, y0 + 3.4, col["caption"],
                    ha="center", va="center", fontsize=7.6,
                    style="italic", color=p["muted"])

        for i in range(len(COLUMNS) - 1):
            xa = x0 + (i + 1) * col_w + i * gap + 0.35
            ax.add_patch(FancyArrowPatch(
                (xa, y0 + col_h * 0.55), (xa + gap - 0.7, y0 + col_h * 0.55),
                arrowstyle="-|>", mutation_scale=16,
                color=p["ink2"], linewidth=1.6))

        ax.add_patch(FancyBboxPatch(
            (x0, 4.2), 4 * col_w + 3 * gap, 6.6,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=p["surface"], edgecolor=p["series"][0],
            linewidth=1.2))
        ax.text(x0 + (4 * col_w + 3 * gap) / 2, 7.5, PROVENANCE,
                ha="center", va="center", fontsize=8.8, color=p["ink2"])
        return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(_REPO / "docs/assets/gallery"))
    ap.add_argument("--out", default=str(_REPO / "docs/assets/readme/architecture"))
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    args = ap.parse_args()
    fig = build(Path(args.assets), mode=args.mode)
    for path in save_figure(fig, Path(args.out), formats=("png",)):
        print("wrote", path)


if __name__ == "__main__":
    main()
