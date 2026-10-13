#!/usr/bin/env python3
"""Sync curated README/docs figures from the gallery output directory.

The gallery (examples/viz_theme_gallery.py) and the TheWell study write
light+dark variants to examples/viz_gallery_output/ (git-ignored). This
script copies the curated subset into the tracked asset folders:

    docs/assets/readme/<name>.png        light variant  (README default)
    docs/assets/readme/dark/<name>.png   dark variant   (README <picture>)
    docs/assets/gallery/<name>.png       light variant  (docs/gallery.md)

Run after regenerating figures:
    python examples/viz_theme_gallery.py
    python scripts/sync_readme_assets.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SRC = _REPO / "examples" / "viz_gallery_output"
README = _REPO / "docs" / "assets" / "readme"
README_DARK = README / "dark"
GALLERY = _REPO / "docs" / "assets" / "gallery"

# gallery-output stem -> tracked asset name
CURATED = {
    "field2d_truth": "field2d_truth",
    "field2d_prediction": "field2d_prediction",
    "field2d_error": "field2d_error",
    "parity_train": "parity_train",
    "parity_test": "parity_test",
    "parity_residuals": "parity_residuals",
    "trio_random_forest": "trio_random_forest",
    "trio_pytorch_mlp": "trio_pytorch_mlp",
    "trio_gaussian_process": "trio_gaussian_process",
    "hpo_convergence": "hpo_convergence",
    "training_curves": "training_curves",
    "uncertainty": "uncertainty",
    "ensemble": "ensemble",
    "leaderboard": "leaderboard",
    "thewell_grayscott": "thewell_grayscott",
    "thewell_grayscott_h1": "thewell_grayscott_h1",
    "parity": "parity",
    "trio": "trio",
    "field2d": "field2d",
    "field_operator": "field_operator",
    "characterization": "characterization",
    "classification": "classification",
    "mission_control": "mission_control",
    "constellaration": "constellaration",
    "scale": "scale",
    "thewell_turbulence": "thewell_turbulence",
    "thewell_helmholtz": "thewell_helmholtz",
    "simformer_sbi": "simformer_sbi",
    "thewell_pod": "thewell_pod",
    "thewell_crps": "thewell_crps",
    "thewell_rollout": "thewell_rollout",
    "qlknn10d": "qlknn10d",
    "grayscott_rollout_curve": "grayscott_rollout_curve", "grayscott_regime_split", "grayscott_rollout_cases",
    "grayscott_regime_split": "grayscott_regime_split",
    "grayscott_rollout_cases": "grayscott_rollout_cases",
}

# gallery.md embeds this subset (light only — RTD theme is light)
GALLERY_SET = {
    "parity", "hpo_convergence", "training_curves", "classification",
    "field_operator", "field2d", "trio", "ensemble", "uncertainty",
    "characterization", "leaderboard", "thewell_grayscott",
    "thewell_grayscott_h1",
    "mission_control", "constellaration", "scale", "thewell_turbulence", "thewell_helmholtz", "simformer_sbi", "thewell_pod", "thewell_crps", "thewell_rollout", "qlknn10d",
    "grayscott_rollout_curve",
}


def main() -> int:
    README_DARK.mkdir(parents=True, exist_ok=True)
    missing = []
    for stem, name in sorted(CURATED.items()):
        light = SRC / f"{stem}_light.png"
        dark = SRC / f"{stem}_dark.png"
        if not light.exists() or not dark.exists():
            missing.append(stem)
            continue
        shutil.copy2(light, README / f"{name}.png")
        shutil.copy2(dark, README_DARK / f"{name}.png")
        if stem in GALLERY_SET:
            shutil.copy2(light, GALLERY / f"{name}.png")
        print(f"synced {name}")
    if missing:
        print(f"[warn] missing light/dark pair, skipped: {', '.join(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
