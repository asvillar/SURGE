"""Tests for pure helpers in examples/thewell_grayscott_study.py.

Only the helpers with no dataset/model dependency are covered here — the
study itself is exercised end-to-end via ``--rollout``/``--rollout-gif``/
``--regime-split`` (long-running, not unit-tested).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from examples.thewell_grayscott_study import regime_label  # noqa: E402


def test_regime_label_parses_regime_and_params():
    path = (
        "/x/gray_scott_reaction_diffusion_maze_F_0.029_k_0.057.hdf5")
    assert regime_label(path) == "maze F=0.029 k=0.057"


def test_regime_label_distinguishes_different_regime_files():
    labels = {
        regime_label("gray_scott_reaction_diffusion_bubbles_F_0.098_k_0.057.hdf5"),
        regime_label("gray_scott_reaction_diffusion_spirals_F_0.018_k_0.051.hdf5"),
        regime_label("gray_scott_reaction_diffusion_worms_F_0.058_k_0.065.hdf5"),
    }
    assert len(labels) == 3


def test_regime_label_falls_back_to_stem_on_unrecognized_name():
    assert regime_label("/x/some_other_file.hdf5") == "some_other_file"
