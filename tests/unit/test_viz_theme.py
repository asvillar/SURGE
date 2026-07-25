"""Tests for the SURGE visual system (surge.viz.theme)."""
from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from surge.viz.theme import (
    PALETTES,
    fmt_metric,
    sequential_cmap,
    series_color,
    surge_theme,
)


def test_palettes_have_both_modes_and_eight_series():
    for mode in ("light", "dark"):
        assert len(PALETTES[mode]["series"]) == 8
        for role in ("surface", "ink", "grid", "good", "critical", "seq"):
            assert role in PALETTES[mode]


def test_theme_applies_and_restores_rcparams():
    before = matplotlib.rcParams["axes.facecolor"]
    with surge_theme("light") as p:
        assert matplotlib.rcParams["axes.facecolor"] == p["surface"]
        assert matplotlib.rcParams["svg.hashsalt"] == "surge"
    assert matplotlib.rcParams["axes.facecolor"] == before


def test_series_color_fixed_order_and_capped():
    assert series_color(0) == PALETTES["light"]["series"][0]
    with pytest.raises(ValueError):
        series_color(8)


def test_sequential_cmap_monotonic_lightness():
    cmap = sequential_cmap("light")
    # luminance approximation must strictly decrease light -> dark
    lums = [sum(cmap(x)[:3]) for x in (0.0, 0.5, 1.0)]
    assert lums[0] > lums[1] > lums[2]


def test_save_figure_is_deterministic(tmp_path: Path):
    """Re-running an export in a fresh process yields identical bytes."""
    import subprocess
    import sys

    script = (
        "import matplotlib; matplotlib.use('Agg');"
        "import matplotlib.pyplot as plt;"
        "from surge.viz.theme import surge_theme, save_figure;"
        "import sys;"
        "ctx = surge_theme('light'); ctx.__enter__();"
        "fig, ax = plt.subplots(figsize=(2, 1.5)); ax.plot([0, 1], [0, 1]);"
        "out = save_figure(fig, sys.argv[1], formats=('svg',))[0];"
        "sys.stdout.write(out.read_text())"
    )

    def render(name: str) -> str:
        res = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / name)],
            capture_output=True, text=True, check=True)
        return res.stdout

    assert render("a") == render("b")


def test_fmt_metric_conventions():
    assert fmt_metric(None) == "—"
    assert fmt_metric(0.98765) == "0.988"
    assert fmt_metric(54.137, "rmse") == "54.1"
    assert fmt_metric(0.4, "runtime") == "400ms"
    assert fmt_metric(3.14, "runtime") == "3.1s"
    assert fmt_metric((0.9, 0.02), "pm") == "0.900 ± 0.020"
