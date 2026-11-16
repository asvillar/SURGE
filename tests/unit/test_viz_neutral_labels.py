"""Default plot labels must be application-neutral; caller labels must win."""
import json

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surge.viz.analysis import plot_output_distributions, plot_profile_mean_std
from surge.viz.comparison import plot_inference_comparison_grid, plot_mse_comparison
from surge.viz.hpo import plot_hpo_convergence
from surge.viz.profiles import plot_profile_comparison_with_inset

BANNED = ("TORIC", "P_c", "P_e", "MW", "gB", "RFR")


def _texts(fig):
    out = []
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        leg = ax.get_legend()
        if leg:
            out += [t.get_text() for t in leg.get_texts()]
        out += [ln.get_label() for ln in ax.get_lines()]
        out += [t.get_text() for t in ax.texts]
    return " | ".join(out)


def _clean(fig):
    text = _texts(fig)
    plt.close(fig)
    for b in BANNED:
        assert b not in text, (b, text)
    return text


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])


def test_output_distributions_labels():
    fig, ax = plot_output_distributions(_df())
    _clean(fig)
    fig, ax = plot_output_distributions(_df(), ylabel="speed [m/s]", title="Mine")
    assert ax.get_ylabel() == "speed [m/s]" and ax.get_title() == "Mine"
    plt.close(fig)


def test_profile_mean_std_labels():
    fig, ax = plot_profile_mean_std(_df(), ["a", "b", "c"])
    _clean(fig)
    fig, ax = plot_profile_mean_std(_df(), ["a", "b", "c"], output_name="temp", units="[K]")
    assert ax.get_ylabel() == "temp [K]"
    plt.close(fig)


def test_profile_comparison_labels():
    x = np.linspace(0, 1, 10)
    fig, ax = plot_profile_comparison_with_inset(x, x, {"m": x * 1.1})
    text = _clean(fig)
    assert "Reference" in text and "Surrogate (m)" in text
    fig, ax = plot_profile_comparison_with_inset(
        x, x, {"m": x}, reference_label="Truth", prediction_label="Net {model}")
    text = _texts(fig)
    plt.close(fig)
    assert "Truth" in text and "Net m" in text


def test_hpo_title(tmp_path):
    data = {"trials": [{"number": i, "value": 0.9 + 0.001 * i} for i in range(10)]}
    f = tmp_path / "m_hpo.json"
    f.write_text(json.dumps(data))
    fig, ax, _ = plot_hpo_convergence([f], metric="r2", method_names=["m"])
    assert ax.get_title() == "Hyperparameter optimization"
    _clean(fig)
    fig, ax, _ = plot_hpo_convergence([f], metric="r2", method_names=["m"], model_name="gbm")
    assert "gbm" in ax.get_title()
    plt.close(fig)


def _grid_results():
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    d = {"train": (y, y + 0.1), "test": (y, y + 0.2)}
    return {"out": {"m": d}}


def test_inference_grid_labels():
    fig, axes, _ = plot_inference_comparison_grid(_grid_results())
    text = _clean(fig)
    assert "Reference" in text and "Surrogate" in text
    fig, axes, _ = plot_inference_comparison_grid(
        _grid_results(), xlabel_prefix="Sim", ylabel_prefix="Net")
    text = _texts(fig)
    plt.close(fig)
    assert "Sim" in text and "Net" in text


def test_mse_comparison_labels():
    m = np.array([1e-3, 2e-3, 3e-3])
    fig, ax = plot_mse_comparison(m, m * 2)[:2]
    _clean(fig)


def test_mission_control_model_not_assumed(tmp_path):
    from surge.viz import animate
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "mymodel_test.parquet").write_bytes(b"")
    assert animate._model_from_predictions(tmp_path) == "mymodel"
    assert animate._model_from_predictions(tmp_path / "none") is None
