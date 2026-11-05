"""Tests for the SOLPS-NN publication figures (surge.viz.solpsnn)."""
from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from surge.viz.solpsnn import (  # noqa: E402
    dataset_overview_figure,
    error_cdf_figure,
    field_figure,
    metrics_from_predictions,
    scalar_parity_figure,
    target_profile_figure,
)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scalars(rng, n=200):
    out = {}
    for k in ("psol", "qpeak", "gamma"):
        t = 10 ** rng.uniform(3, 6, n)
        out[k] = (t, t * np.exp(rng.normal(0, 0.1, n)))
    return out


def _grid(ni=104, nj=50):
    r = np.linspace(1.0, 2.0, ni + 1)
    th = np.linspace(0.0, 1.0, nj + 1)
    R = r[:, None] * np.cos(th)[None, :]
    Z = r[:, None] * np.sin(th)[None, :]
    crx = np.stack([R[:-1, :-1], R[1:, :-1], R[:-1, 1:], R[1:, 1:]], -1)
    cry = np.stack([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]], -1)
    return crx, cry


def test_metrics_perfect_and_shifted():
    t = np.array([1.0, 10.0, 100.0, 1e3])
    m = metrics_from_predictions(t, t)
    assert m["median_rel_err"] == 0.0
    assert m["r2_log10"] == pytest.approx(1.0)
    m = metrics_from_predictions(t, 1.1 * t)
    assert m["median_rel_err"] == pytest.approx(0.1)
    assert m["mape_linear"] == pytest.approx(0.1)


def test_scalar_parity(tmp_path):
    preds = _scalars(np.random.default_rng(0))
    fig = scalar_parity_figure(preds, labels={"psol": "P_SOL"},
                               units={"psol": "W"}, out_stem=tmp_path / "par")
    assert len(fig.axes) == 4  # 3 data + 1 colourbar
    assert (tmp_path / "par.png").exists()


def test_error_cdf(tmp_path):
    rng = np.random.default_rng(1)
    errs = {"a": rng.lognormal(-3, 1, 100), "b": rng.lognormal(-2, 1, 100)}
    fig = error_cdf_figure(errs, reference={"thesis": 0.05},
                           out_stem=tmp_path / "cdf")
    assert len(fig.axes) == 1
    assert (tmp_path / "cdf.png").exists()
    with pytest.raises(ValueError):
        error_cdf_figure({str(i): errs["a"] for i in range(9)})


def test_target_profile(tmp_path):
    rng = np.random.default_rng(2)
    t = 10 ** rng.uniform(0, 2, (6, 48))
    fig = target_profile_figure(t, t * 1.05, cases=[0, 1, 2, 3, 4],
                                out_stem=tmp_path / "prof")
    assert len(fig.axes) == 4
    assert (tmp_path / "prof.png").exists()


def test_field_figure(tmp_path):
    crx, cry = _grid()
    rng = np.random.default_rng(3)
    t = 10 ** rng.uniform(0, 2, (2, 104, 50))
    fig = field_figure(t, t * np.exp(rng.normal(0, 0.05, t.shape)), crx, cry,
                       case_ids=[7, 9], out_stem=tmp_path / "fld")
    assert len(fig.axes) == 2 * 3 + 2 * 2  # panels + colourbars
    assert (tmp_path / "fld.png").exists()


def test_dataset_overview(tmp_path):
    rng = np.random.default_rng(4)
    crx, cry = _grid()
    X = 10 ** rng.uniform(0, 1, (80, 8))
    sc = {k: 10 ** rng.uniform(3, 6, 80) for k in ("a", "b", "c")}
    fig = dataset_overview_figure(
        X, X[:20], sc, crx, cry, param_names=[f"p{i}" for i in range(8)],
        param_units=[""] * 8, out_stem=tmp_path / "ov")
    assert sum(ax.get_visible() for ax in fig.axes) == 12
    assert (tmp_path / "ov.png").exists()


def test_te_cases_figure_writes_png(tmp_path):
    from surge.viz.solpsnn import te_cases_figure
    crx, cry = _grid()
    te = np.exp(np.random.default_rng(0).normal(2, 1, (4, 104, 50)))
    fig = te_cases_figure(te, crx, cry, titles=["a", "b", "c", "d"], separatrix=25,
                          out_stem=tmp_path / "cases")
    assert fig is not None and (tmp_path / "cases.png").exists()


def test_te_atlas_figure_writes_png(tmp_path):
    from surge.viz.solpsnn import te_atlas_figure
    crx, cry = _grid()
    rng = np.random.default_rng(1)
    rows = {"type A": np.exp(rng.normal(0, 1, (3, 104, 50))), "type B": np.exp(rng.normal(1, 1, (2, 104, 50)))}
    titles = {"type A": ["a1", "a2", "a3"], "type B": ["b1", "b2"]}
    fig = te_atlas_figure(rows, crx, cry, titles=titles, separatrix=25, out_stem=tmp_path / "atlas")
    assert fig is not None and (tmp_path / "atlas.png").exists()
    assert len([a for a in fig.axes if a.get_label() != "<colorbar>"]) >= 6
