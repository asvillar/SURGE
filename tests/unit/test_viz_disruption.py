import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from surge.viz.disruption import roc_figure, tpr_at_fpr  # noqa: E402


def _data(n=4000, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    return y, y * sep + rng.normal(size=n)


def test_roc_figure_two_axes_legend_and_png(tmp_path):
    y, s = _data()
    yb, sb = _data(sep=1.0, seed=1)
    fig = roc_figure({"model": (y, s)}, baseline=("n/n_G", yb, sb),
                     out_stem=tmp_path / "roc")
    assert len(fig.axes) == 2
    assert (tmp_path / "roc.png").exists()
    legend = "\n".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "AUROC" in legend and "n/n_G" in legend


def test_roc_figure_no_baseline_dark():
    y, s = _data()
    fig = roc_figure({"a": (y, s), "b": (y, s + 0.1)}, mode="dark")
    assert len(fig.axes) == 2
    assert len(fig.axes[0].get_legend().get_texts()) == 2


def test_perfect_ranking_tpr_one_at_one_percent():
    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(500), np.ones(50)].astype(int)
    fpr, tpr, _ = roc_curve(y, y.astype(float))
    assert tpr_at_fpr(fpr, tpr, 0.01) == 1.0
    fig = roc_figure({"perfect": (y, y.astype(float))})
    assert "1.000" in fig.axes[0].texts[-1].get_text()
