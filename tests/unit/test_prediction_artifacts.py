"""predictions.npz persistence + prediction thumbnails."""
from __future__ import annotations

import numpy as np

from surge.benchmarks.base import PREDICTIONS_MAX_ROWS, write_predictions
from surge.report.prediction_viz import prediction_figure_html, prediction_thumbnail


def _run(tmp_path, *extra):
    from surge.benchmarks.run import main

    rc = main(["--benchmark", "synthetic.regression_1d", "--model", "sklearn.ridge",
               "--save-dir", str(tmp_path), *extra])
    assert rc in (0, 1)
    return list(tmp_path.glob("*/*/result.json"))


def test_bench_writes_predictions(tmp_path):
    (rj,) = _run(tmp_path)
    npz = rj.parent / "predictions.npz"
    assert npz.is_file()
    z = np.load(npz)
    assert {"y_true", "y_pred"} <= set(z.files)
    assert z["y_true"].dtype == np.float32 and z["y_pred"].dtype == np.float32
    assert len(z["y_true"]) == len(z["y_pred"]) > 0
    import json

    assert json.loads(rj.read_text())["extra"]["predictions_file"] == "predictions.npz"


def test_no_predictions_flag(tmp_path):
    (rj,) = _run(tmp_path, "--no-predictions")
    assert not (rj.parent / "predictions.npz").exists()
    assert "predictions_file" not in rj.read_text()


def test_no_save_writes_nothing(tmp_path):
    assert _run(tmp_path, "--no-save") == []
    assert not list(tmp_path.rglob("predictions.npz"))


def test_cap_and_flag(tmp_path):
    n = PREDICTIONS_MAX_ROWS + 500
    y = np.arange(n, dtype=np.float64)
    meta = write_predictions(tmp_path, y, y + 1)
    z = np.load(tmp_path / "predictions.npz")
    assert len(z["y_true"]) == PREDICTIONS_MAX_ROWS and meta["subsampled"] is True
    assert bool(z["subsampled"])
    assert np.all(z["y_pred"] - z["y_true"] == 1)  # same rows kept in both


def test_thumbnails(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    reg = tmp_path / "reg"
    reg.mkdir()
    write_predictions(reg, y, y + 0.1 * rng.normal(size=300))
    svg = prediction_thumbnail(reg, "regression", "light")
    assert svg and svg.startswith("<svg")

    clf = tmp_path / "clf"
    clf.mkdir()
    yt = rng.integers(0, 3, 200)
    write_predictions(clf, yt, np.where(rng.random(200) < 0.8, yt, 0),
                      rng.random((200, 3)))
    assert prediction_thumbnail(clf, "classification", "dark").startswith("<svg")

    fld = tmp_path / "fld"
    fld.mkdir()
    f = rng.normal(size=(12, 8, 10))
    write_predictions(fld, f, f + 0.1 * rng.normal(size=f.shape))
    assert prediction_thumbnail(fld, "regression", "light").startswith("<svg")


def test_missing_predictions_is_none(tmp_path):
    assert prediction_thumbnail(tmp_path, "regression", "light") is None


def test_figure_html_absent_without_predictions(tmp_path):
    from surge.report import prediction_viz

    prediction_viz.CONTEXT.update(base=tmp_path, mode="light")
    res = {"m": {"metrics": {"test_r2": {"mean": 0.9}}, "runs": [
        {"_path": "r/b/t/result.json", "metrics": {"test_r2": 0.9}}]}}
    assert prediction_figure_html(res, "test_r2", "regression") == ""
    d = tmp_path / "r/b/t"
    d.mkdir(parents=True)
    (d / "result.json").write_text('{"model_key": "m", "task_type": "regression"}')
    y = np.linspace(0, 1, 50)
    write_predictions(d, y, y)
    out = prediction_figure_html(res, "test_r2", "regression")
    assert "best model predictions · m · test" in out and "<svg" in out
