"""C-Mod density-limit benchmark: grouped split, rare-event metrics, gate, headline (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from surge.benchmarks import leaderboard as lb
from surge.benchmarks import status
from surge.benchmarks.tasks import disruption_metrics
from surge.report import leaderboard as rep

KEY = "plasma.cmod_density_limit"


def _synthetic(n_dis=60, per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows, ys, gs = [], [], []
    for g in range(n_dis):
        pos = g % 6 == 0
        a, ip = 0.22, rng.uniform(0.5, 1.0)
        ng = ip / (np.pi * a**2)
        frac = np.linspace(0.3, 1.2 if pos else 0.6, per)
        n = frac * ng
        x = np.column_stack([n, np.full(per, 1.5), np.full(per, a), np.full(per, ip),
                             np.full(per, 5.0), np.full(per, 0.4)])
        rows.append(x)
        ys.append((frac > 0.9).astype(int) if pos else np.zeros(per, int))
        gs.append(np.full(per, g))
    return np.vstack(rows), np.concatenate(ys), np.concatenate(gs)


def test_grouped_split_keeps_discharges_whole_and_imbalance():
    X, y, g = _synthetic()
    Xtr, ytr, Xte, yte, gtr, gte = lb._cmod_grouped_split(X, y, g)
    assert not set(gtr) & set(gte)
    assert len(Xtr) + len(Xte) == len(X)
    assert 0 < ytr.mean() < 0.2 and 0 < yte.mean() < 0.2  # natural imbalance, not 50/50


def test_loader_reads_csv_and_caches(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("SURGE_DATA_DIR", str(tmp_path))
    X, y, g = _synthetic()
    df = pd.DataFrame(X, columns=lb._CMOD_FEATURES)
    df.insert(0, "discharge_ID", g)
    df["density_limit_phase"] = y
    root = lb._bench_data_root() / "plasma"
    (root / "cmod_raw").mkdir(parents=True)
    df.to_csv(root / "cmod_raw" / "DL_DataFrame.csv", index=False)
    Xtr, ytr, Xte, yte = lb._load_cmod_density_limit()
    assert Xtr.shape[1] == 6 and len(Xtr) + len(Xte) == len(X)
    d = np.load(root / "cmod_density_limit_grouped.npz")
    assert set(d.files) == {"X_train", "y_train", "X_test", "y_test", "groups_train", "groups_test"}
    assert not set(d["groups_train"]) & set(d["groups_test"])


def test_disruption_metrics_perfect_and_random():
    X, y, _ = _synthetic()
    m, base = disruption_metrics(X, y, y.astype(float))
    assert m["test_auroc"] == 1.0 and m["test_tpr_at_fpr05"] == 1.0 and m["test_tpr_at_fpr01"] == 1.0
    assert m["test_average_precision"] == pytest.approx(1.0)
    assert base["baseline_greenwald_auroc"] > 0.9
    rng = np.random.default_rng(1)
    Xr = rng.random((20000, 6)) + 0.1
    m, _ = disruption_metrics(Xr, rng.random(20000) < 0.1, rng.random(20000))
    assert m["test_auroc"] == pytest.approx(0.5, abs=0.04)


def test_run_leaderboard_emits_baseline_and_gate(monkeypatch):
    X, y, g = _synthetic()
    split = lb._cmod_grouped_split(X, y, g)[:4]
    monkeypatch.setattr(lb, "_load_cmod_density_limit", lambda: split)
    res = lb.run_leaderboard([KEY], model_keys=["sklearn.logistic_regression"], save_root=None)[KEY][0]
    assert "test_accuracy" in res.metrics and "test_f1_macro" in res.metrics
    for k in ("test_auroc", "test_average_precision", "test_tpr_at_fpr05", "test_tpr_at_fpr01"):
        assert k in res.metrics
    assert "baseline_greenwald_auroc" in res.extra
    assert "baseline_greenwald_tpr_at_fpr05" in res.extra
    assert res.passed == (res.metrics["test_auroc"] >= 0.99)


def test_gate_uses_auroc():
    assert lb._GATE_THRESHOLDS[KEY] == ("test_auroc", 0.99)
    assert lb._check_pass(KEY, {"test_auroc": 0.995, "test_accuracy": 0.1}) is True
    assert lb._check_pass(KEY, {"test_auroc": 0.98, "test_accuracy": 0.999}) is False
    assert lb._check_pass(KEY, {"test_accuracy": 0.999}) is False


def test_primary_metric_headline():
    card = {"primary_metric": "test_auroc"}
    r = {"metrics": {"test_accuracy": 0.99, "test_auroc": 0.9999}, "passed": True,
         "model_key": "m", "_ts": "2026-11-04T00:00:00Z"}
    assert status._fmt_result(r, None, card).startswith("AUROC 0.9999")
    assert status._fmt_result(r).startswith("accuracy")  # global order without a card key
    res = {"m": {"metrics": {"test_accuracy": {"mean": 1.0}, "test_auroc": {"mean": 0.9}}}}
    assert rep.primary_metric_key(res, card) == "test_auroc"
    assert rep.primary_metric_key(res) == "test_accuracy"
    assert rep._LABELS["test_average_precision"] == "AP"
    assert rep._LABELS["test_tpr_at_fpr05"] == "TPR@5%FPR"


def test_card_and_listing():
    meta = rep.load_metadata()
    assert meta[KEY]["primary_metric"] == "test_auroc" and meta[KEY]["status"] == "ready"
    from surge.benchmarks.registry import list_benchmarks, resolve_benchmark_key

    assert resolve_benchmark_key("cmod") == KEY
    assert KEY in str(list_benchmarks())
