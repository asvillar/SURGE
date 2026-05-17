"""Regression helpers plus thin wrappers around sklearn classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, X, y):
    start = __import__("time").time()
    preds = model.predict(X)
    return (
        mean_squared_error(y, preds),
        r2_score(y, preds),
        (__import__("time").time() - start) / len(X),
    )


def summarize(results):
    return {
        "mse": results["mse_list"],
        "r2": results["r2_list"],
        "time": results["time_list"],
        "mse_mean": float(np.mean(results["mse_list"])),
        "mse_std": float(np.std(results["mse_list"])),
        "r2_mean": float(np.mean(results["r2_list"])),
        "r2_std": float(np.std(results["r2_list"])),
        "time_mean": float(np.mean(results["time_list"])),
    }


# ---------------------------------------------------------------------------
# Classification (Phase 2 — benchmark plan; all delegate to scikit-learn)
# ---------------------------------------------------------------------------


def accuracy_score(
    y_true,
    y_pred,
    *,
    normalize: bool = True,
    sample_weight=None,
) -> float:
    from sklearn.metrics import accuracy_score as sk_accuracy_score

    return float(
        sk_accuracy_score(
            y_true, y_pred, normalize=normalize, sample_weight=sample_weight
        )
    )


def f1_score(
    y_true,
    y_pred,
    *,
    average: str = "macro",
    sample_weight=None,
    zero_division: str = "warn",
) -> float:
    from sklearn.metrics import f1_score as sk_f1_score

    return float(
        sk_f1_score(
            y_true,
            y_pred,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
    )


def auroc(
    y_true,
    y_prob,
    *,
    average: str = "macro",
    multi_class: str = "ovr",
    sample_weight=None,
) -> float:
    """Area under ROC. For binary, ``y_prob`` can be ``(n,)`` positive-class scores."""
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(
            y_true,
            y_prob,
            average=average,
            multi_class=multi_class,
            sample_weight=sample_weight,
        )
    )


def log_loss(
    y_true,
    y_prob,
    *,
    labels=None,
    sample_weight=None,
    **kwargs,
) -> float:
    from sklearn.metrics import log_loss as sk_log_loss

    return float(
        sk_log_loss(
            y_true,
            y_prob,
            labels=labels,
            sample_weight=sample_weight,
            **kwargs,
        )
    )


def top_k_accuracy_score(
    y_true,
    y_prob,
    k: int = 5,
    *,
    labels=None,
    sample_weight=None,
) -> float:
    """``y_prob`` shape ``(n_samples, n_classes)``."""
    from sklearn.metrics import top_k_accuracy_score

    return float(
        top_k_accuracy_score(
            y_true,
            y_prob,
            k=k,
            labels=labels,
            sample_weight=sample_weight,
        )
    )


def expected_calibration_error(
    y_true,
    y_prob,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Multiclass-safe reliability ECE: bin by prediction confidence (max prob).

    Binary: ``y_prob`` may be ``(n,)`` scores for the positive class.
    Multiclass: ``y_prob`` is ``(n, n_classes)``.
    """
    y_true = np.asarray(y_true).astype(int, copy=False)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        conf = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
        correct = (y_true == 1).astype(float)
    else:
        pred = np.argmax(y_prob, axis=1)
        conf = np.clip(y_prob.max(axis=1), 1e-6, 1.0 - 1e-6)
        correct = (pred == y_true).astype(float)

    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(conf, qs)
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(conf)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        w = float(mask.mean())
        if w <= 0.0:
            continue
        acc = float(correct[mask].mean())
        conf_m = float(conf[mask].mean())
        ece += w * abs(acc - conf_m)

    return float(ece)


__all__ = [
    "accuracy_score",
    "auroc",
    "evaluate",
    "expected_calibration_error",
    "f1_score",
    "log_loss",
    "summarize",
    "top_k_accuracy_score",
]
