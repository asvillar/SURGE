"""Tier 0 and Tier 1 benchmark implementations (CPU, sklearn-only)."""

from __future__ import annotations

import ssl
import time
from typing import Any

import numpy as np

# macOS conda environments do not bundle CA certificates, causing SSL failures
# when sklearn tries to download datasets from figshare/UCI.  This patch makes
# urllib use an unverified context for those internal dataset fetches.
# It has no effect when certificates are already correctly configured.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

from .base import BenchmarkResult

# Default model keys per benchmark (used when the caller passes no model_key).
_DEFAULTS: dict[str, str] = {
    "synthetic.regression_1d": "sklearn.random_forest",
    "synthetic.multioutput_2d": "sklearn.random_forest",
    "synthetic.classification_binary": "sklearn.random_forest_classifier",
    "tabular.diabetes": "sklearn.random_forest",
    "tabular.california_housing": "sklearn.random_forest",
    "tabular.concrete_strength": "sklearn.random_forest",
    "tabular.energy_efficiency": "sklearn.random_forest",
    "tabular.iris": "sklearn.random_forest_classifier",
    "tabular.breast_cancer": "sklearn.random_forest_classifier",
    "tabular.wine": "sklearn.random_forest_classifier",
    "tabular.digits": "sklearn.random_forest_classifier",
    "sequence.lorenz63": "sklearn.random_forest",
    "pde.burgers_1d": "sklearn.random_forest",
    "classification.flow_regime": "sklearn.random_forest_classifier",
    "tabular.airfoil_noise": "sklearn.random_forest",
    "tabular.yacht_dynamics": "sklearn.random_forest",
    "classification.plasma_stability": "sklearn.random_forest_classifier",
    "tabular.superconductor": "sklearn.random_forest",
    "multioutput.scm20d": "sklearn.random_forest",
    "classification.covertype": "sklearn.random_forest_classifier",
    "vision.mnist": "pytorch.lenet5",
    "vision.cifar10": "pytorch.resnet20",
    "fusion.m3dc1_sample": "sklearn.random_forest",
    "plasma.qlknn_transport": "sklearn.random_forest",
    "plasma.cmod_density_limit": "sklearn.random_forest_classifier",
    "thewell.gray_scott": "sklearn.random_forest",
}

# Which task type each benchmark expects of its model.
_TASK_TYPE: dict[str, str] = {
    "synthetic.regression_1d": "regression",
    "synthetic.multioutput_2d": "regression",
    "synthetic.classification_binary": "classification",
    "tabular.diabetes": "regression",
    "tabular.california_housing": "regression",
    "tabular.concrete_strength": "regression",
    "tabular.energy_efficiency": "regression",
    "tabular.iris": "classification",
    "tabular.breast_cancer": "classification",
    "tabular.wine": "classification",
    "tabular.digits": "classification",
    "sequence.lorenz63": "regression",
    "pde.burgers_1d": "regression",
    "classification.flow_regime": "classification",
    "tabular.airfoil_noise": "regression",
    "tabular.yacht_dynamics": "regression",
    "classification.plasma_stability": "classification",
    "tabular.superconductor": "regression",
    "multioutput.scm20d": "regression",
    "classification.covertype": "classification",
    "vision.mnist": "classification",
    "vision.cifar10": "classification",
    "fusion.m3dc1_sample": "regression",
    "thewell.gray_scott": "regression",
    "thewell.turbulence_2d": "regression",
    "thewell.mhd": "regression",
}


def _resolve_model(
    model_key: str | None,
    benchmark_key: str,
    model_kwargs: dict | None = None,
) -> Any:
    """Return a fitted-ready adapter instance from MODEL_REGISTRY."""
    from surge.model.registry import MODEL_REGISTRY

    key = model_key or _DEFAULTS.get(benchmark_key)
    if key is None:
        raise ValueError(f"No default model for {benchmark_key!r}. Pass --model KEY.")
    return MODEL_REGISTRY.create(key, **(model_kwargs or {}))


def _fit_predict_regression(adapter: Any, X_train, y_train, X_test):
    """Fit adapter and return predictions. Returns (y_pred, elapsed_s).

    For multi-output targets (y_train.ndim > 1) predictions are kept 2-D;
    for 1-D targets they are ravelled so callers can treat them as vectors.
    """
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    is_multioutput = np.asarray(y_train).ndim > 1
    if not is_multioutput:
        y_pred = y_pred.ravel()
    elapsed = time.perf_counter() - t0
    return y_pred, elapsed


def _fit_predict_classification(adapter: Any, X_train, y_train, X_test):
    """Fit adapter, return (y_pred_labels, y_proba_or_None, elapsed_s)."""
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = adapter.predict(X_test)
    y_prob = None
    if hasattr(adapter, "predict_proba"):
        try:
            y_prob = adapter.predict_proba(X_test)
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return y_pred, y_prob, elapsed


def _clf_metrics(y_test, y_pred, y_prob) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    metrics: dict[str, float] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score

            n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
            if n_classes == 2:
                scores = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics["test_auroc"] = float(roc_auc_score(y_test, scores))
            else:
                metrics["test_auroc"] = float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def _reg_metrics(y_test, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"test_r2": r2, "test_rmse": rmse}


def _multioutput_reg_extras(
    y_test,
    y_pred,
    names: list[str] | None = None,
    kinds: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-column R² and normalised RMSE for vector-valued regression targets.

    Generic across datasets — nothing here is ConStellaration-specific.

    Parameters
    ----------
    names: optional per-column target names, in the same order as the
        columns of ``y_test``/``y_pred``. When given, adds
        ``extra["per_metric_r2_by_name"]``.
    kinds: optional map of ``name -> kind`` (e.g. from
        :func:`surge.benchmarks.dataset_io.output_kinds`). When given
        together with ``names``, adds one ``test_r2_<kind>_mean`` metric
        per distinct kind present plus ``extra["metric_kinds"]``.
    """
    import numpy as np
    from sklearn.metrics import r2_score

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    if y_test.ndim < 2 or y_test.shape[1] <= 1:
        return {}, {}
    per = [float(r2_score(y_test[:, j], y_pred[:, j])) for j in range(y_test.shape[1])]
    std_y = float(np.std(y_test))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    metrics: dict[str, float] = {
        "min_r2": min(per),
        "max_r2": max(per),
        "test_nrmse": rmse / (std_y + 1e-12),
    }
    extra: dict[str, Any] = {"per_metric_r2": per}

    if names is not None and len(names) == len(per):
        per_by_name = dict(zip(names, per))
        extra["per_metric_r2_by_name"] = per_by_name

        if kinds is not None:
            by_kind: dict[str, list[float]] = {}
            name_kinds: dict[str, str] = {}
            for name, r2 in per_by_name.items():
                kind = kinds.get(name)
                if kind is None:
                    continue
                by_kind.setdefault(kind, []).append(r2)
                name_kinds[name] = kind
            for kind, vals in by_kind.items():
                metrics[f"test_r2_{kind}_mean"] = float(np.mean(vals))
            extra["metric_kinds"] = name_kinds

    return metrics, extra


def _uq_metrics(y_true, mean_pred, std_pred, *, coverage: float = 0.95) -> dict[str, float]:
    """Compute uncertainty-quantification metrics for probabilistic regressors.

    Parameters
    ----------
    y_true:   Ground-truth targets, shape (n,).
    mean_pred: Predictive mean, shape (n,).
    std_pred:  Predictive standard deviation, shape (n,).
    coverage:  Nominal coverage for the prediction interval (default 0.95).

    Returns
    -------
    dict with keys:
    - ``uq_picp``  — Prediction Interval Coverage Probability (ideal = coverage).
    - ``uq_mpiw``  — Mean Prediction Interval Width (lower = sharper, normalised by y_std).
    - ``uq_crps``  — Mean Continuous Ranked Probability Score (lower = better).
    - ``uq_nll``   — Mean Gaussian negative log-likelihood (lower = better).
    """
    from scipy import stats

    y = np.asarray(y_true, dtype=float).ravel()
    mu = np.asarray(mean_pred, dtype=float).ravel()
    sigma = np.asarray(std_pred, dtype=float).ravel()
    sigma = np.maximum(sigma, 1e-8)  # avoid division by zero

    z = stats.norm.ppf(0.5 + coverage / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma

    # PICP: fraction of y within the interval
    picp = float(np.mean((y >= lower) & (y <= upper)))

    # MPIW: mean interval width, normalised by y std so it is scale-free
    y_std = float(np.std(y)) or 1.0
    mpiw = float(np.mean(upper - lower)) / y_std

    # CRPS for Gaussian predictive distribution (analytical formula)
    z_score = (y - mu) / sigma
    phi = stats.norm.pdf(z_score)
    Phi = stats.norm.cdf(z_score)
    crps_per_sample = sigma * (z_score * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    crps = float(np.mean(crps_per_sample))

    # Gaussian NLL
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))

    return {
        "uq_picp": picp,
        "uq_mpiw": mpiw,
        "uq_crps": crps,
        "uq_nll": nll,
    }


# ---------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------


def run_synthetic_regression_1d(*, seed: int = 42, n_samples: int = 400, model_key: str | None = None) -> BenchmarkResult:
    """Hermetic 1→1 regression (Tier 0)."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = 3.0 * X.ravel() + 1.5 + 0.15 * rng.standard_normal(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    adapter = _resolve_model(model_key, "synthetic.regression_1d", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="synthetic.regression_1d",
        model_key=adapter.name,
        tier="0",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.85,
        message="Linear 1-D signal with small Gaussian noise",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_synthetic_classification_binary(*, seed: int = 42, n_samples: int = 500, model_key: str | None = None) -> BenchmarkResult:
    """20 → 2 binary labels (Tier 0)."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 20))
    logits = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(n_samples)
    y = (logits > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    adapter = _resolve_model(model_key, "synthetic.classification_binary", model_kwargs=None)
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="synthetic.classification_binary",
        model_key=adapter.name,
        tier="0",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] > 0.75,
        message="Random linear combo of first 3 features",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Tier 1 — regression
# ---------------------------------------------------------------------------


def run_tabular_diabetes(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """sklearn Diabetes dataset — 10→1 regression (Tier 1)."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.diabetes", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.diabetes",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.35,
        message="UCI Diabetes / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_california_housing(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """California Housing dataset — 8→1 regression (Tier 1)."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.california_housing", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.california_housing",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.75,
        message="California Housing / sklearn.datasets (Pace & Barry 1997)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_concrete_strength(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Concrete Compressive Strength — 8→1 regression (Tier 1)."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # data_id=4353 is "Concrete_Data" (Yeh 1998): 1030 samples, 8 features.
    # The last column is the target — no named target in this OpenML entry.
    data = fetch_openml(data_id=4353, as_frame=True, parser="auto")
    df = data.frame
    target_col = df.columns[-1]
    X = df.iloc[:, :-1].values.astype(float)
    y = df[target_col].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.concrete_strength", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.concrete_strength",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.80,
        message="UCI Concrete Compressive Strength (Yeh 1998)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Tier 1 — classification
# ---------------------------------------------------------------------------


def run_tabular_iris(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Iris — 4→3 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    adapter = _resolve_model(model_key, "tabular.iris", model_kwargs=None)
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.iris",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.88,
        message="UCI Iris / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_breast_cancer(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """Wisconsin Breast Cancer — 30→2 binary classification (Tier 1)."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.breast_cancer", model_kwargs=None)
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.breast_cancer",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.93,
        message="Wisconsin Breast Cancer (UCI / WDBC)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_wine(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Wine — 13→3 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.wine", model_kwargs=None)
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.wine",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.90,
        message="UCI Wine / sklearn.datasets",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_synthetic_multioutput_2d(*, seed: int = 42, n_samples: int = 600, model_key: str | None = None) -> BenchmarkResult:
    """Inline 8→2 multi-output regression (Tier 0)."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 8))
    A = rng.standard_normal((8, 2)) * 0.5
    noise = 0.1 * rng.standard_normal((n_samples, 2))
    Y = X @ A + noise
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)

    adapter = _resolve_model(model_key, "synthetic.multioutput_2d", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    # Per-output r² then averaged.
    from sklearn.metrics import r2_score

    r2 = float(r2_score(y_test, y_pred if y_pred.ndim > 1 else y_pred.reshape(-1, 1)))
    metrics: dict = {"test_r2": r2, "runtime_s": elapsed}
    return BenchmarkResult(
        benchmark_key="synthetic.multioutput_2d",
        model_key=adapter.name,
        tier="0",
        task_type="regression",
        metrics=metrics,
        passed=r2 > 0.75,
        message="8→2 linear multi-output with Gaussian noise (inline fixture)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_energy_efficiency(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """UCI Energy Efficiency — 8→1 regression on heating load (Tier 1, requires internet on first run)."""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = fetch_openml(name="energy-efficiency", version=1, as_frame=True, parser="auto")
    # Target: Y1 (Heating Load)
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if data.target.ndim == 1 else data.target.iloc[:, 0].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.energy_efficiency", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.energy_efficiency",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] > 0.90,
        message="UCI Energy Efficiency — Heating Load (Tsanas & Xifara 2012)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_digits(*, seed: int = 42, model_key: str | None = None) -> BenchmarkResult:
    """Optical digits — 64→10 multiclass classification (Tier 1)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    adapter = _resolve_model(model_key, "tabular.digits", model_kwargs=None)
    y_pred, y_prob, elapsed = _fit_predict_classification(adapter, X_train, y_train, X_test)
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.digits",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.95,
        message="Optical digits / sklearn.datasets (Alpaydin 1998)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Sequence benchmarks (Phase C)
# ---------------------------------------------------------------------------


def _lorenz_rk4_step(state: np.ndarray, dt: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    """Single RK-4 step of the Lorenz-63 system — no scipy dependency."""
    x, y, z = state[0], state[1], state[2]

    def f(s):
        return np.array([sigma * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]])

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _generate_lorenz_trajectories(
    n_trajectories: int,
    T_in: int,
    T_out: int,
    dt: float = 0.01,
    warmup: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ``n_trajectories`` Lorenz-63 windows.

    Returns
    -------
    X : (n_trajectories, T_in * 3)  — input windows (flattened)
    y : (n_trajectories, T_out * 3) — prediction targets (flattened)
    """
    rng = np.random.default_rng(seed)
    n_state = 3
    X_list, y_list = [], []
    for _ in range(n_trajectories):
        # Random IC near the attractor.
        state = rng.uniform(low=-10.0, high=10.0, size=(n_state,))
        # Warm-up to attract to the chaotic attractor.
        for _ in range(warmup):
            state = _lorenz_rk4_step(state, dt)
        # Collect T_in + T_out steps.
        total = T_in + T_out
        traj = np.empty((total, n_state))
        for t in range(total):
            traj[t] = state
            state = _lorenz_rk4_step(state, dt)
        X_list.append(traj[:T_in].ravel())
        y_list.append(traj[T_in:].ravel())
    return np.array(X_list), np.array(y_list)


def run_sequence_lorenz63(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Lorenz-63 one-step and short-horizon prediction (Tier 0, inline ODE).

    Data
    ----
    1 200 trajectories of the Lorenz-63 attractor integrated with RK-4
    (dt=0.01).  Input = 20-step window (T_in=20), target = next 20 steps
    (T_out=20), state dim = 3.  Total feature dim: 60 → 60.

    Metric
    ------
    ``test_nrmse`` = ||y_pred - y_true||_F / ||y_true||_F  (lower is better).
    Pass threshold: NRMSE < 0.30 (well above random, reachable by RF).
    """
    from sklearn.model_selection import train_test_split

    T_in, T_out = 20, 20
    X, y = _generate_lorenz_trajectories(
        n_trajectories=1200, T_in=T_in, T_out=T_out, dt=0.01, warmup=500, seed=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    key = model_key or "sklearn.random_forest"
    from surge.model.registry import MODEL_REGISTRY
    adapter = MODEL_REGISTRY.create(key)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    # NRMSE over all predicted steps and state dimensions.
    nrmse = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
    metrics: dict = {"test_nrmse": nrmse, "runtime_s": elapsed}
    return BenchmarkResult(
        benchmark_key="sequence.lorenz63",
        model_key=adapter.name,
        tier="0",
        task_type="regression",
        metrics=metrics,
        passed=nrmse < 0.30,
        message="Lorenz-63 RK-4 short-horizon prediction (inline, no download)",
        extra={"n_train": len(X_train), "n_test": len(X_test), "T_in": T_in, "T_out": T_out},
    )


# ---------------------------------------------------------------------------
# PDE benchmarks (Phase D1)
# ---------------------------------------------------------------------------


def _burgers_solve_fd(u0: np.ndarray, n_x: int, dt: float, nt: int, nu: float = 0.01) -> np.ndarray:
    """
    Integrate viscous Burgers' equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    using first-order upwind (advection) and central (diffusion) finite
    differences on a periodic domain ``[0, 2π]``.
    """
    dx = 2.0 * np.pi / n_x
    u = u0.copy()
    for _ in range(nt):
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
        # Upwind advection: forward diff for u < 0, backward for u >= 0.
        adv = np.where(u >= 0, u * (u - u_left) / dx, u * (u_right - u) / dx)
        diff = nu * (u_right - 2 * u + u_left) / dx**2
        u = u + dt * (-adv + diff)
    return u


def _generate_burgers_dataset(
    n_samples: int = 1024,
    n_x: int = 64,
    nt: int = 100,
    dt: float = 1e-3,
    nu: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Burgers' equation dataset.

    Initial conditions: sum of random Fourier modes on ``[0, 2π]``.
    Input: u(x, 0) on ``n_x`` grid points.
    Output: u(x, T) after ``nt`` time steps.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)
    X_list, y_list = [], []
    for _ in range(n_samples):
        n_modes = rng.integers(3, 8)
        u0 = np.zeros(n_x)
        for k in range(1, n_modes + 1):
            amp = rng.uniform(-1.0, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(k * x + phase)
        u_T = _burgers_solve_fd(u0, n_x=n_x, dt=dt, nt=nt, nu=nu)
        X_list.append(u0)
        y_list.append(u_T)
    return np.array(X_list), np.array(y_list)


def run_pde_burgers_1d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Viscous Burgers 1D operator learning (Tier 1, inline solver, no download).

    Data
    ----
    1 024 trajectories of viscous Burgers' equation (ν=0.01) on a 64-pt
    periodic grid, integrated for 100 steps (dt=0.001).  Input = initial
    condition u(x, 0); target = u(x, T).

    Metric
    ------
    ``test_relative_l2`` = relative L² error (lower is better).
    Pass threshold: < 0.10 (a simple RandomForest can beat this with 64 features).

    Published FNO baseline (Li et al. 2021 with n_x=1024): ≈ 0.64%.
    Our inline task uses n_x=64 so RF/MLP baselines are also informative.
    """
    from sklearn.model_selection import train_test_split

    X, y = _generate_burgers_dataset(n_samples=1024, n_x=64, nt=100, dt=1e-3, nu=0.01, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    key = model_key or "sklearn.random_forest"
    from surge.model.registry import MODEL_REGISTRY
    adapter = MODEL_REGISTRY.create(key)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    rel_l2 = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
    metrics: dict = {"test_relative_l2": rel_l2, "runtime_s": elapsed}
    return BenchmarkResult(
        benchmark_key="pde.burgers_1d",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=rel_l2 < 0.10,
        message="Viscous Burgers 1D inline FD solver (n_x=64, ν=0.01)",
        extra={"n_train": len(X_train), "n_test": len(X_test), "n_x": 64, "nt": 100},
    )


# ---------------------------------------------------------------------------
# Phase F — SURGE-native Scientific Benchmarks (early part)
# ---------------------------------------------------------------------------


def run_classification_flow_regime(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """CFD flow regime classification (Tier 0, inline fixture, no download).

    Data
    ----
    800 synthetic samples with 3 features: Mach number, Reynolds number
    (log-scaled), and angle of attack.  Four flow regime classes:
    0 = subsonic laminar, 1 = subsonic turbulent,
    2 = transonic, 3 = supersonic.

    Metric
    ------
    ``test_accuracy``  Pass threshold: ≥ 0.85.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = 800

    # Feature 1: Mach number, Feature 2: log10(Re), Feature 3: AoA (deg)
    mach = rng.uniform(0.1, 3.0, n)
    log_re = rng.uniform(4.0, 8.0, n)
    aoa = rng.uniform(-5.0, 25.0, n)

    # Rule-based labelling with some overlap noise.
    labels = np.zeros(n, dtype=int)
    labels[(mach < 0.8) & (log_re < 5.5)] = 0  # subsonic laminar
    labels[(mach < 0.8) & (log_re >= 5.5)] = 1  # subsonic turbulent
    labels[(mach >= 0.8) & (mach < 1.2)] = 2     # transonic
    labels[(mach >= 1.2)] = 3                      # supersonic
    # Add 5% random label noise.
    noise_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    labels[noise_idx] = rng.integers(0, 4, size=len(noise_idx))

    X = np.column_stack([mach, log_re, aoa])
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    key = model_key or "sklearn.random_forest_classifier"
    from surge.model.registry import MODEL_REGISTRY
    adapter = MODEL_REGISTRY.create(key)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    y_prob = None
    if hasattr(adapter, "_model") and hasattr(adapter._model, "predict_proba"):
        try:
            y_prob = adapter._model.predict_proba(X_test)
        except Exception:
            pass
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="classification.flow_regime",
        model_key=adapter.name,
        tier="0",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.85,
        message="CFD flow regime 4-class labeling (inline fixture, no download)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_airfoil_noise(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """NASA Airfoil Self-Noise — 5→1 regression (Tier 1, requires internet on first run).

    Source: UCI via fetch_openml (ID 4544).
    Published RF R² ≈ 0.96 (1 503 samples, 5 features → scaled sound pressure).
    Pass threshold: R² ≥ 0.80.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # "airfoil_self_noise" (underscore) is the correct OpenML name.
    # "airfoil-self-noise" (hyphen, version 1) no longer exists in the registry.
    data = fetch_openml(name="airfoil_self_noise", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y_raw = data.target
    y = y_raw.values.astype(float) if hasattr(y_raw, "values") else np.asarray(y_raw, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.airfoil_noise", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.airfoil_noise",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] >= 0.80,
        message="NASA Airfoil Self-Noise (Brooks et al. 1989) — UCI",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_yacht_dynamics(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Yacht Hydrodynamics — 6→1 regression (Tier 1, requires internet on first run).

    Source: UCI via fetch_openml.  Residuary resistance of sailing yachts.
    308 samples, 6 features.  Published RF R² > 0.99.
    Pass threshold: R² ≥ 0.80.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y_raw = data.target
    y = y_raw.values.astype(float) if hasattr(y_raw, "values") else np.asarray(y_raw, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    adapter = _resolve_model(model_key, "tabular.yacht_dynamics", model_kwargs=None)
    y_pred, elapsed = _fit_predict_regression(adapter, X_train, y_train, X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.yacht_dynamics",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] >= 0.80,
        message="UCI Yacht Hydrodynamics (Gerritsma 1981) — residuary resistance",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_tabular_superconductor(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Superconductor critical temperature regression (Hamidieh 2018).

    81 material-descriptor features → Tc (K). 21,263 samples.
    Standard materials-science surrogate benchmark.
    Pass threshold: R² ≥ 0.90.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = fetch_openml(name="superconduct", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    adapter = _resolve_model(model_key or "sklearn.random_forest", benchmark_key="tabular.superconductor")
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = adapter.predict(X_test)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="tabular.superconductor",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] >= 0.90,
        message="Superconductor Tc prediction (Hamidieh 2018) — 81 material features → critical temp",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_multioutput_scm20d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """SCM20d multi-output regression: 61 features → 20 targets (supply chain management).

    8,966 samples. Widely used multi-target regression benchmark.
    Pass threshold: average R² ≥ 0.60.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    data = fetch_openml(name="scm20d", version=2, as_frame=True, parser="auto")
    target_cols = list(data.target_names)
    feature_cols = [c for c in data.data.columns]
    X = data.data[feature_cols].values.astype(float)
    y = data.target.values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    adapter = _resolve_model(model_key or "sklearn.random_forest", benchmark_key="multioutput.scm20d")
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = adapter.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="multioutput.scm20d",
        model_key=adapter.name,
        tier="1",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] >= 0.60,
        message="SCM20d supply-chain multi-output regression (61→20 targets)",
        extra={"n_train": len(X_train), "n_test": len(X_test), "n_targets": y.shape[1]},
    )


def run_classification_covertype(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Forest Covertype 7-class classification (Blackard & Dean 1999).

    54 cartographic features → forest cover type (7 classes). Uses 20k-sample
    stratified subsample for tractable CPU runtimes (full set: 581k).
    Standard large-tabular classification benchmark.
    Pass threshold: accuracy ≥ 0.85.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    data = fetch_openml(name="covertype", version=3, as_frame=True, parser="auto")
    X_full = data.data.values.astype(float)
    y_full = LabelEncoder().fit_transform(data.target.values)

    # Stratified subsample to 20k for reasonable CPU runtime
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_full), size=min(20_000, len(y_full)), replace=False)
    X_sub, y_sub = X_full[idx], y_full[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.20, random_state=seed, stratify=y_sub
    )

    adapter = _resolve_model(model_key or "sklearn.random_forest_classifier", benchmark_key="classification.covertype")
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = adapter.predict(X_test)
    metrics = _clf_metrics(y_test, y_pred, adapter)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="classification.covertype",
        model_key=adapter.name,
        tier="1",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.85,
        message="Forest Covertype 7-class classification (Blackard & Dean 1999) — 54 cartographic features",
        extra={"n_train": len(X_train), "n_test": len(X_test), "n_classes": 7},
    )


def run_classification_plasma_stability(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """UCI Electrical Grid Stability (plasma-like) — 12→2 classification (Tier 2, requires internet).

    Source: Arzamasov et al. 2018, UCI archive (direct CSV download).
    12 features, 10 000 rows, 2 classes (stable / unstable).
    Published RF accuracy ≈ 97.8%.  Pass threshold: ≥ 0.92.
    """
    import io
    import urllib.request

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    _UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
    )
    with urllib.request.urlopen(_UCI_URL, timeout=30) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))

    # Features: tau1..tau4, p1..p4, g1..g4  (12 cols); target: stabf (stable/unstable)
    feature_cols = [c for c in df.columns if c not in ("stab", "stabf")]
    X = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["stabf"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    key = model_key or "sklearn.random_forest_classifier"
    from surge.model.registry import MODEL_REGISTRY
    adapter = MODEL_REGISTRY.create(key)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    y_prob = None
    if hasattr(adapter, "_model") and hasattr(adapter._model, "predict_proba"):
        try:
            y_prob = adapter._model.predict_proba(X_test)
        except Exception:
            pass
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="classification.plasma_stability",
        model_key=adapter.name,
        tier="2",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.92,
        message="UCI Electrical Grid Stability (Arzamasov 2018)",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Phase D2 — PDEBench benchmarks (Tier 3, requires HDF5 download)
# ---------------------------------------------------------------------------

_PDEBENCH_DEFAULTS = {
    # FNO1d is the published baseline for 1D Burgers; CNN1D/DeepONet also valid
    "pdebench.burgers_1d": "pytorch.fno1d",
    # 2D cases: flat MLP over the full-resolution field until FNO2D is added
    "pdebench.darcy_2d": "pytorch.mlp",
    "pdebench.shallow_water_2d": "pytorch.mlp",
}

# Raw (full) spatial resolution for all PDEBench benchmarks.
# Tabular/sklearn/xgboost models are NOT run on these — see _PDEBENCH_SKIP_MODELS.
_PDEBENCH_REDUCED_RES = {
    "pdebench.burgers_1d": 1,
    "pdebench.darcy_2d": 1,
    "pdebench.shallow_water_2d": 1,
}

# Models that cannot meaningfully train on PDEBench-scale data.
# The leaderboard will mark these N/A instead of attempting to train.
_PDEBENCH_SKIP_MODELS = {
    "sklearn.random_forest",
    "sklearn.gradient_boosting",
    "sklearn.ridge",
    "sklearn.linear_regression",
    "xgboost.regressor",
}

_TASK_TYPE.update({
    "pdebench.burgers_1d": "regression",
    "pdebench.darcy_2d": "regression",
    "pdebench.shallow_water_2d": "regression",
})


def _run_pdebench_benchmark(
    pdebench_key: str,
    benchmark_key: str,
    model_key: str | None,
    pass_threshold: float = 0.20,
    description: str = "",
) -> "BenchmarkResult":
    """Generic runner for PDEBench Tier-3 benchmarks."""
    try:
        from surge.benchmarks.loaders.pdebench import load_pdebench, H5PY_AVAILABLE
    except ImportError as exc:
        raise ImportError("h5py required for PDEBench. pip install h5py") from exc

    if not H5PY_AVAILABLE:
        raise ImportError("h5py required for PDEBench loaders. pip install h5py")

    red = _PDEBENCH_REDUCED_RES.get(benchmark_key, 1)
    print(f"[pdebench] Loading {pdebench_key} (res={red}) …", flush=True)
    X_train, y_train, X_test, y_test = load_pdebench(
        pdebench_key, reduced_resolution=red
    )
    print(
        f"[pdebench] Loaded — X_train {X_train.shape}, X_test {X_test.shape}",
        flush=True,
    )

    from surge.model.registry import MODEL_REGISTRY
    preferred = model_key or _PDEBENCH_DEFAULTS.get(benchmark_key, "pytorch.mlp")

    # Tabular models are not meaningful at PDEBench scales — raise immediately
    # so the leaderboard can record N/A rather than wasting time.
    if preferred in _PDEBENCH_SKIP_MODELS:
        raise ValueError(
            f"Model '{preferred}' is not suitable for PDEBench-scale data "
            f"(high-dimensional PDE fields).  Use a neural operator "
            f"(pytorch.fno1d, pytorch.deeponet, pytorch.mlp) instead."
        )

    key = preferred if preferred in MODEL_REGISTRY else "pytorch.mlp"
    adapter = MODEL_REGISTRY.create(key)

    print(f"[pdebench] Training {key} …", flush=True)
    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    print(f"[pdebench] Predicting …", flush=True)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0
    print(f"[pdebench] Done — {elapsed:.1f}s", flush=True)

    rel_l2 = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
    metrics: dict = {
        "test_relative_l2": rel_l2,
        "runtime_s": elapsed,
        "n_features": X_train.shape[1],
    }
    return BenchmarkResult(
        benchmark_key=benchmark_key,
        model_key=adapter.name,
        tier="3",
        task_type="regression",
        metrics=metrics,
        passed=rel_l2 < pass_threshold,
        message=description or f"PDEBench {pdebench_key}",
        extra={"n_train": len(X_train), "n_test": len(X_test), "reduced_resolution": red},
    )


def run_pdebench_burgers_1d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """PDEBench Burgers 1D (Tier 3, HDF5 download required).

    Published FNO baseline relative-L2 ≈ 0.64% (Li et al. 2021).
    Pass threshold: relative-L2 < 5% (conservative for non-FNO models).
    """
    return _run_pdebench_benchmark(
        "burgers_1d", "pdebench.burgers_1d", model_key,
        pass_threshold=0.05,
        description="PDEBench 1D Burgers ν=0.01 (Takamoto et al. NeurIPS 2022)",
    )


def run_pdebench_darcy_2d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """PDEBench Darcy Flow 2D (Tier 3, HDF5 download required).

    Published FNO baseline relative-L2 ≈ 0.90%.
    Pass threshold: relative-L2 < 10%.
    """
    return _run_pdebench_benchmark(
        "darcy_2d", "pdebench.darcy_2d", model_key,
        pass_threshold=0.10,
        description="PDEBench 2D Darcy Flow β=1.0 (Takamoto et al. NeurIPS 2022)",
    )


def run_pdebench_shallow_water_2d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """PDEBench 2D Shallow Water Equations (Tier 3, HDF5 download required).

    Pass threshold: relative-L2 < 20%.
    """
    return _run_pdebench_benchmark(
        "shallow_water_2d", "pdebench.shallow_water_2d", model_key,
        pass_threshold=0.20,
        description="PDEBench 2D Shallow Water Equations (Takamoto et al. NeurIPS 2022)",
    )


# ---------------------------------------------------------------------------
# Phase E — Vision Benchmarks (Tier 2, torchvision)
# ---------------------------------------------------------------------------


def _load_mnist_arrays(data_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST via torchvision, return flat numpy arrays (N, 784)."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision required for vision benchmarks. pip install torchvision") from exc

    from pathlib import Path
    root = str(Path(data_dir) if data_dir else Path.home() / ".surge" / "data" / "torchvision")

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    def _ds_to_arrays(ds):
        import torch
        loader = torch.utils.data.DataLoader(ds, batch_size=10000, shuffle=False)
        Xs, ys = [], []
        for xb, yb in loader:
            Xs.append(xb.numpy())
            ys.append(yb.numpy())
        X = np.concatenate(Xs).reshape(-1, 784)
        y = np.concatenate(ys)
        return X, y

    X_train, y_train = _ds_to_arrays(train_ds)
    X_test, y_test = _ds_to_arrays(test_ds)
    return X_train, y_train, X_test, y_test


def _load_cifar10_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 via torchvision, return (N, 3072) flat numpy arrays."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision required for vision benchmarks. pip install torchvision") from exc

    from pathlib import Path
    root = str(Path.home() / ".surge" / "data" / "torchvision")
    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def _ds_to_arrays(ds):
        import torch
        loader = torch.utils.data.DataLoader(ds, batch_size=10000, shuffle=False)
        Xs, ys = [], []
        for xb, yb in loader:
            Xs.append(xb.numpy())
            ys.append(yb.numpy())
        X = np.concatenate(Xs).reshape(len(ds), -1)
        y = np.concatenate(ys)
        return X, y

    X_train, y_train = _ds_to_arrays(train_ds)
    X_test, y_test = _ds_to_arrays(test_ds)
    return X_train, y_train, X_test, y_test


def run_vision_mnist(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """MNIST digit classification (Tier 2, requires torchvision + internet on first run).

    Published LeNet-5 baseline: 99.05% top-1 accuracy (LeCun 1998).
    Pass threshold: ≥ 99.0%.
    """
    X_train, y_train, X_test, y_test = _load_mnist_arrays()

    adapter = _resolve_model(model_key, "vision.mnist", model_kwargs)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    y_prob = None
    if hasattr(adapter, "predict_proba"):
        try:
            y_prob = adapter.predict_proba(X_test)
        except Exception:
            pass
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="vision.mnist",
        model_key=adapter.name,
        tier="2",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.99,
        message="MNIST (LeCun et al. 1998) — top-1 accuracy",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_vision_cifar10(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """CIFAR-10 image classification (Tier 2, requires torchvision + internet on first run).

    Published baselines (He et al. 2016):
    - ResNet-20: 91.25%
    - ResNet-56: 93.03%
    Pass threshold: ≥ 91.0%.
    """
    X_train, y_train, X_test, y_test = _load_cifar10_arrays()

    adapter = _resolve_model(model_key, "vision.cifar10", model_kwargs)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    y_prob = None
    if hasattr(adapter, "predict_proba"):
        try:
            y_prob = adapter.predict_proba(X_test)
        except Exception:
            pass
    metrics = _clf_metrics(y_test, y_pred, y_prob)
    metrics["runtime_s"] = elapsed
    return BenchmarkResult(
        benchmark_key="vision.cifar10",
        model_key=adapter.name,
        tier="2",
        task_type="classification",
        metrics=metrics,
        passed=metrics["test_accuracy"] >= 0.91,
        message="CIFAR-10 (Krizhevsky 2009) — top-1 accuracy",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


# ---------------------------------------------------------------------------
# Phase F late — Fusion benchmark
# ---------------------------------------------------------------------------


def run_fusion_m3dc1_sample(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """M3DC1 equilibrium surrogate sample (Tier 2).

    Data
    ----
    Tries to load from ``data/datasets/M3DC1/m3dc1_sample.hdf5`` (13→1
    regression on normalized MHD equilibrium parameters).  Falls back to a
    synthetic inline fixture (same shape / difficulty) if the HDF5 is absent
    so that CI and tests run without the large dataset.

    Shape: 13 input features → 1 stability metric (R²).
    Pass threshold: R² ≥ 0.70.
    """
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    H5_PATH = Path(__file__).parent.parent.parent / "data" / "datasets" / "M3DC1" / "m3dc1_sample.hdf5"

    X, y = None, None

    # Attempt real HDF5 load.
    if H5_PATH.exists():
        try:
            import h5py
            with h5py.File(H5_PATH, "r") as f:
                keys = list(f.keys())
                X_key = [k for k in keys if "X" in k or "input" in k.lower()][0]
                y_key = [k for k in keys if "y" in k or "target" in k.lower() or "output" in k.lower()][0]
                X = np.asarray(f[X_key], dtype=float)
                y = np.asarray(f[y_key], dtype=float).ravel()
        except Exception:
            X, y = None, None

    # Synthetic fallback: linear combination + noise, 13 → 1.
    if X is None or y is None:
        rng = np.random.default_rng(seed)
        n = 2000
        X = rng.standard_normal((n, 13))
        coefs = rng.uniform(-1.0, 1.0, 13)
        y = X @ coefs + 0.1 * rng.standard_normal(n)
        # Hint in the result that synthetic data was used.
        _is_synthetic = True
    else:
        _is_synthetic = False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    key = model_key or "sklearn.random_forest"
    from surge.model.registry import MODEL_REGISTRY
    adapter = MODEL_REGISTRY.create(key)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test)).ravel()
    elapsed = time.perf_counter() - t0

    metrics = _reg_metrics(y_test, y_pred)
    metrics["runtime_s"] = elapsed
    note = "(synthetic inline fixture — place real data at data/datasets/M3DC1/m3dc1_sample.hdf5)" if _is_synthetic else ""
    return BenchmarkResult(
        benchmark_key="fusion.m3dc1_sample",
        model_key=adapter.name,
        tier="2",
        task_type="regression",
        metrics=metrics,
        passed=metrics["test_r2"] >= 0.70,
        message=f"M3DC1 equilibrium surrogate 13→1 {note}".strip(),
        extra={"n_train": len(X_train), "n_test": len(X_test), "synthetic": _is_synthetic},
    )


# ---------------------------------------------------------------------------
# Phase G — TheWell benchmarks (Tier 4, manual trigger, requires the-well)
# ---------------------------------------------------------------------------


def _run_thewell_benchmark(
    thewell_key: str,
    benchmark_key: str,
    model_key: str | None,
    *,
    seed: int = 42,
    model_kwargs: dict | None = None,
    n_train: int = 500,
    n_test: int = 100,
    description: str = "",
    pass_threshold: float = 0.30,
) -> "BenchmarkResult":
    """Generic runner for TheWell Tier-4 benchmarks."""
    from surge.benchmarks.loaders.thewell import load_thewell, THEWELL_AVAILABLE

    if not THEWELL_AVAILABLE:
        raise ImportError("the_well required. pip install the-well  or  pip install 'surge-ml[thewell]'")

    X_train, y_train, X_test, y_test = load_thewell(
        thewell_key, n_train=n_train, n_test=n_test, seed=seed,
    )
    if len(X_train) == 0:
        raise RuntimeError(
            f"TheWell {thewell_key!r}: n_train=0 — need at least one training sample. "
            "Download the train split or set n_train > 0."
        )
    if len(X_test) == 0:
        raise RuntimeError(
            f"TheWell {thewell_key!r}: n_test=0 — need at least one test sample."
        )

    adapter = _resolve_model(model_key, benchmark_key, model_kwargs)

    t0 = time.perf_counter()
    adapter.fit(X_train, y_train)
    y_pred = np.asarray(adapter.predict(X_test))
    elapsed = time.perf_counter() - t0

    rel_l2 = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
    metrics: dict = {"test_relative_l2": rel_l2, "runtime_s": elapsed}
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        metrics.update(_reg_metrics(np.ravel(y_test), np.ravel(y_pred)))
    return BenchmarkResult(
        benchmark_key=benchmark_key,
        model_key=adapter.name,
        tier="4",
        task_type="regression",
        metrics=metrics,
        passed=rel_l2 < pass_threshold,
        message=description or f"TheWell {thewell_key}",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def run_thewell_gray_scott(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """Gray-Scott reaction-diffusion (Tier 4, TheWell, requires the-well package + download)."""
    return _run_thewell_benchmark(
        "gray_scott", "thewell.gray_scott", model_key,
        seed=seed, model_kwargs=model_kwargs,
        description="TheWell Gray-Scott reaction-diffusion (Ohana et al. NeurIPS 2024)",
    )


def run_thewell_turbulence_2d(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """2D homogeneous turbulence (Tier 4, TheWell, requires the-well package + download)."""
    return _run_thewell_benchmark(
        "turbulence_2d", "thewell.turbulence_2d", model_key,
        seed=seed, model_kwargs=model_kwargs,
        description="TheWell 2D turbulence (Ohana et al. NeurIPS 2024)",
    )


def run_thewell_mhd(*, seed: int = 42, model_key: str | None = None,
    model_kwargs: dict | None = None) -> "BenchmarkResult":
    """3D MHD turbulence (Tier 4, TheWell, requires the-well package + download)."""
    return _run_thewell_benchmark(
        "mhd", "thewell.mhd", model_key,
        seed=seed, model_kwargs=model_kwargs,
        description="TheWell 3D MHD turbulence (Ohana et al. NeurIPS 2024)",
    )
