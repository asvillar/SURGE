"""
Cross-model leaderboard for SURGE benchmarks.

Runs every compatible model against one or more benchmarks and renders a
per-benchmark comparison table (rows = models, columns = metrics) with the
best value per column highlighted.  All results are optionally logged to
MLflow so the comparison is browsable in the tracking UI.

Typical usage
-------------
::

    python -m surge.benchmarks.run --leaderboard --tier 1 --task-type classification
    python -m surge.benchmarks.run --leaderboard --benchmark tabular.iris
    python -m surge.benchmarks.run --leaderboard --all-benchmarks --mlflow
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .base import BenchmarkResult
from .registry import benchmark_info, list_benchmarks, run_benchmark

# ---------------------------------------------------------------------------
# Model compatibility matrix
# ---------------------------------------------------------------------------

# Models to try for each task type.  Listed in the order they appear in the
# table (roughly: ensemble → boosting → linear → neural).
_REGRESSION_MODELS: list[str] = [
    "sklearn.ridge",
    "sklearn.random_forest",
    "sklearn.gradient_boosting_regressor",
    "sklearn.mlp",
    "xgboost.xgbregressor",
    "lgbm.regressor",
    "catboost.regressor",
]

_CLASSIFICATION_MODELS: list[str] = [
    "sklearn.logistic_regression",
    "sklearn.random_forest_classifier",
    "sklearn.gradient_boosting_classifier",
    "xgboost.xgbclassifier",
    "lgbm.classifier",
    "catboost.classifier",
]


# Per-benchmark overrides: map a benchmark key to a specific model list.
# PDEBench benchmarks only use neural-operator / deep-learning models.
# Tabular/sklearn/XGBoost models are not viable at PDEBench spatial scales
# and are marked N/A in the leaderboard instead.
_PDEBENCH_OPERATOR_MODELS: list[str] = []
_PDEBENCH_2D_MODELS: list[str] = []
_VISION_MODELS: list[str] = []
_SEQUENCE_MODELS: list[str] = []
try:
    from surge.model.pytorch import PYTORCH_AVAILABLE as _PT
    if _PT:
        _PDEBENCH_OPERATOR_MODELS = [
            "pytorch.fno1d",
            "pytorch.deeponet",
            "pytorch.mlp",
            "pytorch.residual_mlp",
            "pytorch.cnn1d",
            "pytorch.kan",
            "pytorch.ddpm",
            "pytorch.cgan",
        ]
        _PDEBENCH_2D_MODELS = [
            "pytorch.fno2d",
            "pytorch.unet",
        ]
        _VISION_MODELS = [
            "pytorch.lenet5",
            "pytorch.resnet20",
            "pytorch.resnet56",
            "pytorch.vit",
            "pytorch.alexnet",
        ]
        # CIFAR-10 uses RGB (3-channel) — exclude grayscale-only LeNet5.
        _CIFAR10_MODELS = [m for m in _VISION_MODELS if m != "pytorch.lenet5"]
        # Temporal/sequence models — valid for time-series benchmarks.
        # NOT included in default tabular regression (no spatial structure).
        _SEQUENCE_MODELS = [
            "sklearn.random_forest",
            "sklearn.gradient_boosting_regressor",
            "xgboost.xgbregressor",
            "pytorch.mlp",
            "pytorch.residual_mlp",
            "pytorch.cnn1d",
            "pytorch.lstm",
            "pytorch.gru",
        ]
except Exception:
    pass

_BENCHMARK_MODEL_OVERRIDES: dict[str, list[str]] = {
    # Sequence / time-series benchmarks — include temporal models
    "sequence.lorenz63":         _SEQUENCE_MODELS,
    # Inline PDE benchmark (64-pt grid, no download) — operator models + generative
    "pde.burgers_1d":            _PDEBENCH_OPERATOR_MODELS,
    # Real PDEBench HDF5 1D — operator models only, tabular models are N/A
    "pdebench.burgers_1d":       _PDEBENCH_OPERATOR_MODELS,
    # 2D PDE benchmarks — FNO2d and U-Net
    "pdebench.darcy_2d":         _PDEBENCH_2D_MODELS,
    "pdebench.shallow_water_2d": _PDEBENCH_2D_MODELS,
    # Vision benchmarks
    "vision.mnist":              _VISION_MODELS,
    "vision.cifar10":            _CIFAR10_MODELS,
}

# Per-benchmark model constructor kwargs (e.g. image shape for vision models).
_BENCHMARK_MODEL_KWARGS: dict[str, dict] = {
    "vision.cifar10": {"img_size": 32, "in_channels": 3, "n_classes": 10},
    "vision.mnist":   {"img_size": 28, "in_channels": 1, "n_classes": 10},
}


def _default_models_for(task_type: str, benchmark_key: str | None = None) -> list[str]:
    # Per-benchmark override takes precedence.
    if benchmark_key is not None and benchmark_key in _BENCHMARK_MODEL_OVERRIDES:
        return list(_BENCHMARK_MODEL_OVERRIDES[benchmark_key])

    # Base lists already include sklearn + XGBoost + LightGBM + CatBoost.
    # Filter to only registered models so missing optional deps are silently skipped.
    from surge.model.registry import MODEL_REGISTRY

    if task_type == "regression":
        base = [m for m in _REGRESSION_MODELS if m in MODEL_REGISTRY]
        try:
            from surge.model.pytorch import PYTORCH_AVAILABLE
            if PYTORCH_AVAILABLE:
                base.append("pytorch.mlp")
                base.append("pytorch.residual_mlp")
                base.extend(["pytorch.ft_transformer", "pytorch.kan", "pytorch.vae"])
        except Exception:
            pass
        # NOTE: botorch.gp / botorch.sparse_gp excluded — too slow for default runs.
        return base

    # classification
    base = [m for m in _CLASSIFICATION_MODELS if m in MODEL_REGISTRY]
    try:
        from surge.model.pytorch import PYTORCH_AVAILABLE
        if PYTORCH_AVAILABLE:
            base.append("pytorch.mlp_classifier")
            base.extend(["pytorch.ft_transformer_classifier", "pytorch.kan_classifier"])
    except Exception:
        pass
    return base


# Metrics where lower is better (used to decide which direction to highlight).
_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "runtime_s", "test_rmse", "test_nrmse", "test_relative_l2",
    # UQ: lower = sharper / better calibrated
    "uq_mpiw", "uq_crps", "uq_nll",
    # Plasma paper metrics
    "test_nrmse_mean",
})

# Metrics for which higher is better (supplement _LOWER_IS_BETTER)
_HIGHER_IS_BETTER_EXTRAS: frozenset[str] = frozenset({"test_snr_mean"})

# Preferred column order for display (unknown keys appended alphabetically).
_METRIC_ORDER: list[str] = [
    "test_accuracy",
    "test_f1_macro",
    "test_auroc",
    "test_r2",
    "test_rmse",
    "test_nrmse",
    "test_relative_l2",
    # ConStellaration paper-style metrics
    "test_r2_mean",
    "test_rmse_mean",
    "test_nrmse_mean",
    "test_snr_mean",
    # UQ columns — only shown when ≥1 model returns uncertainty estimates
    "uq_picp",
    "uq_mpiw",
    "uq_crps",
    "uq_nll",
    "runtime_s",
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_leaderboard(
    benchmark_keys: list[str],
    *,
    model_keys: list[str] | None = None,
    seed: int = 42,
    n_seeds: int = 1,
    pytorch_mlp_epochs: int = 50,
    save_root: Path | None = Path("benchmark_reports"),
    use_hpo_cache: bool = False,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run every compatible model against each benchmark.

    Parameters
    ----------
    benchmark_keys:
        Benchmarks to run (must be registered).
    model_keys:
        Override the default model list.  If ``None`` the compatible set for
        each benchmark's ``task_type`` is used.
    seed:
        Base random seed.  When ``n_seeds > 1``, seeds ``seed, seed+1, …``
        are used for each repeat.
    n_seeds:
        Number of independent evaluation seeds.  Results are averaged and
        mean ± std are reported when > 1.  Default 1 (single run).
    use_hpo_cache:
        If True, load previously cached best hyperparameters (written by
        ``--hpo``) from ``save_root/hpo_cache/`` and pass them to the model
        constructor.  Falls back to defaults when no cache exists.
    pytorch_mlp_epochs:
        ``n_epochs`` cap for ``pytorch.mlp`` in leaderboard runs (avoids
        very long waits; default 100).
    save_root:
        Auto-save individual results here (``None`` to skip).

    Returns
    -------
    dict mapping benchmark_key → list[BenchmarkResult] (one per model tried).
    When ``n_seeds > 1`` each BenchmarkResult has its metrics replaced by
    mean values and a ``metric_std`` dict added to ``extra``.
    """
    from surge.model.registry import MODEL_REGISTRY

    # Cap models applied inside the loop.
    _PYTORCH_EPOCH_CAP_MODELS = {
        "pytorch.mlp", "pytorch.residual_mlp", "pytorch.mlp_classifier",
        "pytorch.cnn1d", "pytorch.lstm", "pytorch.gru",
        "pytorch.fno1d", "pytorch.deeponet",
        "pytorch.lenet5", "pytorch.resnet20", "pytorch.resnet56",
        "pytorch.ft_transformer", "pytorch.ft_transformer_classifier",
        "pytorch.kan", "pytorch.kan_classifier",
        "pytorch.vae", "pytorch.vit", "pytorch.alexnet",
        "pytorch.fno2d", "pytorch.unet",
        "pytorch.ddpm", "pytorch.cgan",
        "botorch.gp", "botorch.sparse_gp",
    }
    # Some models need more epochs than the global cap to converge.
    # Keys override pytorch_mlp_epochs for specific models.
    _PER_MODEL_EPOCH_CAP: dict[str, int] = {
        "pytorch.ft_transformer": 100,
        "pytorch.ft_transformer_classifier": 100,
        "pytorch.vae": 100,
    }

    seeds = list(range(seed, seed + n_seeds))
    results: dict[str, list[BenchmarkResult]] = {}

    for key in benchmark_keys:
        info = benchmark_info(key)
        task_type = info["task_type"]
        candidates = model_keys if model_keys is not None else _default_models_for(task_type, key)
        results[key] = []

        for model_key in candidates:
            if model_key not in MODEL_REGISTRY:
                print(f"  [skip] {model_key} not in MODEL_REGISTRY", file=sys.stderr)
                continue

            # Load cached HP when requested.
            cached_hp: dict = {}
            if use_hpo_cache and save_root is not None:
                try:
                    from .hpo import load_hpo_cache
                    hp = load_hpo_cache(key, model_key, root=save_root)
                    if hp:
                        cached_hp = hp
                        print(f"  [hpo-cache] {key}/{model_key}: {hp}", file=sys.stderr)
                except Exception:
                    pass

            # Per-benchmark model kwargs (e.g. image shape for vision models).
            bench_kwargs = _BENCHMARK_MODEL_KWARGS.get(key, {})

            # Collect one result per seed.
            seed_results: list[BenchmarkResult] = []
            for s in seeds:
                try:
                    if key == "plasma.constellaration_paper":
                        # Special per-metric runner matching arXiv:2506.19583 Appendix A.4
                        _ecap = _PER_MODEL_EPOCH_CAP.get(model_key, pytorch_mlp_epochs)
                        epoch_kwargs = {"n_epochs": _ecap} if model_key in _PYTORCH_EPOCH_CAP_MODELS else {}
                        res = _run_constellaration_paper_benchmark(
                            model_key,
                            model_kwargs={**epoch_kwargs, **cached_hp},
                            seed=s,
                        )
                    elif model_key in _PYTORCH_EPOCH_CAP_MODELS or bench_kwargs:
                        _ecap = _PER_MODEL_EPOCH_CAP.get(model_key, pytorch_mlp_epochs)
                        epoch_kwargs = {"n_epochs": _ecap} if model_key in _PYTORCH_EPOCH_CAP_MODELS else {}
                        adapter = MODEL_REGISTRY.create(model_key, **epoch_kwargs, **bench_kwargs, **cached_hp)
                        res = _run_with_adapter(key, adapter, seed=s)
                    elif cached_hp:
                        adapter = MODEL_REGISTRY.create(model_key, **cached_hp)
                        res = _run_with_adapter(key, adapter, seed=s)
                    else:
                        res = run_benchmark(key, seed=s, model_key=model_key)
                    if res is not None:
                        if save_root is not None:
                            res.save(root=save_root)
                        seed_results.append(res)
                except Exception as exc:
                    print(f"  [error] {key}/{model_key} seed={s}: {exc}", file=sys.stderr)

            if not seed_results:
                continue

            if n_seeds == 1:
                results[key].append(seed_results[0])
            else:
                # Aggregate: mean ± std across seeds.
                agg = _aggregate_seed_results(seed_results)
                results[key].append(agg)

    return results


def _aggregate_seed_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    """Merge multiple single-seed BenchmarkResults into one mean±std result."""
    import statistics

    base = results[0]
    all_keys = set()
    for r in results:
        all_keys.update(r.metrics.keys())

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for k in all_keys:
        vals = [r.metrics[k] for r in results if k in r.metrics and isinstance(r.metrics[k], (int, float))]
        if vals:
            mean_metrics[k] = float(np.mean(vals))
            std_metrics[k] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    extra = dict(base.extra)
    extra["n_seeds"] = len(results)
    extra["metric_std"] = std_metrics

    from .base import BenchmarkResult
    return BenchmarkResult(
        benchmark_key=base.benchmark_key,
        model_key=base.model_key,
        tier=base.tier,
        task_type=base.task_type,
        metrics=mean_metrics,
        passed=base.passed,
        message=base.message,
        extra=extra,
    )


def _run_with_adapter(benchmark_key: str, adapter: Any, *, seed: int) -> BenchmarkResult | None:
    """Run a benchmark using a pre-instantiated adapter."""
    import time

    import numpy as np
    from sklearn.model_selection import train_test_split

    from .base import BenchmarkResult
    from .tasks import _clf_metrics, _reg_metrics
    from .registry import benchmark_info

    info = benchmark_info(benchmark_key)
    task_type = info["task_type"]

    try:
        X, y = _load_dataset(benchmark_key)
    except Exception as exc:
        print(f"  [error] could not load {benchmark_key}: {exc}", file=sys.stderr)
        return None

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )

    t0 = time.perf_counter()
    try:
        adapter.fit(X_train, y_train)
        y_pred = np.asarray(adapter.predict(X_test))
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"  [error] fit/predict failed for {benchmark_key}: {exc}", file=sys.stderr)
        return None

    if task_type == "regression":
        # Sequence benchmarks: compute NRMSE instead of / in addition to R².
        if benchmark_key.startswith("sequence.") or benchmark_key.startswith("pde.") or benchmark_key.startswith("pdebench."):
            nrmse = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))
            metrics = {"test_nrmse": nrmse}
            # Also compute R² if single-output, skip for multi-output PDE fields.
            try:
                from sklearn.metrics import r2_score
                r2 = float(r2_score(y_test, y_pred.ravel() if y_test.ndim == 1 else y_pred))
                metrics["test_r2"] = r2
            except Exception:
                pass
        else:
            # Support multi-output: only ravel for 1-D targets.
            metrics = _reg_metrics(y_test, y_pred.ravel() if y_test.ndim == 1 else y_pred)

        # UQ metrics — computed when the model exposes predict_with_uncertainty.
        _uq_model = getattr(adapter, "_model", adapter)
        if hasattr(_uq_model, "predict_with_uncertainty"):
            try:
                _uq_out = _uq_model.predict_with_uncertainty(X_test)
                # Handle both (mean, std) tuples and single array returns.
                if isinstance(_uq_out, tuple) and len(_uq_out) == 2:
                    _uq_mean, _uq_std = _uq_out
                    _uq_mean = np.asarray(_uq_mean).ravel()
                    _uq_std = np.asarray(_uq_std).ravel()
                    if len(_uq_std) == len(y_test.ravel()):
                        from .tasks import _uq_metrics
                        metrics.update(_uq_metrics(y_test.ravel(), _uq_mean, _uq_std))
            except Exception as exc:
                pass  # UQ is optional — silently skip on failure
    else:
        y_prob = None
        if hasattr(adapter, "predict_proba"):
            try:
                y_prob = adapter.predict_proba(X_test)
            except Exception:
                pass
        metrics = _clf_metrics(y_test, y_pred, y_prob)

    metrics["runtime_s"] = elapsed
    passed = _check_pass(benchmark_key, metrics)

    return BenchmarkResult(
        benchmark_key=benchmark_key,
        model_key=adapter.name,
        tier=info["tier"],
        task_type=task_type,
        metrics=metrics,
        passed=passed,
        message=f"leaderboard run via {adapter.name}",
        extra={"n_train": len(X_train), "n_test": len(X_test)},
    )


def _load_dataset(benchmark_key: str):
    """Load the raw (X, y) arrays for a benchmark key."""
    from sklearn.datasets import (
        fetch_california_housing,
        load_breast_cancer,
        load_diabetes,
        load_digits,
        load_iris,
        load_wine,
    )

    loaders = {
        "synthetic.regression_1d": lambda: _synthetic_regression_1d(),
        "synthetic.classification_binary": lambda: _synthetic_classification_binary(),
        "synthetic.multioutput_2d": lambda: _synthetic_multioutput_2d(),
        "tabular.diabetes": lambda: load_diabetes(return_X_y=True),
        "tabular.california_housing": lambda: fetch_california_housing(return_X_y=True),
        "tabular.concrete_strength": lambda: _load_concrete_strength(),
        "tabular.energy_efficiency": lambda: _load_energy_efficiency(),
        "tabular.iris": lambda: load_iris(return_X_y=True),
        "tabular.breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "tabular.wine": lambda: load_wine(return_X_y=True),
        "tabular.digits": lambda: load_digits(return_X_y=True),
        "sequence.lorenz63": lambda: _load_lorenz63(),
        "pde.burgers_1d": lambda: _load_burgers_1d(),
        "classification.flow_regime": lambda: _load_flow_regime(),
        "tabular.airfoil_noise": lambda: _load_airfoil_noise(),
        "tabular.yacht_dynamics": lambda: _load_yacht_dynamics(),
        "classification.plasma_stability": lambda: _load_plasma_stability(),
        "tabular.superconductor": lambda: _load_superconductor(),
        "multioutput.scm20d": lambda: _load_scm20d(),
        "classification.covertype": lambda: _load_covertype(),
        # CTR-23 benchmarks (Grinsztajn et al. 2022, arxiv 2207.08815)
        "ctr23.abalone": lambda: _load_ctr23_abalone(),
        "ctr23.bike_sharing": lambda: _load_ctr23_bike_sharing(),
        "ctr23.diamonds": lambda: _load_ctr23_diamonds(),
        "ctr23.house_sales": lambda: _load_ctr23_house_sales(),
        "ctr23.brazilian_houses": lambda: _load_ctr23_brazilian_houses(),
        # DOE fusion plasma benchmarks
        "plasma.cmod_density_limit": lambda: _load_cmod_density_limit(),
        "plasma.qlknn_transport": lambda: _load_qlknn_transport(),
        "plasma.constellaration": lambda: _load_constellaration(),
        # Vision benchmarks
        "vision.cifar10": lambda: _load_cifar10(),
        "vision.mnist":   lambda: _load_mnist(),
    }
    if benchmark_key not in loaders:
        raise KeyError(f"No dataset loader for {benchmark_key!r}")
    return loaders[benchmark_key]()


def _synthetic_regression_1d():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(400, 1))
    y = 3.0 * X.ravel() + 1.5 + 0.15 * rng.standard_normal(400)
    return X, y


def _synthetic_classification_binary():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, 20))
    logits = X[:, :3].sum(axis=1) + 0.1 * rng.standard_normal(500)
    y = (logits > 0).astype(int)
    return X, y


def _synthetic_multioutput_2d():
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.standard_normal((600, 8))
    A = rng.standard_normal((8, 2)) * 0.5
    noise = 0.1 * rng.standard_normal((600, 2))
    Y = X @ A + noise
    return X, Y


def _load_concrete_strength():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=4353, as_frame=True, parser="auto")
    df = data.frame
    target_col = df.columns[-1]   # last column is compressive strength
    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col]).values.astype(float)
    return X, y


def _load_lorenz63():
    from .tasks import _generate_lorenz_trajectories

    return _generate_lorenz_trajectories(n_trajectories=1200, T_in=20, T_out=20, dt=0.01, warmup=500, seed=42)


def _load_burgers_1d():
    from .tasks import _generate_burgers_dataset

    return _generate_burgers_dataset(n_samples=1024, n_x=64, nt=100, dt=1e-3, nu=0.01, seed=42)


def _load_flow_regime(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 800
    mach = rng.uniform(0.1, 3.0, n)
    log_re = rng.uniform(4.0, 8.0, n)
    aoa = rng.uniform(-5.0, 25.0, n)
    labels = np.zeros(n, dtype=int)
    labels[(mach < 0.8) & (log_re < 5.5)] = 0
    labels[(mach < 0.8) & (log_re >= 5.5)] = 1
    labels[(mach >= 0.8) & (mach < 1.2)] = 2
    labels[(mach >= 1.2)] = 3
    noise_idx = rng.choice(n, size=int(0.05 * n), replace=False)
    labels[noise_idx] = rng.integers(0, 4, size=len(noise_idx))
    X = np.column_stack([mach, log_re, aoa])
    return X, labels


def _load_airfoil_noise():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="airfoil_self_noise", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if hasattr(data.target, "values") else np.asarray(data.target, dtype=float)
    return X, y


def _load_yacht_dynamics():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="yacht_hydrodynamics", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if hasattr(data.target, "values") else np.asarray(data.target, dtype=float)
    return X, y


def _load_plasma_stability():
    import io
    import urllib.request
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    _UCI_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"
    )
    with urllib.request.urlopen(_UCI_URL, timeout=30) as resp:
        df = pd.read_csv(io.BytesIO(resp.read()))
    feature_cols = [c for c in df.columns if c not in ("stab", "stabf")]
    X = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["stabf"].values)
    return X, y


def _load_energy_efficiency():
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="energy-efficiency", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float) if data.target.ndim == 1 else data.target.iloc[:, 0].values.astype(float)
    return X, y


def _load_superconductor():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="superconduct", version=1, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    return X, y


def _load_scm20d():
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="scm20d", version=2, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    y = data.target.values.astype(float)
    return X, y


def _load_covertype():
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder
    data = fetch_openml(name="covertype", version=3, as_frame=True, parser="auto")
    X = data.data.values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(data.target.values if hasattr(data.target, "values") else data.target)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(20_000, len(y)), replace=False)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Vision datasets
# ---------------------------------------------------------------------------

def _load_cifar10():
    """CIFAR-10: 60k 32×32 RGB images, 10 classes.

    Returns flat (N, 3072) float32 array and integer class labels.
    Train+test are concatenated; the leaderboard handles the split.
    Reference: Krizhevsky (2009) https://www.cs.toronto.edu/~kriz/cifar.html
    """
    try:
        import torch
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision required. pip install torchvision") from exc

    from pathlib import Path
    root = str(Path.home() / ".surge" / "data" / "torchvision")
    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def _to_arrays(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=10_000, shuffle=False, num_workers=0)
        Xs, ys = [], []
        for xb, yb in loader:
            Xs.append(xb.numpy())
            ys.append(yb.numpy())
        X = np.concatenate(Xs).reshape(len(ds), -1).astype(np.float32)
        y = np.concatenate(ys)
        return X, y

    X_tr, y_tr = _to_arrays(train_ds)
    X_te, y_te = _to_arrays(test_ds)
    return np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])


def _load_mnist():
    """MNIST: 70k 28×28 grayscale images, 10 digit classes.

    Returns flat (N, 784) float32 array and integer class labels.
    Reference: LeCun et al. (1998) http://yann.lecun.com/exdb/mnist/
    """
    try:
        import torch
        import torchvision
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision required. pip install torchvision") from exc

    from pathlib import Path
    root = str(Path.home() / ".surge" / "data" / "torchvision")
    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root, train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    def _to_arrays(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=10_000, shuffle=False, num_workers=0)
        Xs, ys = [], []
        for xb, yb in loader:
            Xs.append(xb.numpy())
            ys.append(yb.numpy())
        X = np.concatenate(Xs).reshape(len(ds), -1).astype(np.float32)
        y = np.concatenate(ys)
        return X, y

    X_tr, y_tr = _to_arrays(train_ds)
    X_te, y_te = _to_arrays(test_ds)
    return np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])


# ---------------------------------------------------------------------------
# CTR-23 datasets (Grinsztajn et al. 2022, arXiv:2207.08815)
# "Why tree-based models still outperform deep learning on tabular data"
# All loaded from OpenML — IDs are stable public identifiers.
# ---------------------------------------------------------------------------

def _load_ctr23_abalone():
    """Abalone dataset — predict age (rings+1.5) from physical measurements.

    n=4177, d=8 (7 numeric + 1 nominal sex → encoded).
    OpenML ID: 183 | Grinsztajn et al. 2022, Table 1.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder

    data = fetch_openml(data_id=183, as_frame=True, parser="auto")
    df = data.frame.copy()
    # Sex column is nominal; encode to integer.
    cat_cols = [c for c in df.columns[:-1] if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    if cat_cols:
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    target = df.columns[-1]
    y = df[target].values.astype(float)
    X = df.drop(columns=[target]).values.astype(float)
    return X, y


def _load_ctr23_bike_sharing():
    """Bike Sharing Demand — predict hourly bike rentals.

    n=17,389, d=12.  OpenML ID: 42712 | Grinsztajn et al. 2022.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=42712, as_frame=True, parser="auto")
    df = data.frame
    target = df.columns[-1]
    y = df[target].values.astype(float)
    X = df.drop(columns=[target]).values.astype(float)
    return X, y


def _load_ctr23_diamonds():
    """Diamonds dataset — predict price from carat/cut/color/clarity.

    n=53,940, d=9.  OpenML ID: 42225 | Grinsztajn et al. 2022.
    Sub-sampled to 15k for speed.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OrdinalEncoder

    data = fetch_openml(data_id=42225, as_frame=True, parser="auto")
    df = data.frame.copy()
    cat_cols = [c for c in df.columns[:-1] if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    if cat_cols:
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    target = df.columns[-1]
    y = df[target].values.astype(float)
    X = df.drop(columns=[target]).values.astype(float)
    # Sub-sample to keep leaderboard runs tractable.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(15_000, len(y)), replace=False)
    return X[idx], y[idx]


def _load_ctr23_house_sales():
    """House Sales in King County — predict house price.

    n=21,613, d=19.  OpenML ID: 42731 | Grinsztajn et al. 2022.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=42731, as_frame=True, parser="auto")
    df = data.frame.copy()
    cat_cols = [c for c in df.columns[:-1] if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    target = df.columns[-1]
    y = df[target].values.astype(float)
    X = df.drop(columns=[target]).values.astype(float)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(15_000, len(y)), replace=False)
    return X[idx], y[idx]


def _load_ctr23_brazilian_houses():
    """Brazilian Houses — predict rent/sale price from property features.

    n=10,692, d=11.  OpenML ID: 42688 | Grinsztajn et al. 2022.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=42688, as_frame=True, parser="auto")
    df = data.frame.copy()
    cat_cols = [c for c in df.columns[:-1] if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    target = df.columns[-1]
    y = df[target].values.astype(float)
    X = df.drop(columns=[target]).values.astype(float)
    return X, y


# ---------------------------------------------------------------------------
# DOE Fusion plasma dataset loaders
# ---------------------------------------------------------------------------

def _load_cmod_density_limit():
    """Alcator C-Mod density limit disruption classification.

    Binary classification: predict whether the plasma has entered the
    Greenwald density limit phase (disruption precursor).

    Source: MIT-PSFC open_density_limit_database
    URL:    https://github.com/MIT-PSFC/open_density_limit_database
    n = 264,385 time-slices from C-Mod discharges (highly imbalanced: 1.4% positive).
    Features: density, elongation, minor_radius, plasma_current, toroidal_B_field,
              triangularity  (6 normalised physics signals per time-slice).

    We balance the dataset to 50/50 by downsampling the majority class to
    keep n ≤ 40k for fast leaderboard runs.
    """
    import io, urllib.request
    import pandas as pd

    url = (
        "https://raw.githubusercontent.com/MIT-PSFC/"
        "open_density_limit_database/main/data/DL_DataFrame.csv"
    )
    raw = urllib.request.urlopen(url, timeout=60).read()
    df = pd.read_csv(io.BytesIO(raw))

    feature_cols = ["density", "elongation", "minor_radius",
                    "plasma_current", "toroidal_B_field", "triangularity"]
    X = df[feature_cols].values.astype(float)
    y = df["density_limit_phase"].values.astype(int)

    # Balance: sample min(all positive, 20k) from each class.
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_per_class = min(len(pos_idx), 20_000)
    chosen_pos = rng.choice(pos_idx, size=n_per_class, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_per_class, replace=False)
    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _load_qlknn_transport():
    """QuaLiKiz/QLKNN turbulent electron heat flux surrogate benchmark.

    Predict the total electron heat flux (efeITG, gyroBohm units) from 10
    normalised gyrokinetic plasma parameters.  The reference "ground truth"
    is the QLKNN_7_11 model (van de Plassche et al. Nuclear Fusion 2020),
    pre-trained on 15M QuaLiKiz runs and distributed by Google DeepMind as
    the `fusion_surrogates` package.

    This benchmark tests how well a SURGE model can approximate the QLKNN
    transport surrogate — a task of direct relevance to real-time transport
    solvers on burning-plasma tokamaks (ITER, DEMO).

    Input features (10):
        Ati, Ate       — normalised ion/electron temperature gradients
        Ane, Ani       — normalised electron/ion density gradients
        q              — safety factor
        smag           — magnetic shear
        x              — normalised minor radius
        Ti_Te          — ion-to-electron temperature ratio
        LogNuStar      — log collisionality
        normni         — normalised ion density

    Target: efeITG (electron heat flux from ITG mode, gyroBohm normalised).
    n = 20,000 samples covering the physical parameter space.

    Requires: pip install fusion_surrogates
    Reference: van de Plassche et al. Physics of Plasmas 27, 022310 (2020).
               Google DeepMind fusion_surrogates (2024).
    """
    try:
        from fusion_surrogates.qlknn import qlknn_model
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError(
            "pip install fusion_surrogates  (Google DeepMind QLKNN surrogate)"
        ) from e

    model = qlknn_model.QLKNNModel.load_default_model()

    rng = np.random.default_rng(42)
    n = 20_000

    # Realistic parameter ranges from van de Plassche et al. 2020 and
    # the QLKNN training data description (Zenodo 8017522 / 8106431).
    X = np.column_stack([
        rng.uniform(0.5, 10.0, n),   # Ati  — ion temperature gradient
        rng.uniform(0.5, 10.0, n),   # Ate  — electron temperature gradient
        rng.uniform(0.0,  3.0, n),   # Ane  — electron density gradient
        rng.uniform(0.0,  3.0, n),   # Ani  — ion density gradient
        rng.uniform(1.0,  5.0, n),   # q    — safety factor
        rng.uniform(-1.0, 2.0, n),   # smag — magnetic shear
        rng.uniform(0.1,  0.9, n),   # x    — normalised minor radius
        rng.uniform(0.5,  2.0, n),   # Ti_Te
        rng.uniform(-3.0, 2.0, n),   # LogNuStar
        rng.uniform(0.8,  1.0, n),   # normni
    ])

    preds = model.predict(jnp.array(X, dtype=jnp.float32))
    y = np.array(preds["efeITG"]).ravel()  # ITG electron heat flux

    # Keep only samples where QLKNN predicts nonzero transport (mode is active).
    mask = y > 0
    return X[mask].astype(float), y[mask].astype(float)


def _load_constellaration(n_samples: int = 10_000):
    """ConStellaration: stellarator boundary shape → quasi-isodynamic quality.

    Predict the QI quality metric from the Fourier coefficients that define
    the stellarator plasma boundary surface.  Each sample is one VMEC
    equilibrium; the boundary is parameterised by r_cos and z_sin Fourier
    arrays (5×9 each = 90 input features).

    Target: metrics.qi  — quasi-isodynamic quality score (lower = better QI).

    Source: proxima-fusion/constellaration on HuggingFace (Cadena et al. 2025).
    Vacuum subset: 182,156 samples; sub-sampled to `n_samples` for speed.
    DOI: arXiv:2506.19583

    Requires: pip install datasets
    """
    try:
        import datasets as hf_datasets
    except ImportError as e:
        raise ImportError("pip install datasets  (HuggingFace datasets library)") from e

    ds = hf_datasets.load_dataset(
        "proxima-fusion/constellaration",
        split="train",
        streaming=True,
    )

    rows_r = []
    rows_z = []
    rows_qi = []

    for row in ds:
        r = row.get("boundary.r_cos")
        z = row.get("boundary.z_sin")
        qi = row.get("metrics.qi")
        if r is None or z is None or qi is None:
            continue
        # r and z are lists-of-lists (5×9); flatten to 45 values each.
        r_flat = np.asarray(r, dtype=float).ravel()
        z_flat = np.asarray(z, dtype=float).ravel()
        if r_flat.size != 45 or z_flat.size != 45:
            continue
        rows_r.append(r_flat)
        rows_z.append(z_flat)
        rows_qi.append(float(qi))
        if len(rows_qi) >= n_samples:
            break

    X = np.hstack([np.array(rows_r), np.array(rows_z)])   # (n, 90)
    y = np.array(rows_qi, dtype=float)                      # (n,)

    # Remove samples with NaN/Inf (failed VMEC runs).
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


# ── Paper reproduction: all 12 metrics, filtered 23 k dataset ─────────────────

_CONSTELLARATION_METRICS = [
    "aspect_ratio",
    "aspect_ratio_over_edge_rotational_transform",
    "max_elongation",
    "axis_rotational_transform_over_n_field_periods",
    "edge_rotational_transform_over_n_field_periods",
    "axis_magnetic_mirror_ratio",
    "edge_magnetic_mirror_ratio",
    "average_triangularity",
    "vacuum_well",
    "minimum_normalized_magnetic_gradient_scale_length",
    "flux_compression_in_regions_of_bad_curvature",
    "log_10_qi",  # transformed from metrics.qi
]


def _load_constellaration_paper(
    n_field_periods: int = 3,
    outlier_clip_pct: float = 0.05,
    cache_dir: "str | None" = None,
) -> "tuple[np.ndarray, np.ndarray, list[str]]":
    """Load ConStellaration data following Appendix A.4 of arXiv:2506.19583.

    Filters to optimised (DESC or VMEC) configurations with *n_field_periods*
    field periods, removes the bottom/top ``outlier_clip_pct/2`` percentile for
    each metric, and returns:

    * **X**  – (n, 90) boundary Fourier coefficients
    * **Y**  – (n, 12) per-metric outputs (log₁₀ applied to qi)
    * **metric_names** – list of 12 metric names

    Results are cached to ``data/datasets/constellaration/`` to avoid
    re-downloading on repeated runs.
    """
    import math
    from pathlib import Path

    repo_root = Path(__file__).parent.parent.parent
    _cache_dir = Path(cache_dir) if cache_dir else (
        repo_root / "data" / "datasets" / "constellaration"
    )
    _cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_dir / f"paper_nfp{n_field_periods}_clip{outlier_clip_pct}.npz"

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X"],
            data["Y"],
            list(data["metric_names"]),
        )

    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError("pip install datasets  (HuggingFace datasets library)") from exc

    METRIC_COLS_SRC = [m for m in _CONSTELLARATION_METRICS if m != "log_10_qi"] + ["qi"]

    print(
        f"[constellaration] downloading full dataset (first run only, will be cached)…",
        flush=True,
    )
    ds = hf_datasets.load_dataset(
        "proxima-fusion/constellaration",
        split="train",
        streaming=True,
    )

    rows_X: list[np.ndarray] = []
    rows_Y: list[list[float]] = []

    for row in ds:
        # Filter: only 3-field-period optimised configs (DESC or VMEC)
        if row.get("boundary.n_field_periods") != n_field_periods:
            continue
        desc_id = row.get("desc_omnigenous_field_optimization_settings.id")
        vmec_id = row.get("vmec_omnigenous_field_optimization_settings.id")
        if desc_id is None and vmec_id is None:
            continue

        r = row.get("boundary.r_cos")
        z = row.get("boundary.z_sin")
        if r is None or z is None:
            continue
        r_flat = np.asarray(r, dtype=float).ravel()
        z_flat = np.asarray(z, dtype=float).ravel()
        if r_flat.size != 45 or z_flat.size != 45:
            continue

        # Collect all 12 metrics
        metric_vals: list[float] = []
        ok = True
        for col in METRIC_COLS_SRC[:-1]:  # all except qi
            v = row.get(f"metrics.{col}")
            if v is None:
                ok = False
                break
            metric_vals.append(float(v))
        if not ok:
            continue

        qi = row.get("metrics.qi")
        if qi is None or float(qi) <= 0:
            continue
        metric_vals.append(math.log10(float(qi)))

        rows_X.append(np.concatenate([r_flat, z_flat]))
        rows_Y.append(metric_vals)

    if not rows_X:
        raise RuntimeError("No data passed the n_field_periods/optimised filter.")

    X = np.array(rows_X, dtype=float)
    Y = np.array(rows_Y, dtype=float)

    # Remove NaN/Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[valid], Y[valid]

    # Remove outlier tails (0.05 % per metric = 0.025 % each tail)
    mask = np.ones(len(X), dtype=bool)
    half = outlier_clip_pct / 2.0
    for j in range(Y.shape[1]):
        lo = np.percentile(Y[:, j], half)
        hi = np.percentile(Y[:, j], 100.0 - half)
        mask &= (Y[:, j] >= lo) & (Y[:, j] <= hi)
    X, Y = X[mask], Y[mask]

    print(
        f"[constellaration] paper dataset: {len(X):,} samples "
        f"(after nfp={n_field_periods} + optimised filter + outlier clip)",
        flush=True,
    )

    np.savez_compressed(
        cache_file,
        X=X,
        Y=Y,
        metric_names=np.array(_CONSTELLARATION_METRICS),
    )
    return X, Y, _CONSTELLARATION_METRICS


def _run_constellaration_paper_benchmark(
    model_key: str,
    model_kwargs: "dict | None" = None,
    seed: int = 42,
) -> "BenchmarkResult | None":
    """Train one model per metric (matching paper protocol) and return aggregate metrics.

    Reports:
    * ``test_r2_mean``   — mean R² across 12 metrics (paper baseline ≥ 0.97)
    * ``test_rmse_mean`` — mean RMSE
    * ``test_nrmse_mean``— mean normalised RMSE  (RMSE / std(y_test))
    * ``test_snr_mean``  — mean signal-to-noise  (var(y_test) / MSE)
    * Per-metric R² stored in ``extra["per_metric_r2"]``
    """
    import time

    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    from surge.model import create_model

    from .base import BenchmarkResult

    try:
        X, Y, metric_names = _load_constellaration_paper()
    except Exception as exc:
        print(f"  [error] could not load constellaration paper data: {exc}", file=sys.stderr)
        return None

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    per_r2: list[float] = []
    per_rmse: list[float] = []
    per_nrmse: list[float] = []
    per_snr: list[float] = []
    per_metric_r2_dict: dict[str, float] = {}

    kwargs = model_kwargs or {}
    t0 = time.perf_counter()

    for j, metric in enumerate(metric_names):
        try:
            adapter = create_model(model_key, **kwargs)
            adapter.fit(X_train, Y_train[:, j])
            y_pred = np.asarray(adapter.predict(X_test)).ravel()
            y_true = Y_test[:, j]

            r2 = float(r2_score(y_true, y_pred))
            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            std_y = float(np.std(y_true))
            nrmse = rmse / (std_y + 1e-12)
            snr = float(np.var(y_true) / (mse + 1e-12))

            per_r2.append(r2)
            per_rmse.append(rmse)
            per_nrmse.append(nrmse)
            per_snr.append(snr)
            per_metric_r2_dict[metric] = r2

        except Exception as exc:
            print(f"  [warn] metric '{metric}' failed for {model_key}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t0

    if not per_r2:
        return None

    metrics = {
        "test_r2_mean": float(np.mean(per_r2)),
        "test_rmse_mean": float(np.mean(per_rmse)),
        "test_nrmse_mean": float(np.mean(per_nrmse)),
        "test_snr_mean": float(np.mean(per_snr)),
        "min_r2": float(np.min(per_r2)),
        "max_r2": float(np.max(per_r2)),
        "runtime_s": elapsed,
    }

    passed = metrics["test_r2_mean"] >= 0.97  # paper threshold

    return BenchmarkResult(
        benchmark_key="plasma.constellaration_paper",
        model_key=model_key,
        tier=3,
        task_type="regression",
        metrics=metrics,
        passed=passed,
        message=f"paper protocol: {len(per_r2)}/12 metrics trained",
        extra={
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_metrics": len(per_r2),
            "per_metric_r2": per_metric_r2_dict,
        },
    )


def _check_pass(benchmark_key: str, metrics: dict) -> bool:
    """Best-effort pass check using known thresholds."""
    _THRESHOLDS: dict[str, tuple[str, float]] = {
        "synthetic.regression_1d": ("test_r2", 0.85),
        "synthetic.multioutput_2d": ("test_r2", 0.75),
        "tabular.diabetes": ("test_r2", 0.35),
        "tabular.california_housing": ("test_r2", 0.75),
        "tabular.concrete_strength": ("test_r2", 0.80),
        "tabular.energy_efficiency": ("test_r2", 0.90),
        "synthetic.classification_binary": ("test_accuracy", 0.75),
        "tabular.iris": ("test_accuracy", 0.88),
        "tabular.breast_cancer": ("test_accuracy", 0.93),
        "tabular.wine": ("test_accuracy", 0.90),
        "tabular.digits": ("test_accuracy", 0.95),
        "sequence.lorenz63": ("test_nrmse", 0.30),  # lower is better
        "pde.burgers_1d": ("test_relative_l2", 0.10),  # lower is better
        "classification.flow_regime": ("test_accuracy", 0.85),
        "tabular.airfoil_noise": ("test_r2", 0.80),
        "tabular.yacht_dynamics": ("test_r2", 0.80),
        "classification.plasma_stability": ("test_accuracy", 0.92),
        "tabular.superconductor": ("test_r2", 0.90),
        "multioutput.scm20d": ("test_r2", 0.60),
        "classification.covertype": ("test_accuracy", 0.85),
        # CTR-23 — thresholds from Grinsztajn et al. 2022 tree-model baselines
        "ctr23.abalone": ("test_r2", 0.55),
        "ctr23.bike_sharing": ("test_r2", 0.90),
        "ctr23.diamonds": ("test_r2", 0.95),
        "ctr23.house_sales": ("test_r2", 0.80),
        "ctr23.brazilian_houses": ("test_r2", 0.75),
        # DOE fusion plasma benchmarks
        "plasma.cmod_density_limit": ("test_accuracy", 0.85),
        "plasma.qlknn_transport": ("test_r2", 0.90),
        "plasma.constellaration": ("test_r2", 0.50),
        # Paper protocol: MLP ensemble achieves R²>0.97 on all 12 metrics
        "plasma.constellaration_paper": ("test_r2_mean", 0.97),
    }
    if benchmark_key not in _THRESHOLDS:
        return True
    metric_key, threshold = _THRESHOLDS[benchmark_key]
    val = metrics.get(metric_key)
    if val is None:
        return True
    # For lower-is-better metrics, pass if val <= threshold.
    if metric_key in _LOWER_IS_BETTER or metric_key == "test_nrmse":
        return float(val) <= threshold
    return float(val) >= threshold


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def format_leaderboard_table(
    benchmark_key: str,
    results: list[BenchmarkResult],
    *,
    highlight_char: str = "*",
) -> str:
    """
    Return a formatted comparison table string for one benchmark.

    Rows are models; columns are metrics.  The best value per column is
    marked with ``highlight_char``.
    """
    if not results:
        return f"  (no results for {benchmark_key})"

    info = benchmark_info(benchmark_key)

    # Collect all metric keys in preferred order.
    all_keys: list[str] = []
    seen: set[str] = set()
    for mk in _METRIC_ORDER:
        if any(mk in r.metrics for r in results):
            all_keys.append(mk)
            seen.add(mk)
    for r in results:
        for mk in sorted(r.metrics):
            if mk not in seen:
                all_keys.append(mk)
                seen.add(mk)

    # Build value matrix and find best per column.
    model_names = [r.model_key for r in results]
    passed_flags = [r.passed for r in results]
    matrix: list[list[float | None]] = [
        [r.metrics.get(k) for k in all_keys] for r in results
    ]

    best_idx: list[int | None] = []
    for col_idx, metric_key in enumerate(all_keys):
        col_vals = [matrix[row][col_idx] for row in range(len(results))]
        numeric = [(i, v) for i, v in enumerate(col_vals) if v is not None]
        if not numeric:
            best_idx.append(None)
            continue
        if metric_key in _LOWER_IS_BETTER:
            best_i = min(numeric, key=lambda x: x[1])[0]
        else:
            best_i = max(numeric, key=lambda x: x[1])[0]
        best_idx.append(best_i)

    # Column widths.
    col_w = max(len(k) + 3 for k in all_keys) if all_keys else 12
    col_w = max(col_w, 12)
    model_w = max((len(n) for n in model_names), default=10) + 4

    # Header.
    lines: list[str] = []
    lines.append(
        f"\nBenchmark : {benchmark_key}  "
        f"(task={info['task_type']}, tier={info['tier']}, shape={info['shape']})"
    )
    lines.append("─" * (model_w + col_w * len(all_keys) + 8))

    header = f"{'Model':<{model_w}}  {'Pass':4}  "
    header += "".join(f"{k:>{col_w}}" for k in all_keys)
    lines.append(header)
    lines.append("─" * (model_w + col_w * len(all_keys) + 8))

    # Check if any result has multi-seed std data.
    has_std = any(r.extra.get("metric_std") for r in results)

    for row_idx, (result, model_name, passed) in enumerate(
        zip(results, model_names, passed_flags)
    ):
        status = "PASS" if passed else "FAIL"
        n_seeds = result.extra.get("n_seeds", 1)
        seed_tag = f"(n={n_seeds})" if n_seeds > 1 else ""
        row = f"{model_name:<{model_w}}  {status:4}  "
        std_map = result.extra.get("metric_std", {})
        for col_idx, metric_key in enumerate(all_keys):
            val = matrix[row_idx][col_idx]
            is_best = best_idx[col_idx] == row_idx
            if val is None:
                cell = "—"
            elif metric_key == "runtime_s":
                cell = f"{val:.2f}s"
            elif has_std and metric_key in std_map and std_map[metric_key] > 0:
                cell = f"{val:.4f}±{std_map[metric_key]:.4f}"
            else:
                cell = f"{val:.4f}"
            if is_best:
                cell = f"{cell}{highlight_char}"
            row += f"{cell:>{col_w}}"
        if seed_tag:
            row += f"  {seed_tag}"
        lines.append(row)

    lines.append("─" * (model_w + col_w * len(all_keys) + 8))
    lines.append(
        f"  {highlight_char} = best  "
        f"(↑ higher-is-better: accuracy, f1, auroc, r2 | ↓ lower-is-better: rmse, runtime)"
    )
    return "\n".join(lines)


def print_leaderboard(results_by_benchmark: dict[str, list[BenchmarkResult]]) -> None:
    """Print all per-benchmark tables to stdout."""
    for bk, res_list in results_by_benchmark.items():
        print(format_leaderboard_table(bk, res_list))
        print()


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_leaderboard_to_mlflow(
    results_by_benchmark: dict[str, list[BenchmarkResult]],
    *,
    experiment_name: str = "surge_benchmarks",
    tracking_uri: str | None = None,
    save_tables: bool = True,
    tables_dir: Path | None = None,
) -> bool:
    """
    Log all leaderboard results to MLflow.

    Each (benchmark, model) pair becomes one MLflow run tagged with
    ``benchmark_key``, ``model_key``, ``tier``, ``passed``, and
    ``surge_version``.  All numeric metrics are logged so MLflow's
    "Compare runs" panel becomes an instant leaderboard.

    Parameters
    ----------
    save_tables:
        If True, a formatted text table for each benchmark is saved as a
        plain-text artifact attached to the *first* run for that benchmark,
        so it is accessible from the MLflow UI.
    tables_dir:
        Directory to write temporary table text files before uploading.
        Defaults to ``benchmark_reports/.leaderboard_tables/``.

    Returns
    -------
    bool — True on success, False if MLflow unavailable or logging fails.
    """
    from surge.integrations.mlflow_logger import MLFLOW_AVAILABLE

    if not MLFLOW_AVAILABLE:
        print(
            "[warn] MLflow not installed. pip install 'surge-ml[mlflow]'",
            file=sys.stderr,
        )
        return False

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    if save_tables and tables_dir is None:
        tables_dir = Path("benchmark_reports") / ".leaderboard_tables"

    try:
        for bk, res_list in results_by_benchmark.items():
            if not res_list:
                continue

            # Write the table as a text artifact once per benchmark.
            table_path: Path | None = None
            if save_tables and tables_dir is not None:
                tables_dir.mkdir(parents=True, exist_ok=True)
                table_text = format_leaderboard_table(bk, res_list)
                safe_key = bk.replace(".", "_")
                table_path = tables_dir / f"{safe_key}_leaderboard.txt"
                table_path.write_text(table_text + "\n", encoding="utf-8")

            for result_idx, result in enumerate(res_list):
                run_name = f"{result.benchmark_key}__{result.model_key}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tags({
                        "benchmark_key": result.benchmark_key,
                        "model_key": result.model_key,
                        "tier": result.tier,
                        "task_type": result.task_type,
                        "passed": str(result.passed),
                        "surge_version": result.surge_version or "",
                        "timestamp": result.timestamp or "",
                    })
                    mlflow.log_params({
                        "benchmark_key": result.benchmark_key,
                        "model_key": result.model_key,
                        "tier": result.tier,
                        "task_type": result.task_type,
                        "n_train": result.extra.get("n_train", ""),
                        "n_test": result.extra.get("n_test", ""),
                    })
                    numeric = {
                        k: float(v)
                        for k, v in result.metrics.items()
                        if isinstance(v, (int, float))
                    }
                    if numeric:
                        mlflow.log_metrics(numeric)

                    # Attach the leaderboard table to the first run for this benchmark.
                    if result_idx == 0 and table_path is not None and table_path.exists():
                        mlflow.log_artifact(str(table_path), artifact_path="leaderboard")

        return True
    except Exception as exc:
        print(f"[warn] MLflow leaderboard logging failed: {exc}", file=sys.stderr)
        return False
