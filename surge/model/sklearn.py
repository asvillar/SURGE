"""scikit-learn based model adapters."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

from ..hpc import ResourceProfile
from .base import BaseModelAdapter, SklearnRegressorAdapter


_RF_PROFILE = ResourceProfile(
    name="sklearn.random_forest",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="n_jobs",
    notes="RandomForestRegressor uses joblib threads; num_workers -> n_jobs.",
)

_SKMLP_PROFILE = ResourceProfile(
    name="sklearn.mlp",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="sklearn.MLPRegressor is single-threaded; num_workers is ignored.",
)

_SKGPR_PROFILE = ResourceProfile(
    name="sklearn.gpr",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="sklearn GaussianProcessRegressor is single-threaded.",
)


class RandomForestModel(SklearnRegressorAdapter):
    """Random Forest model wrapper compatible with the SURGE API."""

    name = "sklearn.random_forest"
    estimator_cls = RandomForestRegressor
    resource_profile = _RF_PROFILE
    default_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    @property
    def feature_importances_(self) -> Any:  # noqa: D401 - proxy property
        return self._model.feature_importances_  # type: ignore[attr-defined]

    def fit(self, X: Any, y: Any) -> Any:
        # If the user declared a ResourceSpec with a specific num_workers,
        # honor it by overriding the estimator's n_jobs right before fit.
        res = self._last_fit_resources or {}
        n_jobs = res.get("concrete", {}).get("n_jobs")
        if n_jobs is not None and self._model is not None:
            try:
                self._model.set_params(n_jobs=int(n_jobs))
            except Exception:  # pragma: no cover - defensive
                pass
        return super().fit(X, y)


class MLPModel(SklearnRegressorAdapter):
    """Scikit-learn based MLP model."""

    name = "sklearn.mlp"
    estimator_cls = MLPRegressor
    resource_profile = _SKMLP_PROFILE
    default_params = {
        "hidden_layer_sizes": (100, 50),
        "max_iter": 800,
        "random_state": 42,
    }


_GB_REG_PROFILE = ResourceProfile(
    name="sklearn.gradient_boosting_regressor",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="GradientBoostingRegressor is single-threaded.",
)


class GradientBoostingRegressorModel(SklearnRegressorAdapter):
    """Gradient Boosting regressor — strong nonlinear tabular baseline.

    Multi-output targets are handled automatically via
    ``sklearn.multioutput.MultiOutputRegressor``.
    """

    name = "sklearn.gradient_boosting_regressor"
    estimator_cls = GradientBoostingRegressor
    resource_profile = _GB_REG_PROFILE
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }

    def fit(self, X: Any, y: Any) -> None:
        import numpy as _np
        from sklearn.multioutput import MultiOutputRegressor

        params = dict(self.default_params)
        y_arr = _np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[1] > 1:
            self._model = MultiOutputRegressor(self.estimator_cls(**params))
        else:
            self._model = self.estimator_cls(**params)
        self._model.fit(X, y_arr)


class GPRModel(SklearnRegressorAdapter):
    """Gaussian Process regressor using scikit-learn backend."""

    name = "sklearn.gpr"
    estimator_cls = GaussianProcessRegressor
    resource_profile = _SKGPR_PROFILE
    default_params = {}

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_y"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_y(X, n_samples=num_samples, random_state=self.params.get("random_state"))


# ---------------------------------------------------------------------------
# Classification adapters
# ---------------------------------------------------------------------------

_RF_CLF_PROFILE = ResourceProfile(
    name="sklearn.random_forest_classifier",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="n_jobs",
    notes="RandomForestClassifier uses joblib threads; num_workers -> n_jobs.",
)

_GB_CLF_PROFILE = ResourceProfile(
    name="sklearn.gradient_boosting_classifier",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="GradientBoostingClassifier is single-threaded.",
)

_LR_PROFILE = ResourceProfile(
    name="sklearn.logistic_regression",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
    notes="LogisticRegression is single-threaded.",
)


class SklearnClassifierAdapter(BaseModelAdapter):
    """Base adapter for scikit-learn classifiers with predict_proba support."""

    backend = "sklearn"
    supports_sklearn_interface = True
    task_type = "classification"
    estimator_cls: Any = None
    default_params: dict[str, Any] = {}

    def _build_model(self, **kwargs: Any) -> Any:
        if self.estimator_cls is None:
            raise ValueError("estimator_cls must be set on subclasses")
        params = dict(self.default_params)
        params.update(kwargs)
        return self.estimator_cls(**params)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probability estimates (required for AUROC / log-loss)."""
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(f"{self.name} does not support predict_proba")
        return self._model.predict_proba(X)

    def save(self, filepath: str) -> None:
        import joblib
        joblib.dump(self._model, filepath)

    def load(self, filepath: str) -> None:
        import joblib
        self._model = joblib.load(filepath)


class RandomForestClassifierAdapter(SklearnClassifierAdapter):
    """Random Forest classifier."""

    name = "sklearn.random_forest_classifier"
    estimator_cls = RandomForestClassifier
    resource_profile = _RF_CLF_PROFILE
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "n_jobs": -1,
    }


class GradientBoostingClassifierAdapter(SklearnClassifierAdapter):
    """Gradient Boosting classifier."""

    name = "sklearn.gradient_boosting_classifier"
    estimator_cls = GradientBoostingClassifier
    resource_profile = _GB_CLF_PROFILE
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
    }


class LogisticRegressionAdapter(SklearnClassifierAdapter):
    """Logistic Regression classifier — linear baseline."""

    name = "sklearn.logistic_regression"
    estimator_cls = LogisticRegression
    resource_profile = _LR_PROFILE
    default_params = {
        "max_iter": 1000,
        "random_state": 42,
    }
