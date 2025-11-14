"""scikit-learn based model adapters."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from .base import SklearnRegressorAdapter


class RandomForestModel(SklearnRegressorAdapter):
    """Random Forest model wrapper compatible with the SURGE API."""

    name = "sklearn.random_forest"
    estimator_cls = RandomForestRegressor
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


class MLPModel(SklearnRegressorAdapter):
    """Scikit-learn based MLP model."""

    name = "sklearn.mlp"
    estimator_cls = MLPRegressor
    default_params = {
        "hidden_layer_sizes": (100, 50),
        "max_iter": 800,
        "random_state": 42,
    }


class GPRModel(SklearnRegressorAdapter):
    """Gaussian Process regressor using scikit-learn backend."""

    name = "sklearn.gpr"
    estimator_cls = GaussianProcessRegressor
    default_params = {}

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_y"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_y(X, n_samples=num_samples, random_state=self.params.get("random_state"))
