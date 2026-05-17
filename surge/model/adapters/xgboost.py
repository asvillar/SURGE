"""Adapters for XGBoost models (registers xgboost.xgbregressor / xgboost.xgbclassifier)."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ...hpc import ResourceProfile

_REG_PROFILE = ResourceProfile(
    name="xgboost.xgbregressor",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="n_jobs",
    notes="XGBRegressor; num_workers → n_jobs. Multi-output via MultiOutputRegressor.",
)

_CLF_PROFILE = ResourceProfile(
    name="xgboost.xgbclassifier",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="n_jobs",
    notes="XGBClassifier; num_workers → n_jobs. predict_proba supported.",
)


class XGBRegressorAdapter(BaseModelAdapter):
    """
    XGBoost gradient-boosted trees — regression.

    Strong tabular baseline; often outperforms Random Forest on structured data.
    Multi-output targets are handled automatically via ``MultiOutputRegressor``.

    Default hyperparameters
    -----------------------
    n_estimators     : 300
    learning_rate    : 0.1
    max_depth        : 6
    subsample        : 0.8
    colsample_bytree : 0.8
    """

    name = "xgboost.xgbregressor"
    backend = "xgboost"
    task_type = "regression"
    resource_profile = _REG_PROFILE

    default_params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        from surge.model.backends.xgboost import XGBRegressorModel

        params = dict(self.default_params)
        params.update(kwargs)
        return XGBRegressorModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))


class XGBClassifierAdapter(BaseModelAdapter):
    """
    XGBoost gradient-boosted trees — classification.

    Supports ``predict_proba()`` for AUROC and log-loss.

    Default hyperparameters
    -----------------------
    n_estimators     : 300
    learning_rate    : 0.1
    max_depth        : 6
    subsample        : 0.8
    colsample_bytree : 0.8
    """

    name = "xgboost.xgbclassifier"
    backend = "xgboost"
    task_type = "classification"
    resource_profile = _CLF_PROFILE

    default_params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        from surge.model.backends.xgboost import XGBClassifierModel

        params = dict(self.default_params)
        params.update(kwargs)
        return XGBClassifierModel(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Return class probability matrix (n_samples × n_classes)."""
        return self._model.predict_proba(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))
