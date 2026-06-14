"""Optional TabPFN adapters for tabular benchmark comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter


_REG_PROFILE = ResourceProfile(
    name="tabpfn.regressor",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="TabPFNRegressor; device selection is handled by tabpfn.",
)

_CLF_PROFILE = ResourceProfile(
    name="tabpfn.classifier",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="TabPFNClassifier; predict_proba supported.",
)


class TabPFNRegressorAdapter(BaseModelAdapter):
    """TabPFN v2 regressor for small-to-medium tabular benchmark tasks.

    Requires the optional ``tabpfn`` extra. SURGE only registers this adapter
    when the package imports successfully.
    """

    name = "tabpfn.regressor"
    backend = "tabpfn"
    task_type = "regression"
    resource_profile = _REG_PROFILE

    def _build_model(self, **kwargs: Any) -> Any:
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError("tabpfn required. Install with pip install 'surge-ml[tabpfn]'.") from exc
        return TabPFNRegressor(**kwargs)

    def fit(self, X: Any, y: Any) -> Any:
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and y_arr.shape[1] > 1:
            raise ValueError("tabpfn.regressor supports single-output regression only.")
        return self._model.fit(X, y_arr.ravel())

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)


class TabPFNClassifierAdapter(BaseModelAdapter):
    """TabPFN v2 classifier for small-to-medium tabular benchmark tasks."""

    name = "tabpfn.classifier"
    backend = "tabpfn"
    task_type = "classification"
    resource_profile = _CLF_PROFILE

    def _build_model(self, **kwargs: Any) -> Any:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError("tabpfn required. Install with pip install 'surge-ml[tabpfn]'.") from exc
        return TabPFNClassifier(**kwargs)

    def fit(self, X: Any, y: Any) -> Any:
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)
