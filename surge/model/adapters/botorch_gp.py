"""Adapters registering ``botorch.gp`` and ``botorch.sparse_gp``."""

from __future__ import annotations

from typing import Any

from ..base import BaseModelAdapter
from ...hpc import ResourceProfile

_GP_PROFILE = ResourceProfile(
    name="botorch.gp",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Exact GP via GPyTorch; O(n^3) training — best for n < 5000.",
)

_SGP_PROFILE = ResourceProfile(
    name="botorch.sparse_gp",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="none",
    notes="Sparse variational GP (SVGP) with inducing points; scales to large n.",
)


class _GPAdapterBase(BaseModelAdapter):
    resource_profile = _GP_PROFILE
    task_type = "regression"
    _backend_cls_name: str

    def _build_model(self, **kwargs: Any) -> Any:
        import importlib
        mod = importlib.import_module("surge.model.backends.botorch_gp")
        cls = getattr(mod, self._backend_cls_name)
        params = dict(self.default_params)
        params.update(kwargs)
        return cls(**params)

    def fit(self, X: Any, y: Any, **_: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any) -> tuple:
        return self._model.predict_with_uncertainty(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        self._model.load(str(path))


class BoTorchGPAdapter(_GPAdapterBase):
    """Exact GP surrogate (BoTorch/GPyTorch)."""

    name = "botorch.gp"
    backend = "botorch"
    uses_internal_preprocessing = True
    _backend_cls_name = "ExactGPModel"
    default_params: dict[str, Any] = {
        "kernel": "rbf_matern",
        "n_train_iter": 100,
        "learning_rate": 0.1,
        "noise_init": 0.1,
        "random_state": 42,
    }


class BoTorchSparseGPAdapter(_GPAdapterBase):
    """Sparse variational GP for large datasets (n > 2000)."""

    name = "botorch.sparse_gp"
    backend = "botorch"
    uses_internal_preprocessing = True
    resource_profile = _SGP_PROFILE
    _backend_cls_name = "SparseGPModel"
    default_params: dict[str, Any] = {
        "n_inducing": 500,
        "n_train_iter": 200,
        "learning_rate": 0.01,
        "batch_size": 256,
        "random_state": 42,
    }
