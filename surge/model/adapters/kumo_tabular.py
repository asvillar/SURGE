"""Optional NVIDIA Kumo Tabular adapters (``kumo.tabular_regressor`` / ``_classifier``)."""

from __future__ import annotations

from typing import Any

from ...hpc import ResourceProfile
from ..base import BaseModelAdapter, ModelInfo

_REFS = [
    "https://huggingface.co/nvidia/Kumo-Tabular",
    "https://huggingface.co/blog/nvidia/kumo-tabular",
]


def _info(task: str) -> ModelInfo:
    return ModelInfo(
        architecture=(
            "Tabular foundation model (28M-215M params) with column, row and "
            "in-context attention, pretrained on synthetic SCM tables. No "
            "fitting: fit() caches context rows and predict() is in-context."
        ),
        use_cases=[
            f"Small/medium tabular {task} benchmarks",
            "Instant baselines with calibrated quantiles",
        ],
        not_for=[
            "More than 100 columns",
            "Field outputs with thousands of targets",
            "Extrapolation beyond the context range",
        ],
        strengths=["No training or HPO", "Calibrated quantiles (q001..q999)"],
        weaknesses=["Inference cost grows with context rows x columns", "One forward pass per target column"],
        references=_REFS,
        notes="License: OpenMDW 1.1. Weights download from the Hugging Face Hub on first use.",
    )


def _profile(name: str) -> ResourceProfile:
    return ResourceProfile(
        name=name, supports_cpu=True, supports_gpu=True, worker_semantics="none",
        notes="Kumo Tabular; MPS falls back to CPU.",
    )


class _KumoBase(BaseModelAdapter):
    backend = "kumo"
    uses_internal_preprocessing = True
    _task = "regression"
    default_params: dict[str, Any] = {
        "size": "small",
        "num_estimators": 4,
        "max_context_rows": 60_000,
        "max_context_cells": 1_000_000,
        "random_state": 42,
        "device": None,
        "target_transform": None,
        "query_batch_size": 1024,
    }

    def _build_model(self, **kwargs: Any) -> Any:
        try:
            import sdm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "structured-data-models required. Install with pip install 'surge-ml[kumo]'."
            ) from exc
        from ..backends.kumo_tabular import KumoTabularModel

        params = dict(self.default_params)
        params.update(kwargs)
        params["task"] = self._task
        return KumoTabularModel(**params)

    def prepare_for_fit(self, *, resources=None, X_shape=None, y_shape=None, extra=None):
        n = int(X_shape[0]) if X_shape is not None else 0
        d = int(X_shape[1]) if X_shape is not None and len(X_shape) > 1 else 1
        rows = int(self.params.get("max_context_rows", self.default_params["max_context_rows"]))
        cells = int(self.params.get("max_context_cells", self.default_params["max_context_cells"]))
        cap = max(1, min(rows, cells // max(1, d)))
        merged = {
            "pretrained_backbone": "nvidia/Kumo-Tabular",
            "in_context_learning": True,
            "context_rows": min(n, cap),
        }
        merged.update(extra or {})
        return super().prepare_for_fit(resources=resources, X_shape=X_shape, y_shape=y_shape, extra=merged)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(path)

    def load(self, path: Any) -> None:
        self._model.load(path)


class KumoTabularRegressorAdapter(_KumoBase):
    """Kumo Tabular regressor; point estimate is the median, std from q159/q841."""

    name = "kumo.tabular_regressor"
    task_type = "regression"
    _task = "regression"
    resource_profile = _profile("kumo.tabular_regressor")
    _INFO = _info("regression")

    def predict_with_uncertainty(self, X: Any) -> Any:
        return self._model.predict_with_uncertainty(X)


class KumoTabularClassifierAdapter(_KumoBase):
    """Kumo Tabular classifier (single target, up to 10 classes)."""

    name = "kumo.tabular_classifier"
    task_type = "classification"
    _task = "classification"
    resource_profile = _profile("kumo.tabular_classifier")
    _INFO = _info("classification")

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)
