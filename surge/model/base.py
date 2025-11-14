"""Base model adapter classes for SURGE."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type


class BaseModelAdapter(ABC):
    """Unified interface shared by all SURGE model implementations."""

    name: str = "base"
    backend: str = "generic"
    uses_internal_preprocessing: bool = False
    handles_output_scaling: bool = False
    supports_sklearn_interface: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = dict(kwargs)
        self._model: Any = None
        self._initialize()

    def _initialize(self) -> None:
        self._model = self._build_model(**self.params)

    @abstractmethod
    def _build_model(self, **kwargs: Any) -> Any:
        """Construct the underlying estimator."""

    def get_estimator(self) -> Any:
        return self._model

    def update_estimator(self, estimator: Any) -> "BaseModelAdapter":
        self._model = estimator
        return self

    def fit(self, X: Any, y: Any) -> Any:
        if self._model is None:
            raise ValueError("Model has not been initialized")
        return self._model.fit(X, y)

    def predict(self, X: Any) -> Any:
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        return self._model.predict(X)

    def predict_with_uncertainty(self, X: Any) -> Any:
        if self._model is None or not hasattr(self._model, "predict"):
            raise NotImplementedError("Predict with uncertainty not supported for this model")
        return self._model.predict(X, return_std=True)

    def sample_posterior(self, X: Any, num_samples: int = 10) -> Any:
        if self._model is None or not hasattr(self._model, "sample_posterior"):
            raise NotImplementedError("Posterior sampling not supported for this model")
        return self._model.sample_posterior(X, num_samples)

    def score(self, X: Any, y: Any) -> float:
        if self._model is None or not hasattr(self._model, "score"):
            raise NotImplementedError("score not implemented for this model")
        return self._model.score(X, y)

    def save(self, filepath: str) -> Any:
        if self._model is None or not hasattr(self._model, "save"):
            raise NotImplementedError("save not implemented for this model")
        return self._model.save(filepath)

    def load(self, filepath: str) -> Any:
        if self._model is None or not hasattr(self._model, "load"):
            raise NotImplementedError("load not implemented for this model")
        return self._model.load(filepath)


class SklearnRegressorAdapter(BaseModelAdapter):
    """Adapter that wraps a scikit-learn regressor."""

    backend = "sklearn"
    supports_sklearn_interface = True
    estimator_cls: Optional[Type[Any]] = None
    default_params: Dict[str, Any] = {}

    def _build_model(self, **kwargs: Any) -> Any:
        if self.estimator_cls is None:
            raise ValueError("estimator_cls must be defined for SklearnRegressorAdapter subclasses")
        params = dict(self.default_params)
        params.update(kwargs)
        return self.estimator_cls(**params)
