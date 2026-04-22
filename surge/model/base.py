"""Base model adapter classes for SURGE."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from ..hpc import (
    DEFAULT_RESOURCE_PROFILE,
    ResourceProfile,
    ResourceSpec,
    apply_policy,
    log_fit_banner,
)


class BaseModelAdapter(ABC):
    """Unified interface shared by all SURGE model implementations."""

    name: str = "base"
    backend: str = "generic"
    uses_internal_preprocessing: bool = False
    handles_output_scaling: bool = False
    supports_sklearn_interface: bool = False
    # Per-adapter capability advertisement. Subclasses override to declare
    # whether they support GPU, and what the ``num_workers`` knob means
    # for them. Defaults to CPU-only, no worker knob (safest).
    resource_profile: ResourceProfile = DEFAULT_RESOURCE_PROFILE

    def __init__(self, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = dict(kwargs)
        self._model: Any = None
        # Populated by :meth:`prepare_for_fit`; persisted in run summaries.
        self._last_fit_resources: Optional[Dict[str, Any]] = None
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

    def prepare_for_fit(
        self,
        *,
        resources: Optional[ResourceSpec] = None,
        X_shape: Optional[Any] = None,
        y_shape: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve the user's ``ResourceSpec`` against this adapter's
        ``resource_profile``, emit the ``[surge.fit] ...`` banner, and
        return the resolved fields so the engine can persist them.

        Subclasses do **not** need to override this. They only need to set
        ``resource_profile`` (class attribute) and, if they accept
        runtime knobs like ``n_jobs`` / ``num_workers``, read them out of
        ``self._last_fit_resources`` inside ``_build_model`` or ``fit``.

        Parameters
        ----------
        resources
            User-supplied ``ResourceSpec``. ``None`` triggers defaults.
        X_shape, y_shape
            ``(n, f)`` / ``(n, o)`` shapes used for banner fields.
        extra
            Optional extra key/value pairs added to the banner (e.g.
            ``{"epochs": 200, "batch_size": 256}`` for torch adapters).
        """
        spec = resources or ResourceSpec()
        effective, concrete = apply_policy(
            spec,
            self.resource_profile,
            model_name=self.name,
        )
        n_train = int(X_shape[0]) if X_shape is not None and len(X_shape) >= 1 else 0
        n_features = int(X_shape[1]) if X_shape is not None and len(X_shape) >= 2 else 0
        n_outputs = int(y_shape[1]) if y_shape is not None and len(y_shape) >= 2 else (
            1 if y_shape is not None and len(y_shape) == 1 else 0
        )
        fields = log_fit_banner(
            model_name=self.name,
            backend=self.backend,
            concrete=concrete,
            n_train=n_train,
            n_features=n_features,
            n_outputs=n_outputs,
            extra=extra,
        )
        self._last_fit_resources = {
            "requested": spec.to_dict(),
            "effective": effective.to_dict(),
            "concrete": dict(concrete),
            "banner": fields,
        }
        return self._last_fit_resources

    def mark_fitted(self) -> None:
        """Called by SurrogateEngine after ``fit`` completes. Subclasses may override."""

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
