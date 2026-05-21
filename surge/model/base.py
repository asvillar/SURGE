"""Base model adapter classes for SURGE."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from ..hpc import (
    DEFAULT_RESOURCE_PROFILE,
    ResourceProfile,
    ResourceSpec,
    apply_policy,
    log_fit_banner,
)

# ---------------------------------------------------------------------------
# ModelInfo — structured per-model documentation
# ---------------------------------------------------------------------------

class ModelInfo:
    """Structured description of a model's architecture, use cases, and literature.

    Attached to each adapter via the ``_INFO`` class variable and exposed
    through :meth:`BaseModelAdapter.info`.
    """

    def __init__(
        self,
        architecture: str,
        use_cases: List[str],
        not_for: List[str],
        strengths: List[str],
        weaknesses: List[str],
        references: List[str],
        notes: str = "",
    ) -> None:
        self.architecture = architecture
        self.use_cases = use_cases
        self.not_for = not_for
        self.strengths = strengths
        self.weaknesses = weaknesses
        self.references = references
        self.notes = notes

    def __repr__(self) -> str:  # pragma: no cover
        return self._render()

    def _render(self) -> str:
        lines = [
            f"Architecture : {self.architecture}",
            "",
            "Use cases    :",
            *[f"  • {u}" for u in self.use_cases],
            "",
            "Not for      :",
            *[f"  • {n}" for n in self.not_for],
            "",
            "Strengths    :",
            *[f"  + {s}" for s in self.strengths],
            "",
            "Weaknesses   :",
            *[f"  - {w}" for w in self.weaknesses],
        ]
        if self.notes:
            lines += ["", f"Notes        : {self.notes}"]
        lines += ["", "References   :"]
        for ref in self.references:
            lines.append(f"  [{self.references.index(ref)+1}] {ref}")
        return "\n".join(lines)

    def print(self) -> None:  # pragma: no cover
        print(self._render())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "use_cases": self.use_cases,
            "not_for": self.not_for,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "references": self.references,
            "notes": self.notes,
        }

_DEFAULT_INFO = ModelInfo(
    architecture="Not documented — see source.",
    use_cases=["General-purpose supervised learning"],
    not_for=[],
    strengths=[],
    weaknesses=[],
    references=[],
)


class BaseModelAdapter(ABC):
    """Unified interface shared by all SURGE model implementations."""

    name: str = "base"
    backend: str = "generic"
    uses_internal_preprocessing: bool = False
    handles_output_scaling: bool = False
    supports_sklearn_interface: bool = False
    resource_profile: ResourceProfile = DEFAULT_RESOURCE_PROFILE

    # Subclasses override with a :class:`ModelInfo` instance to document
    # architecture, use cases, strengths, weaknesses, and citations.
    _INFO: ModelInfo = _DEFAULT_INFO

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

    @property
    def training_history(self) -> list[dict]:
        """Epoch-level training history from the underlying backend.

        Returns a list of dicts with at minimum ``{"epoch": int, "train_loss": float}``.
        An empty list is returned if the model has not been trained or does not
        record per-epoch losses (e.g. sklearn / XGBoost backends).
        """
        return list(getattr(self._model, "training_history", []))

    def info(self, *, print_: bool = True) -> ModelInfo:
        """Print and return structured model information.

        Each adapter documents its architecture, appropriate use cases,
        known limitations, and literature references.  Override ``_INFO``
        at the class level to customise.

        Parameters
        ----------
        print_ :
            If ``True`` (default) pretty-print the info to stdout.

        Examples
        --------
        >>> from surge.model import create_model
        >>> m = create_model("pytorch.lstm")
        >>> m.info()
        Architecture : Stacked LSTM ...
        """
        if print_:
            print(f"\n{'─'*60}")
            print(f"  {self.name}  [{self.backend}]")
            print(f"{'─'*60}")
            print(self._INFO._render())
            print(f"{'─'*60}\n")
        return self._INFO

    def plot_training_history(self, **kwargs: Any) -> Any:
        """Plot training (and validation) loss curves for this model.

        Delegates to :func:`surge.model.plot_training.plot_training_history`.
        All keyword arguments are forwarded unchanged.

        Examples
        --------
        >>> adapter.fit(X_train, y_train)
        >>> adapter.plot_training_history(save_path="loss.png", show=False)
        """
        from .plot_training import plot_training_history as _plot
        return _plot(self._model, **kwargs)


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
