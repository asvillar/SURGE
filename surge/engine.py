"""Refactored SURGE surrogate engine (data prep + training helpers)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .hpc import ResourceSpec
from .registry import BaseModelAdapter, ModelRegistry
from .model.registry import MODEL_REGISTRY

LOGGER = logging.getLogger(__name__)


@dataclass
class EngineRunConfig:
    """Controls dataset splitting, scaling, and metric computation."""

    test_fraction: float = 0.2
    val_fraction: float = 0.1
    standardize_inputs: bool = True
    standardize_outputs: bool = False
    shuffle: bool = True
    random_state: Optional[int] = 42
    metrics: Tuple[str, ...] = ("r2", "rmse", "mae", "mape")
    # Declarative compute request propagated into every model trained by
    # this engine (device, num_workers, strict policy). See
    # surge.hpc.policy.ResourceSpec for semantics.
    resources: ResourceSpec = field(default_factory=ResourceSpec)

    def with_updates(self, **overrides: Any) -> "EngineRunConfig":
        return replace(self, **overrides)


@dataclass
class ModelSpec:
    """User-facing description of a model to train within the engine."""

    key: str
    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    request_uncertainty: bool = False
    store_predictions: bool = True
    tags: Tuple[str, ...] = ()


@dataclass
class RawSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    train_index: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    val_index: np.ndarray
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    test_index: Optional[np.ndarray]


@dataclass
class ProcessedSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: Optional[np.ndarray]


@dataclass
class ScalerBundle:
    input_scaler: Optional[StandardScaler]
    output_scaler: Optional[StandardScaler]


@dataclass
class ModelRunResult:
    """Structured record describing a single training run."""

    spec: ModelSpec
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]
    adapter: BaseModelAdapter
    predictions: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class SurrogateEngine:
    """
    Minimal-yet-robust training engine that powers the new SURGE workflow.

    Responsibilities:
      * Accept tabular datasets (directly or via SurrogateDataset)
      * Split + standardize inputs/outputs according to EngineRunConfig
      * Create adapters through ModelRegistry and train/evaluate them
      * Capture metrics/predictions in a structured format for downstream
        artifact writers/workflow runners.
    """

    def __init__(
        self,
        *,
        registry: Optional[ModelRegistry] = None,
        run_config: Optional[EngineRunConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.registry = registry or MODEL_REGISTRY
        self.config = run_config or EngineRunConfig()
        self.logger = logger or LOGGER

        self._dataset_df: Optional[pd.DataFrame] = None
        self._input_columns: List[str] = []
        self._output_columns: List[str] = []
        self._raw_splits: Optional[RawSplits] = None
        self._proc_splits: Optional[ProcessedSplits] = None
        self._scalers = ScalerBundle(None, None)
        self._results: List[ModelRunResult] = []

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def configure_dataframe(
        self,
        df: pd.DataFrame,
        input_columns: Sequence[str],
        output_columns: Sequence[str],
        *,
        run_config: Optional[EngineRunConfig] = None,
    ) -> None:
        required_cols = list(input_columns) + list(output_columns)
        # Drop rows with NaNs in required columns to prevent downstream errors
        cleaned_df = df.dropna(subset=required_cols)
        self._dataset_df = cleaned_df.copy(deep=False)
        self._input_columns = list(input_columns)
        self._output_columns = list(output_columns)
        if run_config is not None:
            self.config = run_config

        self.logger.debug(
            "Configured dataframe with %s rows, %s inputs, %s outputs",
            len(self._dataset_df),
            len(self._input_columns),
            len(self._output_columns),
        )
        self._raw_splits = None
        self._proc_splits = None
        self._results.clear()

    def configure_from_dataset(self, dataset: Any, **kwargs: Any) -> None:
        """
        Convenience wrapper that accepts objects exposing `df`, `input_columns`,
        and `output_columns`. The dataset may originate from the legacy SURGE
        loader or the refactored `surge/dataset.py`.
        """
        missing_attrs = [
            attr
            for attr in ("df", "input_columns", "output_columns")
            if not hasattr(dataset, attr)
        ]
        if missing_attrs:
            raise AttributeError(
                f"Dataset object is missing attributes: {', '.join(missing_attrs)}"
            )
        self.configure_dataframe(dataset.df, dataset.input_columns, dataset.output_columns, **kwargs)

    # ------------------------------------------------------------------
    # Public orchestration helpers
    # ------------------------------------------------------------------
    def prepare(
        self,
        *,
        pretrained_scalers: Optional[ScalerBundle] = None,
    ) -> None:
        """Split + (optionally) standardize the configured dataset.

        When pretrained_scalers is provided (e.g. for fine-tuning), use them to
        transform instead of fitting new scalers.
        """
        if self._dataset_df is None:
            raise ValueError("Dataset not configured. Call configure_dataframe() first.")

        self._raw_splits = self._build_raw_splits()
        if pretrained_scalers is not None:
            self._proc_splits, self._scalers = self._standardize_with_pretrained(
                self._raw_splits, pretrained_scalers
            )
        else:
            self._proc_splits, self._scalers = self._standardize_raw_splits(self._raw_splits)
        self.logger.debug("Prepared dataset splits and scalers.")

    def run(self, model_specs: Sequence[ModelSpec]) -> List[ModelRunResult]:
        """Train/evaluate the provided models sequentially."""
        if not model_specs:
            return []
        if self._raw_splits is None or self._proc_splits is None:
            self.prepare()

        results: List[ModelRunResult] = []
        for spec in model_specs:
            result = self.train_model(spec, record=True)
            results.append(result)
        return results

    def train_model(
        self,
        spec: ModelSpec,
        *,
        record: bool = False,
        pretrained_adapter: Optional[BaseModelAdapter] = None,
        finetune_lr_scale: float = 0.1,
    ) -> ModelRunResult:
        """Public hook for training a single model (used by HPO routines).

        When pretrained_adapter is provided (fine-tuning), continue training
        instead of training from scratch. finetune_lr_scale multiplies the
        learning rate for torch models.
        """
        if self._raw_splits is None or self._proc_splits is None:
            self.prepare()
        result = self._train_single_model(
            spec,
            pretrained_adapter=pretrained_adapter,
            finetune_lr_scale=finetune_lr_scale,
        )
        if record:
            self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Internal training workflow
    # ------------------------------------------------------------------
    def _build_raw_splits(self) -> RawSplits:
        df = self._dataset_df
        if df is None:
            raise ValueError("Dataset not configured.")

        X = df[self._input_columns].to_numpy(copy=True)
        y = df[self._output_columns].to_numpy(copy=True)
        indices = df.index.to_numpy()

        cfg = self.config
        test_fraction = max(0.0, min(0.9, cfg.test_fraction))
        val_fraction = max(0.0, min(0.9, cfg.val_fraction))

        if test_fraction > 0:
            (
                X_train_val,
                X_test,
                y_train_val,
                y_test,
                idx_train_val,
                idx_test,
            ) = train_test_split(
                X,
                y,
                indices,
                test_size=test_fraction,
                random_state=cfg.random_state,
                shuffle=cfg.shuffle,
            )
        else:
            X_train_val, y_train_val = X, y
            X_test = y_test = None
            idx_train_val = indices
            idx_test = None

        if val_fraction > 0:
            denom = 1.0 - test_fraction
            if denom <= 0:
                raise ValueError("test_fraction must be < 1.0 when val_fraction > 0.")
            relative_val = val_fraction / denom
            relative_val = max(0.0, min(0.9, relative_val))
            (
                X_train,
                X_val,
                y_train,
                y_val,
                idx_train,
                idx_val,
            ) = train_test_split(
                X_train_val,
                y_train_val,
                idx_train_val,
                test_size=relative_val,
                random_state=cfg.random_state,
                shuffle=cfg.shuffle,
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val = y_val = None
            idx_train = idx_train_val
            idx_val = None

        if X_val is None or y_val is None:
            raise ValueError(
                "Validation split is required for workflow orchestration. "
                "Consider setting val_fraction > 0."
            )

        return RawSplits(
            X_train=X_train,
            y_train=y_train,
            train_index=idx_train,
            X_val=X_val,
            y_val=y_val,
            val_index=idx_val,
            X_test=X_test,
            y_test=y_test,
            test_index=idx_test,
        )

    def _standardize_raw_splits(
        self, raw: RawSplits
    ) -> Tuple[ProcessedSplits, ScalerBundle]:
        cfg = self.config
        input_scaler: Optional[StandardScaler] = None
        output_scaler: Optional[StandardScaler] = None

        X_train = raw.X_train
        X_val = raw.X_val
        X_test = raw.X_test
        y_train = raw.y_train
        y_val = raw.y_val
        y_test = raw.y_test

        if cfg.standardize_inputs:
            input_scaler = StandardScaler()
            input_scaler.fit(X_train)
            X_train = input_scaler.transform(X_train)
            X_val = input_scaler.transform(X_val)
            X_test = input_scaler.transform(X_test) if X_test is not None else None

        if cfg.standardize_outputs:
            output_scaler = StandardScaler()
            output_scaler.fit(y_train)
            y_train = output_scaler.transform(y_train)
            y_val = output_scaler.transform(y_val)
            y_test = output_scaler.transform(y_test) if y_test is not None else None

        processed = ProcessedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_index=raw.train_index,
            val_index=raw.val_index,
            test_index=raw.test_index,
        )
        scalers = ScalerBundle(input_scaler=input_scaler, output_scaler=output_scaler)
        return processed, scalers

    def _standardize_with_pretrained(
        self, raw: RawSplits, pretrained: ScalerBundle
    ) -> Tuple[ProcessedSplits, ScalerBundle]:
        """Transform splits using pretrained scalers (for fine-tuning)."""
        X_train = raw.X_train
        X_val = raw.X_val
        X_test = raw.X_test
        y_train = raw.y_train
        y_val = raw.y_val
        y_test = raw.y_test

        if pretrained.input_scaler is not None:
            X_train = pretrained.input_scaler.transform(X_train)
            X_val = pretrained.input_scaler.transform(X_val)
            X_test = (
                pretrained.input_scaler.transform(X_test)
                if X_test is not None
                else None
            )
        if pretrained.output_scaler is not None:
            y_train = pretrained.output_scaler.transform(y_train)
            y_val = pretrained.output_scaler.transform(y_val)
            y_test = (
                pretrained.output_scaler.transform(y_test)
                if y_test is not None
                else None
            )

        processed = ProcessedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_index=raw.train_index,
            val_index=raw.val_index,
            test_index=raw.test_index,
        )
        return processed, pretrained

    @staticmethod
    def _prepare_target_for_fit(target: np.ndarray) -> np.ndarray:
        arr = np.asarray(target)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        return arr

    @staticmethod
    def _ensure_2d(target: Any) -> np.ndarray:
        arr = np.asarray(target)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _train_single_model(
        self,
        spec: ModelSpec,
        *,
        pretrained_adapter: Optional[BaseModelAdapter] = None,
        finetune_lr_scale: float = 0.1,
    ) -> ModelRunResult:
        if spec.key not in self.registry:
            raise KeyError(f"Model '{spec.key}' not found in registry.")

        _backend = getattr(pretrained_adapter, "backend", None) if pretrained_adapter else None
        if pretrained_adapter is not None and _backend in ("torch", "pytorch"):
            adapter = pretrained_adapter
            # Scale LR for fine-tuning (optimizer lives on inner PyTorchMLP)
            inner = getattr(adapter._model, "model", None) if hasattr(adapter, "_model") else None
            if inner is not None and hasattr(inner, "optimizer") and inner.optimizer is not None:
                base_lr = getattr(adapter._model, "learning_rate", 1e-3) * finetune_lr_scale
                for pg in inner.optimizer.param_groups:
                    pg["lr"] = base_lr
        else:
            adapter = self.registry.create(spec.key, **spec.params)
            if pretrained_adapter is not None:
                self.logger.warning(
                    "pretrained_adapter provided but backend %s does not support fine-tuning; training from scratch",
                    getattr(pretrained_adapter, "backend", "unknown"),
                )

        proc = self._proc_splits
        raw = self._raw_splits
        if proc is None or raw is None:
            raise ValueError("Dataset has not been prepared. Call prepare() first.")

        start = time.perf_counter()
        y_train_for_fit = self._prepare_target_for_fit(proc.y_train)

        # Resource-policy banner: resolves user ResourceSpec against this
        # adapter's resource_profile, emits one [surge.fit] ... line, and
        # stores the resolved fields on adapter._last_fit_resources so the
        # run summary can persist them under "resources_used".
        if hasattr(adapter, "prepare_for_fit"):
            try:
                adapter.prepare_for_fit(
                    resources=getattr(self.config, "resources", None),
                    X_shape=getattr(proc.X_train, "shape", None),
                    y_shape=getattr(y_train_for_fit, "shape", None),
                )
            except Exception:  # never let banner/policy block a fit
                self.logger.warning(
                    "prepare_for_fit failed for %s; continuing without banner",
                    getattr(adapter, "name", "adapter"),
                    exc_info=True,
                )

        _adapter_backend = getattr(adapter, "backend", None)
        finetune = pretrained_adapter is not None and _adapter_backend in ("torch", "pytorch")
        if _adapter_backend in ("torch", "pytorch"):
            adapter.fit(
                proc.X_train,
                y_train_for_fit,
                X_val=proc.X_val,
                y_val=self._prepare_target_for_fit(proc.y_val),
                finetune=finetune,
            )
        else:
            adapter.fit(proc.X_train, y_train_for_fit)
        adapter.mark_fitted()
        train_time = time.perf_counter() - start

        predictions: Dict[str, Any] = {}

        # Training predictions (scaled) with timing metadata
        pred_train_start = time.perf_counter()
        y_pred_train = self._ensure_2d(adapter.predict(proc.X_train))
        pred_train_time = time.perf_counter() - pred_train_start

        pred_val_start = time.perf_counter()
        y_pred_val = self._ensure_2d(adapter.predict(proc.X_val))
        pred_val_time = time.perf_counter() - pred_val_start

        if proc.X_test is not None:
            pred_test_start = time.perf_counter()
            y_pred_test = self._ensure_2d(adapter.predict(proc.X_test))
            pred_test_time = time.perf_counter() - pred_test_start
        else:
            y_pred_test = None
            pred_test_time = None

        if self._scalers.output_scaler is not None:
            scaler = self._scalers.output_scaler
            y_pred_train = scaler.inverse_transform(y_pred_train)
            y_pred_val = scaler.inverse_transform(y_pred_val)
            if y_pred_test is not None:
                y_pred_test = scaler.inverse_transform(y_pred_test)
        else:
            y_pred_train = np.asarray(y_pred_train)
            y_pred_val = np.asarray(y_pred_val)
            if y_pred_test is not None:
                y_pred_test = np.asarray(y_pred_test)

        if spec.store_predictions:
            train_indices = (
                raw.train_index
                if raw.train_index is not None
                else np.arange(len(raw.y_train))
            )
            val_indices = (
                raw.val_index
                if raw.val_index is not None
                else np.arange(len(raw.y_val))
            )
            test_indices = (
                raw.test_index
                if raw.test_index is not None and raw.y_test is not None
                else (np.arange(len(raw.y_test)) if raw.y_test is not None else None)
            )

            predictions["train"] = {
                "index": train_indices,
                "y_true": raw.y_train,
                "y_pred": y_pred_train,
            }
            predictions["val"] = {
                "index": val_indices,
                "y_true": raw.y_val,
                "y_pred": y_pred_val,
            }
            if y_pred_test is not None and raw.y_test is not None:
                predictions["test"] = {
                    "index": test_indices,
                    "y_true": raw.y_test,
                    "y_pred": y_pred_test,
                }

        uq_payload: Optional[Dict[str, Any]] = None
        if spec.request_uncertainty:
            try:
                uq_payload = adapter.predict_with_uncertainty(proc.X_val)
            except NotImplementedError:
                self.logger.warning(
                    "Model '%s' does not implement predict_with_uncertainty.",
                    spec.key,
                )

        metrics_train = self._compute_metrics(raw.y_train, y_pred_train)
        metrics_val = self._compute_metrics(raw.y_val, y_pred_val)
        metrics_test = (
            self._compute_metrics(raw.y_test, y_pred_test)
            if raw.y_test is not None and y_pred_test is not None
            else None
        )

        timings = {
            "train_seconds": train_time,
            "train_inference_seconds": pred_train_time,
            "val_inference_seconds": pred_val_time,
            "train_inference_per_sample": float(
                pred_train_time / max(1, len(proc.X_train))
            ),
            "val_inference_per_sample": float(
                pred_val_time / max(1, len(proc.X_val))
            ),
        }
        if pred_test_time is not None and proc.X_test is not None:
            timings["test_inference_seconds"] = pred_test_time
            timings["test_inference_per_sample"] = float(
                pred_test_time / max(1, len(proc.X_test))
            )
        if uq_payload is not None:
            predictions["val_uq"] = uq_payload

        extra_payload: Dict[str, Any] = {}
        training_history = getattr(adapter, "training_history", None)
        if training_history:
            extra_payload["training_history"] = training_history

        # Lightweight post-training profile: pickled size, parameter count,
        # and median predict() latency on a validation sub-batch. Timed in
        # the scaled-feature space the adapter was actually trained on
        # (proc.X_val), so numbers reflect the real deployment path.
        try:
            from .model.profiling import measure_model_profile

            probe_X = proc.X_val if proc.X_val is not None and len(proc.X_val) else proc.X_train
            extra_payload["profile"] = measure_model_profile(adapter, probe_X)
        except Exception:  # never let profiling block a fit
            self.logger.warning(
                "profile measurement failed for %s", spec.name or spec.key, exc_info=True
            )

        return ModelRunResult(
            spec=spec,
            train_metrics=metrics_train,
            val_metrics=metrics_val,
            test_metrics=metrics_test,
            adapter=adapter,
            predictions=predictions,
            timings=timings,
            extra=extra_payload,
        )

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def _compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        if y_true is None or y_pred is None:
            return {}
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        metrics: Dict[str, float] = {}

        if "r2" in self.config.metrics:
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                metrics["r2"] = float(
                    r2_score(y_true, y_pred, multioutput="variance_weighted")
                )
                metrics["r2_uniform_average"] = float(
                    r2_score(y_true, y_pred, multioutput="uniform_average")
                )
            else:
                metrics["r2"] = float(r2_score(y_true, y_pred))
        if "rmse" in self.config.metrics or "mse" in self.config.metrics:
            mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
            if "mse" in self.config.metrics:
                metrics["mse"] = float(mse)
            if "rmse" in self.config.metrics:
                metrics["rmse"] = float(np.sqrt(mse))
        if "mae" in self.config.metrics:
            metrics["mae"] = float(
                mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
            )
        if "mape" in self.config.metrics:
            with np.errstate(divide="ignore", invalid="ignore"):
                metrics["mape"] = float(
                    mean_absolute_percentage_error(
                        y_true, y_pred, multioutput="uniform_average"
                    )
                )
        return metrics

    # ------------------------------------------------------------------
    # Introspection utilities
    # ------------------------------------------------------------------
    @property
    def dataset_summary(self) -> Dict[str, Any]:
        df = self._dataset_df
        return {
            "n_rows": len(df) if df is not None else 0,
            "n_inputs": len(self._input_columns),
            "n_outputs": len(self._output_columns),
            "input_columns": list(self._input_columns),
            "output_columns": list(self._output_columns),
            "config": self.config,
        }

    @property
    def scalers(self) -> ScalerBundle:
        return self._scalers

    @property
    def results(self) -> List[ModelRunResult]:
        return list(self._results)

    def get_raw_splits(self) -> RawSplits:
        if self._raw_splits is None:
            raise ValueError("Splits not prepared. Call prepare() or run() first.")
        return self._raw_splits

    def get_processed_splits(self) -> ProcessedSplits:
        if self._proc_splits is None:
            raise ValueError("Splits not prepared. Call prepare() or run() first.")
        return self._proc_splits

