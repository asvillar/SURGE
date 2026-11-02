"""NVIDIA Kumo Tabular in-context wrapper (optional dep: ``structured-data-models``).

Kumo Tabular is a pretrained tabular foundation model: ``fit`` only caches the
context rows (no gradient training) and ``predict`` runs a forward pass over
context + query.  Cost scales with the number of outputs (one model call per
target column) and with ``context_rows * n_columns``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_MAX_COLUMNS = 100
_MAX_CLASSES = 10
_QUANTILES = [f"q{i:03d}" for i in range(1, 1000)]


class KumoTabularModel:
    """Wrapper around ``sdm.models.KumoTabular`` with a sklearn-like API.

    Regression: ``predict`` returns the median (``q500``);
    ``predict_with_uncertainty`` returns ``(mean, std)`` with ``std`` estimated
    as ``(q841 - q159) / 2`` (the +-1 sigma span of a Gaussian), per output.
    Multi-output targets fit one sdm model per column.  Classification supports
    one target column and up to 10 classes.
    """

    def __init__(
        self,
        task: str = "regression",
        size: str = "small",
        num_estimators: int = 4,
        max_context_rows: int = 60_000,
        random_state: int = 42,
        device: str | None = None,
        target_transform: str | None = None,
        query_batch_size: int = 4096,
    ) -> None:
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        if target_transform not in (None, "log10"):
            raise ValueError("target_transform must be None or 'log10'")
        if target_transform and task != "regression":
            raise ValueError("target_transform is only valid for regression")
        self.task = task
        self.size = size
        self.num_estimators = int(num_estimators)
        self.max_context_rows = int(max_context_rows)
        self.random_state = random_state
        self.device = device
        self.target_transform = target_transform
        self.query_batch_size = int(query_batch_size)
        self.models_: list[Any] = []
        self.n_outputs_ = 0
        self.classes_: np.ndarray | None = None
        self.X_context_: np.ndarray | None = None
        self.y_context_: np.ndarray | None = None

    # -- helpers -----------------------------------------------------------
    def _kwargs(self) -> dict[str, Any]:
        return dict(
            task=self.task, size=self.size, num_estimators=self.num_estimators,
            max_context_rows=self.max_context_rows, random_state=self.random_state,
            device=self.device, target_transform=self.target_transform,
            query_batch_size=self.query_batch_size,
        )

    def _resolve_device(self) -> str:
        from ...utils import resolve_device

        dev = str(resolve_device(self.device))
        # Kumo does not support MPS; fall back to CPU there (cuda stays cuda).
        return "cpu" if dev.startswith("mps") else dev

    def _check_X(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] > _MAX_COLUMNS:
            raise ValueError(
                f"Kumo Tabular supports at most {_MAX_COLUMNS} columns, got {X.shape[1]}"
            )
        return X

    def _build_and_fit(self) -> None:
        import torch
        from sdm.models import KumoTabular

        device = self._resolve_device()
        self.models_ = []
        xc = torch.from_numpy(self.X_context_)
        for j in range(self.y_context_.shape[1]):
            m = KumoTabular(task=self.task, size=self.size, device=device)
            col = self.y_context_[:, j : j + 1]
            yc = torch.from_numpy(col.astype(np.int64 if self.task == "classification" else np.float32))
            m.fit(xc, yc, num_estimators=self.num_estimators)
            self.models_.append(m)

    # -- API ---------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "KumoTabularModel":
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or len(y) != len(X):
            raise ValueError("y must be (n,) or (n, k) with the same n as X")
        if self.task == "classification":
            if y.shape[1] != 1:
                raise ValueError("classification supports a single target column")
            self.classes_, enc = np.unique(y[:, 0], return_inverse=True)
            if len(self.classes_) > _MAX_CLASSES:
                raise ValueError(
                    f"Kumo Tabular supports at most {_MAX_CLASSES} classes, "
                    f"got {len(self.classes_)}"
                )
            y = enc[:, None].astype(np.int64)
        else:
            y = y.astype(np.float64)
            if self.target_transform == "log10":
                if np.any(y <= 0):
                    raise ValueError("target_transform='log10' requires y > 0")
                y = np.log10(y)
            y = y.astype(np.float32)
        if len(X) > self.max_context_rows:
            idx = np.random.default_rng(self.random_state).choice(
                len(X), self.max_context_rows, replace=False
            )
            idx.sort()
            X, y = X[idx], y[idx]
        self.X_context_, self.y_context_ = X, y
        self.n_outputs_ = y.shape[1]
        self._build_and_fit()
        return self

    def _frames(self, X: Any, model: Any):
        import torch

        X = self._check_X(X)
        step = self.query_batch_size
        # sdm's in-context ensemble is stochastic; reseed so predictions are
        # reproducible across calls and across save/load.
        torch.manual_seed(self.random_state)
        for i in range(0, len(X), step):
            out = model.predict(torch.from_numpy(X[i : i + step]))
            yield out.to_pandas()

    def _require_fit(self) -> None:
        if not self.models_:
            raise ValueError("Model must be fitted before predicting")

    def _quantiles(self, X: Any, qs: list[str]) -> np.ndarray:
        """Return (n, k, len(qs)) array of requested quantiles, original units."""
        self._require_fit()
        per_out = []
        for m in self.models_:
            parts = [df[qs].to_numpy(dtype=np.float64) for df in self._frames(X, m)]
            per_out.append(np.concatenate(parts, axis=0))
        out = np.stack(per_out, axis=1)
        if self.target_transform == "log10":
            out = 10.0 ** out
        return out

    def predict(self, X: Any) -> np.ndarray:
        if self.task == "classification":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        out = self._quantiles(X, ["q500"])[:, :, 0]
        return out[:, 0] if self.n_outputs_ == 1 else out

    def predict_with_uncertainty(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.task != "regression":
            raise NotImplementedError("uncertainty is regression-only")
        q = self._quantiles(X, ["q159", "q500", "q841"])
        mean = q[:, :, 1]
        std = np.maximum((q[:, :, 2] - q[:, :, 0]) / 2.0, 0.0)
        if self.n_outputs_ == 1:
            return mean[:, 0], std[:, 0]
        return mean, std

    def predict_quantiles(self, X: Any, quantiles: list[float]) -> np.ndarray:
        """Quantiles (levels in (0, 1)) as ``(n, k, len(quantiles))``."""
        cols = [f"q{int(round(q * 1000)):03d}" for q in quantiles]
        bad = [c for c in cols if c not in _QUANTILES]
        if bad:
            raise ValueError(f"unsupported quantile columns {bad}")
        return self._quantiles(X, cols)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("predict_proba is classification-only")
        self._require_fit()
        k = len(self.classes_)
        parts = []
        for df in self._frames(X, self.models_[0]):
            p = np.zeros((len(df), k))
            for c in df.columns:  # columns are encoded class indices as strings
                p[:, int(c)] = df[c].to_numpy()
            parts.append(p)
        proba = np.concatenate(parts, axis=0)
        s = proba.sum(axis=1, keepdims=True)
        return proba / np.where(s > 0, s, 1.0)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Any) -> None:
        """Persist context arrays + kwargs (the model has no trained weights)."""
        import json

        self._require_fit()
        np.savez_compressed(
            str(path),
            X_context=self.X_context_,
            y_context=self.y_context_,
            classes=self.classes_ if self.classes_ is not None else np.array([]),
            kwargs=np.array(json.dumps(self._kwargs())),
        )

    def load(self, path: Any) -> "KumoTabularModel":
        """Restore context + kwargs and rebuild the in-context model(s)."""
        import json

        p = str(path)
        if not p.endswith(".npz") and not __import__("os").path.exists(p):
            p += ".npz"
        with np.load(p, allow_pickle=False) as z:
            kw = json.loads(str(z["kwargs"]))
            for k, v in kw.items():
                setattr(self, k, v)
            self.X_context_ = z["X_context"]
            self.y_context_ = z["y_context"]
            self.classes_ = z["classes"] if self.task == "classification" else None
        self.n_outputs_ = self.y_context_.shape[1]
        self._build_and_fit()
        return self
