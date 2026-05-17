"""XGBoost backend wrappers for SURGE.

Thin wrappers around ``xgboost.XGBRegressor`` and ``xgboost.XGBClassifier``
that follow the SURGE fit/predict/save/load contract and handle:
- Multi-output regression via ``sklearn.multioutput.MultiOutputRegressor``
- ``predict_proba`` for AUROC / log-loss metrics
- Consistent ``random_state`` / ``n_jobs`` surface
"""

from __future__ import annotations

from typing import Any

import numpy as np

XGBOOST_AVAILABLE: bool
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore


def _require_xgb() -> None:
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost required. pip install xgboost")


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class XGBRegressorModel:
    """
    sklearn-compatible wrapper for ``xgboost.XGBRegressor``.

    Multi-output targets (``y.ndim > 1``) are handled automatically via
    ``sklearn.multioutput.MultiOutputRegressor``.

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, subsample, colsample_bytree:
        Standard XGBoost knobs.
    n_jobs:
        Parallelism (``-1`` = all cores).
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        **_kwargs: Any,
    ) -> None:
        _require_xgb()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: Any = None
        self._is_multioutput = False

    def _make_estimator(self) -> Any:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
        )

    def fit(self, X, y) -> "XGBRegressorModel":
        from sklearn.multioutput import MultiOutputRegressor

        y_arr = np.asarray(y)
        self._is_multioutput = y_arr.ndim > 1 and y_arr.shape[1] > 1
        if self._is_multioutput:
            self._model = MultiOutputRegressor(self._make_estimator())
        else:
            self._model = self._make_estimator()
        self._model.fit(X, y_arr)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    def load(self, path: str) -> None:
        import joblib

        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class XGBClassifierModel:
    """
    sklearn-compatible wrapper for ``xgboost.XGBClassifier``.

    Exposes ``predict_proba`` for AUROC / log-loss metrics.

    Parameters match ``XGBRegressorModel`` with the addition of ``use_label_encoder``.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        **_kwargs: Any,
    ) -> None:
        _require_xgb()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._model: Any = None

    def fit(self, X, y) -> "XGBClassifierModel":
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(np.asarray(y))
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
            eval_metric="logloss",
        )
        self._model.fit(X, y_enc)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        idx = self._model.predict(X)
        return self._le.inverse_transform(idx)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    def load(self, path: str) -> None:
        import joblib

        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
