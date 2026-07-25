"""Generic tf.keras adapter — use any TensorFlow/Keras model in SURGE.

Two entry points:

* ``keras.mlp`` — a ready-made dense regressor (hidden_layers, activation,
  dropout, lr, epochs, batch_size, ...).
* Bring your own architecture: pass ``build_fn`` (a callable
  ``(n_inputs, n_outputs) -> compiled keras.Model``) and the adapter
  handles fit/predict/save/load and SURGE registration semantics.

TensorFlow is optional: install with ``pip install -e ".[gpflow]"`` or
``".[tensorflow]"``. Without it the adapter is reported as skipped by
``surge models --verbose`` (never a silent absence).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseModelAdapter


def _keras():
    """Import keras through TF lazily so module import stays cheap."""
    import tensorflow as tf  # noqa: F401 - ensures the TF backend loads
    try:
        import tf_keras as keras  # TF >= 2.16 keeps Keras 2 here
    except ImportError:
        from tensorflow import keras
    return keras


class KerasMLPAdapter(BaseModelAdapter):
    """Dense tf.keras regressor with a pluggable ``build_fn`` escape hatch."""

    name = "keras.mlp"
    backend = "tensorflow"
    supports_uq = False
    supports_serialization = True

    def __init__(
        self,
        *,
        hidden_layers: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        loss: str = "mse",
        epochs: int = 100,
        batch_size: int = 64,
        validation_fraction: float = 0.1,
        early_stopping_patience: int = 15,
        verbose: int = 0,
        random_state: int | None = None,
        build_fn: Callable[[int, int], Any] | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            hidden_layers=tuple(hidden_layers), activation=activation,
            dropout=dropout, learning_rate=learning_rate, loss=loss,
            epochs=epochs, batch_size=batch_size,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose, random_state=random_state, **params)
        self._build_fn = build_fn
        self._n_outputs = 1
        self._y_was_1d = True

    def _build_model(self, **kwargs: Any) -> Any:
        # Keras layer sizes need the input dimension, which is only known
        # at fit() time — construction is deferred (base contract allows
        # a None model until fit populates it).
        return None

    # ------------------------------------------------------------------
    def _default_build(self, n_inputs: int, n_outputs: int):
        keras = _keras()
        p = self.params
        layers = [keras.layers.Input(shape=(n_inputs,))]
        for width in p["hidden_layers"]:
            layers.append(keras.layers.Dense(width, activation=p["activation"]))
            if p["dropout"] > 0:
                layers.append(keras.layers.Dropout(p["dropout"]))
        layers.append(keras.layers.Dense(n_outputs))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
            loss=p["loss"])
        return model

    def fit(self, X, y) -> KerasMLPAdapter:
        keras = _keras()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._y_was_1d = y.ndim == 1
        if self._y_was_1d:
            y = y[:, None]
        self._n_outputs = y.shape[1]

        seed = self.params.get("random_state")
        if seed is not None:
            keras.utils.set_random_seed(int(seed))

        build = self._build_fn or self._default_build
        self._model = build(X.shape[1], self._n_outputs)

        callbacks = []
        patience = self.params["early_stopping_patience"]
        val_frac = self.params["validation_fraction"]
        if patience and val_frac > 0:
            callbacks.append(keras.callbacks.EarlyStopping(
                patience=patience, restore_best_weights=True))
        self._model.fit(
            X, y,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=val_frac,
            callbacks=callbacks,
            verbose=self.params["verbose"])
        self.mark_fitted()
        return self

    def predict(self, X):
        if self._model is None:
            raise ValueError("Model must be fitted before predicting")
        pred = np.asarray(self._model.predict(
            np.asarray(X, dtype=np.float32),
            verbose=self.params["verbose"]))
        return pred.ravel() if self._y_was_1d else pred

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("Model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".keras":
            path = path.with_suffix(".keras")
        self._model.save(path)

    def load(self, path: Path) -> KerasMLPAdapter:
        keras = _keras()
        path = Path(path)
        if path.suffix != ".keras" and path.with_suffix(".keras").exists():
            path = path.with_suffix(".keras")
        self._model = keras.models.load_model(path)
        out_shape = self._model.output_shape
        self._n_outputs = out_shape[-1] if out_shape[-1] else 1
        self._y_was_1d = self._n_outputs == 1
        self.mark_fitted()
        return self
