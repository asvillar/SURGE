"""Template for a custom model adapter: a small 1-D CNN on torch.

Copy this file, rename the classes and change the key. The pieces are:

1. A backend class with ``fit`` / ``predict`` (and optionally ``save`` /
   ``load``). ``torch`` is imported inside ``fit`` and ``predict`` only, so the
   module imports without torch installed.
2. An adapter subclassing ``surge.model.base.BaseModelAdapter`` that sets
   ``name``, ``backend`` and implements ``_build_model``.
3. One ``register_model(...)`` call, which makes the key usable from a spec
   (``models: - key: custom.cnn``) and from ``MODEL_REGISTRY.create``.

Registration of the built-in adapters happens in ``surge/model/__init__.py``.
A model that lives outside the package registers itself when its module is
imported, so import it before running a spec.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from surge.model.base import BaseModelAdapter
from surge.model.registry import register_model

MODEL_KEY = "custom.cnn"


class TinyCNNRegressor:
    """Backend: 1-D convolutions over the feature axis, then a linear head."""

    def __init__(
        self,
        channels: int = 16,
        kernel_size: int = 3,
        n_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        random_state: int = 0,
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self._net: Any = None
        self.training_history: list[dict] = []

    def _make_net(self, n_features: int, n_outputs: int) -> Any:
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv1d(1, self.channels, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.channels * n_features, n_outputs),
        )

    def fit(self, X: Any, y: Any) -> "TinyCNNRegressor":
        import torch

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self._net = self._make_net(X.shape[1], y.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        Xt = torch.from_numpy(X)[:, None, :]  # (N, 1, n_features)
        yt = torch.from_numpy(y)
        self.training_history = []
        for epoch in range(self.n_epochs):
            perm = torch.randperm(len(Xt))
            total = 0.0
            for i in range(0, len(Xt), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.training_history.append({"epoch": epoch, "train_loss": total / len(Xt)})
        return self

    def predict(self, X: Any) -> np.ndarray:
        import torch

        if self._net is None:
            raise ValueError("Model must be fitted before predicting")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.from_numpy(np.asarray(X, dtype=np.float32))[:, None, :])
        return out.numpy()

    def save(self, path: str) -> None:
        import torch

        torch.save({"state": self._net.state_dict(), "n_features": self._n_features()}, path)

    def load(self, path: str, n_outputs: int = 1) -> None:
        """Loads a pickle-based file: only load files you created or trust."""
        import torch

        blob = torch.load(path, weights_only=False)
        self._net = self._make_net(blob["n_features"], n_outputs)
        self._net.load_state_dict(blob["state"])

    def _n_features(self) -> int:
        return int(self._net[-1].in_features // self.channels)


class CNNAdapter(BaseModelAdapter):
    """Adapter: the object the engine and the registry talk to."""

    name = MODEL_KEY
    backend = "pytorch"  # the engine passes X_val / y_val to fit() for torch backends

    default_params: Dict[str, Any] = {"channels": 16, "kernel_size": 3, "n_epochs": 50}

    def _build_model(self, **kwargs: Any) -> TinyCNNRegressor:
        params = dict(self.default_params)
        params.update(kwargs)
        return TinyCNNRegressor(**params)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        **_: Any,
    ) -> None:
        self._model.fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        return self._model.predict(X)

    def save(self, path: Any) -> None:
        self._model.save(str(path))

    def load(self, path: Any) -> None:
        """Loads a pickle-based file: only load files you created or trust."""
        self._model.load(str(path))


# Registration: key, adapter class, optional aliases.
register_model(CNNAdapter, key=MODEL_KEY, aliases=["custom_cnn"])


if __name__ == "__main__":
    from surge.model.registry import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 8))
    y = X[:, :2].sum(axis=1)
    model = create_model(MODEL_KEY, n_epochs=20)
    model.fit(X, y)
    print("predictions:", model.predict(X[:3]).ravel())
