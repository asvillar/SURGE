"""Ensemble utilities for neural network models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .base import BaseModelAdapter


@dataclass
class EnsemblePrediction:
    """Container for ensemble statistics."""

    mean: np.ndarray
    std: np.ndarray
    members: List[np.ndarray]


class FNNEnsemble:
    """Simple ensemble of feed-forward neural network adapters for UQ."""

    def __init__(self, models: Sequence[BaseModelAdapter]) -> None:
        if len(models) == 0:
            raise ValueError("At least one model is required for an ensemble")
        self.models = list(models)

    @classmethod
    def from_factory(cls, factory: callable, n_members: int = 5) -> "FNNEnsemble":
        models = [factory(member_idx=i) for i in range(n_members)]
        return cls(models)

    def fit(self, X, y) -> Iterable[Any]:
        return [model.fit(X, y) for model in self.models]

    def predict(self, X) -> EnsemblePrediction:
        preds = [np.asarray(model.predict(X)) for model in self.models]
        stacked = np.stack(preds, axis=0)
        return EnsemblePrediction(mean=stacked.mean(axis=0), std=stacked.std(axis=0), members=preds)
