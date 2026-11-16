"""Dataset loaders for SURGE."""

from __future__ import annotations

from ..dataset import SurrogateDataset

from .stacked_npy import StackedNpyDataset

__all__ = [
    "SurrogateDataset",
    "StackedNpyDataset",
]
