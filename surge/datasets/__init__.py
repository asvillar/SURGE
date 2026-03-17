"""Application-specific dataset loaders for SURGE."""

from __future__ import annotations

from ..dataset import SurrogateDataset

from .mhd_code import MHD-codeDataset
from .stacked import XGCDataset

__all__ = [
    "SurrogateDataset",
    "MHD-codeDataset",
    "XGCDataset",
]
