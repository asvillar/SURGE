"""Smoke tests for the public top-level ``surge`` package surface.

Covers the symbols every SURGE user touches first: the main classes, the
feature-availability flags, and the model registry. Anything deeper
(dataset / engine / workflow behaviour) lives in its own focused test
module.
"""

from surge import (
    GPFLOW_AVAILABLE,
    MODEL_REGISTRY,
    PYTORCH_AVAILABLE,
    SurrogateDataset,
    SurrogateEngine,
)


def test_import() -> None:
    """The main classes are re-exported from the package root."""
    assert SurrogateEngine is not None
    assert SurrogateDataset is not None
    assert MODEL_REGISTRY is not None


def test_availability_flags() -> None:
    """Optional-backend flags are plain booleans, safe to branch on."""
    assert isinstance(PYTORCH_AVAILABLE, bool)
    assert isinstance(GPFLOW_AVAILABLE, bool)


def test_surrogate_dataset_init() -> None:
    """Default-constructed dataset is empty but fully usable."""
    dataset = SurrogateDataset()
    assert dataset is not None
    assert dataset.df is None
    assert dataset.input_columns == []
    assert dataset.output_columns == []


def test_model_registry_access() -> None:
    """Registry lists at least the bundled sklearn/pytorch adapters."""
    models = MODEL_REGISTRY.list_models()
    assert isinstance(models, dict)
    assert len(models) > 0
