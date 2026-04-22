"""
Core functionality tests for SURGE.

Tests basic imports, initialization, and fundamental operations.
"""

import numpy as np
import pandas as pd
import pytest

from surge import (
    GPFLOW_AVAILABLE,
    PYTORCH_AVAILABLE,
    SurrogateEngine,
    SurrogateDataset,
    MODEL_REGISTRY,
)


def test_import():
    """Test that we can import the main classes."""
    assert SurrogateEngine is not None
    assert SurrogateDataset is not None
    assert MODEL_REGISTRY is not None


def test_availability_flags():
    """Test that availability flags are accessible."""
    assert isinstance(PYTORCH_AVAILABLE, bool)
    assert isinstance(GPFLOW_AVAILABLE, bool)


@pytest.mark.skip(
    reason="legacy pre-refactor API (n_features/n_outputs constructor kwargs, "
    "engine.F attribute); pending migration — docs/REFACTORING_PLAN.md §1.9"
)
def test_surrogate_engine_init():
    """Test SurrogateEngine initialization."""
    engine = SurrogateEngine(n_features=4, n_outputs=10)
    assert engine is not None
    assert engine.F == 4
    assert engine.T == 10
    assert len(engine.models) == 0


def test_surrogate_dataset_init():
    """Test SurrogateDataset initialization."""
    dataset = SurrogateDataset()
    assert dataset is not None
    assert dataset.df is None
    assert dataset.input_columns == []
    assert dataset.output_columns == []


def test_model_registry_access():
    """Test that model registry is accessible."""
    models = MODEL_REGISTRY.list_models()
    assert isinstance(models, dict)
    assert len(models) > 0


@pytest.mark.skip(
    reason="legacy pre-refactor API (SurrogateEngine positional constructor, "
    "engine.fit/predict); pending migration — docs/REFACTORING_PLAN.md §1.9"
)
def test_basic_operations():
    """Test basic operations work without errors."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['input1', 'input2', 'input3'])
    df['output1'] = y

    # Test with SurrogateEngine
    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['input1', 'input2', 'input3'], ['output1'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()

    assert engine.X_train_val is not None
    assert engine.X_test is not None
    assert engine.x_scaler is not None
    assert engine.y_scaler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

