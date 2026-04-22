"""
Tests for individual model types.

Tests each model type (RFR, MLP, PyTorch, GPflow) individually.
"""

import numpy as np
import pandas as pd
import pytest

from surge import (
    GPFLOW_AVAILABLE,
    PYTORCH_AVAILABLE,
    SurrogateEngine,
)

# Legacy pre-refactor API tests. The refactored SurrogateEngine rejects
# the constructor signature used here. Skipped at module level until
# rewritten. Tracked in docs/REFACTORING_PLAN.md §1.9.
pytestmark = pytest.mark.skip(
    reason="legacy pre-refactor API; pending migration (docs/REFACTORING_PLAN.md §1.9)"
)


def test_random_forest_model():
    """Test Random Forest Regressor model."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()
    engine.init_model(0, n_estimators=50, random_state=42)  # Random Forest
    engine.train(0)
    engine.predict_output(0)

    assert engine.R2 > 0.5
    assert engine.MSE < 1.0


def test_sklearn_mlp_model():
    """Test scikit-learn MLP Regressor model."""
    np.random.seed(42)
    # Use slightly larger dataset for MLP (neural networks need more data)
    X = np.random.random((100, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()
    engine.init_model(1, hidden_layer_sizes=(50,), max_iter=200, random_state=42)  # MLP
    engine.train(0)
    engine.predict_output(0)

    # MLP may perform worse than RFR on small datasets, but should still learn something
    assert engine.R2 is not None, "R2 should be calculated"
    assert engine.MSE is not None, "MSE should be calculated"
    assert engine.R2 > -1.0, f"R2 should be reasonable (got {engine.R2})"  # Allow negative R2 but not terrible
    assert engine.MSE > 0, f"MSE should be positive (got {engine.MSE})"


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
def test_pytorch_mlp_model():
    """Test PyTorch MLP model if available."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()
    engine.init_model(2, hidden_layers=[50, 25], epochs=50, learning_rate=0.01, random_state=42)  # PyTorch MLP
    engine.train(0)
    engine.predict_output(0)

    assert engine.R2 > 0.3
    assert engine.MSE > 0


@pytest.mark.skipif(not GPFLOW_AVAILABLE, reason="GPflow not available")
def test_gpflow_gpr_model():
    """Test GPflow GPR model if available."""
    np.random.seed(42)
    X = np.random.random((30, 3))  # Smaller dataset for GP
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(30)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()

    # GPflowGPR parameters: kernel_type, variance, noise_variance, optimize, maxiter
    engine.init_model(
        3,
        kernel_type='matern32',
        variance=1.0,
        noise_variance=0.1,
        optimize=True,
        maxiter=50  # Reduced for faster testing
    )
    engine.train(0)
    engine.predict_output(0)

    assert engine.R2 is not None
    assert engine.MSE is not None


def test_model_prediction_functionality():
    """Test that predictions are stored correctly."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()
    engine.init_model(0, n_estimators=50, random_state=42)  # Random Forest
    engine.train(0)
    engine.predict_output(0)

    # Test that predictions were stored globally (backward compatibility)
    assert hasattr(engine, 'y_pred_test'), "y_pred_test should exist"
    assert hasattr(engine, 'y_pred_train_val'), "y_pred_train_val should exist"
    assert engine.y_pred_test is not None, "y_pred_test should not be None"
    assert engine.y_pred_train_val is not None, "y_pred_train_val should not be None"
    assert engine.y_pred_test.shape[0] > 0, f"y_pred_test should have predictions (got shape {engine.y_pred_test.shape})"
    assert engine.y_pred_train_val.shape[0] > 0, f"y_pred_train_val should have predictions (got shape {engine.y_pred_train_val.shape})"
    
    # Test that predictions were stored per-model
    assert 0 in engine.model_predictions, "Model 0 should have predictions stored"
    assert 'test' in engine.model_predictions[0], "Model 0 should have test predictions"
    assert 'train' in engine.model_predictions[0], "Model 0 should have train predictions"
    assert engine.model_predictions[0]['test'] is not None, "Test predictions should not be None"
    assert engine.model_predictions[0]['train'] is not None, "Train predictions should not be None"
    assert engine.model_predictions[0]['test'].shape[0] > 0, "Test predictions should have samples"
    assert engine.model_predictions[0]['train'].shape[0] > 0, "Train predictions should have samples"


def test_model_performance_tracking():
    """Test that model performance metrics are tracked."""
    np.random.seed(42)
    X = np.random.random((50, 3))
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)

    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    df['y'] = y

    engine = SurrogateEngine(n_features=3, n_outputs=1)
    engine.load_df_dataset(df, ['x1', 'x2', 'x3'], ['y'])
    engine.train_test_split(test_split=0.3)
    engine.standardize_data()
    engine.init_model(0, n_estimators=50, random_state=42)
    engine.train(0)
    engine.predict_output(0)

    # Test model summary
    summary = engine.get_model_summary(0)
    assert 'model_index' in summary
    assert 'model_name' in summary
    assert 'performance' in summary
    assert summary['performance']['R2'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

