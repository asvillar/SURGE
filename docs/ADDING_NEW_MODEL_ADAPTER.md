# Adding a New Model Adapter to SURGE

This guide explains how to extend SURGE with a new model type. We'll use a **Convolutional Neural Network (CNN)** as a concrete example, but the same pattern applies to any model architecture.

---

## Overview

To add a new model to SURGE, you need to:

1. **Create an adapter class** that inherits from `BaseModelAdapter`
2. **Implement required methods**: `fit()`, `predict()`
3. **Optionally implement**: `predict_with_uncertainty()`, `save()`, `load()`
4. **Register the adapter** in the model registry

---

## Step-by-Step Guide: Adding a CNN Adapter

### Step 1: Create the Adapter File

Create a new file: `surge/models/adapters/cnn_adapter.py`

```python
"""PyTorch CNN adapter for SURGE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ...registry import BaseModelAdapter, MODEL_REGISTRY

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


# Step 2: Define your model architecture
if TORCH_AVAILABLE:
    class CNNModel(nn.Module):
        """1D CNN for tabular data (treats features as sequence)."""
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            conv_channels: Sequence[int] = (64, 128, 256),
            kernel_sizes: Sequence[int] = (3, 3, 3),
            dropout: float = 0.2,
            fc_layers: Sequence[int] = (128,),
        ) -> None:
            super().__init__()
            
            # Convolutional layers
            self.conv_layers = nn.ModuleList()
            prev_channels = 1  # Treat each feature as a channel
            for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                        nn.ReLU(),
                        nn.BatchNorm1d(out_channels),
                        nn.Dropout(dropout),
                    )
                )
                prev_channels = out_channels
            
            # Calculate flattened size after convolutions
            # Assuming input is reshaped to (batch, 1, input_dim)
            self.flattened_size = conv_channels[-1] * input_dim
            
            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            prev_size = self.flattened_size
            for fc_size in fc_layers:
                self.fc_layers.append(
                    nn.Sequential(
                        nn.Linear(prev_size, fc_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                )
                prev_size = fc_size
            
            # Output layer
            self.output_layer = nn.Linear(prev_size, output_dim)
        
        def forward(self, x):
            # Reshape input: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            
            # Apply convolutional layers
            for conv in self.conv_layers:
                x = conv(x)
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Apply fully connected layers
            for fc in self.fc_layers:
                x = fc(x)
            
            # Output
            return self.output_layer(x)

else:
    class CNNModel:
        pass


# Step 3: Create the Adapter Class
class CNNAdapter(BaseModelAdapter):
    """
    PyTorch CNN adapter for SURGE.
    
    This adapter wraps a 1D CNN model that treats tabular features as a sequence,
    allowing the model to learn local patterns in the feature space.
    """
    
    # Required class attributes
    name = "CNN"
    backend = "torch"
    supports_uq = False  # Set to True if you implement uncertainty quantification
    supports_serialization = True  # Set to True if you implement save/load
    
    def __init__(
        self,
        *,
        conv_channels: Sequence[int] = (64, 128, 256),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        dropout: float = 0.2,
        fc_layers: Sequence[int] = (128,),
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 200,
        device: Optional[str] = None,
        **params: Any,
    ) -> None:
        """
        Initialize CNN adapter.
        
        Parameters
        ----------
        conv_channels : Sequence[int]
            Number of channels in each convolutional layer
        kernel_sizes : Sequence[int]
            Kernel size for each convolutional layer
        dropout : float
            Dropout rate
        fc_layers : Sequence[int]
            Sizes of fully connected layers
        batch_size : int
            Training batch size
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            L2 regularization weight
        epochs : int
            Number of training epochs
        device : Optional[str]
            Device to use ('cpu' or 'cuda'). Auto-detected if None.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNAdapter.")
        
        # Store parameters
        super().__init__(
            conv_channels=tuple(conv_channels),
            kernel_sizes=tuple(kernel_sizes),
            dropout=dropout,
            fc_layers=tuple(fc_layers),
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            device=device,
            **params,
        )
        
        # Set device
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Model will be created in fit()
        self.model: Optional[CNNModel] = None
        self.training_history: list[Dict[str, float]] = []
    
    # Step 4: Implement required methods
    
    def fit(self, X, y) -> "CNNAdapter":
        """
        Train the CNN model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Training target data
        
        Returns
        -------
        self : CNNAdapter
            Returns self for method chaining
        """
        # Convert to numpy arrays
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        
        # Handle single output
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        
        # Get dimensions
        input_dim = X_np.shape[1]
        output_dim = y_np.shape[1]
        
        # Create model
        self.model = CNNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            conv_channels=self.params["conv_channels"],
            kernel_sizes=self.params["kernel_sizes"],
            dropout=self.params["dropout"],
            fc_layers=self.params["fc_layers"],
        ).to(self.device)
        
        # Store dimensions for later use (e.g., in save/load)
        self.params["input_dim"] = input_dim
        self.params["output_dim"] = output_dim
        
        # Create data loader
        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.MSELoss()
        
        # Training loop
        self.training_history.clear()
        self.model.train()
        n_samples = len(loader.dataset)
        
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().item() * len(batch_x)
            
            # Record training history
            avg_loss = epoch_loss / max(1, n_samples)
            self.training_history.append({
                "epoch": epoch,
                "loss": float(avg_loss),
            })
        
        # Set to evaluation mode
        self.model.eval()
        self.mark_fitted()
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        predictions : ndarray of shape (n_samples, n_outputs)
            Model predictions
        """
        self.ensure_fitted()
        assert self.model is not None
        
        self.model.eval()
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    # Step 5: Optional - Implement uncertainty quantification
    def predict_with_uncertainty(self, X) -> Mapping[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        This is optional. If you don't implement it, set `supports_uq = False`.
        
        For CNNs, you could use:
        - Ensemble predictions
        - Monte Carlo dropout
        - Bayesian neural networks
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        
        Returns
        -------
        dict with keys:
            'mean' : ndarray of shape (n_samples, n_outputs)
                Predictive mean
            'variance' : ndarray of shape (n_samples, n_outputs)
                Predictive variance
        """
        # Example: Simple ensemble approach
        # You could train multiple models and average their predictions
        raise NotImplementedError(
            "Uncertainty quantification not implemented for CNN adapter. "
            "Consider implementing ensemble or MC-dropout."
        )
    
    # Step 6: Optional - Implement model serialization
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        path : Path
            Path to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        
        if not self.supports_serialization:
            raise RuntimeError("Serialization not supported for this adapter.")
        
        # Save model state and parameters
        payload = {
            "state_dict": self.model.state_dict(),
            "params": self.params,
            "model_class": "CNNModel",
        }
        torch.save(payload, path)
    
    def load(self, path: Path) -> "CNNAdapter":
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        path : Path
            Path to the saved model
        
        Returns
        -------
        self : CNNAdapter
            Returns self for method chaining
        """
        if not self.supports_serialization:
            raise RuntimeError("Serialization not supported for this adapter.")
        
        # Load saved data
        payload = torch.load(path, map_location=self.device)
        self.params.update(payload.get("params", {}))
        
        # Reconstruct model
        input_dim = self.params.get("input_dim")
        output_dim = self.params.get("output_dim")
        if input_dim is None or output_dim is None:
            raise ValueError("Saved model missing input/output dimensions.")
        
        self.model = CNNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            conv_channels=self.params["conv_channels"],
            kernel_sizes=self.params["kernel_sizes"],
            dropout=self.params["dropout"],
            fc_layers=self.params["fc_layers"],
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(payload["state_dict"])
        self.mark_fitted()
        return self


# Step 7: Register the adapter
if TORCH_AVAILABLE:
    MODEL_REGISTRY.register(
        CNNAdapter,
        key="torch.cnn",  # Primary key
        name="PyTorch CNN",
        backend="torch",
        description="1D Convolutional Neural Network for tabular data",
        tags=["cnn", "torch", "deep_learning"],
        aliases=["cnn", "conv1d", "torch_cnn"],  # Alternative keys users can use
        default_params={
            "conv_channels": (64, 128, 256),
            "kernel_sizes": (3, 3, 3),
            "dropout": 0.2,
            "fc_layers": (128,),
            "batch_size": 256,
            "learning_rate": 1e-3,
            "epochs": 200,
        },
    )


__all__ = ["CNNAdapter"]
```

---

## Step 8: Update the Adapters Module

Edit `surge/models/adapters/__init__.py` to include your new adapter:

```python
"""Adapter registry bootstrap."""

from .sklearn_adapter import RandomForestRegressorAdapter
from .torch_adapter import TorchMLPAdapter
from .gpflow_adapter import GPflowGPRAdapter
from .cnn_adapter import CNNAdapter  # Add this line

__all__ = [
    "RandomForestRegressorAdapter",
    "TorchMLPAdapter",
    "GPflowGPRAdapter",
    "CNNAdapter",  # Add this line
]
```

---

## Step 9: Test Your Adapter

Create a test script to verify everything works:

```python
"""Test the CNN adapter."""

import numpy as np
from surge import SurrogateEngine, SurrogateDataset, ModelSpec, EngineRunConfig

# Create synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 20
n_outputs = 2

X = np.random.randn(n_samples, n_features)
y = X[:, :2] + 0.1 * np.random.randn(n_samples, n_outputs)

# Create dataset
dataset = SurrogateDataset.from_dataframe(
    pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]),
    input_columns=[f"feature_{i}" for i in range(n_features)],
    output_columns=[f"output_{i}" for i in range(n_outputs)],
)

# Create engine
engine = SurrogateEngine(
    run_config=EngineRunConfig(test_fraction=0.2, val_fraction=0.1)
)
engine.configure_from_dataset(dataset)
engine.prepare()

# Create model spec
spec = ModelSpec(
    key="torch.cnn",  # Use the registered key
    name="my_cnn_model",
    params={
        "conv_channels": (32, 64),
        "kernel_sizes": (3, 3),
        "dropout": 0.1,
        "fc_layers": (64,),
        "epochs": 10,  # Small number for testing
        "batch_size": 128,
    },
)

# Train
results = engine.run([spec])
print(f"Validation R²: {results[0].val_metrics['r2']:.4f}")
print(f"Validation RMSE: {results[0].val_metrics['rmse']:.4f}")
```

---

## Step 10: Use in Workflow

Your new adapter can now be used in YAML workflow specs:

```yaml
models:
  - key: torch.cnn
    name: cnn_profiles
    params:
      conv_channels: [64, 128, 256]
      kernel_sizes: [3, 3, 3]
      dropout: 0.2
      fc_layers: [128]
      epochs: 200
      batch_size: 256
      learning_rate: 0.001
    hpo:
      enabled: true
      n_trials: 20
      metric: val_rmse
      direction: minimize
      search_space:
        dropout:
          type: float
          low: 0.0
          high: 0.5
        learning_rate:
          type: loguniform
          low: 1e-4
          high: 1e-2
```

---

## Required vs Optional Methods

### Required Methods (Must Implement)

1. **`fit(X, y)`** - Train the model
   - Must call `self.mark_fitted()` when done
   - Should return `self` for method chaining

2. **`predict(X)`** - Make predictions
   - Must call `self.ensure_fitted()` first
   - Should return numpy array

### Optional Methods (Implement if Needed)

1. **`predict_with_uncertainty(X)`** - Uncertainty quantification
   - Only needed if `supports_uq = True`
   - Must return dict with `'mean'` and `'variance'` keys

2. **`save(path)`** - Save model to disk
   - Only needed if `supports_serialization = True`
   - Should save everything needed to reconstruct the model

3. **`load(path)`** - Load model from disk
   - Only needed if `supports_serialization = True`
   - Must call `self.mark_fitted()` after loading

---

## Key Points to Remember

1. **Class Attributes**:
   - `name`: Human-readable model name
   - `backend`: Backend identifier (e.g., "torch", "sklearn", "gpflow")
   - `supports_uq`: Whether uncertainty quantification is supported
   - `supports_serialization`: Whether save/load is implemented

2. **Registration**:
   - Use `MODEL_REGISTRY.register()` to make your adapter available
   - Provide a primary `key` and optional `aliases`
   - Add helpful `tags` and `description`

3. **State Management**:
   - Always call `self.mark_fitted()` after training
   - Always call `self.ensure_fitted()` before prediction
   - Store necessary parameters in `self.params` for save/load

4. **Data Handling**:
   - SURGE provides standardized/scaled data to `fit()` and `predict()`
   - Handle both single and multi-output cases
   - Convert to appropriate types (numpy arrays, tensors, etc.)

5. **Training History** (Optional):
   - Store training metrics in `self.training_history`
   - This will be automatically included in workflow summaries

---

## Example: Minimal Adapter

Here's a minimal example showing the absolute essentials:

```python
from ...registry import BaseModelAdapter, MODEL_REGISTRY
import numpy as np
from sklearn.linear_model import LinearRegression

class SimpleLinearAdapter(BaseModelAdapter):
    name = "SimpleLinear"
    backend = "sklearn"
    supports_uq = False
    supports_serialization = False
    
    def __init__(self, **params):
        super().__init__(**params)
        self.model = LinearRegression(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.mark_fitted()
        return self
    
    def predict(self, X):
        self.ensure_fitted()
        return self.model.predict(X)

# Register
MODEL_REGISTRY.register(
    SimpleLinearAdapter,
    key="sklearn.linear",
    aliases=["linear"],
)
```

---

## Troubleshooting

### Model not found in registry
- Make sure you've imported the adapter module (it auto-registers on import)
- Check that `surge/models/adapters/__init__.py` imports your adapter
- Verify the key/alias matches what you're using

### "Model not fitted" error
- Make sure `self.mark_fitted()` is called in `fit()`
- Check that `fit()` was called before `predict()`

### Serialization errors
- Set `supports_serialization = False` if you haven't implemented save/load
- Make sure `save()` and `load()` handle all necessary state

### Uncertainty quantification not working
- Set `supports_uq = False` if not implemented
- Or implement `predict_with_uncertainty()` returning `{'mean': ..., 'variance': ...}`

---

## Next Steps

- Add your adapter to the test suite
- Document any special parameters or requirements
- Consider adding HPO search space suggestions
- Update the main SURGE documentation

---

*For questions or issues, refer to existing adapters in `surge/models/adapters/` as reference implementations.*

