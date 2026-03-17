"""
Template for creating a custom CNN adapter for SURGE.

This is a complete, working example that you can copy and modify for your own model.
To use this:
1. Copy this file to surge/models/adapters/cnn_adapter.py
2. Update surge/models/adapters/__init__.py to import it
3. Use it in your workflows with key="torch.cnn"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from surge.registry import BaseModelAdapter, MODEL_REGISTRY

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


# Define your model architecture
if TORCH_AVAILABLE:
    class CNNModel(nn.Module):
        """1D CNN for tabular data."""
        
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
            prev_channels = 1
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
            
            # Calculate flattened size
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
            # Reshape: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            
            # Convolutional layers
            for conv in self.conv_layers:
                x = conv(x)
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Fully connected layers
            for fc in self.fc_layers:
                x = fc(x)
            
            return self.output_layer(x)

else:
    class CNNModel:
        pass


# Create the adapter
class CNNAdapter(BaseModelAdapter):
    """PyTorch CNN adapter for SURGE."""
    
    name = "CNN"
    backend = "torch"
    supports_uq = False
    supports_serialization = True
    
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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNAdapter.")
        
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
        
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[CNNModel] = None
        self.training_history: list[Dict[str, float]] = []
    
    def fit(self, X, y) -> "CNNAdapter":
        """Train the CNN model."""
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        
        input_dim = X_np.shape[1]
        output_dim = y_np.shape[1]
        
        self.model = CNNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            conv_channels=self.params["conv_channels"],
            kernel_sizes=self.params["kernel_sizes"],
            dropout=self.params["dropout"],
            fc_layers=self.params["fc_layers"],
        ).to(self.device)
        
        self.params["input_dim"] = input_dim
        self.params["output_dim"] = output_dim
        
        dataset = TensorDataset(
            torch.from_numpy(X_np),
            torch.from_numpy(y_np)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
        )
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )
        criterion = nn.MSELoss()
        
        self.training_history.clear()
        self.model.train()
        n_samples = len(loader.dataset)
        
        for epoch in range(self.params["epochs"]):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.detach().item() * len(batch_x)
            
            avg_loss = epoch_loss / max(1, n_samples)
            self.training_history.append({
                "epoch": epoch,
                "loss": float(avg_loss),
            })
        
        self.model.eval()
        self.mark_fitted()
        return self
    
    def predict(self, X):
        """Make predictions."""
        self.ensure_fitted()
        assert self.model is not None
        
        self.model.eval()
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save(self, path: Path) -> None:
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        
        payload = {
            "state_dict": self.model.state_dict(),
            "params": self.params,
        }
        torch.save(payload, path)
    
    def load(self, path: Path) -> "CNNAdapter":
        """Load a trained model."""
        payload = torch.load(path, map_location=self.device)
        self.params.update(payload.get("params", {}))
        
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
        
        self.model.load_state_dict(payload["state_dict"])
        self.mark_fitted()
        return self


# Register the adapter
if TORCH_AVAILABLE:
    MODEL_REGISTRY.register(
        CNNAdapter,
        key="torch.cnn",
        name="PyTorch CNN",
        backend="torch",
        description="1D Convolutional Neural Network for tabular data",
        tags=["cnn", "torch", "deep_learning"],
        aliases=["cnn", "conv1d", "torch_cnn"],
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


# Example usage:
if __name__ == "__main__":
    import pandas as pd
    from surge import SurrogateEngine, SurrogateDataset, ModelSpec, EngineRunConfig
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_outputs = 2
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, :2] + 0.1 * np.random.randn(n_samples, n_outputs)
    
    # Create dataset
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    for i in range(n_outputs):
        df[f"output_{i}"] = y[:, i]
    
    dataset = SurrogateDataset.from_dataframe(
        df,
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
        key="torch.cnn",
        name="test_cnn",
        params={
            "conv_channels": (32, 64),
            "kernel_sizes": (3, 3),
            "dropout": 0.1,
            "fc_layers": (64,),
            "epochs": 10,
            "batch_size": 128,
        },
    )
    
    # Train
    print("Training CNN model...")
    results = engine.run([spec])
    print(f"Validation R²: {results[0].val_metrics['r2']:.4f}")
    print(f"Validation RMSE: {results[0].val_metrics['rmse']:.4f}")

