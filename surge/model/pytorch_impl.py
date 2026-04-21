"""
PyTorch-based model implementations for SURGE package.
Integrates PyTorch MLP models with the unified SURGE interface.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

from sklearn.preprocessing import StandardScaler

_LOGGER = logging.getLogger("surge.pytorch.mlp")

# Guard torch imports for environments without PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    Subset = None


def _make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
):
    """Standard PyTorch DataLoader; batches are produced on CPU for `.to(device)` in the loop."""
    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(dataset, **kw)


def _append_training_progress_jsonl(entry: dict[str, Any]) -> None:
    """Append one epoch record for live monitoring (tail -f friendly)."""
    path = os.environ.get("SURGE_TRAINING_PROGRESS_JSONL")
    if not path:
        return
    row = dict(entry)
    hpo_trial = os.environ.get("SURGE_HPO_TRIAL")
    if hpo_trial is not None:
        try:
            row["hpo_trial"] = int(hpo_trial)
        except ValueError:
            row["hpo_trial"] = hpo_trial
    mirror = os.environ.get("SURGE_TRAINING_PROGRESS_JSONL_MIRROR")
    targets = [path]
    if mirror:
        targets.append(mirror)
    for target in targets:
        try:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass


class PyTorchMLP(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch Multi-Layer Perceptron implementation compatible with SURGE interface.

    Training uses a DataLoader (mini-batches on CPU, batches moved to device). Optional
    ``epoch_train_sample_fraction`` (e.g. 0.6) draws a fresh random subset of training
    rows each epoch before batching. Set ``log_progress`` to log train/val RMSE (scaled
    targets, sqrt of MSE over all profile dimensions) each epoch.
    """

    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.1,
                 activation_fn=None, learning_rate=1e-3, n_epochs=300,
                 batch_size=32, device=None, dataloader_num_workers: int = 0,
                 pin_memory: Optional[bool] = None,
                 inference_batch_size: Optional[int] = None,
                 log_progress: bool = False,
                 epoch_train_sample_fraction: float = 1.0,
                 checkpoint_dir: Optional[str] = None,
                 checkpoint_every_n_epochs: int = 0,
                 restore_best_weights: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchMLP. Install with: pip install torch")
        if activation_fn is None:
            activation_fn = nn.ReLU
        super(PyTorchMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dataloader_num_workers = int(dataloader_num_workers)
        self._pin_memory_setting = pin_memory
        self.inference_batch_size = inference_batch_size
        self.log_progress = bool(log_progress)
        frac = float(epoch_train_sample_fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError("epoch_train_sample_fraction must be in (0, 1].")
        self.epoch_train_sample_fraction = frac
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = max(0, int(checkpoint_every_n_epochs))
        self.restore_best_weights = bool(restore_best_weights)

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.to(self.device)

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # For sklearn-like interface
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_history = []

    def forward(self, x):
        return self.network(x)

    def _maybe_save_epoch_checkpoint(self, epoch_num: int) -> None:
        if not self.checkpoint_dir or self.checkpoint_every_n_epochs <= 0:
            return
        if epoch_num % self.checkpoint_every_n_epochs != 0:
            return
        try:
            d = Path(self.checkpoint_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"epoch_{epoch_num:04d}.pt"
            torch.save(
                {
                    "epoch": epoch_num,
                    "model_state_dict": self.state_dict(),
                    "scaler_X": self.scaler_X,
                    "scaler_y": self.scaler_y,
                    "input_size": self.input_size,
                    "output_size": self.output_size,
                },
                path,
            )
        except OSError:
            pass

    def fit(self, X, y, X_val=None, y_val=None, patience=None, finetune=False):
        """
        Fit the model using sklearn-like interface.
        If X_val, y_val and patience are provided, use early stopping.
        When finetune=True and scalers exist (from load), use transform instead of fit_transform.
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale the data. When finetune=True, engine already scaled with pretrained scalers.
        if finetune:
            X_scaled = X
            y_scaled = y.reshape(-1, 1) if y.ndim == 1 else y
        else:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)

        use_pin = self._pin_memory_setting
        if use_pin is None:
            use_pin = self.device.type == "cuda"

        non_blocking = bool(use_pin and self.device.type == "cuda")

        # CPU tensors + DataLoader: batches moved to device inside the loop (standard PyTorch pattern).
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        full_train_ds = TensorDataset(X_tensor, y_tensor)
        n_train_rows = len(full_train_ds)

        # Validation data for early stopping
        use_early_stop = (
            patience is not None and patience > 0
            and X_val is not None and y_val is not None
        )
        val_loader = None
        if use_early_stop:
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.cpu().numpy()
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy()
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            if finetune:
                X_val_scaled = X_val
                y_val_scaled = y_val
            else:
                X_val_scaled = self.scaler_X.transform(X_val)
                y_val_scaled = self.scaler_y.transform(y_val)
            X_val_cpu = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_cpu = torch.tensor(y_val_scaled, dtype=torch.float32)
            val_loader = _make_dataloader(
                TensorDataset(X_val_cpu, y_val_cpu),
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=use_pin,
                num_workers=self.dataloader_num_workers,
            )

        # Training loop
        self.train()
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state: Optional[dict[str, Any]] = None
        best_epoch_num = 0
        self.training_history.clear()

        for epoch in range(self.n_epochs):
            if self.epoch_train_sample_fraction < 1.0 - 1e-12:
                k = max(1, int(n_train_rows * self.epoch_train_sample_fraction))
                idx = torch.randperm(n_train_rows)[:k].tolist()
                epoch_train_ds = Subset(full_train_ds, idx)
            else:
                epoch_train_ds = full_train_ds

            dataloader = _make_dataloader(
                epoch_train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=use_pin,
                num_workers=self.dataloader_num_workers,
            )

            epoch_loss = 0.0
            train_sse = 0.0
            train_nel = 0
            self.train()
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=non_blocking)
                batch_y = batch_y.to(self.device, non_blocking=non_blocking)
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                train_sse += torch.sum((outputs.detach() - batch_y) ** 2).item()
                train_nel += int(outputs.numel())

            nb = max(len(dataloader), 1)
            avg_train_loss = epoch_loss / nb
            train_rmse = math.sqrt(train_sse / max(train_nel, 1))
            entry = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_rmse_scaled": train_rmse,
                "train_rows_this_epoch": int(len(epoch_train_ds)),
            }

            if use_early_stop:
                self.eval()
                with torch.no_grad():
                    assert val_loader is not None
                    val_sse = 0.0
                    val_nel = 0
                    for vx, vy in val_loader:
                        vx = vx.to(self.device, non_blocking=non_blocking)
                        vy = vy.to(self.device, non_blocking=non_blocking)
                        vpred = self.forward(vx)
                        val_sse += torch.sum((vpred - vy) ** 2).item()
                        val_nel += int(vpred.numel())
                    val_loss = val_sse / max(val_nel, 1)
                self.train()
                val_rmse = math.sqrt(val_loss)
                entry["val_loss"] = val_loss
                entry["val_rmse_scaled"] = val_rmse
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if self.restore_best_weights:
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.state_dict().items()
                        }
                        best_epoch_num = epoch + 1
                else:
                    epochs_without_improvement += 1
                if self.log_progress:
                    _LOGGER.info(
                        "[torch.mlp] epoch %d/%d | train_RMSE(scaled profiles)=%.6f | "
                        "val_RMSE(scaled profiles)=%.6f | train_rows=%d/%d | patience=%d/%d",
                        epoch + 1,
                        self.n_epochs,
                        train_rmse,
                        val_rmse,
                        entry["train_rows_this_epoch"],
                        n_train_rows,
                        epochs_without_improvement,
                        patience,
                    )
                if epochs_without_improvement >= patience:
                    self.training_history.append(entry)
                    _append_training_progress_jsonl(entry)
                    self._maybe_save_epoch_checkpoint(epoch + 1)
                    break
            else:
                entry["val_loss"] = None
                entry["val_rmse_scaled"] = None
                if self.log_progress:
                    _LOGGER.info(
                        "[torch.mlp] epoch %d/%d | train_RMSE(scaled profiles)=%.6f | "
                        "train_rows=%d/%d (no val — set patience + val in engine)",
                        epoch + 1,
                        self.n_epochs,
                        train_rmse,
                        entry["train_rows_this_epoch"],
                        n_train_rows,
                    )
                self.training_history.append(entry)
                _append_training_progress_jsonl(entry)
                self._maybe_save_epoch_checkpoint(epoch + 1)
                continue

            self.training_history.append(entry)
            _append_training_progress_jsonl(entry)
            self._maybe_save_epoch_checkpoint(epoch + 1)

        if (
            use_early_stop
            and self.restore_best_weights
            and best_state is not None
        ):
            device_state = {k: v.to(self.device) for k, v in best_state.items()}
            self.load_state_dict(device_state)
            _LOGGER.info(
                "[torch.mlp] restored weights from best validation (epoch %d / val_loss=%.8f)",
                best_epoch_num,
                best_val_loss,
            )

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using sklearn-like interface.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Scale the input
        X_scaled = self.scaler_X.transform(X)
        X_cpu = torch.tensor(X_scaled, dtype=torch.float32)
        ibs = self.inference_batch_size
        if ibs is None:
            ibs = max(4096, self.batch_size * 4)

        pieces = []
        self.eval()
        with torch.no_grad():
            for start in range(0, X_cpu.shape[0], ibs):
                chunk = X_cpu[start : start + ibs].to(self.device)
                out = self.forward(chunk).cpu()
                pieces.append(out)
            y_pred_scaled = torch.cat(pieces, dim=0).numpy()

        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # Return 1D array if output_size is 1
        if self.output_size == 1:
            y_pred = y_pred.ravel()

        return y_pred


class PyTorchMLPModel:
    """
    SURGE-compatible wrapper for PyTorch MLP models.
    Supports early stopping via patience (epochs without val improvement).
    """

    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.1, activation_fn="relu",
                 learning_rate=1e-3, n_epochs=300, batch_size=32, patience=None,
                 max_epochs=None, epochs=None, dropout=0.1,
                 dataloader_num_workers: int = 0, pin_memory=None,
                 inference_batch_size=None, log_progress: bool = False,
                 epoch_train_sample_fraction: float = 1.0,
                 checkpoint_every_n_epochs: int = 0,
                 restore_best_weights: bool = True,
                 **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchMLPModel. Install with: pip install torch")

        # Map activation functions
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid
        }

        if isinstance(activation_fn, str):
            activation_fn = activation_map.get(activation_fn.lower(), nn.ReLU)

        n_epochs = max_epochs or epochs or n_epochs
        dropout_rate = dropout if dropout is not None else dropout_rate

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.pin_memory = pin_memory
        self.inference_batch_size = inference_batch_size
        self.log_progress = bool(log_progress)
        self.epoch_train_sample_fraction = float(epoch_train_sample_fraction)
        self.checkpoint_every_n_epochs = max(0, int(checkpoint_every_n_epochs))
        self.restore_best_weights = bool(restore_best_weights)
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None, finetune=False):
        """
        Fit the PyTorch MLP model.
        When finetune=True and self.model exists (from load), continue training.
        """
        input_size = X.shape[1]
        output_size = 1 if y.ndim == 1 else y.shape[1]

        ckpt_dir = os.environ.get("SURGE_CHECKPOINT_DIR")
        if self.model is None or not finetune:
            self.model = PyTorchMLP(
                input_size=input_size,
                hidden_layers=self.hidden_layers,
                output_size=output_size,
                dropout_rate=self.dropout_rate,
                activation_fn=self.activation_fn,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                dataloader_num_workers=self.dataloader_num_workers,
                pin_memory=self.pin_memory,
                inference_batch_size=self.inference_batch_size,
                log_progress=self.log_progress,
                epoch_train_sample_fraction=self.epoch_train_sample_fraction,
                checkpoint_dir=ckpt_dir,
                checkpoint_every_n_epochs=self.checkpoint_every_n_epochs,
                restore_best_weights=self.restore_best_weights,
            )

        return self.model.fit(
            X, y, X_val=X_val, y_val=y_val, patience=self.patience, finetune=finetune
        )

    def predict(self, X):
        """
        Make predictions with the fitted model.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def save(self, filepath):
        """
        Save the PyTorch model state.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.model.scaler_X,
            'scaler_y': self.model.scaler_y,
            'training_history': list(getattr(self.model, "training_history", []) or []),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.model.output_size,
                'dropout_rate': self.dropout_rate,
                'activation_fn': self.activation_fn,
                'learning_rate': self.learning_rate,
                'n_epochs': self.model.n_epochs,
                'batch_size': self.model.batch_size,
                'dataloader_num_workers': self.model.dataloader_num_workers,
                'pin_memory': self.model._pin_memory_setting,
                'inference_batch_size': self.model.inference_batch_size,
                'log_progress': self.model.log_progress,
                'epoch_train_sample_fraction': self.model.epoch_train_sample_fraction,
                'checkpoint_every_n_epochs': self.checkpoint_every_n_epochs,
                'restore_best_weights': self.restore_best_weights,
            }
        }, filepath)

    def load(self, filepath):
        """
        Load a saved PyTorch model.
        """
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        config = dict(checkpoint["model_config"])
        config.setdefault("checkpoint_dir", None)
        config.setdefault("checkpoint_every_n_epochs", 0)
        config.setdefault("restore_best_weights", True)

        # Recreate model with saved configuration
        self.checkpoint_every_n_epochs = int(config.get("checkpoint_every_n_epochs", 0) or 0)
        self.restore_best_weights = bool(config.get("restore_best_weights", True))
        self.model = PyTorchMLP(**config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.scaler_X = checkpoint['scaler_X']
        self.model.scaler_y = checkpoint['scaler_y']
        if isinstance(checkpoint.get("training_history"), list):
            self.model.training_history = checkpoint["training_history"]
        self.model.is_fitted = True

        return self
