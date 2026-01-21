"""
PyTorch-based model implementations for SURGE package.
Integrates PyTorch MLP models with the unified SURGE interface.
"""

from sklearn.preprocessing import StandardScaler

# Guard torch imports for environments without PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None


class PyTorchMLP(nn.Module if TORCH_AVAILABLE else object):
    """
    PyTorch Multi-Layer Perceptron implementation compatible with SURGE interface.
    """

    def __init__(self, input_size, hidden_layers, output_size=1, dropout_rate=0.1,
                 activation_fn=None, learning_rate=1e-3, n_epochs=300,
                 batch_size=32, device=None):
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def forward(self, x):
        return self.network(x)

    def fit(self, X, y):
        """
        Fit the model using sklearn-like interface.
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

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
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

        # Make predictions
        self.eval()
        with torch.no_grad():
            y_pred_scaled = self.forward(X_tensor)
            y_pred_scaled = y_pred_scaled.cpu().numpy()

        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # Return 1D array if output_size is 1
        if self.output_size == 1:
            y_pred = y_pred.ravel()

        return y_pred


class PyTorchMLPModel:
    """
    SURGE-compatible wrapper for PyTorch MLP models.
    """

    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.1, activation_fn="relu",
                 learning_rate=1e-3, n_epochs=300, batch_size=32, **kwargs):
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

        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        """
        Fit the PyTorch MLP model.
        """
        # Initialize model with correct input size
        input_size = X.shape[1]
        output_size = 1 if y.ndim == 1 else y.shape[1]

        self.model = PyTorchMLP(
            input_size=input_size,
            hidden_layers=self.hidden_layers,
            output_size=output_size,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size
        )

        return self.model.fit(X, y)

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
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.model.output_size,
                'dropout_rate': self.dropout_rate,
                'activation_fn': self.activation_fn,
                'learning_rate': self.learning_rate
            }
        }, filepath)

    def load(self, filepath):
        """
        Load a saved PyTorch model.
        """
        checkpoint = torch.load(filepath)
        config = checkpoint['model_config']

        # Recreate model with saved configuration
        self.model = PyTorchMLP(**config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.scaler_X = checkpoint['scaler_X']
        self.model.scaler_y = checkpoint['scaler_y']
        self.model.is_fitted = True

        return self
