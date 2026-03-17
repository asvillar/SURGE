# Quick Start: Adding a New Model to SURGE

## TL;DR - 3 Steps

1. **Create adapter class** inheriting from `BaseModelAdapter`
2. **Implement `fit()` and `predict()` methods**
3. **Register with `MODEL_REGISTRY.register()`**

---

## Minimal Example

```python
# File: surge/models/adapters/my_model_adapter.py

from ...registry import BaseModelAdapter, MODEL_REGISTRY
import numpy as np

class MyModelAdapter(BaseModelAdapter):
    name = "MyModel"
    backend = "my_backend"
    supports_uq = False  # Set True if you implement uncertainty
    supports_serialization = False  # Set True if you implement save/load
    
    def __init__(self, **params):
        super().__init__(**params)
        # Initialize your model here
        self.model = None
    
    def fit(self, X, y):
        # Train your model
        # X: (n_samples, n_features) numpy array
        # y: (n_samples,) or (n_samples, n_outputs) numpy array
        self.model = ...  # Your training code
        self.mark_fitted()  # REQUIRED!
        return self
    
    def predict(self, X):
        self.ensure_fitted()  # REQUIRED!
        # X: (n_samples, n_features) numpy array
        return self.model.predict(X)  # Return numpy array

# Register
MODEL_REGISTRY.register(
    MyModelAdapter,
    key="my_backend.mymodel",
    aliases=["mymodel"],
)
```

---

## Required Methods

### `fit(X, y) -> self`
- Train the model
- **Must call** `self.mark_fitted()` at the end
- Return `self` for chaining

### `predict(X) -> np.ndarray`
- Make predictions
- **Must call** `self.ensure_fitted()` first
- Return numpy array of shape `(n_samples, n_outputs)`

---

## Optional Methods

### `predict_with_uncertainty(X) -> dict`
Only if `supports_uq = True`:
```python
return {
    "mean": np.array(...),      # (n_samples, n_outputs)
    "variance": np.array(...),  # (n_samples, n_outputs)
}
```

### `save(path: Path) -> None`
Only if `supports_serialization = True`:
```python
# Save model state to path
```

### `load(path: Path) -> self`
Only if `supports_serialization = True`:
```python
# Load model state from path
self.mark_fitted()  # REQUIRED!
return self
```

---

## Registration

```python
MODEL_REGISTRY.register(
    YourAdapterClass,
    key="backend.model_name",      # Primary key
    name="Human Readable Name",    # Optional
    backend="backend_name",        # Optional (from class)
    description="Model description", # Optional
    tags=["tag1", "tag2"],         # Optional
    aliases=["alias1", "alias2"],  # Optional alternative keys
    default_params={...},          # Optional default parameters
)
```

---

## Usage

After registration, use in code:

```python
from surge import SurrogateEngine, ModelSpec

engine = SurrogateEngine(...)
engine.configure_from_dataset(dataset)
engine.prepare()

spec = ModelSpec(
    key="my_backend.mymodel",  # or use alias "mymodel"
    params={"param1": value1, ...}
)
results = engine.run([spec])
```

Or in YAML:

```yaml
models:
  - key: my_backend.mymodel
    params:
      param1: value1
```

---

## Class Attributes

```python
class YourAdapter(BaseModelAdapter):
    name = "ModelName"              # Required: Human-readable name
    backend = "backend_name"        # Required: Backend identifier
    supports_uq = False             # Optional: Can do uncertainty?
    supports_serialization = False  # Optional: Can save/load?
```

---

## Important Notes

1. **Data is pre-processed**: SURGE provides standardized/scaled data to `fit()` and `predict()`
2. **Handle multi-output**: `y` can be 1D or 2D
3. **State management**: Always call `mark_fitted()` after training, `ensure_fitted()` before prediction
4. **Auto-registration**: Import your adapter module to register it (add to `surge/models/adapters/__init__.py`)

---

## Full Example: CNN Adapter

See `docs/ADDING_NEW_MODEL_ADAPTER.md` for a complete CNN implementation example.

---

## Reference Implementations

- **Simple**: `surge/models/adapters/sklearn_adapter.py` (RandomForest)
- **PyTorch**: `surge/models/adapters/torch_adapter.py` (MLP)
- **GPflow**: `surge/models/adapters/gpflow_adapter.py` (GPR)

