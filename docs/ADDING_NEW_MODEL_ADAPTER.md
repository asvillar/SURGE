# Adding a new model adapter

An adapter is a thin class that lets the workflow engine train and call your
model. The engine owns data loading, splits, scaling, metrics, HPO and run
artifacts; the adapter owns `fit` and `predict` on NumPy arrays.

The worked example is [`examples/custom_cnn_adapter_template.py`](../examples/custom_cnn_adapter_template.py),
a small 1-D CNN on PyTorch. It is exercised by
[`tests/unit/test_adapter_template.py`](../tests/unit/test_adapter_template.py), so
the code below matches the current API. For a one-page version see
[QUICK_START_NEW_MODEL.md](QUICK_START_NEW_MODEL.md).

## 1. The base-class contract

Subclass `surge.model.base.BaseModelAdapter`.

| Member | Required | Notes |
|--------|----------|-------|
| `name` | yes | Class attribute; the registry key by default (`backend.model`). |
| `backend` | yes | Class attribute, e.g. `"pytorch"` or `"sklearn"`. For `"pytorch"` the engine calls `fit(X, y, X_val=..., y_val=..., finetune=...)`; otherwise `fit(X, y)`. |
| `_build_model(**kwargs)` | yes | Abstract. Return the underlying estimator. The base `__init__(**params)` stores `params` and calls it. |
| `fit(X, y)` | yes | The default delegates to `self._model.fit(X, y)`; override it to match your estimator. `y` may be 1-D or 2-D. |
| `predict(X)` | yes | Return an array of shape `(n_samples, n_outputs)`. |
| `save(path)` / `load(path)` | recommended | Without them runs are not reloadable. |
| `predict_with_uncertainty(X)` | optional | Return whatever your UQ payload is; the engine stores it when the spec asks for it. |
| `training_history` | optional | List of `{"epoch": int, "train_loss": float}` read from the estimator. |
| `uses_internal_preprocessing`, `handles_output_scaling` | optional | Set to `True` if the estimator scales inputs or targets itself. |
| `resource_profile` | optional | A `surge.hpc.ResourceProfile`; drives the `[surge.fit]` banner. |
| `_INFO` | optional | A `ModelInfo` describing architecture, use cases and references; shown by `adapter.info()`. |

Do not define your own `__init__` unless you need to: keep constructor
arguments as keyword params and build the estimator in `_build_model`. After
`fit` the engine calls `mark_fitted()`, a no-op hook you may override.

Keep the heavy import out of module scope. The template imports `torch` inside
`fit` and `predict`, so the module imports on a machine without torch.

```python
from surge.model.base import BaseModelAdapter
from surge.model.registry import register_model


class CNNAdapter(BaseModelAdapter):
    name = "custom.cnn"
    backend = "pytorch"
    default_params = {"channels": 16, "kernel_size": 3, "n_epochs": 50}

    def _build_model(self, **kwargs):
        params = {**self.default_params, **kwargs}
        return TinyCNNRegressor(**params)      # backend class from the template

    def fit(self, X, y, X_val=None, y_val=None, **_):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)
```

## 2. Registering

Registration is one call. The signature is
`register_model(adapter_cls, *, key=None, aliases=None)`; `key` defaults to the
class `name`. The lower-level equivalent is
`MODEL_REGISTRY.register(key, adapter_cls, aliases=...)`.

```python
register_model(CNNAdapter, key="custom.cnn", aliases=["custom_cnn"])
```

Where it goes:

- **A model outside the package** (your own project): put the call at the
  bottom of the module that defines the adapter, and import that module before
  the registry is used, for example `import my_models` at the top of the script
  that runs the spec. The template does this.
- **A model shipped in SURGE:** put the adapter in `surge/model/adapters/<name>.py`
  and register it in `surge/model/__init__.py`. Optional-dependency adapters go
  through `_register_optional`, which records a "skipped" reason instead of
  failing the whole import when the dependency is missing. Registration records
  are listed by `surge models --verbose`.

Key convention: `backend.model_name`, for example `pytorch.residual_mlp`.

## 3. Using the model from a spec

Once the module is imported, the key works anywhere the registry is used:

```yaml
models:
  - key: custom.cnn
    name: my_cnn
    params:
      channels: 32
      n_epochs: 100
```

`params` are passed to the adapter constructor as keyword arguments. Run the
spec from Python so your module is imported first:

```python
import yaml

import my_models                      # registers "custom.cnn"
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

payload = yaml.safe_load(open("spec.yaml"))
run_surrogate_workflow(SurrogateWorkflowSpec.from_dict(payload))
```

`surge run spec.yaml` only sees adapters that SURGE itself registers, so for an
out-of-package model use the script above (see also
[`examples/run_workflow.py`](../examples/run_workflow.py)).

Or create it directly:

```python
from surge.model.registry import create_model
model = create_model("custom.cnn", channels=8, n_epochs=20)
model.fit(X, y)
pred = model.predict(X)
```

## 4. Saving and loading

`save(path)` and `load(path)` receive a path. The template stores a `torch.save`
archive, which is pickle-based. Say so in the docstring of `load`:

> Loads a pickle-based file: only load files you created or trust.

## 5. Testing

Keep a test that registers the adapter under its key, builds it from the
registry, and, when torch is present, fits and predicts on a tiny random array.
`tests/unit/test_adapter_template.py` shows the pattern: load the module by
path, register, and restore the registry afterwards so the global registry is
left as found.

```bash
.venv/bin/python -m pytest -q tests/unit/test_adapter_template.py
```

## 6. Checklist

- `name` and `backend` set; `_build_model` implemented.
- `fit` accepts 1-D and 2-D `y`; `predict` returns `(n_samples, n_outputs)`.
- Heavy imports deferred so the module imports without the optional dependency.
- One `register_model(...)` call, reachable on import.
- `save`/`load` implemented, with the pickle note on `load`.
- A test that registers, instantiates, and (with the dependency) fits and predicts.

See [BUILD_YOUR_OWN_SURROGATE.md](BUILD_YOUR_OWN_SURROGATE.md) (Part II) for the
data contract, HPO search spaces and benchmark registration.
