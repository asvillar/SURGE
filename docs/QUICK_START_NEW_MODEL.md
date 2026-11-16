# Quick start: add a model

Full guide: [ADDING_NEW_MODEL_ADAPTER.md](ADDING_NEW_MODEL_ADAPTER.md). Working
code: [`examples/custom_cnn_adapter_template.py`](../examples/custom_cnn_adapter_template.py).

Three steps:

1. Subclass `BaseModelAdapter`, set `name` and `backend`, and implement `_build_model`.
2. Implement `fit(X, y)` and `predict(X)` (the defaults delegate to the estimator).
3. Call `register_model(...)` once, at import time.

```python
import numpy as np
from sklearn.linear_model import Ridge

from surge.model.base import BaseModelAdapter
from surge.model.registry import create_model, register_model


class MyRidge(BaseModelAdapter):
    name = "custom.my_ridge"
    backend = "sklearn"

    def _build_model(self, **kwargs):
        return Ridge(**kwargs)          # fit/predict are delegated to it


register_model(MyRidge, key="custom.my_ridge", aliases=["my_ridge"])

model = create_model("custom.my_ridge", alpha=0.5)
X = np.random.default_rng(0).normal(size=(50, 4))
model.fit(X, X.sum(axis=1))
print(model.predict(X[:3]))
```

Use it from a spec after importing the module that registers it:

```yaml
models:
  - key: custom.my_ridge
    params: {alpha: 0.5}
```

```python
import yaml

import my_models                      # runs register_model(...)
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

payload = yaml.safe_load(open("spec.yaml"))
run_surrogate_workflow(SurrogateWorkflowSpec.from_dict(payload))
```

Adapters shipped inside SURGE are registered in `surge/model/__init__.py`.

For a torch model, import `torch` inside `fit` and `predict` so the module
imports without it; the template shows this, plus `save` and `load` (whose
docstring notes that the file is pickle-based and must come from a trusted source).

Test it the way `tests/unit/test_adapter_template.py` does: register, build from
the registry, fit and predict on a tiny random array.
