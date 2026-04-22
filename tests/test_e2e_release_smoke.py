# Copyright (c) 2026, SURGE authors. Licensed under BSD-3-Clause.
"""
End-to-end release smoke test.

This is the CI regression referenced by docs/RELEASE_DEMO_PLAN.md and the
``e2e-regression`` job in ``.github/workflows/ci.yml``. It exercises the full
public path a first-time user is supposed to touch:

    load CSV -> SurrogateEngine -> fit -> predict -> (if torch) export ONNX
    -> (if onnxruntime) reload + verify parity against the native model.

If any of those links break, CI fails. Heavy optional dependencies are
soft-skipped, not hard-failed, so the test also survives environments that
only have the sklearn baseline installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Data fixture (shared across sub-tests)
# ---------------------------------------------------------------------------
# Uses scikit-learn's diabetes benchmark (442 rows, 10 inputs, 1 output). It
# ships with scikit-learn, so CI does not depend on any dataset being vendored
# into the repo.
@pytest.fixture(scope="module")
def sample_xy() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes(as_frame=True)
    df = bunch.frame

    input_cols = list(bunch.feature_names)
    # Fan out the single regression target into a small multi-output problem
    # so the PyTorch -> ONNX export path exercises n_out > 1, matching the
    # shape constraints used by SURGE's multi-output adapters.
    df = df.copy()
    df["target_b"] = df["target"] * 0.5 + 1.0
    df["target_c"] = df["target"] * -0.25 + 0.5
    output_cols = ["target", "target_b", "target_c"]

    X = df[input_cols].to_numpy(dtype=np.float64)
    y = df[output_cols].to_numpy(dtype=np.float64)
    return X, y, input_cols, output_cols


# ---------------------------------------------------------------------------
# (1) Train + predict via the public SurrogateEngine API
# ---------------------------------------------------------------------------
def test_engine_sklearn_train_predict(sample_xy) -> None:
    """Registry -> engine -> fit -> predict on the release sample CSV."""
    from surge import EngineRunConfig, ModelSpec, SurrogateEngine
    from surge.model import MODEL_REGISTRY  # side-effect: registers adapters

    X, y, input_cols, output_cols = sample_xy
    df = pd.DataFrame(
        np.hstack([X, y]),
        columns=[*input_cols, *output_cols],
    )

    engine = SurrogateEngine(
        registry=MODEL_REGISTRY,
        run_config=EngineRunConfig(
            test_fraction=0.2,
            val_fraction=0.1,
            random_state=0,
        ),
    )
    engine.configure_dataframe(df, input_columns=input_cols, output_columns=output_cols)

    # Pick a model key that exists in the registry; sklearn.random_forest is
    # the lowest-dependency backend and is always registered.
    candidates = ["sklearn.random_forest", "random_forest", "sklearn.rf"]
    chosen = next((k for k in candidates if k in MODEL_REGISTRY), None)
    assert chosen is not None, (
        f"None of {candidates} are registered. Registry contents: "
        f"{list(MODEL_REGISTRY.keys()) if hasattr(MODEL_REGISTRY, 'keys') else 'n/a'}"
    )

    results = engine.run([
        ModelSpec(key=chosen, params={"n_estimators": 20, "random_state": 0}),
    ])
    assert len(results) == 1
    res = results[0]
    assert res.val_metrics, "val metrics missing"
    assert np.isfinite(list(res.val_metrics.values())).all()

    # Prediction path
    y_pred = res.adapter.predict(X[:8])
    assert y_pred.shape == (8, y.shape[1]), (y_pred.shape, y.shape)
    assert np.isfinite(y_pred).all()


# ---------------------------------------------------------------------------
# (2) PyTorch -> ONNX export -> onnxruntime parity
# ---------------------------------------------------------------------------
def test_torch_export_onnx_roundtrip(sample_xy, tmp_path) -> None:
    """Tiny MLP -> torch.onnx.export -> onnxruntime -> numeric parity."""
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    # torch>=2.11 routes torch.onnx.export through onnxscript. Skip cleanly
    # (don't fail) when it's missing — common on minimal installs.
    if tuple(int(p) for p in torch.__version__.split(".")[:2]) >= (2, 11):
        pytest.importorskip("onnxscript")

    X, y, _, _ = sample_xy
    n_in = X.shape[1]
    n_out = y.shape[1]

    # Minimal MLP: whatever changes in surge.model.pytorch_impl, CI still
    # proves that "a torch module can be exported and re-run" from this repo.
    class TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_in, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, n_out),
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    model = TinyMLP().eval()

    # One forward pass to make sure the graph is valid before export.
    sample = torch.tensor(X[:4], dtype=torch.float32)
    with torch.no_grad():
        native = model(sample).numpy()

    onnx_path = tmp_path / "tiny_mlp.onnx"
    torch.onnx.export(
        model,
        sample,
        onnx_path.as_posix(),
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        opset_version=17,
    )
    assert onnx_path.is_file() and onnx_path.stat().st_size > 0

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    (ort_out,) = session.run(
        None, {"x": sample.numpy().astype(np.float32)}
    )

    # Numerical parity: ONNX/CPU vs. native PyTorch/CPU with the same weights
    # should match to well within float32 noise.
    np.testing.assert_allclose(ort_out, native, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# (3) Package import health (catches the "from loader import ..." regression)
# ---------------------------------------------------------------------------
def test_package_imports_cleanly() -> None:
    """`import surge` must not raise, and key public symbols must exist."""
    import surge

    for name in (
        "SurrogateEngine",
        "SurrogateDataset",
        "EngineRunConfig",
        "ModelSpec",
        "run_surrogate_workflow",
        "SurrogateWorkflowSpec",
        "__version__",
    ):
        assert hasattr(surge, name), f"surge.{name} is missing from the public API"

    assert surge.__version__, "surge.__version__ must be set"
