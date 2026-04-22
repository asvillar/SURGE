# SURGE Test Suite

Unit + end-to-end tests covering the **current** SURGE public API
(`SurrogateDataset`, `SurrogateEngine`, `run_surrogate_workflow`, model
registry, ONNX round-trip). All tests exercise the refactored interface;
there is no pre-refactor API left in the tree.

## Layout

| File | Scope |
|---|---|
| `test_core.py` | Package-level smoke: main imports, availability flags, default `SurrogateDataset` constructor, `MODEL_REGISTRY` access |
| `test_workflow_new.py` | `SurrogateDataset.from_dataframe` / `.from_path` auto-detection + `SurrogateEngine.prepare()` / `.run([ModelSpec])` unit tests |
| `test_e2e_release_smoke.py` | End-to-end regression fixture used by CI: sklearn RF train → predict → ONNX export → ONNX-runtime inference on `sklearn.datasets.load_diabetes` (no vendored data) |
| `test_model_registry.py` | Listing, retrieval, and custom registration against `MODEL_REGISTRY` |
| `test_model_profiling.py` | `model_size_bytes` / `inference_ms_per_sample` profile fields produced by the workflow |
| `test_preprocessing.py` | Input / output column detection heuristics |
| `test_mhd_code_workflow_modes.py` | MHD-code dataset loader paths (pickle today, `h5py`-gated batch modes) |
| `test_xgc_workflow_modes.py` | kinetic-code dataset loader paths |
| `test_model_comparison.py` | Multi-model comparison scaffold — currently guarded by a legacy `VISUALIZATION_AVAILABLE` flag and skips until refactored against `surge.viz`. |

## Running

```bash
# Full suite
pytest -q

# Verbose, showing skip reasons
pytest -q -rs

# Single file
pytest -q tests/test_e2e_release_smoke.py
```

Expected in a fresh `.[torch,onnx,dev]` env: **52 passed, 1 skipped**
(`h5py` is included in `dev` so the MHD-code batch tests run). The skip:

- 1 × `test_model_comparison.py` — legacy visualization scaffold;
  pending migration to `surge.viz.run_viz`.

## Test data

All tests use synthetic data (`numpy.random` with fixed seeds) or
public `sklearn.datasets`. No proprietary or large vendored files are
required or downloaded.
