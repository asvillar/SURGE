# SURGE Test Suite

Unit and end-to-end tests for the SURGE public API (`SurrogateDataset`,
`SurrogateEngine`, `run_surrogate_workflow`, the model registry, ONNX
round-trip, the benchmark leaderboard and the viz module).

## Layout

| Path | Scope |
|---|---|
| `tests/` | Package-level tests: imports, dataset and engine construction, model registry and profiling, workflow and HPO, data generation, end-to-end release smoke (sklearn RF train, predict, ONNX export and inference on `sklearn.datasets.load_diabetes`) |
| `tests/unit/` | Focused unit tests: CLI, metrics, dataset I/O, viz, paths, rollout, leaderboard report, loaders |
| `tests/benchmarks/` | Benchmark registry, metadata, thresholds, leaderboard, adapters and dataset loaders |
| `tests/fixtures/` | Small static inputs used by the tests |

## Running

```bash
# Whole suite
pytest -q tests/

# Show skip reasons
pytest -q -rs tests/

# Single file
pytest -q tests/test_e2e_release_smoke.py

# Include tests marked slow
pytest -q tests/ -m slow
```

Tests marked `slow` are deselected by default; select them with `-m slow`.
Some tests skip when an optional dependency (for example `h5py`, `torch`,
`onnxruntime`) or a local dataset is not installed; `-rs` lists the reasons.

## Test data

Tests use synthetic data (`numpy.random` with fixed seeds) or public
`sklearn.datasets`. No proprietary or large vendored files are required or
downloaded.
