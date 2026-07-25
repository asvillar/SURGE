# Getting started with SURGE

One page from zero to a trained surrogate with figures. Python 3.10–3.12.

## 1 · Install (2 minutes)

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/S-Villar/SURGE.git && cd SURGE
uv venv && source .venv/bin/activate
uv pip install -e ".[torch,dev]"
```

With plain pip: replace the last two lines with
`python3.11 -m venv .venv && source .venv/bin/activate` and
`pip install -e ".[torch,dev]"`.

Minimal install (`uv pip install -e .`) gives you the scikit-learn model
families; `[torch]` adds the neural adapters (MLPs, FNO, U-Net, vision,
sequence models).

Check what you got — including *why* any optional model family is absent:

```bash
surge models --verbose
```

## 2 · First surrogate (30 seconds)

```bash
python -m examples.quickstart --dataset diabetes --model rf --infer
```

This downloads nothing exotic (scikit-learn's diabetes set), trains a
random forest, and writes a complete, reproducible run:

```text
runs/diabetes_rf/
├── metrics.json         # train/val/test R², RMSE, MAE per model
├── spec.yaml            # re-run with: surge run runs/diabetes_rf/spec.yaml
├── models/  scalers/  predictions/   # artifacts for inference
└── git_rev.txt  env.txt  model_card_*.json   # provenance
```

## 3 · Your own data

Point a YAML spec at any CSV / Parquet / Pickle / HDF5 / NetCDF file.
Columns named `y_*`, `output_*`, or `target_*` are auto-detected as
outputs; otherwise list them explicitly in a metadata file.

```yaml
# my_surrogate.yaml
dataset_path: path/to/data.csv
models:
  - key: sklearn.random_forest
  - key: pytorch.residual_mlp
    hpo: {n_trials: 20, metric: val_rmse}
run_tag: my_first_surrogate
```

```bash
surge run my_surrogate.yaml
```

Model shortlist by task: tabular → `sklearn.random_forest`,
`xgboost.xgbregressor`, `pytorch.residual_mlp` · 1D/2D fields →
`pytorch.fno1d` / `pytorch.fno2d`, `pytorch.unet` · sequences →
`pytorch.lstm` · uncertainty → `sklearn.gpr`, `botorch.gp`.
Full walkthrough: [BUILD_YOUR_OWN_SURROGATE.md](BUILD_YOUR_OWN_SURROGATE.md).

## 4 · Benchmarks and reports

```bash
surge list                                                  # what exists
surge bench -b synthetic.regression_1d -m sklearn.random_forest --no-save
surge bench -b tabular.california_housing -m all --seeds 5  # mean ± std
surge report --out leaderboard.html                         # dashboard
python examples/viz_theme_gallery.py                        # figure gallery
```

`surge report` builds a self-contained HTML leaderboard (spider charts,
dataset previews, sortable tables) from your local results — no server,
no network.

## 5 · Verify your installation

```bash
pytest -q tests/test_e2e_release_smoke.py    # seconds
pytest -q                                    # full suite
```

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| A model key is "not registered" | Its optional dependency is missing or broken. `surge models --verbose` shows the exact reason per adapter. |
| `lgbm.*` / `xgboost.*` missing on macOS | The OpenMP runtime is absent: `brew install libomp`, then reinstall/re-import. |
| First benchmark run is slow / downloads | Network-backed benchmarks (OpenML, MNIST/CIFAR, ConStellaration) cache under `data/datasets/benchmarks/` on first use; later runs are offline. |
| `pytorch.*` models absent | Install the extra: `uv pip install -e ".[torch]"`. |
| GPflow models error on Apple Silicon | See the arm64 notes in `requirements.txt` (mainline `tensorflow`, `gpflow --no-deps`). |
| Grouped samples (shots, trajectories) | Splits are random-only today — leakage risk. Track the split-strategy roadmap in `docs/design/ARCHITECTURE_RECOMMENDATIONS.md` (R4). |
