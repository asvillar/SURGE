# SURGE — Surrogate Unified Robust Generation Engine

SURGE is a surrogate modeling framework for scientific workflows that
integrates Scientific Machine Learning (SciML) and AutoML features,
uncertainty quantification (UQ), and MLOps-style provenance in a single
declarative pipeline. One YAML spec (or Python API) covers
**load → schema → split → train → HPO → metrics → UQ → artifacts → figures**,
backed by an extensible registry of 40+ model adapters across scikit-learn,
PyTorch, TensorFlow/Keras, and Gaussian-process backends — with every run
reproducible from its own `runs/<tag>/` directory.

**v0.1.0** · License: BSD-3-Clause · [DOE CODE 179819](https://www.osti.gov/doecode/biblio/179819) · DOI [10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5)

![SURGE architecture: ingest & characterize, model & optimize, evaluate & quantify, report & deploy](https://raw.githubusercontent.com/asvillar/SURGE/v0.1.0/docs/assets/readme/architecture.png)

## Install

```bash
pip install surge-ml
```

| Extra | Adds | Install |
|-------|------|---------|
| *(base)* | sklearn, pandas, Optuna, plotting | `pip install surge-ml` |
| `torch` | PyTorch model families | `pip install "surge-ml[torch]"` |
| `onnx` | ONNX export + runtime parity tests | `pip install "surge-ml[onnx]"` |
| `benchmarks` | h5py for the benchmark suite | `pip install "surge-ml[benchmarks]"` |
| `tensorflow` | TF ≥ 2.21 + tf-keras → `keras.mlp` adapter | `pip install "surge-ml[tensorflow]"` |
| `gpflow` | TF stack for GPflow GPs — finish with `pip install gpflow --no-deps` (its published metadata pins numpy<2, unresolvable next to the TF floor above, though GPflow itself runs fine on NumPy 2.x) | `pip install "surge-ml[gpflow]"` |
| `all` | torch + onnx + shap + mlflow + benchmarks + tensorflow (not gpflow — see the no-deps note above) | `pip install "surge-ml[all]"` |

Requires Python 3.10–3.12.

## Quickstart

```bash
surge init --data my_data.csv --target y --yes --out spec.yaml   # write a commented spec
surge validate spec.yaml                                         # schema-check it
surge run spec.yaml                                               # train + evaluate
surge models                                                      # list registered adapters
surge bench -b tabular.california_housing -m all --seeds 5       # leaderboard, mean ± std
surge report --out leaderboard.html                               # self-contained HTML dashboard
```

Every run writes `metrics.json`, trained models, scalers, and per-split
predictions to `runs/<tag>/`.

## Benchmarks

SURGE is built for the fusion energy sciences community: its benchmark
suite leads with public fusion datasets (QLKNN10D turbulent transport,
ConStellaration stellarator equilibria, SOLPS-NN divertor exhaust, the
C-Mod density-limit database), followed by operator-learning fields
(PDEBench, TheWell) and a tutorial tier of general tabular and vision
datasets (California housing, superconductor critical temperature,
MNIST/CIFAR-10, …) you can pull and run in a minute. Every benchmark has a
citation, a published pass threshold and machine-readable `result.json`
artifacts. Headline numbers from the fusion benchmarks:

- **QLKNN10D plasma transport** — R² 0.989 on 200k held-out QuaLiKiz ITG
  fluxes, reproduced end-to-end from public data (gate R² ≥ 0.95).
- **ConStellaration stellarator design** — mean R² 0.916 across 8 VMEC++
  equilibrium metrics (gate R² ≥ 0.85).
- **C-Mod density-limit precursor detection** — discharge-grouped split of the
  MIT-PSFC open database on the natural 1.4 % imbalance: random forest AUROC
  0.9999 against 0.973 for the Greenwald-fraction scaling (gate AUROC ≥ 0.99).
- **SOLPS-NN divertor exhaust** (in progress) — an in-context tabular
  foundation model (`kumo.tabular_regressor`) reaches 9.3 % median relative
  error on the peak target heat flux with no training, against the
  published 13 % from tuned five-fold networks.

Full ledger with per-benchmark status: [docs/benchmarks/README.md](https://github.com/asvillar/SURGE/blob/v0.1.0/docs/benchmarks/README.md).

![QLKNN10D reproduction: residual MLP R2 0.989 on held-out QuaLiKiz ITG fluxes, parity density plot](https://raw.githubusercontent.com/asvillar/SURGE/v0.1.0/docs/assets/readme/qlknn10d.png)

## Learn more

- [Full README](https://github.com/asvillar/SURGE/blob/v0.1.0/README.md) — architecture, highlights, visualization system, all 40+ model adapters
- [Getting started](https://github.com/asvillar/SURGE/blob/v0.1.0/docs/GETTING_STARTED.md)
- [Benchmark ledger](https://github.com/asvillar/SURGE/blob/v0.1.0/docs/benchmarks/README.md) — status, gates and data citations per benchmark
- [References](https://github.com/asvillar/SURGE/blob/v0.1.0/README.md#references) — the datasets, architectures and software behind the suite
- [Changelog](https://github.com/asvillar/SURGE/blob/v0.1.0/CHANGELOG.md)
- [Citation](https://github.com/asvillar/SURGE/blob/v0.1.0/CITATION.cff) — DOE CODE 179819, DOI [10.11578/dc.20260422.5](https://doi.org/10.11578/dc.20260422.5)

## License

BSD-3-Clause — see [LICENSE](https://github.com/asvillar/SURGE/blob/v0.1.0/LICENSE) and [NOTICE](https://github.com/asvillar/SURGE/blob/v0.1.0/NOTICE).
