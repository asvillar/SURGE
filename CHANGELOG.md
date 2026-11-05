# Changelog

All notable changes to SURGE (`surge-ml`). Follows
[Keep a Changelog](https://keepachangelog.com); versions follow PEP 440.

## [0.1.0] — 2026-10-30 (first public PyPI release)

### Added

- **Model portfolio**: 44-adapter registry across scikit-learn baselines,
  gradient boosting (XGBoost, LightGBM, CatBoost), PyTorch MLP families
  (MLP, residual, ensemble, classifier), operator learners (FNO 1D/2D,
  DeepONet, U-Net), sequence models (LSTM, GRU, CNN1D), vision models
  (LeNet-5, ResNet-20/56, AlexNet, ViT), tabular deep learning (KAN,
  FT-Transformer), generative baselines (VAE, DDPM, CGAN), Gaussian
  processes with uncertainty (sklearn GPR, BoTorch exact/sparse, GPflow),
  TabPFN, and TensorFlow/Keras (`keras.mlp`, with a `build_fn` hook for
  arbitrary user-compiled tf.keras architectures).
- **Transparent registration**: every adapter attempt is recorded as
  registered / skipped (with the true reason) / error; `surge models
  --verbose` shows the table; `SURGE_STRICT_REGISTRY=1` makes internal
  registration bugs fatal (enabled in CI). Broken native libraries
  (e.g. LightGBM without libomp) are reported at import, not at fit time.
- **`surge` CLI**: `surge run <spec.yaml>` (workflow execution),
  `surge bench` (benchmark runner), `surge list`, `surge models
  [--verbose]`, `surge report` (HTML leaderboard), `surge version`,
  `surge init` (interactive or non-interactive `--data --target --goal
  --budget --yes` spec wizard), and `surge validate <spec>` (offline
  schema check with did-you-mean suggestions, exit 0/2) against a
  workflow JSON Schema generated from the spec dataclasses and shipped
  as package data.
- **Benchmark suite**: 30+ curated benchmarks (UCI/OpenML tabular,
  MNIST/CIFAR-10, Burgers/PDEBench/TheWell operator tasks, Lorenz-63,
  QLKNN transport, ConStellaration stellarator protocols) with
  citations, tiers, and thresholds in `surge/benchmarks/metadata.yaml`;
  multi-seed leaderboards with mean ± std.
- **Visual system** (`surge.viz.theme`): colorblind-validated light/dark
  palette, signature reversed-plasma parity density style, reserved
  status colors, deterministic PNG/SVG/PDF export; publication figure
  gallery (`examples/viz_theme_gallery.py`) covering parity, training,
  HPO (starred bests), classification, field/operator diagnostics,
  GP uncertainty bands, and dataset characterization.
- **Reports**: self-contained HTML benchmark leaderboard
  (`surge.report.leaderboard`) with spider charts, per-benchmark dataset
  previews rendered offline from local caches, citations, and sortable
  mean ± std tables — generated exclusively from run artifacts.
- **Provenance artifacts** per run: spec snapshot, git revision,
  environment capture, scalers, per-split parquet predictions, model
  card, per-epoch training logs, HPO trial logs.
- **Fusion-first benchmark suite**: the ledger leads with the fusion
  benchmarks (QLKNN10D, ConStellaration and the re-posed C-Mod density
  limit ready; SOLPS-NN in progress); the
  general tabular datasets (California housing, concrete, energy
  efficiency, airfoil noise, superconductor) move to the tutorial tier as
  pull-and-run demos. Relative L2 is a headline ledger metric for the
  field benchmarks.
- **SOLPS-NN divertor exhaust** (`plasma.solpsnn_exhaust_scalars`,
  `plasma.solpsnn_te_field`; Dasbach & Wiesen 2023, Zenodo
  10.5281/zenodo.19237127, CC BY 4.0): loader that extracts only the
  needed members of the 17.6 GB record by HTTP byte ranges (~290 MB),
  fixed Sobol-train / random-test split, thesis metric `test_median_rel_err`
  with per-target keys, scenario-atlas and B2-grid field figures in
  `surge.viz.solpsnn`. The scalar benchmark is public in the in-progress
  tier; the field benchmark is hidden until its gate settles.
- **C-Mod density limit re-posed** (`plasma.cmod_density_limit`, MIT-PSFC
  Open Density Limit Database, CC BY): fixed discharge-grouped split on the
  natural class imbalance, scored by AUROC, average precision and TPR at
  1 %/5 % false-alarm rate with the Greenwald-fraction scaling as the
  physics baseline (0.973 AUROC on the same slices); random forest 0.9999,
  gradient boosting 0.998 against the 0.99 gate. Cards may declare a
  `primary_metric`. New `surge.viz.disruption.roc_figure`.
- **NVIDIA Kumo Tabular** in-context adapters (`kumo.tabular_regressor`,
  `kumo.tabular_classifier`, extra `kumo`): no training, up to 60k context
  rows under a rows×columns memory budget, 999-quantile uncertainty. First
  SURGE result where a tabular foundation model beats the published tuned
  networks: 8.4 % median relative error on the SOLPS-NN peak target heat
  flux against 13 %.
- `surge.viz.theme.field_cmap`: perceptual (inferno) colormap for fields
  spanning many decades.
- **DINOv2 frozen-backbone linear probe** (`pytorch.dino_linear_probe`):
  frozen DINOv2 ViT-S/14 (`torch.hub`) features plus a logistic-regression
  head, cached per content hash and extractor identity, labelled as
  transfer learning and reloadable through `load_model_compat`;
  `vision.cifar10` passes its accuracy gate (0.965 vs. 0.91) and is kept
  as a tutorial-tier plumbing benchmark (no surrogate-modelling content).
- **QLKNN10D** (`plasma.qlknn10d`): reproduces QLKNN from its public
  290M-row QuaLiKiz training table, residual MLP R² 0.9898 on 200k
  held-out fluxes (gate 0.95). `BENCH_MODEL_DEFAULTS` lets a benchmark
  declare the hyperparameters it needs at its data scale, so qlknn10d
  trains with its reproduction-study settings instead of registry
  defaults that ran 11h+ on CPU without finishing.
- **Benchmark status ledger**: every benchmark carries `status` /
  `status_note` (`ready | in_progress | needs_rerun | tutorial | sanity`);
  `surge bench --status-md` renders the ledger to Markdown.
- **Maintainer selection**: per-benchmark `visible` / `focus` fields and
  `surge bench --apply-selection PATH.json` to apply a reviewed selection
  in bulk.
- **Scientific-discipline taxonomy**: every benchmark is tagged with a
  discipline (Fusion & Plasma Physics, Nonlinear Dynamics & Chaos, Fluid
  Dynamics & PDE Operators, Materials Science, Engineering, Environmental
  Sciences, Biology & Medicine, Chemistry, Operations Research, Computer
  Vision), surfaced as a filter and sortable column in the index, cards,
  and ledger.
- **Benchmark index and report upgrades**: a searchable, filterable
  index (name, key, category, discipline, model) with sortable columns;
  a per-benchmark task explainer, published-in citation, and
  state-of-the-art reference line; held-out predictions saved per run
  (`predictions.npz`, 20k-row cap, `--no-predictions` to skip) and
  rendered on the card (parity density / confusion matrix / field
  triptych); `surge report --layout full|status` (full expands every
  visible benchmark with its category spider chart; status keeps the
  collapsed default view).
- **Demo video and campaign replay**: the end-to-end animation and a
  scrubbable HPO / mission-control campaign replay (`surge.viz.animate`)
  embed in the report for benchmarks run with `--demo-run KEY=runs/dir`.
- **Autoregressive rollout API** (`surge.inference.rollout`): shared
  rollout, valid-prediction-time, and climate-error helpers used by the
  Lorenz-63 and Gray-Scott studies.
- **Streaming training path**: `fit_from_loader` plus lazy TheWell pair
  loading for the ConvNeXt U-Net backend, for datasets too large to
  materialize in memory.
- **POD / reduced-order helpers**: `pod_fit` / `pod_transform` /
  `pod_inverse` in `surge.preprocessing`.
- **Simformer** (`pytorch.simformer`): all-in-one simulation-based
  inference, validated against the analytic linear-Gaussian posterior.
- **ConvNeXt U-Net** backend, used for probabilistic turbulence
  forecasting (CRPS).
- **Packaging**: PyPI-ready sdist/wheel (`uv build`), benchmark metadata
  shipped as package data, Python 3.10–3.12, a PyPI-safe long description
  (`README_PYPI.md`), a `SURGE_DATA_DIR` data-root resolver with a
  `~/.cache/surge` fallback for installed wheels, an sdist that no longer
  ships `tests/`, `docs/`, or `examples/`, an `all` extra, and a single
  version source (`surge/__init__.__version__`).

### Changed

- **Benchmark honesty review**: closed-form / linear-identity results no
  longer count toward headline surrogate claims.
  `plasma.constellaration_multioutput` is re-scoped 90→8 trained metrics —
  the closed-form aspect ratio, max elongation, average triangularity, and
  the derived ι ratio leave the trained average and are instead computed
  analytically (`loaders/constellaration.geometry_metrics`, exact to
  1e-15) and exposed as optional model inputs via `surge bench
  --input-features stellarator_geometry`; a geometry-sanity warning fires
  if a closed-form target ever scores below R² 0.99.
- **Lorenz-63** (`sequence.lorenz63`) is rescored from a windowed 20-step
  sanity check to a one-step surrogate judged by a 10-Lyapunov-time
  autoregressive rollout from 100 held-out initial conditions: the
  primary metric is valid prediction time (median Lyapunov times until
  the climate-normalised error exceeds 0.4), plus attractor-climate error
  and one-step NRMSE. The tier-0 CLI test now accepts the benchmark's
  honest non-zero exit when every current model fails the provisional
  VPT ≥ 2λ gate. The benchmark stays `in_progress` and hidden from the
  public report and index until a model passes.
- Single model registry (`surge.model.registry`); workflow summaries now
  record the live registry (previously an empty list from a duplicate,
  never-populated registry).
- Custom-adapter template registers into the live registry (previously a
  dead one, making user adapters unresolvable).
- Documentation: science-first README with real result figures, one-page
  Getting Started, uv-first environment guidance, release guide.

### Removed

- Legacy modules (`surge/legacy/`, `surge/models.py`,
  `surge/visualization.py`, duplicate `surge/registry.py`).
- Editor-specific tooling (Cursor canvas dashboards and refresh scripts);
  the leaderboard is now artifact-driven HTML.
- Generated artifacts from version control (MLflow state, benchmark
  report trees, backup files, tooling demos).

[Unreleased]: https://github.com/asvillar/SURGE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/asvillar/SURGE/releases/tag/v0.1.0
