# Changelog

All notable changes to SURGE (`surge-ml`). Follows
[Keep a Changelog](https://keepachangelog.com); versions follow PEP 440.

## [0.1.0] — 2026-07 (first public PyPI release)

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
  [--verbose]`, `surge report` (HTML leaderboard), `surge version`.
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
- **Packaging**: PyPI-ready sdist/wheel (`uv build`), benchmark metadata
  shipped as package data, Python 3.10–3.12.

### Changed

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
- Editor-specific tooling (editor canvas dashboards and refresh scripts);
  the leaderboard is now artifact-driven HTML.
- Generated artifacts from version control (MLflow state, benchmark
  report trees, backup files, tooling demos).

[0.1.0]: https://github.com/S-Villar/SURGE/releases/tag/v0.1.0
