# `scripts/` — optional utilities

Reusable library code lives under `surge/`. This folder holds small helper
scripts that are **not** part of the importable `surge` API:

- `benchmark_scale.py` — measure the scaling levers (device, threads, batch
  fan-out) and cache the numbers behind the scale figure.
- `dataset_explorer.py` — quick look at a dataset's columns and distributions.
- `extract_benchmark_metadata.py` — the one-off that seeded
  `surge/benchmarks/metadata.yaml`.
- `plot_training.py`, `utils/show_surge_training_progress.py` — live/offline
  training-loss viewers for a run.
- `utils/check_installation.py` — verify an install.

Facility-specific batch pipelines and launchers (M3D-C1, Slurm) are kept out
of the public tree; use `surge` from Python or the YAML workflow for
installable, documented behaviour. See `docs/SURGE_OVERVIEW.md`.
