# `scripts/` — optional utilities

Reusable library code lives under `surge/`. This folder holds small helper
scripts that are **not** part of the importable `surge` API:

- `benchmark_scale.py` — measure the scaling levers (device, threads, batch
  fan-out) and cache the numbers behind the scale figure.
- `plot_training.py`, `utils/show_surge_training_progress.py` — live/offline
  training-loss viewers for logs written with `surge bench --train-log-file`.
- `utils/check_installation.py` — print the surge version and which optional backends import.

Facility-specific batch pipelines and launchers (Slurm) are kept out
of the public tree; use `surge` from Python or the YAML workflow for
installable, documented behaviour.
