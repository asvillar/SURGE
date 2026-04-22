# `scripts/` — optional and application-specific

Most **reusable** library code lives under `surge/`. This folder holds **ad hoc**
utilities (batch jobs, MHD-code helpers, HPC launchers) that are **not** part of
the importable `surge` API.

- **MHD-code / device** batch pipelines and interfaces live under `scripts/mhd_code/`.
- **kinetic-code-oriented** one-off scripts were removed from the public-release tree;
  develop those privately or in a separate application repo if they contain
  collaboration-specific paths or data assumptions.

For installable, documented behavior, use **`surge`** from Python or the
**YAML workflow** (`run_surrogate_workflow`). See `docs/SURGE_OVERVIEW.md`.
