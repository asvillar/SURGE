# SURGE — development policy for AI-assisted work

## Integrate, don't script

Every capability needed for a task is built into the package and then used:

- new physics or data helpers → a module under `surge/` (e.g. `surge/physics/`,
  `surge/benchmarks/`), imported by benchmarks, workflows and figures alike;
- new benchmark behaviour → registry entry + `metadata.yaml` + dataset spec, run
  through `surge bench`;
- new training/HPO behaviour → a workflow spec option or CLI flag, never a driver
  that re-implements the workflow;
- new figures → a function in `surge/viz/` reachable from `examples/viz_theme_gallery.py`
  and `surge report`;
- every feature ships with a test under `tests/`.

Standalone scripts are allowed only as thin drivers that call package APIs, or as
throwaway checks under `audit/` (untracked). If finishing a task seems to require an
ad-hoc script that duplicates package logic, the missing package feature is the task.

## Benchmarks are proposals

Changes to benchmark definitions, thresholds, datasets, model inputs or training
recipes are proposals for human review before they are committed. Status docs,
reports and figures regenerated from existing artifacts are autonomous.

## Skills

Project skills live in `.claude/skills/` (`surge-run`, `surge-benchmark`, `surge-viz`,
`surge-add-model`, `surge-build-surrogate`, `surge-scale`, `surge-wizard`). Read the
relevant one before touching its area; `surge-viz` is mandatory before any plotting code.
