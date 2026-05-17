# Benchmark policy (SURGE)

This repo ships **CPU-only, sklearn-based** standard tasks (Tier 0–1) under
`surge/benchmarks/`. Larger tiers (vision, PDE, ImageNet) are described in
[benchmark plan.md](../../benchmark plan.md) and are not
required for CI.

## Tiers (summary)

| Tier | Intent | CI |
|------|--------|-----|
| **0** | Hermetic smoke: synthetic data, no downloads | Recommended |
| **1** | Tabular classics via `sklearn.datasets` | Optional / cached runners |
| **2+** | Vision, fusion, HPC | Manual or scheduled workflows |

## Pass/fail

Each task returns `BenchmarkResult.passed` using conservative thresholds
documented in `surge/benchmarks/tasks.py`. Adjust thresholds when Swarming CI
hardware is consistently slower or noisier.

## Commands

```bash
PYTHONPATH=. python -m surge.benchmarks.run --list
PYTHONPATH=. python -m surge.benchmarks.run --benchmark synthetic.regression_1d
PYTHONPATH=. python -m pytest tests/benchmarks/test_smoke_benchmarks.py -q
```

Artifact JSON (`--output`) is intended for `benchmark_reports/` in human-run jobs.
