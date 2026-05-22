"""
SURGE — PDE / Operator Learning Leaderboard Example
=====================================================
Runs FNO and baseline models against 1D and 2D PDE benchmarks.

Usage
-----
    # Inline Burgers 1D (no download):
    python examples/leaderboard_pde.py --quick

    # Full PDEBench suite (requires HDF5 data download ~GB):
    python examples/leaderboard_pde.py --full

    # Via the surge CLI:
    surge run --category pde -m all

Benchmarks
----------
pde.burgers_1d      — Inline 1D Burgers solver (n=1024, 64-pt grid, no download)
pdebench.burgers_1d — PDEBench 1D Burgers ν=0.01 (n=9000, 1024-pt grid)
pdebench.darcy_2d   — PDEBench 2D Darcy Flow (n=10000, 128×128 grid)
pdebench.shallow_water_2d — PDEBench 2D Shallow Water (n=1000, 128×128)

Reference: Takamoto et al. (2022) "PDEBench: An Extensive Benchmark for
Scientific Machine Learning" NeurIPS 2022. arXiv:2210.07182.
"""

import argparse

from surge.benchmarks.leaderboard import print_leaderboard, run_leaderboard

INLINE_BENCHMARKS = ["pde.burgers_1d"]
FULL_BENCHMARKS = [
    "pde.burgers_1d",
    "pdebench.burgers_1d",
    "pdebench.darcy_2d",
    "pdebench.shallow_water_2d",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="Inline Burgers only (no download), 5 epochs")
    ap.add_argument("--full", action="store_true",
                    help="Include PDEBench datasets (requires HDF5 download)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    keys = FULL_BENCHMARKS if args.full else INLINE_BENCHMARKS
    epochs = args.epochs or (5 if args.quick else 50)

    print(f"\nSURGE PDE / Operator Learning Leaderboard")
    print(f"Benchmarks  : {', '.join(keys)}")
    print(f"Epochs: {epochs}  |  seeds: {args.seeds}\n")

    results = run_leaderboard(
        benchmark_keys=keys,
        pytorch_mlp_epochs=epochs,
        n_seeds=args.seeds,
    )
    print_leaderboard(results)


if __name__ == "__main__":
    main()
