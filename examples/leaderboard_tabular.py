"""
SURGE — Tabular Regression Leaderboard Example
===============================================
Runs all compatible models against the CTR-23 tabular regression suite
and prints a per-benchmark leaderboard table.

Usage
-----
    # Quick run (2 epochs for PyTorch models):
    python examples/leaderboard_tabular.py --quick

    # Full run (default 50 epochs):
    python examples/leaderboard_tabular.py

    # Via the surge CLI:
    surge run --category tabular --task-type regression -m all

Benchmarks
----------
ctr23.abalone         — 4k samples, 8 features
ctr23.bike_sharing    — 17k samples, 12 features
ctr23.diamonds        — 15k samples, 9 features
ctr23.house_sales     — 15k samples, 19 features
ctr23.brazilian_houses — 10k samples, 11 features

Reference: Grinsztajn et al. (2022) "Why tree-based models still outperform
deep learning on tabular data" arXiv:2207.08815.
"""

import argparse

from surge.benchmarks.leaderboard import print_leaderboard, run_leaderboard

CTR23_BENCHMARKS = [
    "ctr23.abalone",
    "ctr23.bike_sharing",
    "ctr23.diamonds",
    "ctr23.house_sales",
    "ctr23.brazilian_houses",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="Run with 5 epochs (fast smoke check)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of random seeds (reports mean ± std). Default: 1")
    args = ap.parse_args()

    epochs = 5 if args.quick else 50
    print(f"\nSURGE Tabular Regression Leaderboard")
    print(f"Benchmarks  : {', '.join(CTR23_BENCHMARKS)}")
    print(f"PyTorch epochs: {epochs}  |  seeds: {args.seeds}\n")

    results = run_leaderboard(
        benchmark_keys=CTR23_BENCHMARKS,
        pytorch_mlp_epochs=epochs,
        n_seeds=args.seeds,
    )
    print_leaderboard(results)


if __name__ == "__main__":
    main()
