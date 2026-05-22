"""
SURGE — Plasma / Fusion Leaderboard Example
============================================
Runs all compatible models against the DOE-relevant plasma physics benchmarks
and prints a per-benchmark leaderboard table.

Usage
-----
    # All plasma benchmarks (downloads data on first run):
    python examples/leaderboard_plasma.py

    # Only the QLKNN transport benchmark:
    python examples/leaderboard_plasma.py --benchmarks qlknn

    # ConStellaration paper-exact reproduction:
    python examples/leaderboard_plasma.py --benchmarks constellaration_paper

    # Via the surge CLI:
    surge run --category plasma -m all

Benchmarks
----------
plasma.qlknn_transport
    QuaLiKiz/QLKNN turbulent electron heat flux surrogate.
    10 gyrokinetic parameters → efeITG (heat flux).
    Reference: van de Plassche et al. Nuclear Fusion 60 (2020) 066019.
    Requires: pip install fusion_surrogates

plasma.constellaration
    ConStellaration stellarator boundary shape → quasi-isodynamic quality.
    90-dimensional shape → QI metric (R² target > 0.97).
    Reference: Goodman et al. (2025) arXiv:2506.19583.
    Requires: pip install datasets

plasma.constellaration_paper
    Paper-exact reproduction: 12 metrics, per-metric training, 23k samples,
    0.05% outlier removal, log10(qi), Z-score normalisation.
    Paper baseline: mean R² > 0.97 with 10-member MLP ensemble.

plasma.cmod_density_limit
    Alcator C-Mod density limit disruption classification.
    6 plasma signals → binary label (disruption / no disruption).
    Requires: MIT-PSFC GitHub data (auto-downloaded).
"""

import argparse

from surge.benchmarks.leaderboard import print_leaderboard, run_leaderboard
from surge.benchmarks.registry import resolve_benchmark_key

ALL_PLASMA = [
    "plasma.qlknn_transport",
    "plasma.constellaration",
    "plasma.cmod_density_limit",
]

PAPER_BENCHMARK = ["plasma.constellaration_paper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--benchmarks", nargs="+", default=None,
        metavar="KEY",
        help="Benchmark keys or short aliases (default: all plasma benchmarks).",
    )
    ap.add_argument("--quick", action="store_true",
                    help="Run with 5 epochs (fast smoke check)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of random seeds. Default: 1")
    args = ap.parse_args()

    if args.benchmarks:
        keys = [resolve_benchmark_key(k) for k in args.benchmarks]
    else:
        keys = ALL_PLASMA

    epochs = 5 if args.quick else 50
    print(f"\nSURGE Plasma / Fusion Leaderboard")
    print(f"Benchmarks  : {', '.join(keys)}")
    print(f"PyTorch epochs: {epochs}  |  seeds: {args.seeds}\n")

    results = run_leaderboard(
        benchmark_keys=keys,
        pytorch_mlp_epochs=epochs,
        n_seeds=args.seeds,
    )
    print_leaderboard(results)


if __name__ == "__main__":
    main()
