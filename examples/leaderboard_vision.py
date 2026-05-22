"""
SURGE — Vision Leaderboard Example
====================================
Runs all vision models against CIFAR-10 and/or MNIST and prints leaderboard tables.

Usage
-----
    # CIFAR-10 (5 epochs — smoke check):
    python examples/leaderboard_vision.py --quick

    # MNIST only:
    python examples/leaderboard_vision.py --benchmarks mnist --quick

    # Both, 30 epochs:
    python examples/leaderboard_vision.py --epochs 30

    # Via the surge CLI:
    surge run -b cifar10 mnist -m all

Models
------
pytorch.alexnet   — AlexNet (Krizhevsky et al. NeurIPS 2012)
pytorch.resnet20  — ResNet-20 (He et al. CVPR 2016)  ~91.3% CIFAR-10
pytorch.resnet56  — ResNet-56 (He et al. CVPR 2016)  ~93.0% CIFAR-10
pytorch.vit       — Vision Transformer (Dosovitskiy et al. ICLR 2021)
pytorch.lenet5    — LeNet-5 (LeCun et al. 1998)  ~99.2% MNIST
"""

import argparse

from surge.benchmarks.leaderboard import print_leaderboard, run_leaderboard
from surge.benchmarks.registry import resolve_benchmark_key

ALL_VISION = ["vision.cifar10", "vision.mnist"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    metavar="KEY",
                    help="Benchmark keys or short aliases (default: cifar10 mnist).")
    ap.add_argument("--quick", action="store_true",
                    help="Run with 5 epochs (fast smoke check)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override number of training epochs.")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    keys = (
        [resolve_benchmark_key(k) for k in args.benchmarks]
        if args.benchmarks
        else ALL_VISION
    )
    epochs = args.epochs or (5 if args.quick else 30)

    print(f"\nSURGE Vision Leaderboard")
    print(f"Benchmarks  : {', '.join(keys)}")
    print(f"Epochs: {epochs}  |  seeds: {args.seeds}\n")
    print("Note: paper-level accuracy requires 100+ epochs on a GPU.")
    print("This example uses a reduced epoch count for demonstration.\n")

    results = run_leaderboard(
        benchmark_keys=keys,
        pytorch_mlp_epochs=epochs,
        n_seeds=args.seeds,
    )
    print_leaderboard(results)


if __name__ == "__main__":
    main()
