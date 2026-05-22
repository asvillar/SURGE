"""Top-level ``surge`` CLI dispatcher.

Subcommands
-----------
surge run      — benchmark runner (train + evaluate models)
surge list     — alias for ``surge run --list``
surge models   — alias for ``surge run --list-models``

All arguments after the subcommand are forwarded to the relevant main().

Examples
--------
::

    surge run --list
    surge run -b cifar10 -m all
    surge run -b qlknn constellaration -m all --seeds 3
    surge list
    surge list --category plasma
    surge models
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if not argv:
        # No subcommand — print brief help and forward to benchmark runner help.
        print("SURGE — Surrogate modelling framework\n")
        print("Usage: surge <subcommand> [options]\n")
        print("Subcommands:")
        print("  run      Run benchmarks and leaderboards")
        print("  list     List available benchmarks  (alias: surge run --list)")
        print("  models   List available models      (alias: surge run --list-models)")
        print()
        print("Run 'surge run --help' for full options.")
        return 0

    sub = argv[0]

    if sub in ("run",):
        from surge.benchmarks.run import main as run_main
        return run_main(argv[1:])

    if sub in ("list", "ls"):
        from surge.benchmarks.run import main as run_main
        return run_main(["--list"] + argv[1:])

    if sub in ("models",):
        from surge.benchmarks.run import main as run_main
        return run_main(["--list-models"] + argv[1:])

    # Unknown subcommand — try forwarding everything to benchmark runner so
    # that ``surge --list``, ``surge -b iris``, etc. still work for users who
    # skip the subcommand.
    from surge.benchmarks.run import main as run_main
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
