"""Top-level ``surge`` CLI dispatcher.

Subcommands
-----------
surge init              — interactive wizard: inspect data, write a
                          commented spec.yaml (also non-interactive via
                          --data/--target/--goal/--budget --yes)
surge validate <spec>   — schema-check a spec without running it
surge run <spec.yaml>   — execute a surrogate workflow from a YAML spec
surge bench …           — benchmark runner (train + evaluate on benchmarks)
surge list              — list available benchmarks
surge models            — list registered models (``--verbose``: show WHY
                          optional adapters were skipped)
surge report            — build the self-contained HTML leaderboard
surge viz <run_dir>     — render a run's figures (mission_control,
                          hpo_convergence, training_curves, parity) to
                          <run_dir>/figures/ (--out, --modes, --only, --model,
                          --units)
surge monitor <run_dir> — text status of a running/finished campaign from its
                          artifacts (trials, best, test metrics); --watch N
                          redraws every N seconds until metrics.json appears
surge data list         — benchmarks that have a dataset loader
surge data export <benchmark> <out.csv|out.parquet> [--split train|test|all]
                        — write a benchmark's dataset with named columns
surge version           — package version

Back-compat: ``surge run -b <benchmark> …`` (benchmark flags instead of a
YAML path) still forwards to the benchmark runner.

Examples
--------
::

    surge run examples/configs/qlknn_multi_hpo.yaml
    surge bench -b tabular.california_housing -m all --seeds 5
    surge models --verbose
    surge report --out leaderboard.html
    surge viz runs/qlknn_hpo_40 --modes light dark
    surge monitor runs/qlknn_hpo_40 --watch 10
    surge data export plasma.qlknn_transport qlknn.csv
"""

from __future__ import annotations

import sys
from pathlib import Path


def _workflow_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="surge run",
        description="Run a surrogate workflow from a YAML spec "
                    "(load -> split -> train -> HPO -> metrics -> artifacts).")
    parser.add_argument("spec", help="Path to a workflow spec YAML file")
    parser.add_argument("--tag", help="Override the run tag (runs/<tag>/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing run directory")
    args = parser.parse_args(argv)

    import yaml

    from surge.workflow.run import run_surrogate_workflow
    from surge.workflow.spec import SurrogateWorkflowSpec

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        print(f"error: spec file not found: {spec_path}\n"
              f"       starter specs live in examples/configs/ "
              f"(e.g. qlknn_multi_hpo.yaml)", file=sys.stderr)
        return 2
    payload = yaml.safe_load(spec_path.read_text())
    if args.tag:
        payload["run_tag"] = args.tag
    if args.overwrite:
        payload["overwrite_existing_run"] = True
    spec = SurrogateWorkflowSpec.from_dict(payload)

    result = run_surrogate_workflow(
        spec, invocation={"argv": ["surge", "run", *argv],
                          "config_source": str(Path(args.spec).resolve())})

    print(flush=True)
    for entry in result.get("models") or []:
        name = entry.get("name", "?")
        test = (entry.get("metrics") or {}).get("test") or {}
        parts = "  ".join(f"{k}={v:.4g}" for k, v in test.items()
                          if isinstance(v, (int, float)))
        if parts:
            print(f"  {name:32s} test: {parts}", flush=True)
    run_dir = (result.get("artifacts") or {}).get("root")
    if run_dir:
        print(f"\nArtifacts: {run_dir}", flush=True)
    return 0


def _models_main(argv: list[str]) -> int:
    verbose = "--verbose" in argv or "-v" in argv
    from surge.model import list_models, registration_table

    if verbose:
        print(registration_table())
        return 0
    for key, cls in sorted(list_models().items()):
        print(f"  {key:36s}{cls}")
    print("\nUse 'surge models --verbose' to see skipped adapters and why.")
    return 0


def _report_main(argv: list[str]) -> int:
    from surge.report.leaderboard import main as report_main

    sys.argv = ["surge report", *argv]
    report_main()
    return 0


def _viz_main(argv: list[str]) -> int:
    import argparse

    from surge.viz.campaign import FIGURE_NAMES, render_run

    parser = argparse.ArgumentParser(
        prog="surge viz",
        description="Render the figures of a finished (or running) run directory.")
    parser.add_argument("run_dir", help="run directory (contains metrics.json)")
    parser.add_argument("--out", help="output directory (default <run_dir>/figures)")
    parser.add_argument("--modes", nargs="+", default=["light"],
                        choices=["light", "dark"])
    parser.add_argument("--only", nargs="+", choices=FIGURE_NAMES,
                        help="subset of figures")
    parser.add_argument("--model", help="model name (default: the HPO model, "
                                        "else the first model in metrics.json)")
    parser.add_argument("--units", default="a.u.",
                        help="target units for axis labels (default a.u.)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "metrics.json").is_file():
        print(f"error: {run_dir} has no metrics.json -- not a finished run "
              f"directory (use 'surge monitor' for a run in progress)",
              file=sys.stderr)
        return 2
    written = render_run(run_dir, out_dir=args.out, modes=tuple(args.modes),
                         only=args.only, model_name=args.model, units=args.units)
    if not written:
        print("no figures could be rendered (artifacts missing)", file=sys.stderr)
        return 1
    for path in written.values():
        print(f"wrote {path}")
    return 0


def _monitor_main(argv: list[str]) -> int:
    import argparse

    from surge.workflow.monitor import monitor

    parser = argparse.ArgumentParser(
        prog="surge monitor",
        description="Text status of a running or finished campaign.")
    parser.add_argument("run_dir", help="run directory")
    parser.add_argument("--watch", type=float, metavar="SECONDS",
                        help="redraw every SECONDS until metrics.json appears")
    args = parser.parse_args(argv)
    if not Path(args.run_dir).is_dir():
        print(f"error: run directory not found: {args.run_dir}", file=sys.stderr)
        return 2
    return monitor(args.run_dir, watch=args.watch)


def _data_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="surge data", description="Benchmark dataset helpers.")
    sub = parser.add_subparsers(dest="verb", required=True)
    sub.add_parser("list", help="benchmarks that have a dataset loader")
    exp = sub.add_parser("export", help="write a benchmark dataset to CSV/Parquet")
    exp.add_argument("benchmark", help="benchmark key or short alias")
    exp.add_argument("out", help="output path (.csv or .parquet)")
    exp.add_argument("--split", choices=["train", "test", "all"], default="all",
                     help="fixed-split benchmarks only (default all)")
    args = parser.parse_args(argv)

    if args.verb == "list":
        from surge.benchmarks.leaderboard import dataset_loader_keys
        from surge.benchmarks.run import _print_list
        _print_list(verbose=True, include_smoke=True,
                    only_keys=dataset_loader_keys())
        return 0

    from surge.benchmarks.export import export_benchmark
    try:
        df = export_benchmark(args.benchmark, args.out, split=args.split)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - loader/cache failures
        print(f"error: could not load {args.benchmark}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}  shape={df.shape[0]} x {df.shape[1]}")
    print("columns: " + ", ".join(df.columns))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.split("Examples")[0].strip())
        print("\nRun 'surge <subcommand> --help' for full options.")
        return 0

    sub, rest = argv[0], argv[1:]

    if sub in ("version", "--version", "-V"):
        from surge import __version__
        print(f"surge-ml {__version__}")
        return 0

    if sub == "init":
        from surge.wizard import main as wizard_main
        return wizard_main(rest)

    if sub == "validate":
        import argparse as _ap
        p = _ap.ArgumentParser(prog="surge validate",
                               description="Schema-check a workflow spec "
                                           "without running it.")
        p.add_argument("spec", help="path to a spec YAML")
        a = p.parse_args(rest)
        from surge.workflow.schema import validate_file
        errors = validate_file(a.spec)
        if errors:
            print(f"INVALID — {len(errors)} problem(s):", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print(f"OK — {a.spec} is a valid SURGE workflow spec")
        return 0

    if sub == "run":
        # YAML path => workflow; anything else => benchmark runner (compat)
        if rest and rest[0].endswith((".yaml", ".yml")):
            return _workflow_main(rest)
        from surge.benchmarks.run import main as bench_main
        return bench_main(rest)

    if sub in ("bench", "benchmark"):
        from surge.benchmarks.run import main as bench_main
        return bench_main(rest)

    if sub in ("list", "ls"):
        from surge.benchmarks.run import main as bench_main
        return bench_main(["--list", *rest])

    if sub == "models":
        return _models_main(rest)

    if sub == "report":
        return _report_main(rest)

    if sub == "viz":
        return _viz_main(rest)

    if sub == "monitor":
        return _monitor_main(rest)

    if sub == "data":
        return _data_main(rest)

    # Unknown subcommand — forward to the benchmark runner so legacy
    # invocations like ``surge --list`` or ``surge -b iris`` keep working.
    from surge.benchmarks.run import main as bench_main
    return bench_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
