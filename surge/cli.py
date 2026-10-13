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

    # Unknown subcommand — forward to the benchmark runner so legacy
    # invocations like ``surge --list`` or ``surge -b iris`` keep working.
    from surge.benchmarks.run import main as bench_main
    return bench_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
