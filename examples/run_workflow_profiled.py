"""
Run a SURGE surrogate workflow with cProfile enabled.

Saves profile to runs/<run_tag>/workflow.prof and prints a top-N summary by
cumulative time so you can see where time is spent (e.g. torch_mlp fit, data loading).

Note: Profiling adds overhead (often 2–5x slower per trial). If the run looks stuck
at "Trial 0 started...", wait a few minutes or run the same config without profiling
first to confirm trials complete: surge run spec.yaml

Usage:
  conda activate surge
  python -m examples.run_workflow_profiled --spec spec.yaml --run-tag debug_5trials_profiled

Then inspect:
  python -c "import pstats; p=pstats.Stats('runs/debug_5trials_profiled/workflow.prof'); p.sort_stats('cumulative'); p.print_stats(40)"
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

try:
    import yaml
except ImportError as e:
    raise SystemExit("PyYAML is required.") from e

import surge  # noqa: F401
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec


def parse_args():
    p = argparse.ArgumentParser(description="Run SURGE workflow with cProfile.")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--top", type=int, default=40, help="Number of top functions to print in summary")
    return p.parse_args()


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    with args.spec.open("r", encoding="utf-8") as f:
        spec = SurrogateWorkflowSpec.from_dict(yaml.safe_load(f))
    if args.run_tag:
        spec.run_tag = args.run_tag
    if args.output_dir:
        spec.output_dir = str(args.output_dir)

    run_tag = spec.run_tag or "profiled_run"
    profile_path = Path(spec.output_dir or ".") / "runs" / run_tag / "workflow.prof"
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Profiling workflow (run_tag={run_tag}). Profile will be saved to {profile_path}", flush=True)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        summary = run_surrogate_workflow(spec)
        print("\nWorkflow completed.", flush=True)
    finally:
        profiler.disable()
        profiler.dump_stats(str(profile_path))

    # Print summary by cumulative time
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(args.top)
    summary_txt = stream.getvalue()
    print("\n--- Profile (top {} by cumulative time) ---\n{}".format(args.top, summary_txt), flush=True)

    # Save summary to run dir
    summary_path = profile_path.parent / "profile_summary.txt"
    summary_path.write_text(summary_txt, encoding="utf-8")
    print(f"Profile summary saved to {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
