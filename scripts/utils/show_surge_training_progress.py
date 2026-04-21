#!/usr/bin/env python3
"""Inspect per-epoch training metrics from a SURGE run (PyTorch MLP).

Examples
--------
  # Live (same as: tail -f .../training_progress_<model>.jsonl)
  python scripts/utils/show_surge_training_progress.py runs/my_run --follow

  # Last 15 epochs from JSONL
  python scripts/utils/show_surge_training_progress.py runs/my_run -n 15

  # Summary from training_history_*.json (written after training completes)
  python scripts/utils/show_surge_training_progress.py runs/my_run --json

  # HPO runs (explicit file)
  python scripts/utils/show_surge_training_progress.py runs/my_run --follow --hpo-current
  python scripts/utils/show_surge_training_progress.py runs/my_run --follow --hpo-best
  python scripts/utils/show_surge_training_progress.py runs/my_run --follow --hpo-all

Looks for:
  runs/<tag>/training_progress_*.jsonl  (one JSON object per line, flushed each epoch)
  runs/<tag>/training_progress_hpo_{current,all,best}.jsonl  (during/after Optuna HPO)
  runs/<tag>/training_history_*.json   (full list, after a full train step finishes)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _find_progress_files(run_dir: Path) -> Tuple[List[Path], List[Path]]:
    jsonl = sorted(run_dir.glob("training_progress_*.jsonl"))
    hist = sorted(run_dir.glob("training_history_*.json"))
    return jsonl, hist


def _pick_jsonl(jsonl: List[Path]) -> Path:
    """Prefer the most recently modified file when multiple matches (e.g. HPO vs final train)."""
    if not jsonl:
        raise ValueError("empty jsonl list")
    if len(jsonl) == 1:
        return jsonl[0]
    return max(jsonl, key=lambda p: p.stat().st_mtime)


def _print_line(obj: dict) -> None:
    ep = obj.get("epoch", "?")
    tr = obj.get("train_rmse_scaled")
    vl = obj.get("val_rmse_scaled")
    tl = obj.get("train_loss")
    trial = obj.get("hpo_trial")
    prefix = f"trial {trial}  " if trial is not None else ""
    if vl is not None:
        print(f"{prefix}epoch {ep:>3}  train_RMSE={tr}  val_RMSE={vl}  train_loss={tl}")
    else:
        print(f"{prefix}epoch {ep:>3}  train_RMSE={tr}  train_loss={tl}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("Examples")[0].strip())
    p.add_argument("run_dir", type=Path, help="SURGE run directory (e.g. runs/<run_tag>)")
    p.add_argument("-n", type=int, default=20, help="Show last N lines of JSONL (default 20)")
    p.add_argument(
        "--json",
        action="store_true",
        help="Load training_history_*.json instead of JSONL",
    )
    p.add_argument(
        "--follow",
        action="store_true",
        help="Follow JSONL growth while training (Ctrl+C to stop)",
    )
    p.add_argument(
        "--hpo-current",
        action="store_true",
        help="Use training_progress_hpo_current.jsonl (live epochs for the active HPO trial)",
    )
    p.add_argument(
        "--hpo-best",
        action="store_true",
        help="Use training_progress_hpo_best.jsonl (epochs for the best HPO trial so far)",
    )
    p.add_argument(
        "--hpo-all",
        action="store_true",
        help="Use training_progress_hpo_all.jsonl (all trials; each line has hpo_trial)",
    )
    args = p.parse_args()
    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1

    hpo_flags = int(args.hpo_current) + int(args.hpo_best) + int(args.hpo_all)
    if hpo_flags > 1:
        print("Choose at most one of --hpo-current, --hpo-best, --hpo-all", file=sys.stderr)
        return 1

    if args.hpo_current:
        jsonl_files = [run_dir / "training_progress_hpo_current.jsonl"]
        hist_files = []
    elif args.hpo_best:
        jsonl_files = [run_dir / "training_progress_hpo_best.jsonl"]
        hist_files = []
    elif args.hpo_all:
        jsonl_files = [run_dir / "training_progress_hpo_all.jsonl"]
        hist_files = []
    else:
        jsonl_files, hist_files = _find_progress_files(run_dir)

    if args.json:
        if not hist_files:
            print("No training_history_*.json found.", file=sys.stderr)
            return 1
        path = hist_files[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"# {path.name} ({len(data)} epochs)")
        for row in data[-args.n :]:
            _print_line(row)
        return 0

    if not jsonl_files:
        if not hist_files:
            print(
                "No training_progress_*.jsonl or training_history_*.json found.\n"
                "Typical causes: wrong run_dir, or PyTorch training has not started yet.",
                file=sys.stderr,
            )
            return 1
        path = hist_files[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"# {path.name} ({len(data)} epochs)")
        for row in data[-args.n :]:
            _print_line(row)
        return 0

    jpath = jsonl_files[0]
    if not jpath.is_file():
        if not hist_files:
            hint = ""
            if hpo_flags:
                hint = (
                    f"\nExpected {jpath.name} — start an HPO run with a current SURGE workflow, "
                    "or wait until the first training epoch is written."
                )
            print(
                "No usable training_progress JSONL yet.\n"
                "Wait for the first epoch, use `tail -f` on the Slurm log, or try "
                "training_history_*.json after a train step completes."
                + hint,
                file=sys.stderr,
            )
            return 1
        path = hist_files[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"# {path.name} ({len(data)} epochs)")
        for row in data[-args.n :]:
            _print_line(row)
        return 0

    path = _pick_jsonl(jsonl_files)
    if len(jsonl_files) > 1:
        print(f"# Using {path.name} (newest of {len(jsonl_files)} training_progress_*.jsonl)\n")

    if args.follow:
        print(f"# tail -f {path}\n")
        with path.open(encoding="utf-8") as handle:
            handle.seek(0, 2)
            while True:
                line = handle.readline()
                if not line:
                    time.sleep(1.0)
                    continue
                try:
                    _print_line(json.loads(line))
                except json.JSONDecodeError:
                    print(line.rstrip())
        return 0

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    tail = lines[-args.n :] if args.n > 0 else lines
    print(f"# {path.name} (showing {len(tail)}/{len(lines)} lines)")
    for line in tail:
        if not line.strip():
            continue
        try:
            _print_line(json.loads(line))
        except json.JSONDecodeError:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
