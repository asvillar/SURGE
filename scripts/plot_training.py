#!/usr/bin/env python
"""Live/offline training-loss viewer for SURGE leaderboard runs.

Usage examples
--------------
# List all available training log files
python scripts/plot_training.py --list

# Print a quick text summary of the latest epoch for every running model
python scripts/plot_training.py --status

# Plot loss curves for one file and save to PNG
python scripts/plot_training.py benchmark_reports/training_logs/vision_cifar10/pytorch_resnet56_seed42.jsonl

# Plot loss curves and open an interactive window (requires a display)
python scripts/plot_training.py --show benchmark_reports/training_logs/vision_mnist/pytorch_resnet20_seed42.jsonl

# Overlay val_loss from multiple runs on one chart
python scripts/plot_training.py --compare benchmark_reports/training_logs/plasma_constellaration_paper/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_logs(root: Path) -> list[Path]:
    """Return all *.jsonl files under root, sorted newest-first."""
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _read_epochs(path: Path) -> list[dict]:
    """Read a JSONL file, skip sentinel lines, return only epoch records."""
    records: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("__run_start__"):
            continue
        if "epoch" in obj:
            records.append(obj)
    return records


def _fmt_record(r: dict) -> str:
    parts = [f"ep={r['epoch']}"]
    for k in ("train_loss", "val_loss", "val_r2", "val_rmse_scaled", "lr"):
        v = r.get(k)
        if v is not None:
            parts.append(f"{k}={v:.5g}")
    if r.get("early_stop"):
        parts.append("early_stop=✓")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_list(root: Path) -> None:
    logs = _find_logs(root)
    if not logs:
        print(f"No training logs found under {root}")
        return
    print(f"Found {len(logs)} training log(s) under {root}:\n")
    for p in logs:
        records = _read_epochs(p)
        n = len(records)
        last = records[-1] if records else {}
        status = _fmt_record(last) if last else "(no epoch records yet)"
        age = os.path.getmtime(p)
        import time
        age_s = int(time.time() - age)
        print(f"  {p.relative_to(root.parent.parent) if root.parent.parent.exists() else p}")
        print(f"    epochs logged: {n}   last modified: {age_s}s ago")
        print(f"    latest: {status}\n")


def cmd_status(root: Path) -> None:
    logs = _find_logs(root)
    if not logs:
        print(f"No training logs found under {root}")
        return
    rows = []
    for p in logs:
        records = _read_epochs(p)
        if not records:
            continue
        last = records[-1]
        rows.append((p.stem, len(records), last))

    if not rows:
        print("All training log files are empty (training may not have started yet)")
        return

    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'model / seed':<{col_w}}  {'epochs':>6}  {'latest record'}"
    print(header)
    print("-" * len(header))
    for stem, n, last in rows:
        print(f"{stem:<{col_w}}  {n:>6}  {_fmt_record(last)}")


def cmd_plot(paths: list[Path], *, show: bool, log_scale: bool, save: Path | None) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    from surge.viz.training import load_training_history, plot_training_dashboard

    for src in paths:
        records = _read_epochs(src)
        if not records:
            print(f"[warn] {src}: no epoch records yet, skipping")
            continue
        out = save or src.with_suffix(".png")
        title = src.stem
        fig = plot_training_dashboard(records, model_name=title, save_path=None)
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved: {save}")
        else:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved: {out}")
        if show:
            plt.show()
        plt.close(fig)


def cmd_compare(directory: Path, *, metric: str, show: bool, save: Path | None) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    from surge.viz.training import compare_training_curves

    logs = _find_logs(directory)
    if not logs:
        print(f"No training logs found under {directory}")
        return
    histories = {}
    for p in logs:
        records = _read_epochs(p)
        if records:
            histories[p.stem] = records
    if not histories:
        print("All logs empty, nothing to compare")
        return
    out = save or (directory / f"compare_{metric}.png")
    fig = compare_training_curves(histories, metric=metric, title=f"{directory.name} — {metric}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved comparison chart: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View SURGE training-loss logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more *.jsonl log files (or a directory for --compare)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available training log files with their latest epoch info",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="One-line summary per running model",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay curves from every log in the given directory",
    )
    parser.add_argument(
        "--metric",
        default="val_loss",
        help="Metric to compare when using --compare (default: val_loss)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open matplotlib interactive window in addition to saving",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale on y-axis",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Override output file path for plots",
    )
    parser.add_argument(
        "--log-root",
        default="benchmark_reports/training_logs",
        metavar="DIR",
        help="Root directory for training logs (default: benchmark_reports/training_logs)",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    save = Path(args.save) if args.save else None

    if args.list:
        cmd_list(log_root)
        return

    if args.status:
        cmd_status(log_root)
        return

    if not args.paths:
        parser.print_help()
        return

    if args.compare:
        for p in args.paths:
            cmd_compare(Path(p), metric=args.metric, show=args.show, save=save)
        return

    # Default: plot each given file
    cmd_plot([Path(p) for p in args.paths], show=args.show, log_scale=args.log_scale, save=save)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
