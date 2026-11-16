#!/usr/bin/env python
"""Live/offline training-loss viewer for JSONL logs written by SURGE.

``surge bench ... --train-log-file PATH`` appends one JSON record per epoch to
PATH; this script lists, summarises and plots such files. There is no default
log location: pass the folder or file you gave to ``--train-log-file``.

Usage examples
--------------
# Write a log while benchmarking
surge bench -b synthetic.regression_1d -m pytorch.mlp --train-log-file logs/mlp.jsonl

# List the training logs under a folder
python scripts/plot_training.py --list --log-root logs

# Print a quick text summary of the latest epoch of every log
python scripts/plot_training.py --status --log-root logs

# Plot loss curves for one file and save next to it as PNG
python scripts/plot_training.py logs/mlp.jsonl

# Same, with a log-scaled loss axis and an interactive window
python scripts/plot_training.py --show --log-scale logs/mlp.jsonl

# Overlay val_loss from every log in a folder on one chart
python scripts/plot_training.py --compare logs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
        age_s = int(time.time() - age)
        print(f"  {p}")
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

    from surge.viz.training import plot_training_dashboard

    for src in paths:
        records = _read_epochs(src)
        if not records:
            print(f"[warn] {src}: no epoch records yet, skipping")
            continue
        out = save or src.with_suffix(".png")
        title = src.stem
        fig = plot_training_dashboard(records, model_name=title, save_path=None)
        if log_scale and fig.axes:
            fig.axes[0].set_yscale("log")  # first panel is the loss
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
            print(f"Saved: {save}")
        else:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved: {out}")
        if show:
            plt.show()
        plt.close(fig)


def cmd_compare(directory: Path, *, metric: str, show: bool, log_scale: bool, save: Path | None) -> None:
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
    fig = compare_training_curves(histories, metric=metric, log_scale=log_scale, title=f"{directory.name} — {metric}")
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
        help="Use a log scale on the loss axis (plots) or y-axis (--compare)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Override output file path for plots",
    )
    parser.add_argument(
        "--log-root",
        default=".",
        metavar="DIR",
        help="Folder searched (recursively) for *.jsonl logs by --list/--status (default: .)",
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
            cmd_compare(Path(p), metric=args.metric, show=args.show, log_scale=args.log_scale, save=save)
        return

    # Default: plot each given file
    cmd_plot([Path(p) for p in args.paths], show=args.show, log_scale=args.log_scale, save=save)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
