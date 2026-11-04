"""Text status of a running or finished campaign, from the run's artifacts only.

Reads ``hpo_trials_manifest.jsonl``, ``hpo_trial_XXXX_training_history.json``,
``metrics.json`` and ``run.log`` -- never the training process itself, so it
works from another terminal, over ssh, or after the run is gone. Pure text
(no curses): safe to pipe. Used by ``surge monitor``.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")


def _is_loss_like(metric: str) -> bool:
    return any(t in metric.lower() for t in ("loss", "mse", "rmse", "nrmse", "mae", "error"))


def _params(params: dict, width: int = 64) -> str:
    parts = []
    for k, v in (params or {}).items():
        parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    text = " ".join(parts)
    return text if len(text) <= width else text[: width - 1] + "…"


def _fmt_secs(sec: float) -> str:
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _read_history(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []  # file mid-write
    return data if isinstance(data, list) else []


def _elapsed(run_dir: Path, finished: bool) -> float | None:
    log = run_dir / "run.log"
    if log.exists():
        stamps = []
        for line in log.read_text(errors="replace").splitlines():
            m = _TS.search(line)
            if m:
                try:
                    stamps.append(datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}"))
                except ValueError:
                    pass
        if len(stamps) >= 2:
            end = stamps[-1] if finished else datetime.now()
            return (end - stamps[0]).total_seconds()
    files = [f for f in run_dir.iterdir() if f.is_file()]
    if not files:
        return None
    start = min(f.stat().st_mtime for f in files)
    end = max(f.stat().st_mtime for f in files) if finished else time.time()
    return end - start


def format_status(run_dir) -> str:
    """Multi-line text status of ``run_dir``."""
    run_dir = Path(run_dir)
    lines: list[str] = []
    manifest = run_dir / "hpo_trials_manifest.jsonl"
    rows = []
    if manifest.exists():
        for ln in manifest.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass  # partially written last line
    metrics_f = run_dir / "metrics.json"
    finished = metrics_f.exists()
    metric = rows[0].get("metric", "value") if rows else "value"
    maximize = not _is_loss_like(metric)

    lines.append(f"SURGE monitor  {run_dir}  [{'FINISHED' if finished else 'RUNNING'}]")
    best = None
    best_n = None
    if rows:
        lines.append(f"HPO trials ({metric}, {'max' if maximize else 'min'}):")
        for i, r in enumerate(rows):
            n = r.get("trial", i)
            val = r.get("value")
            hist = _read_history(run_dir / f"hpo_trial_{n:04d}_training_history.json")
            last = hist[-1] if hist else {}
            tl, vl = last.get("train_loss"), last.get("val_loss")
            new_best = isinstance(val, (int, float)) and val == val and (
                best is None or (val > best if maximize else val < best))
            if new_best:
                best, best_n = val, n
            loss = (f"train={tl:.4g}  val={vl:.4g}  "
                    if tl is not None and vl is not None else "")
            marker = "  <- best so far" if new_best else ""
            lines.append(f"  trial {n:>3d}  {metric}={val:.4f}  epochs={len(hist):>3d}  "
                         f"{loss}{_params(r.get('params', {}))}{marker}")
    # trial in progress: a history file with no manifest line yet
    seen = {r.get("trial", i) for i, r in enumerate(rows)}
    for f in sorted(run_dir.glob("hpo_trial_*_training_history.json")):
        m = re.search(r"hpo_trial_(\d+)_training_history", f.name)
        if not m or int(m.group(1)) in seen:
            continue
        hist = _read_history(f)
        last = hist[-1] if hist else {}
        lines.append(
            f"  trial {int(m.group(1)):>3d}  (running)  epochs={len(hist):>3d}  "
            f"train={last.get('train_loss', float('nan')):.4g}  "
            f"val={last.get('val_loss', float('nan')):.4g}")
    if not rows and not finished and not list(run_dir.glob("hpo_trial_*")):
        lines.append("  (no HPO trials recorded yet)")

    lines.append("")
    if best_n is not None:
        lines.append(f"best: trial {best_n}  {metric}={best:.4f}  "
                     f"({len(rows)} trial(s) complete)")
    el = _elapsed(run_dir, finished) if run_dir.is_dir() else None
    if el is not None:
        lines.append(f"elapsed: {_fmt_secs(el)}")
    if finished:
        try:
            metrics = json.loads(metrics_f.read_text())
        except ValueError:
            metrics = {}
        lines.append("final metrics:")
        for model, entry in metrics.items():
            test = (entry or {}).get("test") or {}
            body = "  ".join(f"{k}={v:.4g}" for k, v in test.items()
                             if isinstance(v, (int, float)))
            lines.append(f"  {model:32s} test: {body}")
    return "\n".join(lines)


def monitor(run_dir, watch: float | None = None, out=None) -> int:
    """Print the status once, or redraw every ``watch`` seconds until metrics.json."""
    out = out or sys.stdout
    run_dir = Path(run_dir)
    try:
        while True:
            print(format_status(run_dir), file=out, flush=True)
            if watch is None or (run_dir / "metrics.json").exists():
                return 0
            print("", file=out)
            time.sleep(watch)
    except KeyboardInterrupt:
        print("", file=out)
        return 0
