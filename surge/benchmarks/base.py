"""Benchmark result container and persistence helpers."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows (NTFS) forbids ':' in path components. Use hyphens in the time portion
# so benchmark_reports/<key>/<timestamp>/ is checkout-safe on every OS.
BENCHMARK_TIMESTAMP_FMT = "%Y-%m-%dT%H-%M-%SZ"


def benchmark_timestamp(dt: datetime | None = None) -> str:
    """Return a filesystem-safe UTC timestamp for benchmark result directories."""
    when = dt or datetime.now(timezone.utc)
    return when.strftime(BENCHMARK_TIMESTAMP_FMT)


PREDICTIONS_FILE = "predictions.npz"
PREDICTIONS_MAX_ROWS = 20_000
_PREDICTIONS_MAX_ELEMENTS = 4_000_000  # keeps field tasks small
_stash = threading.local()


def capture_predictions(y_true, y_pred, y_prob=None) -> None:
    """Remember the held-out predictions of the run in progress (this thread).

    Called by the metric helpers; the next :class:`BenchmarkResult` built in
    the same thread adopts (and clears) them, and ``save`` writes them to
    ``predictions.npz`` next to ``result.json``.  Never raises.
    """
    try:
        import numpy as np

        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        if yt.size == 0 or len(yt) != len(yp):
            return
        _stash.value = (yt, yp, None if y_prob is None else np.asarray(y_prob))
    except Exception:
        _stash.value = None


def write_predictions(out_dir: Path, y_true, y_pred, y_prob=None, *,
                      seed: int = 0) -> dict[str, Any]:
    """Write ``predictions.npz`` (float32, <= 20k rows) into *out_dir*.

    Arrays with more than two dims (spatial fields) are flattened per sample and
    their original shape stored as ``field_shape``.  Returns metadata
    (``predictions_file``, ``subsampled``, ``n_rows``).
    """
    import numpy as np

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    arrays: dict[str, Any] = {}
    if yt.ndim > 2:
        arrays["field_shape"] = np.asarray(yt.shape[1:], dtype=np.int64)
    if yp.shape != yt.shape and yt.ndim <= 2 and yp.ndim == 2 and yt.ndim == 1:
        yp = yp.ravel() if yp.shape[1] == 1 else yp
    row_size = max(1, int(np.prod(yt.shape[1:])) if yt.ndim > 1 else 1)
    cap = max(8, min(PREDICTIONS_MAX_ROWS, _PREDICTIONS_MAX_ELEMENTS // row_size))
    idx = np.arange(n)
    subsampled = n > cap
    if subsampled:
        idx = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
    yt, yp = yt[idx], yp[idx]
    if yt.ndim > 2:
        yt, yp = yt.reshape(len(idx), -1), yp.reshape(len(idx), -1)
    arrays["y_true"] = yt.astype(np.float32)
    arrays["y_pred"] = yp.astype(np.float32)
    if y_prob is not None:
        arrays["y_prob"] = np.asarray(y_prob)[idx].astype(np.float32)
    arrays["subsampled"] = np.asarray(subsampled)
    arrays["n_total"] = np.asarray(n)
    np.savez_compressed(out_dir / PREDICTIONS_FILE, **arrays)
    return {"predictions_file": PREDICTIONS_FILE, "subsampled": subsampled,
            "n_rows": int(len(idx))}


@dataclass
class BenchmarkResult:
    """Outcome of a single benchmark run (Tier 0 / Tier 1 smoke)."""

    benchmark_key: str
    tier: str
    task_type: str
    metrics: dict[str, float]
    passed: bool
    model_key: str = ""
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # Populated by save(); readable by downstream tooling.
    timestamp: str = ""
    surge_version: str = ""

    def __post_init__(self) -> None:
        # adopt the predictions captured by the metric helpers (then clear,
        # so a later result in this thread can never inherit them)
        self._predictions = getattr(_stash, "value", None)
        _stash.value = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, root: Path = Path("benchmark_reports"),
             predictions: bool = True) -> Path:
        """
        Persist this result under ``<root>/<benchmark_key>/<timestamp>/result.json``.

        Returns the path written.
        """
        from surge import __version__

        if not self.timestamp:
            self.timestamp = benchmark_timestamp()
        if not self.surge_version:
            self.surge_version = __version__

        # include the model in the directory name: second-resolution
        # timestamps collide when jobs run in parallel (surge bench --parallel)
        slug = self.model_key.replace(".", "_") if self.model_key else ""
        out_dir = root / self.benchmark_key / (
            f"{self.timestamp}__{slug}" if slug else self.timestamp)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = getattr(self, "_predictions", None)
        if predictions and pred is not None:
            try:
                self.extra.update(write_predictions(out_dir, *pred))
            except Exception as exc:  # predictions are optional
                self.extra["predictions_error"] = str(exc)
        out_path = out_dir / "result.json"
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return out_path
