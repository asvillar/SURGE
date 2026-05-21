"""
Training-progress instrumentation for SURGE model backends.

Usage in a backend's ``fit()``::

    from ._progress import ProgressList

    self.training_history = ProgressList(
        total_epochs=self.n_epochs,
        verbose=self.verbose,
        log_file=self.log_file,
        desc=type(self).__name__,
    )
    # ... training loop calls self.training_history.append(record) as usual ...
    self.training_history.close()

``ProgressList`` is a plain ``list`` subclass so all downstream code that reads
``model.training_history`` continues to work unchanged.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import IO, Any, Optional


class ProgressList(list):
    """A ``list[dict]`` that drives a live tqdm bar and/or a JSONL log file.

    Parameters
    ----------
    total_epochs:
        Total number of expected epochs (used to size the tqdm bar).
    verbose:
        If ``True``, print a tqdm progress bar to stderr showing live loss values.
    log_file:
        Path to a ``.jsonl`` file.  Each call to ``append()`` writes one JSON
        line so the file can be ``tail -f``'ed or loaded with
        ``plot_training_history(log_file=...)``.  Parent directories are created
        automatically.  If the file already exists its contents are **not**
        cleared; records from the new run are appended with a run-separator.
    mlflow_run:
        Optional active ``mlflow.ActiveRun`` context.  When provided, each
        epoch record is logged as MLflow metrics.
    desc:
        Label shown in the tqdm bar (typically the model class name).
    """

    def __init__(
        self,
        total_epochs: int = 0,
        *,
        verbose: bool = False,
        log_file: Optional[str] = None,
        mlflow_run: Any = None,
        desc: str = "Training",
    ) -> None:
        super().__init__()
        self._pbar: Any = None
        self._fh: Optional[IO[str]] = None
        self._mlflow_run = mlflow_run
        self._total = total_epochs

        if verbose:
            try:
                from tqdm.auto import tqdm
                self._pbar = tqdm(
                    total=total_epochs,
                    desc=desc,
                    unit="ep",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    leave=True,
                )
            except ImportError:
                # tqdm not installed — fall back to plain stderr printing
                self._pbar = _SimplePrinter(desc, total_epochs)

        if log_file:
            p = pathlib.Path(log_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("a", encoding="utf-8")
            # Write a run-start sentinel so multiple runs in the same file are
            # distinguishable.
            import time
            self._fh.write(
                json.dumps({"__run_start__": True, "ts": time.time(), "desc": desc})
                + "\n"
            )
            self._fh.flush()

    # ------------------------------------------------------------------
    # Core intercept
    # ------------------------------------------------------------------

    def append(self, record: dict) -> None:  # type: ignore[override]
        super().append(record)
        self._emit(record)

    def _emit(self, record: dict) -> None:
        epoch = record.get("epoch")
        loss_items = {k: v for k, v in record.items() if k.endswith("_loss")}

        if self._pbar is not None:
            postfix = {k.replace("_loss", ""): f"{v:.5f}" for k, v in loss_items.items()}
            if record.get("early_stop"):
                postfix["early_stop"] = "✓"
            self._pbar.set_postfix(postfix, refresh=True)
            self._pbar.update(1)

        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

        if self._mlflow_run is not None and epoch is not None:
            try:
                import mlflow
                for k, v in record.items():
                    if isinstance(v, (int, float)) and k != "epoch":
                        mlflow.log_metric(k, float(v), step=int(epoch))
            except Exception:
                pass

    def close(self) -> None:
        """Must be called at the end of training to flush/close resources."""
        if self._pbar is not None:
            # If training stopped early, fill remaining steps so bar shows 100 %.
            remaining = self._total - len(self)
            if remaining > 0:
                self._pbar.update(remaining)
            self._pbar.close()
            self._pbar = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fallback printer when tqdm is not installed
# ---------------------------------------------------------------------------

class _SimplePrinter:
    """Minimal progress reporter that works without tqdm."""

    def __init__(self, desc: str, total: int) -> None:
        self._desc = desc
        self._total = total
        self._n = 0
        self._print_every = max(1, total // 20)  # ~20 updates

    def set_postfix(self, postfix: dict, **_: Any) -> None:
        self._postfix = postfix

    def update(self, n: int = 1) -> None:
        self._n += n
        if self._n % self._print_every == 0 or self._n == self._total:
            losses = "  ".join(f"{k}={v}" for k, v in getattr(self, "_postfix", {}).items())
            pct = 100 * self._n // self._total
            print(
                f"\r{self._desc}  [{pct:3d}%  {self._n}/{self._total}]  {losses}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if self._n == self._total:
                print(file=sys.stderr)

    def close(self) -> None:
        pass
