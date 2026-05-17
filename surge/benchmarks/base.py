"""Benchmark result container and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, root: Path = Path("benchmark_reports")) -> Path:
        """
        Persist this result under ``<root>/<benchmark_key>/<timestamp>/result.json``.

        Returns the path written.
        """
        from surge import __version__

        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.surge_version:
            self.surge_version = __version__

        out_dir = root / self.benchmark_key / self.timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.json"
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return out_path
