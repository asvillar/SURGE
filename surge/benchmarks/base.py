"""Benchmark result container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
