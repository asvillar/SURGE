"""Standard benchmark registry and CLI (benchmark plan)."""

from __future__ import annotations

from .base import BenchmarkResult
from .registry import list_benchmarks, run_benchmark

__all__ = ["BenchmarkResult", "list_benchmarks", "run_benchmark"]
