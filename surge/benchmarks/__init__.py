"""Standard benchmark registry and CLI (benchmark plan)."""

from __future__ import annotations

from .base import BenchmarkResult
from .registry import benchmark_info, list_benchmarks, run_benchmark

__all__ = ["BenchmarkResult", "benchmark_info", "list_benchmarks", "run_benchmark"]
