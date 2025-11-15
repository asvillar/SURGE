"""Lightweight HPC/resource detection utilities."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


try:  # pragma: no cover - optional
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None
    PSUTIL_AVAILABLE = False

try:  # pragma: no cover
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False


@dataclass(slots=True)
class ComputeResources:
    scheduler: Optional[str]
    n_cpus: int
    n_gpus: int
    gpu_type: Optional[str]
    device: str
    hostname: str
    extras: Dict[str, str]

    def to_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        payload["extras"] = dict(self.extras)
        return payload


SCHEDULER_ENV_VARS = {
    "slurm": "SLURM_JOB_ID",
    "pbs": "PBS_JOBID",
    "lsf": "LSB_JOBID",
    "rose": "ROSE_JOB_ID",
}


def detect_compute_resources() -> ComputeResources:
    scheduler = _detect_scheduler()
    n_cpus = _detect_cpu_count()
    gpu_info = _detect_gpus()
    hostname = platform.node()
    device = "cuda" if gpu_info["count"] > 0 else "cpu"
    extras = {"platform": platform.platform()}
    extras.update(gpu_info.get("extras", {}))
    return ComputeResources(
        scheduler=scheduler,
        n_cpus=n_cpus,
        n_gpus=gpu_info["count"],
        gpu_type=gpu_info.get("name"),
        device=device,
        hostname=hostname,
        extras=extras,
    )


def _detect_scheduler() -> Optional[str]:
    for scheduler, env_var in SCHEDULER_ENV_VARS.items():
        if os.environ.get(env_var):
            return scheduler
    return None


def _detect_cpu_count() -> int:
    if PSUTIL_AVAILABLE and psutil is not None:
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    return os.cpu_count() or 1


def _detect_gpus() -> Dict[str, Any]:
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else None
        return {"count": count, "name": name, "extras": {"torch_cuda_version": torch.version.cuda or "unknown"}}

    # Fallback to nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        names = [line.strip() for line in output.splitlines() if line.strip()]
        return {"count": len(names), "name": names[0] if names else None, "extras": {}}
    except Exception:
        return {"count": 0, "name": None, "extras": {}}


def summarize_resources(resources: Optional[ComputeResources] = None) -> str:
    resources = resources or detect_compute_resources()
    lines = [
        f"Scheduler: {resources.scheduler or 'local'}",
        f"CPUs: {resources.n_cpus}",
        f"GPUs: {resources.n_gpus} ({resources.gpu_type or 'N/A'})",
        f"Device: {resources.device}",
        f"Hostname: {resources.hostname}",
    ]
    for key, value in resources.extras.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)

