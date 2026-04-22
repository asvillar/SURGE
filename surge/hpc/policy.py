"""
Resource policy for SURGE training runs.

This module defines the public ``ResourceSpec`` dataclass that users place in
their workflow spec or engine config to declare *what compute they want*, and
the small helpers that turn that declaration into concrete backend knobs
(``device`` string, sklearn ``n_jobs``, torch DataLoader workers) with a
readable policy decision logged to the user.

Design goals (v0.1.0):

* One dataclass, stable field names — users write YAML like:
      resources:
        device: cuda
        num_workers: 8
* Never silently ignore a user choice. Either honor it, adjust it with a
  ``UserWarning``, or — when ``strict=True`` — raise ``ResourcePolicyError``.
* Emit exactly one ``[surge.fit] ...`` banner per model train so users can
  tell "what worker count is being used" at a glance, as asked.

Out of scope for v0.1.0 (tracked in docs/ROADMAP.md):

* Multi-GPU DDP/FSDP — ``max_gpus > 1`` currently produces a warning and is
  clamped to 1.
* Per-process DataLoader sharding for ``IterableDataset`` feeds.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

LOGGER = logging.getLogger("surge.hpc.policy")


class ResourcePolicyError(RuntimeError):
    """Raised when ``ResourceSpec.strict`` is True and a combo is unsupported."""


# ---------------------------------------------------------------------------
# ResourceSpec — the user-facing dataclass
# ---------------------------------------------------------------------------
@dataclass
class ResourceSpec:
    """Declarative compute request attached to a workflow or engine run.

    Attributes
    ----------
    device
        Target device for the training step. One of:
        ``"auto"`` (prefer CUDA if available, else CPU),
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"`` for a specific GPU index.
    num_workers
        Per-model worker count. For sklearn backends this maps to
        ``n_jobs``; for torch backends it maps to DataLoader
        ``num_workers``. ``0`` means "pick a sensible default
        (``min(cpu_count, 8)`` for sklearn, ``0`` for torch in-memory
        datasets)".
    max_gpus
        Cap on the number of GPUs actually used. ``1`` in v0.1.0; higher
        values currently warn and clamp (see docs/ROADMAP.md).
    pin_memory
        Torch DataLoader ``pin_memory`` flag. Ignored by sklearn.
    persistent_workers
        Torch DataLoader ``persistent_workers`` flag. Ignored by sklearn.
    strict
        If True, unsupported combos (e.g. RandomForest + ``device=cuda``)
        raise ``ResourcePolicyError`` instead of warning and adjusting.
    """

    device: str = "auto"
    num_workers: int = 0
    max_gpus: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Optional[Mapping[str, Any]]) -> "ResourceSpec":
        """Construct a ``ResourceSpec`` from a YAML/JSON mapping.

        ``None`` and ``{}`` both yield a default spec, so downstream code
        can rely on ``resources`` always being present.
        """
        if not payload:
            return ResourceSpec()
        # Accept only known fields; ignore unknown keys with a warning so
        # users catch typos without their run failing.
        known = {f for f in ResourceSpec.__dataclass_fields__}
        unknown = set(payload.keys()) - known
        if unknown:
            warnings.warn(
                f"ResourceSpec: ignoring unknown fields {sorted(unknown)}. "
                f"Known fields: {sorted(known)}.",
                UserWarning,
                stacklevel=2,
            )
        clean = {k: v for k, v in payload.items() if k in known}
        return ResourceSpec(**clean)


# ---------------------------------------------------------------------------
# Resource-profile metadata that each adapter can declare
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceProfile:
    """Static per-adapter capability advertisement.

    An adapter whose profile has ``supports_gpu=False`` cannot run on GPU
    no matter what the user requests; the policy engine will either warn
    and fall back to CPU (``strict=False``) or raise (``strict=True``).
    """

    name: str = "generic"
    supports_cpu: bool = True
    supports_gpu: bool = False
    # "n_jobs" for sklearn, "dataloader_workers" for torch, "none" for models
    # that ignore the worker knob entirely.
    worker_semantics: str = "none"
    notes: str = ""


# Default profile — conservative: CPU-only, no worker knob.
DEFAULT_RESOURCE_PROFILE = ResourceProfile(
    name="default",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="none",
)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cpu"`` or ``"cuda"`` based on availability.

    ``"cuda"``/``"cuda:N"`` are passed through verbatim; the caller is
    responsible for detecting whether torch is installed and CUDA is
    really available (we avoid importing torch at module import time so
    the sklearn-only install path stays light).
    """
    if device != "auto":
        return device
    try:
        import torch  # local import: optional dep

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Policy application: translate ResourceSpec -> concrete backend knobs
# ---------------------------------------------------------------------------
def apply_policy(
    resources: ResourceSpec,
    profile: ResourceProfile,
    *,
    model_name: str = "",
    logger: Optional[logging.Logger] = None,
) -> Tuple[ResourceSpec, Dict[str, Any]]:
    """Reconcile a user ``ResourceSpec`` against an adapter ``ResourceProfile``.

    Returns a tuple of (effective_spec, concrete_kwargs) where:

    * ``effective_spec`` is a possibly-adjusted copy of ``resources`` (e.g.
      ``device="cuda"`` was requested but the adapter is CPU-only → the
      effective spec will have ``device="cpu"``).
    * ``concrete_kwargs`` is a plain dict of backend-flavoured keys ready
      to hand to sklearn / torch:

        {
          "device": "cuda:0",
          "n_jobs": 8,              # if worker_semantics == "n_jobs"
          "dataloader_num_workers": 4,  # if worker_semantics == "dataloader_workers"
          "pin_memory": True,
          "persistent_workers": True,
          "max_gpus": 1,
        }

    Violations either warn (``strict=False``) or raise
    ``ResourcePolicyError`` (``strict=True``).
    """
    log = logger or LOGGER
    spec = replace(resources)

    def _violate(msg: str) -> None:
        if spec.strict:
            raise ResourcePolicyError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)
        log.warning("[%s] policy adjusted: %s", model_name or profile.name, msg)

    # Device reconciliation
    requested = resolve_device(spec.device)
    effective_device = requested
    if requested.startswith("cuda") and not profile.supports_gpu:
        _violate(
            f"{profile.name} does not support GPU; falling back to CPU "
            f"(requested device={spec.device!r})."
        )
        effective_device = "cpu"
    if effective_device == "cpu" and not profile.supports_cpu:
        _violate(f"{profile.name} does not support CPU execution.")
    spec = replace(spec, device=effective_device)

    # Multi-GPU cap (v0.1.0)
    if spec.max_gpus > 1:
        _violate(
            "max_gpus>1 is not supported in v0.1.0 (multi-GPU DDP is a "
            "v0.2.0 item; see docs/ROADMAP.md). Clamping to 1."
        )
        spec = replace(spec, max_gpus=1)

    # Worker knob
    concrete: Dict[str, Any] = {
        "device": effective_device,
        "max_gpus": spec.max_gpus,
        "pin_memory": spec.pin_memory,
        "persistent_workers": spec.persistent_workers,
    }
    n = spec.num_workers
    if profile.worker_semantics == "n_jobs":
        if n <= 0:
            n = min(os.cpu_count() or 1, 8)
        concrete["n_jobs"] = int(n)
    elif profile.worker_semantics == "dataloader_workers":
        if n < 0:
            _violate("num_workers must be >= 0 for torch DataLoader; clamping to 0.")
            n = 0
        concrete["dataloader_num_workers"] = int(n)
    # "none" -> expose nothing; the adapter ignores workers.

    return spec, concrete


# ---------------------------------------------------------------------------
# Fit banner — the single [surge.fit] log line users see per model
# ---------------------------------------------------------------------------
def log_fit_banner(
    *,
    model_name: str,
    backend: str,
    concrete: Mapping[str, Any],
    n_train: int,
    n_features: int,
    n_outputs: int,
    extra: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Emit the canonical ``[surge.fit] ...`` banner and return the same
    dict of fields so callers can persist it to ``run_summary.json`` under
    ``resources_used``.
    """
    log = logger or LOGGER
    payload: Dict[str, Any] = {
        "model": model_name,
        "backend": backend,
        "device": concrete.get("device", "cpu"),
        "max_gpus": concrete.get("max_gpus", 1),
        "n_train": n_train,
        "n_features": n_features,
        "n_outputs": n_outputs,
    }
    if "n_jobs" in concrete:
        payload["n_jobs"] = concrete["n_jobs"]
    if "dataloader_num_workers" in concrete:
        payload["dataloader_num_workers"] = concrete["dataloader_num_workers"]
        payload["pin_memory"] = concrete.get("pin_memory", True)
        payload["persistent_workers"] = concrete.get("persistent_workers", True)
    if extra:
        payload.update({k: v for k, v in extra.items() if k not in payload})

    formatted = " ".join(f"{k}={v}" for k, v in payload.items())
    log.info("[surge.fit] %s", formatted)
    return payload


__all__ = [
    "ResourceSpec",
    "ResourceProfile",
    "ResourcePolicyError",
    "DEFAULT_RESOURCE_PROFILE",
    "resolve_device",
    "apply_policy",
    "log_fit_banner",
]
