"""HPC/resource detection helpers and user-facing resource policy."""

from .policy import (
    DEFAULT_RESOURCE_PROFILE,
    ResourcePolicyError,
    ResourceProfile,
    ResourceSpec,
    apply_policy,
    log_fit_banner,
    resolve_device,
)
from .resources import ComputeResources, detect_compute_resources, summarize_resources

__all__ = [
    "ComputeResources",
    "detect_compute_resources",
    "summarize_resources",
    "ResourceSpec",
    "ResourceProfile",
    "ResourcePolicyError",
    "DEFAULT_RESOURCE_PROFILE",
    "resolve_device",
    "apply_policy",
    "log_fit_banner",
]


