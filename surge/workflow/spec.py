"""Workflow specification dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..hpc import ResourceSpec
from ..utils import posix_str


@dataclass
class HPOConfig:
    enabled: bool = False
    n_trials: int = 20
    timeout: Optional[int] = None
    direction: str = "minimize"  # or "maximize"
    metric: str = "val_rmse"
    search_space: Dict[str, Any] = field(default_factory=dict)
    sampler: str = "tpe"  # or "botorch"

    @staticmethod
    def from_dict(payload: Optional[Mapping[str, Any]]) -> Optional["HPOConfig"]:
        if not payload:
            return None
        data = dict(payload)
        data.setdefault("enabled", True)
        return HPOConfig(**data)


@dataclass
class ModelConfig:
    key: str
    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    request_uncertainty: bool = False
    store_predictions: bool = True
    tags: Sequence[str] = field(default_factory=list)
    hpo: Optional[HPOConfig] = None

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ModelConfig":
        data = dict(payload)
        data["hpo"] = HPOConfig.from_dict(data.get("hpo"))
        return ModelConfig(**data)


@dataclass
class SurrogateWorkflowSpec:
    dataset_path: Union[str, Path]
    dataset_format: str = "auto"
    dataset_source: Optional[str] = None  # "mhd_code_batch" or "mhd_code_batch_per_mode" = load from batch dir
    metadata_path: Optional[Union[str, Path]] = None
    models: List[ModelConfig] = field(default_factory=list)
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    standardize_inputs: bool = True
    standardize_outputs: bool = False
    # Optional, reserved for a future k-fold path: k>0 is accepted but the unified
    # workflow currently performs a single random split only (see docs/ROADMAP.md).
    cv_folds: int = 0
    predictions_scope: Sequence[str] = field(default_factory=lambda: ("train", "val", "test"))
    predictions_format: str = "parquet"
    output_dir: Union[str, Path] = "."
    run_tag: Optional[str] = None
    seed: int = 42
    sample_rows: Optional[int] = None
    analyzer: Dict[str, Any] = field(default_factory=dict)
    metadata_overrides: Dict[str, Any] = field(default_factory=dict)
    batch_dir_mode_step: Optional[int] = None
    batch_dir_psi_step: Optional[int] = None
    batch_dir_target_shape: Optional[tuple] = None  # (n_modes, n_psi) for fixed resolution
    batch_dir_include_eigenmodes: bool = False
    # HDF5 leaf name under run*/case_data (default: sdata_pertfields_grid_complex_v2.h5).
    # Use sdata_complex_v2.h5 on partner-style trees where that file holds nonsymmetric δp modes.
    batch_dir_filename: Optional[str] = None
    notes: Optional[str] = None
    overwrite_existing_run: bool = False
    pretrained_run_dir: Optional[Union[str, Path]] = None
    finetune_lr_scale: float = 0.1
    mlflow_tracking: bool = False
    mlflow_experiment: Optional[str] = None
    # Declarative compute request applied to every model in this workflow.
    # See surge.hpc.policy.ResourceSpec for field semantics. Omitting this
    # block (or passing {}) preserves v0.0.x behaviour (auto device, default
    # n_jobs).
    resources: ResourceSpec = field(default_factory=ResourceSpec)

    def __post_init__(self) -> None:
        # Allow the ergonomic dict form in `models=[{...}, ...]` and
        # `resources={...}` — users who construct a spec directly in Python
        # (README example, notebooks) should get the same coercion that
        # from_dict() does for YAML-loaded dicts. Entries that are already
        # typed are passed through untouched.
        coerced_models: List[ModelConfig] = []
        for m in self.models:
            if isinstance(m, ModelConfig):
                coerced_models.append(m)
            elif isinstance(m, Mapping):
                coerced_models.append(ModelConfig.from_dict(m))
            else:
                raise TypeError(
                    f"SurrogateWorkflowSpec.models entries must be ModelConfig "
                    f"or dict; got {type(m).__name__}"
                )
        self.models = coerced_models

        if not isinstance(self.resources, ResourceSpec):
            self.resources = ResourceSpec.from_dict(self.resources)  # type: ignore[arg-type]

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SurrogateWorkflowSpec":
        data = dict(payload)
        model_dicts = data.get("models", [])
        data["models"] = [ModelConfig.from_dict(model) for model in model_dicts]
        # Accept resources as dict, ResourceSpec, or absent.
        if "resources" in data and not isinstance(data["resources"], ResourceSpec):
            data["resources"] = ResourceSpec.from_dict(data["resources"])
        return SurrogateWorkflowSpec(**data)

    def to_dict(self) -> Dict[str, Any]:
        spec_dict = asdict(self)
        spec_dict["dataset_path"] = posix_str(self.dataset_path)
        spec_dict["metadata_path"] = posix_str(self.metadata_path) if self.metadata_path else None
        spec_dict["output_dir"] = posix_str(self.output_dir)
        return spec_dict


