"""Workflow specification dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


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
    metadata_path: Optional[Union[str, Path]] = None
    models: List[ModelConfig] = field(default_factory=list)
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    standardize_inputs: bool = True
    standardize_outputs: bool = False
    cv_folds: int = 0
    predictions_scope: Sequence[str] = field(default_factory=lambda: ("train", "val", "test"))
    predictions_format: str = "parquet"
    output_dir: Union[str, Path] = "."
    run_tag: Optional[str] = None
    seed: int = 42
    sample_rows: Optional[int] = None
    analyzer: Dict[str, Any] = field(default_factory=dict)
    metadata_overrides: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    overwrite_existing_run: bool = False

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SurrogateWorkflowSpec":
        data = dict(payload)
        model_dicts = data.get("models", [])
        data["models"] = [ModelConfig.from_dict(model) for model in model_dicts]
        return SurrogateWorkflowSpec(**data)

    def to_dict(self) -> Dict[str, Any]:
        spec_dict = asdict(self)
        spec_dict["dataset_path"] = str(self.dataset_path)
        spec_dict["metadata_path"] = str(self.metadata_path) if self.metadata_path else None
        spec_dict["output_dir"] = str(self.output_dir)
        return spec_dict


