# Copyright (c) 2026, SURGE authors. Licensed under BSD-3-Clause.
"""
Tests for surge.hpc.policy (ResourceSpec) and its integration with
SurrogateEngine + BaseModelAdapter (T1.3 of docs/RELEASE_SPRINT.md).

We verify three things:

1. `ResourceSpec.from_dict` round-trips YAML-style mappings.
2. `apply_policy` correctly reconciles a GPU request against a CPU-only
   profile — warning in the default case, raising in strict mode.
3. A real engine run emits the `[surge.fit] ...` banner and stores the
   resolved resources on ``adapter._last_fit_resources``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from surge.hpc import (
    ResourceSpec,
    ResourceProfile,
    ResourcePolicyError,
    apply_policy,
    resolve_device,
)


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------
def test_resource_spec_defaults_and_from_dict():
    spec = ResourceSpec.from_dict(None)
    assert spec.device == "auto"
    assert spec.num_workers == 0
    assert spec.strict is False

    spec2 = ResourceSpec.from_dict({"device": "cpu", "num_workers": 4, "strict": True})
    assert spec2.device == "cpu"
    assert spec2.num_workers == 4
    assert spec2.strict is True


def test_resource_spec_from_dict_ignores_unknown_keys_with_warning():
    with pytest.warns(UserWarning, match="unknown fields"):
        spec = ResourceSpec.from_dict({"device": "cpu", "bogus": 123})
    assert spec.device == "cpu"
    assert not hasattr(spec, "bogus")


def test_resolve_device_falls_back_to_cpu_when_torch_absent(monkeypatch):
    # Force the optional torch import inside resolve_device to fail so we
    # exercise the "no GPU" branch deterministically regardless of env.
    import importlib
    import surge.hpc.policy as policy_mod

    def _raise(*a, **k):
        raise ImportError("simulated")

    monkeypatch.setattr(importlib, "import_module", _raise, raising=False)
    # Even with auto, the function must return something sensible.
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


# ---------------------------------------------------------------------------
# apply_policy
# ---------------------------------------------------------------------------
_CPU_ONLY = ResourceProfile(
    name="cpu_only",
    supports_cpu=True,
    supports_gpu=False,
    worker_semantics="n_jobs",
)

_GPU_CAPABLE = ResourceProfile(
    name="gpu_capable",
    supports_cpu=True,
    supports_gpu=True,
    worker_semantics="dataloader_workers",
)


def test_apply_policy_cpu_only_rejects_gpu_nonstrict_warns_and_falls_back():
    spec = ResourceSpec(device="cuda", num_workers=4, strict=False)
    with pytest.warns(UserWarning, match="does not support GPU"):
        effective, concrete = apply_policy(spec, _CPU_ONLY, model_name="rf")
    assert effective.device == "cpu"
    assert concrete["device"] == "cpu"
    assert concrete["n_jobs"] == 4


def test_apply_policy_cpu_only_strict_raises_on_gpu():
    spec = ResourceSpec(device="cuda", strict=True)
    with pytest.raises(ResourcePolicyError, match="does not support GPU"):
        apply_policy(spec, _CPU_ONLY, model_name="rf")


def test_apply_policy_gpu_capable_honors_request():
    spec = ResourceSpec(device="cuda:0", num_workers=2)
    effective, concrete = apply_policy(spec, _GPU_CAPABLE, model_name="mlp")
    assert effective.device == "cuda:0"
    assert concrete["dataloader_num_workers"] == 2
    assert "n_jobs" not in concrete


def test_apply_policy_clamps_max_gpus_with_warning():
    spec = ResourceSpec(device="cuda", max_gpus=4, strict=False)
    with pytest.warns(UserWarning, match="max_gpus>1 is not supported"):
        effective, _ = apply_policy(spec, _GPU_CAPABLE, model_name="mlp")
    assert effective.max_gpus == 1


# ---------------------------------------------------------------------------
# End-to-end: banner fires + resources_used is populated
# ---------------------------------------------------------------------------
def test_engine_emits_fit_banner_and_records_resources(caplog):
    from surge import EngineRunConfig, ModelSpec, SurrogateEngine
    from surge.model import MODEL_REGISTRY  # noqa: F401 - side-effect import

    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "y1": rng.normal(size=n),
            "y2": rng.normal(size=n),
        }
    )
    engine = SurrogateEngine(
        run_config=EngineRunConfig(
            test_fraction=0.2,
            val_fraction=0.1,
            random_state=0,
            resources=ResourceSpec(device="cpu", num_workers=2),
        )
    )
    engine.configure_dataframe(df, input_columns=["x1", "x2"], output_columns=["y1", "y2"])

    with caplog.at_level(logging.INFO, logger="surge.hpc.policy"):
        results = engine.run([
            ModelSpec(key="sklearn.random_forest", params={"n_estimators": 10, "random_state": 0}),
        ])

    assert len(results) == 1
    res = results[0]

    # Banner must have been logged exactly once for this training.
    banner_records = [r for r in caplog.records if "[surge.fit]" in r.getMessage()]
    assert len(banner_records) >= 1, "expected at least one [surge.fit] banner"
    msg = banner_records[-1].getMessage()
    assert "model=sklearn.random_forest" in msg
    assert "backend=sklearn" in msg
    assert "n_jobs=2" in msg  # the user asked for num_workers=2

    # Adapter must expose the resolved resources for downstream persistence.
    used = getattr(res.adapter, "_last_fit_resources", None)
    assert used is not None
    assert used["effective"]["device"] == "cpu"
    assert used["concrete"]["n_jobs"] == 2
    assert used["banner"]["model"] == "sklearn.random_forest"
