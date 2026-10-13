"""Tests for surge.workflow.run HPO param reconstruction.

Covers the bug where `variable_list` search-space entries (e.g. per-layer
`hidden_layers` with `logint` elements) were not replayed after
`study.optimize`, so the refit spec got raw Optuna keys like
`n_blocks`/`hidden_layers_0_log` instead of a resolved `hidden_layers` list.
"""

from __future__ import annotations

import optuna
import pytest

from surge.workflow.run import _sample_hpo_params


VARIABLE_LIST_SPACE = {
    "hidden_layers": {
        "type": "variable_list",
        "num_layers": {"low": 2, "high": 5},
        "num_layers_name": "n_blocks",
        "element": {"type": "logint", "low": 32, "high": 512},
    },
    "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "dropout_rate": {"type": "float", "low": 0.0, "high": 0.3},
}


def test_sample_hpo_params_variable_list_logint_custom_num_layers_name():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    sampled = _sample_hpo_params(trial, VARIABLE_LIST_SPACE)

    assert isinstance(sampled["hidden_layers"], list)
    assert sampled["hidden_layers"], "hidden_layers must not be empty"
    assert all(isinstance(v, int) for v in sampled["hidden_layers"])
    # num_layers_name was customized to "n_blocks" -> that's the raw Optuna key,
    # not the resolved "hidden_layers" name.
    assert "n_blocks" in trial.params
    assert len(sampled["hidden_layers"]) == trial.params["n_blocks"]
    # Raw per-element log keys, not a resolved "hidden_layers" key, live in trial.params.
    assert "hidden_layers" not in trial.params
    assert any(k.startswith("hidden_layers_") and k.endswith("_log") for k in trial.params)


def test_fixed_trial_replay_reproduces_hidden_layers():
    """This is the fix under test: replaying trial.params through FixedTrial
    must reconstruct the exact same `hidden_layers` list that _run_hpo needs
    for the refit spec, instead of leaving raw n_blocks/hidden_layers_i_log keys.
    """
    from optuna.trial import FixedTrial

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    trial = study.ask()
    sampled = _sample_hpo_params(trial, VARIABLE_LIST_SPACE)

    # Simulate what _run_hpo now does with study.best_trial.params after optimize().
    replayed = _sample_hpo_params(FixedTrial(dict(trial.params)), VARIABLE_LIST_SPACE)

    assert replayed["hidden_layers"] == sampled["hidden_layers"]
    assert replayed["learning_rate"] == pytest.approx(sampled["learning_rate"])
    assert replayed["batch_size"] == sampled["batch_size"]
    assert replayed["dropout_rate"] == pytest.approx(sampled["dropout_rate"])


def test_fixed_trial_replay_many_seeds_variable_num_layers():
    """Same replay property across several trials with different sampled
    num_layers, to make sure the fix isn't accidentally tied to one depth.
    """
    from optuna.trial import FixedTrial

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    for _ in range(8):
        trial = study.ask()
        sampled = _sample_hpo_params(trial, VARIABLE_LIST_SPACE)
        replayed = _sample_hpo_params(FixedTrial(dict(trial.params)), VARIABLE_LIST_SPACE)
        assert replayed["hidden_layers"] == sampled["hidden_layers"]
        study.tell(trial, 0.0)
