"""Generic (dataset-agnostic) behaviour of the importance and run_viz helpers."""

import inspect

import numpy as np
import pytest

from surge.viz import run_viz
from surge.viz.importance import plot_shap_grouped_bar, stage_group_specs


def test_stage_group_specs_with_extra():
    specs = stage_group_specs(11, 4, n_extra=3)
    assert [len(idx) for _, idx in specs] == [4, 4, 3]
    assert specs[-1][1] == [8, 9, 10]
    assert sorted(i for _, idx in specs for i in idx) == list(range(11))


def test_stage_group_specs_no_extra():
    specs = stage_group_specs(6, 2)
    assert len(specs) == 3 and specs[1][1] == [2, 3]


def test_stage_group_specs_rejects_bad_layout():
    with pytest.raises(ValueError):
        stage_group_specs(10, 4)


def test_grouped_bar_requires_group_specs():
    with pytest.raises(ValueError, match="group_specs"):
        plot_shap_grouped_bar(np.zeros((5, 8)))


def test_model_label_falls_back_to_model_name():
    assert run_viz._model_short_name("my_custom_model") == "My Custom Model"
    assert run_viz._model_short_name("torch_mlp_mc_dropout") == "Torch Mlp Mc Dropout"


def test_default_output_is_first_and_set_name_optional():
    sig = inspect.signature(run_viz.viz_datastreamset_evaluation)
    assert sig.parameters["output_key"].default == "output_0"
    assert inspect.signature(run_viz._plot_datastreamset_evaluation).parameters["output_key"].default == "output_0"
    assert inspect.signature(run_viz.viz_shap).parameters["output_index"].default is None
