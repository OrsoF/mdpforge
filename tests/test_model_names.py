import inspect

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdpforge import list_models
from mdpforge.models.barto import Model as Barto
from mdpforge.models.barto import build_track
from mdpforge.models.elevator import Model as Elevator
from mdpforge.models.impatience import Model as Impatience
from tests._contracts import reproducible_rng


@pytest.mark.parametrize("entry", list_models(), ids=lambda entry: entry["name"])
@pytest.mark.parametrize("size", [None, "small", "medium", "large"])
def test_model_name_matches_actual_dimensions_and_filename(entry, size):
    options = {}
    if size is not None:
        preset = entry["sizes"][size]
        if preset["parameters"] is None:
            pytest.skip("Preset unavailable")
        options = {"state_dim": preset["requested_state_dim"], **preset["parameters"]}
    module = pytest.importorskip(f"mdpforge.models.{entry['name']}")
    with reproducible_rng(0):
        model = module.Model(**options)
    prefix = f"{model.state_dim}_{model.action_dim}_{entry['name']}"
    assert model.name == prefix or model.name.startswith(prefix + "_")


def test_barto_uses_only_the_generated_track():
    model = Barto(state_dim=600)
    assert_array_equal(model._track, build_track(600, model._max_speed))
    assert "custom_track" not in inspect.signature(Barto).parameters
    with pytest.raises(TypeError, match="custom_track"):
        Barto(custom_track=np.ones((3, 3)))


def test_elevator_names_distinguish_floor_counts_at_equal_dimensions():
    three = Elevator(state_dim=100, n_floors=3)
    four = Elevator(state_dim=100, n_floors=4)
    assert three.name == "100_5_elevator_floors3"
    assert four.name == "100_5_elevator_floors4"


def test_impatience_names_distinguish_kernel_tolerances():
    precise = Impatience(truncation_tol=1e-8)
    approximate = Impatience(truncation_tol=1e-4)
    assert precise.name == "200_10_impatience_tol1e-08"
    assert approximate.name == "200_10_impatience_tol0.0001"
