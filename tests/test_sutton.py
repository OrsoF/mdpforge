import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model
from mdpforge.models.sutton import METADATA, TEST_PARAMETERS, Model, build_track
from mdpforge.solvers.mdpforge_vi import Solver


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_generated_track_preserves_preset_dimensions(size):
    preset = METADATA["sizes"][size]
    model = Model(**preset["parameters"])
    assert model.state_dim == preset["state_dim"]
    assert model.action_dim == 9


@pytest.mark.parametrize("state_dim", [0, -1])
def test_track_requires_positive_size(state_dim):
    with pytest.raises(ValueError, match="state_dim must be positive"):
        build_track(state_dim)


def test_small_generated_racetrack_dynamics_and_value(isolated_model_cache):
    model = Model(**TEST_PARAMETERS)
    model.create_model(save=False)
    validate_model(model)
    assert model.track.shape == (2, 2)
    assert model.state_dim == 107

    terminal = model.state_space_index[model.terminal_state]
    start = model.state_space_index[(0, 0, 0, 0)]
    next_state = model.state_space_index[(0, 1, 0, 1)]
    accelerate_up = model.action_space_index[(0, 1)]
    row = model.transition_matrix[accelerate_up].getrow(start)
    assert set(row.indices) == {start, next_state}
    assert_allclose(row[0, start], 0.1)
    assert_allclose(row[0, next_state], 0.9)

    # The blocked bottom-right cell causes a reset to the start line.
    crash = model.state_space_index[(0, 0, 1, 0)]
    coast = model.action_space_index[(0, 0)]
    assert model.transition_matrix[coast][crash, start] == 1.0

    # Acceleration crosses the finish line; failure keeps the old velocity.
    near_finish = model.state_space_index[(0, 1, 1, 0)]
    accelerate_right = model.action_space_index[(1, 0)]
    assert_allclose(
        model.transition_matrix[accelerate_right][near_finish, terminal], 0.9
    )
    assert np.all(model.reward_matrix[terminal] == 0)
    assert np.all(np.delete(model.reward_matrix, terminal, axis=0) == -1)
    for matrix in model.transition_matrix:
        assert matrix[terminal, terminal] == 1.0

    discount, precision = 0.9, 1e-6
    solver = Solver(model, discount, final_precision=precision)
    solver.run()
    residual = np.max(
        np.abs(optimal_bellman_operator(model, solver.value, discount) - solver.value)
    )
    assert residual / (1 - discount) <= precision
    assert_allclose(solver.value[terminal], 0, atol=precision)
