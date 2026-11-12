"""Check total-reward solver convergence against the exact Rooms value."""

from copy import deepcopy
from importlib import import_module
from inspect import signature

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.models.total.rooms import Model as Rooms
from mdpforge.solvers import total
from tests._contracts import assert_model_unchanged, reproducible_rng
from tests._solver_support import run_external_test, solver_names

PRECISION = 1e-3


@pytest.fixture(scope="module")
def rooms_total():
    """Build the cyclic 100-state total-reward Rooms model without caching."""
    model = Rooms(state_dim=100)
    model.create_model(check_transition=True, cache="off")
    assert model.state_dim == 100
    return model


@pytest.mark.parametrize("solver_name", solver_names(total))
def test_all_total_solvers_converge_rooms_100(
    solver_name, rooms_total, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(f"{total.__name__}.{solver_name}").Solver
    model = deepcopy(rooms_total)
    options = (
        {"final_precision": PRECISION}
        if "final_precision" in signature(solver_class).parameters
        else {}
    )
    with reproducible_rng():
        try:
            result = solver_class(model, **options).run()
        finally:
            assert_model_unchanged(model, rooms_total)

    # A rewarded exit attempt succeeds with probability 0.8; failures stay put.
    # Every nonterminal state can reach the exit, so V* = 1 / 0.8 there.
    expected = np.full(rooms_total.state_dim, 1.25)
    expected[list(rooms_total.terminal_states)] = 0.0
    assert isinstance(result, SolveResult)
    assert result.value.shape == expected.shape
    assert np.isfinite(result.value).all()
    assert_allclose(result.value, expected, rtol=0.0, atol=PRECISION)
    backup = compact_optimal_bellman_operator(rooms_total, result.value, 1.0)
    assert_allclose(backup, result.value, rtol=0.0, atol=PRECISION)
