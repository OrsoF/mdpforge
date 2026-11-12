"""Check average-reward solver results and policies on 100-state Rooms."""

from copy import deepcopy
from importlib import import_module
from inspect import signature

import numpy as np
import pytest
from scipy.sparse import vstack
from scipy.sparse.csgraph import breadth_first_order

from mdpforge.core.solve_result import SolveResult
from mdpforge.models.discounted.rooms import Model as Rooms
from mdpforge.solvers import average
from tests._contracts import assert_model_unchanged, reproducible_rng
from tests._solver_support import run_external_test, solver_names

PRECISION = 1e-3
AVERAGE_SOLVERS = solver_names(average) or [
    pytest.param(
        None,
        marks=pytest.mark.skip(reason="No average-reward solvers are registered"),
        id="no-average-solvers",
    )
]


@pytest.fixture(scope="module")
def rooms_average():
    """Build a 100-state Rooms model without persistent caching."""
    model = Rooms(state_dim=100)
    model.create_model(check_transition=True, cache="off")
    assert model.state_dim == 100
    return model


@pytest.mark.parametrize("solver_name", AVERAGE_SOLVERS)
def test_all_average_solvers_rooms_100(solver_name, rooms_average, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(f"{average.__name__}.{solver_name}").Solver
    model = deepcopy(rooms_average)
    options = (
        {"final_precision": PRECISION}
        if "final_precision" in signature(solver_class).parameters
        else {}
    )
    with reproducible_rng():
        try:
            solver = solver_class(model, **options)
            assert_model_unchanged(model, rooms_average)
            result = solver.run()
        finally:
            assert_model_unchanged(model, rooms_average)

    assert isinstance(result, SolveResult)
    assert isinstance(result.value, np.ndarray)
    assert result.value.shape == (rooms_average.state_dim,)
    assert np.isfinite(result.value).all()
    if result.policy is not None:
        assert_policy_reaches_exit(rooms_average, result.policy)


def assert_policy_reaches_exit(model, policy):
    """Check that the policy earns the optimal average reward of zero."""
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (model.state_dim,)
    assert np.issubdtype(policy.dtype, np.integer)
    assert ((0 <= policy) & (policy < model.action_dim)).all()
    exit_states = np.flatnonzero(np.all(model.reward_matrix == 0.0, axis=1))
    assert exit_states.size == 1
    transitions = vstack(
        [
            model.transition_matrix[action].getrow(state)
            for state, action in enumerate(policy)
        ],
        format="csr",
    )
    transitions.eliminate_zeros()
    # All other rewards are -1. Reaching the absorbing exit from every state
    # rules out any closed class with negative long-run reward.
    reachable = breadth_first_order(
        transitions.T, int(exit_states[0]), directed=True, return_predecessors=False
    )
    assert reachable.size == model.state_dim
