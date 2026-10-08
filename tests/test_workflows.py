import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge.core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from mdpforge.solvers.personal_pim import Solver as MPI
from mdpforge.solvers.personal_vi import Solver as VI


@pytest.mark.parametrize("solver_class", [VI, MPI], ids=["vi", "mpi"])
def test_model_to_solver_workflow(
    model_spec, isolated_model_cache, solver_class, monkeypatch
):
    model_class, state_dim, action_dim, _ = model_spec
    model = model_class(state_dim, action_dim)
    model.create_model(save=False)
    discount = 0.9
    # MPI initializes a random policy; isolate the global RNG used by that solver.
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    solver = solver_class(model, discount, final_precision=1e-4)
    if solver_class is MPI:
        # Allow policy evaluation to reach the requested numerical accuracy.
        solver.max_iter_eval = 100_000_000
    solver.run()
    assert solver.value.shape == (model.state_dim,)
    assert np.all(np.isfinite(solver.value))
    policy = bellman_operator(model, solver.value, discount).argmax(axis=1)
    if solver_class is VI:
        assert solver.bellman_residual() < 1e-4
    else:
        assert solver.policy.shape == (model.state_dim,)
        assert np.issubdtype(solver.policy.dtype, np.integer)
        assert np.all((0 <= solver.policy) & (solver.policy < model.action_dim))
        np.testing.assert_array_equal(solver.policy, policy)
    transition, reward = compute_transition_reward_policy(model, policy)
    policy_value = np.linalg.solve(
        # RiverSwim stores rewards as float16, which NumPy linalg does not accept.
        np.eye(model.state_dim) - discount * transition.toarray(),
        reward.astype(float),
    )
    # A residual of 1e-4 implies a value error of at most 1e-3 at discount 0.9.
    assert_allclose(solver.value, policy_value, atol=1e-3, rtol=0)
    assert_allclose(
        optimal_bellman_operator(model, solver.value, discount),
        solver.value,
        atol=1e-4,
        rtol=0,
    )
    assert np.isfinite(solver.runtime) and solver.runtime >= 0
