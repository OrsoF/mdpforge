"""Catalog-wide scientific contracts: new models/solvers inherit these checks."""

import os
import subprocess
import sys
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model
from tests._contracts import (
    assert_optimal_solution,
    matrix_model,
    reproducible_rng,
    run_solver_preserving_input,
)


def run_external_test(request, tmp_path):
    """Isolate installed native backends that may own global process state."""
    marker = request.node.get_closest_marker("external")
    if marker is None or os.environ.get("MDPFORGE_TEST_CHILD") == "1":
        return False
    for dependency in marker.kwargs["dependencies"]:
        if find_spec(dependency) is None:
            pytest.skip(f"Dependency not installed: {dependency}")
    path, test_name = request.node.nodeid.split("::", 1)
    nodeid = f"{request.config.rootpath / path}::{test_name}"
    result = subprocess.run(
        [sys.executable, "-I", "-m", "pytest", nodeid, "-q", "--tb=short"],
        cwd=tmp_path,
        env={**os.environ, "MDPFORGE_TEST_CHILD": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if "1 xfailed" in result.stdout:
        pytest.xfail(result.stdout[-4000:].strip())
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return True


@pytest.mark.parametrize("discount", [0.5, 0.99])
def test_all_solvers_meet_precision_and_preserve_inputs(
    solver_module, mdp_case, discount, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(solver_module).Solver
    precision = 1e-4
    previous = None
    for _ in range(2):
        solver = run_solver_preserving_input(
            solver_class, mdp_case, discount, precision
        )
        actual = assert_optimal_solution(solver, mdp_case, discount, precision)
        if previous is not None:
            # Value and policy value must reproduce, even if there are action ties.
            for current, prior in zip(actual, previous):
                assert_allclose(current, prior, atol=1e-12, rtol=0)
        previous = actual


def test_all_models_produce_valid_csr(
    model_module, isolated_model_cache, monkeypatch, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    module = import_module(model_module)
    parameters = getattr(module, "TEST_PARAMETERS", {"state_dim": 7, "action_dim": 4})
    with reproducible_rng(0):
        monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
        model = module.Model(**parameters)
        model.create_model(save=False)
    validate_model(model)
    assert all(isinstance(p, csr_matrix) for p in model.transition_matrix)


@pytest.mark.parametrize("requested", [1, 113])
def test_all_model_constructors_accept_positive_requests(
    model_module, requested, request, tmp_path, isolated_model_cache
):
    if run_external_test(request, tmp_path):
        return
    with reproducible_rng(0):
        model = import_module(model_module).Model(state_dim=requested)
    assert model.requested_state_dim == requested
    assert isinstance(model.state_dim, (int, np.integer)) and model.state_dim > 0
    assert isinstance(model.action_dim, (int, np.integer)) and model.action_dim > 0


@pytest.mark.parametrize("discount", [-0.1, 0, 1, 1.1, np.nan])
def test_all_solvers_reject_invalid_discount(
    solver_module, chain, discount, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(solver_module).Solver
    with pytest.raises(AssertionError, match="discount"):
        precision_options = (
            {"final_precision": 1e-4}
            if "final_precision" in signature(solver_class).parameters
            else {}
        )
        solver_class(chain, discount, **precision_options)


def test_all_solvers_accept_plain_model_protocol(
    solver_module, chain, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    model = SimpleNamespace(
        name=chain.name,
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=deepcopy(chain.transition_matrix),
        reward_matrix=chain.reward_matrix.copy(),
    )
    validate_model(model)
    solver_class = import_module(solver_module).Solver
    solver = run_solver_preserving_input(solver_class, model, 0.9, 1e-4)
    assert_optimal_solution(solver, model, 0.9, 1e-4)


@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param(
            "mdptoolbox_mpi", marks=pytest.mark.external(dependencies=["mdptoolbox"])
        ),
        pytest.param(
            "mdpsolver_pi", marks=pytest.mark.external(dependencies=["mdpsolver"])
        ),
    ],
)
def test_span_stopping_rule_does_not_lose_constant_value(
    solver_name, chain, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    chain.transition_matrix = [csr_matrix(np.eye(3)) for _ in range(2)]
    chain.reward_matrix = np.zeros((3, 2))
    chain.reward_matrix[:, 0] = 1
    solver = import_module(f"mdpforge.solvers.{solver_name}").Solver(
        chain, 0.99, final_precision=1e-3
    )
    solver.run()
    assert_allclose(solver.value, 100, atol=1e-3, rtol=0)
    assert np.all(solver.policy == 0)


@pytest.mark.parametrize("discount", [0.5, 0.99])
@pytest.mark.parametrize("constant", [False, True])
def test_all_vi_solvers_meet_precision_on_mixing_chain(
    vi_module, chain, discount, constant, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(vi_module).Solver
    if constant and not getattr(solver_class, "supports_constant_rewards", True):
        pytest.skip("The solver does not support constant rewards")
    transition = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.1, 0.9]])
    rewards = np.ones((3, 2)) if constant else [[1, 0.2], [2, -0.1], [0.5, 0]]
    model = matrix_model("mixing_chain", [transition, transition], rewards)
    if constant and vi_module == "mdpforge.solvers.mdptoolbox_vi":
        # Native pymdptoolbox 4.0b3 cannot initialise its bound for a zero span.
        with pytest.raises(OverflowError):
            run_solver_preserving_input(solver_class, model, discount, 1e-4)
        return
    solver = run_solver_preserving_input(solver_class, model, discount, 1e-4)
    # First action is optimal; linear solve is an independent reference.
    expected = np.linalg.solve(
        np.eye(3) - discount * transition, model.reward_matrix[:, 0]
    )
    assert_allclose(solver.value, expected, atol=1e-4, rtol=0)
    residual = np.abs(
        optimal_bellman_operator(model, solver.value, discount) - solver.value
    ).max()
    assert residual <= 1e-4 * (1 - discount) * (1 + 1e-8)
