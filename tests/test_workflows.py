import os
import subprocess
import sys
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from mdpforge.core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from mdpforge.core.validation import validate_model


def run_external_test(request, tmp_path):
    marker = request.node.get_closest_marker("external")
    if marker is None or os.environ.get("MDPFORGE_TEST_CHILD") == "1":
        return False
    for dependency in marker.kwargs["dependencies"]:
        if find_spec(dependency) is None:
            pytest.skip(f"Dependency not installed: {dependency}")
    test_file, test_name = request.node.nodeid.split("::", 1)
    test_id = f"{request.config.rootpath / test_file}::{test_name}"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            test_id,
            "-q",
            "--tb=short",
        ],
        cwd=tmp_path,
        env={**os.environ, "MDPFORGE_TEST_CHILD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[:4000]
    if "1 xfailed" in result.stdout:
        pytest.xfail(result.stdout[-4000:].strip())
    return True


def test_model_contract(
    model_module, isolated_model_cache, request, tmp_path, monkeypatch
):
    if run_external_test(request, tmp_path):
        return
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    module = import_module(model_module)
    parameters = getattr(module, "TEST_PARAMETERS", {"state_dim": 7, "action_dim": 4})
    model = module.Model(**parameters)
    model.create_model(save=False)
    validate_model(model)
    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)


def test_solver_contract(solver_module, chain, monkeypatch, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    # Solver input needs data attributes, without MDP methods.
    chain = SimpleNamespace(
        name=chain.name,
        state_dim=chain.state_dim,
        action_dim=chain.action_dim,
        transition_matrix=chain.transition_matrix,
        reward_matrix=chain.reward_matrix,
    )
    validate_model(chain)
    solver_class = import_module(solver_module).Solver
    for invalid_discount in (-0.1, 0, 1, 1.1, np.nan):
        with pytest.raises(AssertionError, match="discount"):
            solver_class(chain, invalid_discount, final_precision=1e-4)
    reference_model = deepcopy(chain)
    discount = 0.9
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    solver = solver_class(chain, discount, final_precision=1e-4)
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)
    known_sor_bug = solver_module == "mdpforge.solvers.mdpsolver_visor"
    if known_sor_bug:
        request.applymarker(
            pytest.mark.xfail(
                reason="Upstream MDPSolver VI/SOR bug on transient rewards",
                raises=RuntimeError,
                strict=True,
            )
        )
    try:
        solver.run()
    except RuntimeError as exc:
        if known_sor_bug:
            assert str(exc) == ("VI precision not reached: error bound 91 > 0.0001"), (
                f"Unexpected VISOR failure: {exc}"
            )
        raise
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)
    assert solver.value.shape == (chain.state_dim,)
    assert np.all(np.isfinite(solver.value))
    assert_allclose(solver.value, [1.8, 2, 0], atol=1e-3, rtol=0)
    assert_allclose(
        optimal_bellman_operator(reference_model, solver.value, discount),
        solver.value,
        atol=1e-4,
        rtol=0,
    )

    policy = solver.policy
    if policy is None:
        policy = bellman_operator(reference_model, solver.value, discount).argmax(
            axis=1
        )
    assert policy.shape == (chain.state_dim,)
    assert np.issubdtype(policy.dtype, np.integer)
    assert np.all((0 <= policy) & (policy < chain.action_dim))
    transition, reward = compute_transition_reward_policy(reference_model, policy)
    policy_value = np.linalg.solve(
        np.eye(chain.state_dim) - discount * transition.toarray(), reward.astype(float)
    )
    assert_allclose(policy_value, [1.8, 2, 0], atol=1e-3, rtol=0)
    assert np.isfinite(solver.runtime) and solver.runtime >= 0


@pytest.mark.parametrize(
    "solver_module",
    [
        pytest.param(
            "mdptoolbox_mpi",
            marks=pytest.mark.external(dependencies=["mdptoolbox"]),
        ),
        pytest.param(
            "mdpsolver_pi",
            marks=pytest.mark.external(dependencies=["mdpsolver"]),
        ),
    ],
)
def test_span_based_solvers_center_values(solver_module, chain, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(f"mdpforge.solvers.{solver_module}").Solver

    chain.transition_matrix = [
        csr_matrix(np.eye(chain.state_dim)) for _ in range(chain.action_dim)
    ]
    chain.reward_matrix = np.zeros((chain.state_dim, chain.action_dim))
    chain.reward_matrix[:, 0] = 1
    # A constant residual has zero span, but the exact value is 100, not 1.
    solver = solver_class(chain, discount=0.99, final_precision=1e-3)
    solver.run()
    assert_allclose(solver.value, 100, atol=1e-3, rtol=0)
    assert np.all(solver.policy == 0)


def test_vi_final_precision(vi_module, chain, monkeypatch, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(vi_module).Solver
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    transition = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.1, 0.9]])
    for discount, epsilon, constant_reward in (
        (0.5, 1e-3, False),
        (0.99, 1e-5, False),
        (0.95, 1e-3, True),
    ):
        if constant_reward and not getattr(
            solver_class, "supports_constant_rewards", True
        ):
            continue
        model = deepcopy(chain)
        model.transition_matrix = [csr_matrix(transition), csr_matrix(transition)]
        model.reward_matrix = np.array([[1, 0.2], [2, -0.1], [0.5, 0]])
        if constant_reward:
            model.reward_matrix[:] = 1
        reference = deepcopy(model)
        # Action zero dominates everywhere, so this linear solve gives V*.
        expected = np.linalg.solve(
            np.eye(3) - discount * transition, model.reward_matrix[:, 0]
        )
        solver = solver_class(model, discount, final_precision=epsilon)
        solver.run()
        residual = np.abs(
            optimal_bellman_operator(reference, solver.value, discount) - solver.value
        ).max()
        assert residual <= epsilon * (1 - discount)
        assert_allclose(solver.value, expected, atol=epsilon, rtol=0)
