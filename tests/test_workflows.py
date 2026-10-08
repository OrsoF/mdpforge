import os
import subprocess
import sys
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec

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


def run_optional_test(request, tmp_path):
    marker = request.node.get_closest_marker("optional")
    if marker is None or os.environ.get("MDPFORGE_TEST_CHILD") == "1":
        return False
    for dependency in marker.kwargs["dependencies"]:
        if find_spec(dependency) is None:
            pytest.skip(f"Optional dependency not installed: {dependency}")
    test_file, test_name = request.node.nodeid.split("::", 1)
    test_id = f"{request.config.rootpath / test_file}::{test_name}"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            test_id,
            "--optional",
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
    return True


def test_model_contract(
    model_module, isolated_model_cache, request, tmp_path, monkeypatch
):
    if run_optional_test(request, tmp_path):
        return
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    module = import_module(model_module)
    parameters = getattr(module, "TEST_PARAMETERS", {"state_dim": 7, "action_dim": 4})
    model = module.Model(**parameters)
    model.create_model(save=False)
    validate_model(model)
    assert all(isinstance(matrix, csr_matrix) for matrix in model.transition_matrix)


def test_solver_contract(solver_module, chain, monkeypatch, request, tmp_path):
    if run_optional_test(request, tmp_path):
        return
    solver_class = import_module(solver_module).Solver
    for invalid_discount in (-0.1, 0, 1, 1.1, np.nan):
        with pytest.raises(AssertionError, match="discount"):
            solver_class(chain, invalid_discount, final_precision=1e-4)
    reference_model = deepcopy(chain)
    discount = 0.9
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    solver = solver_class(chain, discount, final_precision=1e-4)
    assert all(isinstance(matrix, csr_matrix) for matrix in chain.transition_matrix)
    solver.run()
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


def test_vi_final_precision(vi_module, chain, monkeypatch, request, tmp_path):
    if run_optional_test(request, tmp_path):
        return
    solver_class = import_module(vi_module).Solver
    monkeypatch.setattr(np.random, "randint", np.random.RandomState(0).randint)
    transition = np.array([[0.7, 0.3, 0], [0.2, 0.5, 0.3], [0, 0.1, 0.9]])
    for discount, epsilon, constant_reward in (
        (0.5, 1e-3, False),
        (0.99, 1e-5, False),
        (0.95, 1e-3, True),
    ):
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
