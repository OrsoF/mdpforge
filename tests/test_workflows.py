"""Catalog-wide scientific contracts: new models/solvers inherit these checks."""

import os
import subprocess
import sys
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from numpy.testing import assert_allclose

from mdpforge.core.validation import validate_model
from tests._contracts import assert_optimal_solution, run_solver_preserving_input


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


def test_all_solvers_meet_precision_and_preserve_inputs(
    solver_case, request, tmp_path
):
    if run_external_test(request, tmp_path):
        return
    solver_module, mdp_case, discount = solver_case
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
