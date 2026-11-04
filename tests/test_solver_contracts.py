"""Catalogue-wide solver interface and input-preservation checks."""

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

from mdpforge.core.validation import validate_model
from tests._contracts import run_solver_preserving_input
from tests._solver_support import run_external_test


def test_all_solvers_preserve_inputs(solver_case, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_module, mdp_case, discount = solver_case
    solver_class = import_module(solver_module).Solver
    run_solver_preserving_input(solver_class, mdp_case, discount, 1e-4)


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
    run_solver_preserving_input(solver_class, model, 0.9, 1e-4)
