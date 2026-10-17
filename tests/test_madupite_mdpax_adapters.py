"""Reusable backend-adapter contracts, without optional native dependencies."""

from importlib import import_module

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdpforge.utils.madupite import madupite_matrices
from mdpforge.utils.mdpax import padded_transitions
from tests._contracts import assert_optimal_solution
from tests._fake_backends import install_madupite, install_mdpax


def test_madupite_conversion_preserves_transition_rows(mdp_case):
    """madupite flattens (state, action) rows into (S*A, S)."""
    reward, flattened = madupite_matrices(mdp_case)
    assert_array_equal(reward, mdp_case.reward_matrix)
    assert flattened.shape == (
        mdp_case.state_dim * mdp_case.action_dim,
        mdp_case.state_dim,
    )
    for state in range(mdp_case.state_dim):
        for action in range(mdp_case.action_dim):
            assert_allclose(
                flattened[state * mdp_case.action_dim + action].toarray().ravel(),
                mdp_case.transition_matrix[action][state].toarray().ravel(),
            )


def test_mdpax_conversion_preserves_every_sparse_transition(mdp_case):
    """Padding is semantically invisible, including on uneven CSR rows."""
    states, probabilities = padded_transitions(mdp_case, max_event_entries=2_000_000)
    assert states.shape == probabilities.shape
    assert states.shape[:2] == (mdp_case.state_dim, mdp_case.action_dim)
    assert_allclose(probabilities.sum(axis=2), 1)
    for state in range(mdp_case.state_dim):
        for action in range(mdp_case.action_dim):
            actual = np.bincount(
                states[state, action],
                weights=probabilities[state, action],
                minlength=mdp_case.state_dim,
            )
            assert_allclose(
                actual, mdp_case.transition_matrix[action][state].toarray().ravel()
            )


def test_mdpax_conversion_rejects_insufficient_capacity(mdp_case):
    with pytest.raises(ValueError, match="event slots"):
        padded_transitions(mdp_case, max_event_entries=1)


def test_madupite_common_contract(chain, monkeypatch):
    fake = install_madupite(monkeypatch, [1.8, 2, 0])
    solver = import_module("mdpforge.solvers.madupite_ipi").Solver(
        chain, 0.9, final_precision=1e-5
    )
    solver.run()
    assert_optimal_solution(solver, chain, 0.9, 1e-5)
    opts = fake.calls[-1].options
    assert opts["-mode"] == "MAXREWARD"
    assert float(opts["-atol_pi"]) == pytest.approx(1e-6)
    assert fake.calls[-1].transition_path.endswith("P.bin")


def test_madupite_explicitly_rejects_multi_rank_execution(chain, monkeypatch):
    fake = install_madupite(monkeypatch, [1.8, 2, 0])
    fake.state.ranks = (0, 2)
    with pytest.raises(RuntimeError, match="single MPI rank"):
        import_module("mdpforge.solvers.madupite_ipi").Solver(chain, 0.9).run()


@pytest.mark.parametrize("name", ["mdpax_vi", "mdpax_pi"])
def test_mdpax_common_contract(chain, monkeypatch, name):
    fake = install_mdpax(monkeypatch, [1.8, 2, 0])
    adapter = import_module(f"mdpforge.solvers.{name}")
    cls_name = "ValueIteration" if name.endswith("vi") else "PolicyIteration"
    monkeypatch.setattr(adapter, cls_name, getattr(fake.backend, cls_name))
    monkeypatch.setattr(adapter, "make_mdpax_problem", lambda model, cap: (model, cap))
    solver = adapter.Solver(chain, 0.9, final_precision=1e-5)
    solver.run()
    assert_optimal_solution(solver, chain, 0.9, 1e-5)
    call = fake.calls[-1]
    assert call.problem == (chain, 2_000_000)
    assert call.options["gamma"] == 0.9
    assert call.options["epsilon"] == pytest.approx(1e-6)
    assert call.options["checkpoint_frequency"] == 0
    assert call.run_options["max_iterations"] > 0


@pytest.mark.parametrize("name", ["mdpax_vi", "mdpax_pi"])
def test_mdpax_rejects_inaccurate_backend_results(chain, monkeypatch, name):
    fake = install_mdpax(monkeypatch, [0, 0, 0])
    adapter = import_module(f"mdpforge.solvers.{name}")
    cls_name = "ValueIteration" if name.endswith("vi") else "PolicyIteration"
    monkeypatch.setattr(adapter, cls_name, getattr(fake.backend, cls_name))
    monkeypatch.setattr(adapter, "make_mdpax_problem", lambda *_: object())
    with pytest.raises(RuntimeError, match="precision not reached"):
        adapter.Solver(chain, 0.9, final_precision=1e-4).run()


def test_madupite_rejects_inaccurate_backend_results(chain, monkeypatch):
    install_madupite(monkeypatch, [0, 0, 0])
    with pytest.raises(RuntimeError, match="precision not reached"):
        import_module("mdpforge.solvers.madupite_ipi").Solver(
            chain, 0.9, final_precision=1e-4
        ).run()
