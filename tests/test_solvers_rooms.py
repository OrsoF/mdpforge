"""Validate every installed catalogue solver on a small shared Rooms instance."""

from importlib import import_module

import pytest

from mdpforge.core.validation import validate_result
from mdpforge.models.discounted.rooms import Model as Rooms
from tests._contracts import run_solver_preserving_input
from tests._solver_support import run_external_test, solver_names

DISCOUNT = 0.8
PRECISION = 1e-3


@pytest.fixture(scope="module")
def rooms_small():
    """Build a 100-state Rooms instance once, without persistent caching."""
    model = Rooms(state_dim=100)
    model.create_model(cache="off")
    return model


@pytest.mark.parametrize("solver_name", solver_names())
def test_all_solvers_validate_rooms_small(solver_name, rooms_small, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(f"mdpforge.solvers.{solver_name}").Solver
    result = run_solver_preserving_input(solver_class, rooms_small, DISCOUNT, PRECISION)
    validate_result(rooms_small, result, discount=DISCOUNT, precision=PRECISION)
