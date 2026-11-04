"""Validate every installed catalogue solver on the shared Rooms small preset."""

from importlib import import_module

import pytest

from mdpforge import solvers
from mdpforge.core.validation import validate_result
from mdpforge.models.rooms import METADATA as ROOMS_METADATA
from mdpforge.models.rooms import Model as Rooms
from tests._contracts import run_solver_preserving_input
from tests.conftest import module_names
from tests.test_workflows import run_external_test

DISCOUNT = 0.8
PRECISION = 1e-3


@pytest.fixture(scope="module")
def rooms_small():
    """Build the catalogue's small Rooms instance once, without persistent caching."""
    preset = ROOMS_METADATA["sizes"]["small"]
    model = Rooms(state_dim=preset["requested_state_dim"], **preset["parameters"])
    model.create_model(cache="off")
    return model


@pytest.mark.parametrize("solver_name", module_names(solvers))
def test_all_solvers_validate_rooms_small(solver_name, rooms_small, request, tmp_path):
    if run_external_test(request, tmp_path):
        return
    solver_class = import_module(f"mdpforge.solvers.{solver_name}").Solver
    result = run_solver_preserving_input(
        solver_class, rooms_small, DISCOUNT, PRECISION
    )
    validate_result(rooms_small, result, discount=DISCOUNT, precision=PRECISION)
