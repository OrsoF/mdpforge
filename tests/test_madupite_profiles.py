"""Each named PETSc configuration is only a profile of one madupite adapter."""

from importlib import import_module

import pytest

from mdpforge.solvers import list_solvers
from tests._contracts import assert_optimal_solution
from tests._fake_backends import install_madupite


# (name, family, Krylov solver, preconditioner, inner iteration budget)
PROFILES = [
    ("madupite_vi", "vi", "richardson", "none", 1),
    ("madupite_pi", "pi", "richardson", "lu", 1),
    ("madupite_opi", "mpi", "richardson", "none", 20),
    ("madupite_ipi_bicgstab", "pi", "bcgs", "none", 1000),
    ("madupite_ipi_tfqmr", "pi", "tfqmr", "none", 1000),
    ("madupite_ipi_gmres", "pi", "gmres", "none", 1000),
]


@pytest.mark.parametrize("name,family,ksp,pc,limit", PROFILES)
def test_profile_options_and_shared_solution_contract(
    chain, monkeypatch, name, family, ksp, pc, limit
):
    fake = install_madupite(monkeypatch, [1.8, 2, 0])
    solver = import_module(f"mdpforge.solvers.{name}").Solver(
        chain, discount=0.9, final_precision=1e-5
    )
    solver.run()
    assert_optimal_solution(solver, chain, 0.9, 1e-5)
    assert solver.solver_type == family
    opts = fake.calls[-1].options
    assert opts["-ksp_type"] == ksp
    assert opts["-pc_type"] == pc
    assert opts["-ksp_max_it"] == str(limit)
    assert opts["-max_iter_ksp"] == str(limit)
    assert opts["-mode"] == "MAXREWARD"
    assert float(opts["-atol_pi"]) == pytest.approx(1e-6)


def test_opi_custom_evaluation_budget(chain, monkeypatch):
    fake = install_madupite(monkeypatch, [1.8, 2, 0])
    cls = import_module("mdpforge.solvers.madupite_opi").Solver
    cls(chain, 0.9, max_evaluation_iterations=7, max_iterations=120).run()
    opts = fake.calls[-1].options
    assert opts["-ksp_max_it"] == opts["-max_iter_ksp"] == "7"
    assert opts["-max_iter_pi"] == "120"
    with pytest.raises(ValueError, match="positive"):
        cls(chain, 0.9, max_evaluation_iterations=0)


def test_krylov_limit_isolation_across_profiles(chain, monkeypatch):
    fake = install_madupite(monkeypatch, [1.8, 2, 0])
    import_module("mdpforge.solvers.madupite_vi").Solver(chain, 0.9).run()
    import_module("mdpforge.solvers.madupite_ipi_bicgstab").Solver(
        chain, 0.9, max_evaluation_iterations=84
    ).run()
    assert fake.calls[-2].options["-ksp_max_it"] == "1"
    assert fake.calls[-1].options["-ksp_max_it"] == "84"
    assert fake.calls[-1].options["-ksp_type"] == "bcgs"


def test_all_profiles_advertised_without_importing_native_backend():
    catalog = {entry["name"]: entry for entry in list_solvers(backend="madupite")}
    for name, family, *_ in PROFILES:
        assert catalog[name]["family"] == family
    assert "madupite_ipi" in catalog
