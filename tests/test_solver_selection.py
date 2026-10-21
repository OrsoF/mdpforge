from types import SimpleNamespace

import numpy as np
import pytest

from mdpforge import Benchmark, list_solvers
from mdpforge.solvers.mdpforge_vi import Solver as VI


def select_fake_solvers(monkeypatch, constructors, backend="mdpforge"):
    monkeypatch.setattr(
        "mdpforge.core.benchmark.list_solvers",
        lambda **filters: [
            {"name": name, "backend": backend} for name in sorted(constructors)
        ],
    )
    monkeypatch.setattr(
        "mdpforge.core.benchmark.import_module",
        lambda path: SimpleNamespace(Solver=constructors[path.rsplit(".", 1)[1]]),
    )


def test_filtered_pi_solvers_preserve_solution_quality(chain):
    bench = Benchmark().add_mdp(chain)
    assert bench.add_solver(family="pi", backend="mdpforge") is bench
    expected = {e["name"] for e in list_solvers(family="pi", backend="mdpforge")}
    result = bench.run(discount=0.9, precision=1e-4, repeats=1)
    assert {row["solver"] for row in result.rows} == expected
    for row in result.rows:
        assert row["status"] == "success", row["error"]
        assert row["error_bound"] <= 1e-4


def test_filtered_addition_preserves_default_selection_and_is_idempotent(chain):
    bench = Benchmark().add_mdp(chain).add_default_solvers()
    bench.add_solver(family="vi", backend="mdpforge")
    bench.add_solver(family="vi", backend="mdpforge")
    result = bench.run(discount=0.9, precision=1e-4, repeats=1)
    assert {row["solver"] for row in result.rows} == {"mdpforge_vi", "mdpforge_mpi"}
    assert all(row["status"] == "success" for row in result.rows)


def test_group_options_are_forwarded_and_checked_before_registration(
    chain, monkeypatch
):
    budgets = []

    class WithBudget(VI):
        def __init__(self, model, discount, final_precision=1e-3, *, budget=10):
            budgets.append(budget)
            super().__init__(model, discount, final_precision)

    select_fake_solvers(monkeypatch, {"a_budget": WithBudget, "b_plain": VI})
    bench = Benchmark().add_mdp(chain)
    with pytest.raises(TypeError, match="Invalid options for solver 'b_plain'.*budget"):
        bench.add_solver(family="vi", budget=4)
    with pytest.raises(ValueError, match="No solver registered"):
        bench.run(discount=0.9)

    select_fake_solvers(monkeypatch, {"a_budget": WithBudget})
    bench.add_solver(family="vi", budget=4)
    result = bench.run(discount=0.9, repeats=1)
    assert [row["solver"] for row in result.rows] == ["a_budget"]
    assert result.rows[0]["status"] == "success"
    assert budgets and all(budget == 4 for budget in budgets)


def test_group_array_options_deduplicate_and_conflicts_preserve_state(
    chain, monkeypatch
):
    select_fake_solvers(monkeypatch, {"z_existing": VI})
    values = np.array([1.8, 2, 0])
    bench = Benchmark().add_mdp(chain)
    bench.add_solver("z_existing", solve_function=VI, initial_value=values)
    bench.add_solver(family="vi", initial_value=values.copy())
    assert [row["solver"] for row in bench.run(0.9).rows] == ["z_existing"]
    np.testing.assert_array_equal(values, [1.8, 2, 0])

    select_fake_solvers(monkeypatch, {"a_new": VI, "z_existing": VI})
    with pytest.raises(ValueError, match="different implementation or configuration"):
        bench.add_solver(family="vi", initial_value=np.zeros(3))
    assert [row["solver"] for row in bench.run(0.9).rows] == ["z_existing"]


def test_missing_lazy_backend_is_skipped_without_importing_solver(chain, monkeypatch):
    select_fake_solvers(monkeypatch, {"gurobi_primal": VI}, backend="gurobi")
    monkeypatch.setattr("mdpforge.core.benchmark.find_spec", lambda name: None)

    def fail_import(path):
        pytest.fail("An absent lazy backend must be detected before solver import")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark().add_mdp(chain)
    with pytest.warns(UserWarning, match="Skipping solver 'gurobi_primal'.*gurobipy"):
        assert bench.add_solver(family="lp") is bench
    with pytest.raises(ValueError, match="No solver registered"):
        bench.run(discount=0.9)


def test_missing_transitive_dependency_only_skips_the_affected_solver(
    chain, monkeypatch
):
    select_fake_solvers(monkeypatch, {"a_missing": VI, "b_good": VI})

    def import_solver(path):
        if path.endswith("a_missing"):
            raise ModuleNotFoundError("Missing backend helper", name="backend_helper")
        return SimpleNamespace(Solver=VI)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_solver)
    bench = Benchmark().add_mdp(chain)
    with pytest.warns(UserWarning, match="Skipping solver 'a_missing'.*backend_helper"):
        bench.add_solver(family="vi")
    result = bench.run(discount=0.9, repeats=1)
    assert [row["solver"] for row in result.rows] == ["b_good"]
    assert result.rows[0]["status"] == "success"


def test_group_function_options_follow_the_function_calling_convention(
    chain, monkeypatch
):
    calls = []

    def solver(transitions, rewards, discount, precision, *, budget):
        calls.append(budget)
        return np.array([2 * discount, 2, 0])

    select_fake_solvers(monkeypatch, {"example": solver})
    bench = Benchmark().add_mdp(chain)
    bench.add_solver(family="vi", budget=4)
    results = bench.run(0.9, precision=1e-4, repeats=1)
    assert calls == [4]
    assert results.rows[0]["status"] == "success"
    assert results.rows[0]["error_bound"] <= 1e-4


def test_import_errors_do_not_leave_partial_group_registration(chain, monkeypatch):
    select_fake_solvers(monkeypatch, {"a_good": VI, "b_broken": VI})

    def import_solver(path):
        if path.endswith("b_broken"):
            raise RuntimeError("broken integration")
        return SimpleNamespace(Solver=VI)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_solver)
    bench = Benchmark().add_mdp(chain)
    with pytest.raises(RuntimeError, match="broken integration"):
        bench.add_solver(family="vi")
    with pytest.raises(ValueError, match="No solver registered"):
        bench.run(discount=0.9)


def test_named_filters_constrain_but_do_not_reconfigure_solver(monkeypatch):
    bench = Benchmark()
    bench.add_solver("mdpforge_pi", family="pi", backend="mdpforge")

    def fail_import(path):
        pytest.fail("Mismatching named filters must be checked before import")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    with pytest.raises(ValueError, match="does not match"):
        bench.add_solver("mdpforge_vi", family="pi")


def test_custom_solver_and_run_parameters_cannot_be_used_as_group_filters():
    bench = Benchmark()
    with pytest.raises(ValueError, match="custom solve_function"):
        bench.add_solver("custom", solve_function=VI, family="vi")
    with pytest.raises(ValueError, match="parameters controlled by run"):
        bench.add_solver(family="vi", discount=0.9)
    with pytest.raises(ValueError, match="nonempty string name"):
        bench.add_solver()
