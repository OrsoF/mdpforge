from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdpforge import Benchmark, list_solvers
from mdpforge.solvers.mdpforge_vi import Solver as VI
from tests._contracts import run_solver_preserving_input


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
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    assert bench.add_solver(family="pi", backend="mdpforge") is bench
    expected = {e["name"] for e in list_solvers(family="pi", backend="mdpforge")}
    assert set(bench._solvers) == expected
    result = bench.run(discount=0.9, precision=1e-4, repeats=1)
    for row in result.rows:
        assert row["status"] == "success", row["error"]
        assert row["error_bound"] <= 1e-4
    for constructor, options in bench._solvers.values():
        solver = run_solver_preserving_input(constructor, chain, 0.9, 1e-4, **options)
        assert_allclose(solver.value, [1.8, 2, 0], atol=1e-4, rtol=0)


def test_filtered_addition_preserves_default_vi_and_is_idempotent():
    bench = Benchmark()
    bench.add_solver(family="vi", backend="mdpforge")
    assert "VI" in bench._solvers
    assert "mdpforge_vi" not in bench._solvers
    assert sum(constructor is VI for constructor, _ in bench._solvers.values()) == 1
    original = bench._solvers.copy()
    bench.add_solver(family="vi", backend="mdpforge")
    assert bench._solvers == original


def test_group_options_are_forwarded_and_checked_before_registration(monkeypatch):
    class WithBudget(VI):
        def __init__(self, model, discount, final_precision=1e-3, *, budget=10):
            super().__init__(model, discount, final_precision)

    select_fake_solvers(monkeypatch, {"a_budget": WithBudget, "b_plain": VI})
    bench = Benchmark(default_solvers=False)
    with pytest.raises(TypeError, match="Invalid options for solver 'b_plain'.*budget"):
        bench.add_solver(family="vi", budget=4)
    assert bench._solvers == {}

    select_fake_solvers(monkeypatch, {"a_budget": WithBudget})
    bench.add_solver(family="vi", budget=4)
    assert bench._solvers == {"a_budget": (WithBudget, {"budget": 4})}


def test_group_array_options_deduplicate_and_conflicts_preserve_state(monkeypatch):
    select_fake_solvers(monkeypatch, {"z_existing": VI})
    values = np.array([1.8, 2, 0])
    bench = Benchmark(default_solvers=False)
    bench.add_solver("z_existing", solve_function=VI, initial_value=values)
    bench.add_solver(family="vi", initial_value=values.copy())
    assert list(bench._solvers) == ["z_existing"]
    assert bench._solvers["z_existing"][1]["initial_value"] is values

    select_fake_solvers(monkeypatch, {"a_new": VI, "z_existing": VI})
    with pytest.raises(ValueError, match="different implementation or configuration"):
        bench.add_solver(family="vi", initial_value=np.zeros(3))
    assert list(bench._solvers) == ["z_existing"]


def test_missing_lazy_backend_is_skipped_without_importing_solver(monkeypatch):
    select_fake_solvers(monkeypatch, {"gurobi_primal": VI}, backend="gurobi")
    monkeypatch.setattr("mdpforge.core.benchmark.find_spec", lambda name: None)

    def fail_import(path):
        pytest.fail("An absent lazy backend must be detected before solver import")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    bench = Benchmark()
    with pytest.warns(UserWarning, match="Skipping solver 'gurobi_primal'.*gurobipy"):
        assert bench.add_solver(family="lp") is bench
    assert list(bench._solvers) == ["VI"]


def test_missing_transitive_dependency_only_skips_the_affected_solver(monkeypatch):
    select_fake_solvers(monkeypatch, {"a_missing": VI, "b_good": VI})

    def import_solver(path):
        if path.endswith("a_missing"):
            raise ModuleNotFoundError("Missing backend helper", name="backend_helper")
        return SimpleNamespace(Solver=VI)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_solver)
    bench = Benchmark(default_solvers=False)
    with pytest.warns(UserWarning, match="Skipping solver 'a_missing'.*backend_helper"):
        bench.add_solver(family="vi")
    assert bench._solvers == {"b_good": (VI, {})}


def test_group_function_options_follow_the_function_calling_convention(
    chain, monkeypatch
):
    calls = []

    def solver(transitions, rewards, discount, precision, *, budget):
        calls.append(budget)
        return np.array([2 * discount, 2, 0])

    select_fake_solvers(monkeypatch, {"example": solver})
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver(family="vi", budget=4)
    results = bench.run(0.9, precision=1e-4, repeats=1)
    assert calls == [4]
    assert results.rows[0]["status"] == "success"
    assert results.rows[0]["error_bound"] <= 1e-4


def test_import_errors_do_not_leave_partial_group_registration(monkeypatch):
    select_fake_solvers(monkeypatch, {"a_good": VI, "b_broken": VI})

    def import_solver(path):
        if path.endswith("b_broken"):
            raise RuntimeError("broken integration")
        return SimpleNamespace(Solver=VI)

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", import_solver)
    bench = Benchmark(default_solvers=False)
    with pytest.raises(RuntimeError, match="broken integration"):
        bench.add_solver(family="vi")
    assert bench._solvers == {}


def test_named_filters_constrain_but_do_not_reconfigure_solver(monkeypatch):
    bench = Benchmark(default_solvers=False)
    bench.add_solver("mdpforge_pi", family="pi", backend="mdpforge")

    def fail_import(path):
        pytest.fail("Mismatching named filters must be checked before import")

    monkeypatch.setattr("mdpforge.core.benchmark.import_module", fail_import)
    with pytest.raises(ValueError, match="does not match"):
        bench.add_solver("mdpforge_vi", family="pi")


def test_custom_solver_and_run_parameters_cannot_be_used_as_group_filters():
    bench = Benchmark(default_solvers=False)
    with pytest.raises(ValueError, match="custom solve_function"):
        bench.add_solver("custom", solve_function=VI, family="vi")
    with pytest.raises(ValueError, match="parameters controlled by run"):
        bench.add_solver(family="vi", discount=0.9)
    with pytest.raises(ValueError, match="nonempty string name"):
        bench.add_solver()
