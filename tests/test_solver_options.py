"""Constructor contracts and forwarding of explicit solver options."""

import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mdpforge import Benchmark
from mdpforge.solvers.mdpforge_pi import Solver as PI
from tests._contracts import assert_optimal_solution


def test_exact_pi_runs_through_benchmark_without_precision_argument(chain):
    with pytest.raises(TypeError, match="final_precision"):
        PI(chain, 0.9, final_precision=1e-4)
    solver = PI(chain, 0.9, max_iterations=10)
    solver.run()
    assert_optimal_solution(solver, chain, 0.9, 1e-4)

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("exact", solve_function=PI, max_iterations=10)
    result = bench.run(discount=0.9, precision=1e-4, repeats=1)
    assert result.rows[0]["status"] == "success", result.rows[0]["error"]
    assert result.rows[0]["error_bound"] <= 1e-4
    assert result.experiment["solvers"]["exact"]["parameters"]["max_iterations"] == 10


@pytest.mark.parametrize(
    "name",
    [
        "mdpsolver_mpi",
        "mdpsolver_mpigs",
        "mdpsolver_mpigs_init",
        "mdpsolver_mpisor",
        "mdpsolver_mpisor_init",
    ],
)
def test_mdpsolver_variants_share_default_evaluation_budget(chain, monkeypatch, name):
    calls = []
    backend = ModuleType("mdpsolver")

    class Model:
        def mdp(self, **options):
            pass

        def solve(self, **options):
            calls.append(options)

        def getValueVector(self):
            return [1.8, 2, 0]

        def getPolicy(self):
            return [0, 0, 0]

    backend.model = Model
    monkeypatch.setitem(sys.modules, "mdpsolver", backend)
    adapter = import_module(f"mdpforge.solvers.{name}")
    monkeypatch.setattr(adapter, "mdpsolver", backend)
    solver = adapter.Solver(chain, 0.9, final_precision=1e-4)
    solver.run()
    assert calls[-1]["parIterLim"] == 10
    assert_optimal_solution(solver, chain, 0.9, 1e-4)

    options = {"max_evaluation_iterations": 7}
    if "sor" in name:
        options["relaxation"] = 1.2
    if name.endswith("_init"):
        value = np.zeros(3)
        policy = np.zeros(3, dtype=int)
        with pytest.raises(TypeError, match="initial_value"):
            adapter.Solver(chain, 0.9, 1e-3, initial_value=value)
        with pytest.raises(TypeError, match="initial_policy"):
            adapter.Solver(chain, 0.9, 1e-3, initial_policy=policy)
    else:
        value = np.array([1.8, 2, 0])
        policy = np.zeros(3, dtype=int)
        options.update(initial_value=value, initial_policy=policy)
    solver = adapter.Solver(chain, 0.9, final_precision=1e-3, **options)
    solver.run()
    assert calls[-1]["parIterLim"] == 7
    assert calls[-1]["initValueVector"] == value.tolist()
    assert calls[-1]["initPolicy"] == policy.tolist()
    if "sor" in name:
        assert calls[-1]["SORrelaxation"] == 1.2
    assert_optimal_solution(solver, chain, 0.9, 1e-3)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, np.inf])
@pytest.mark.parametrize(
    "name",
    [
        "mdpforge_mpi",
        "mdpforge_mpi_reward_init",
        "aggregated_vi",
        "aggregated_mpi",
        "envelope_vi",
    ],
)
def test_solver_rejects_invalid_explicit_budgets(chain, name, budget):
    solver_class = import_module(f"mdpforge.solvers.{name}").Solver
    with pytest.raises(ValueError, match="max_iterations"):
        solver_class(chain, 0.9, max_iterations=budget)


def test_gurobi_resource_options_are_forwarded(monkeypatch):
    from mdpforge.utils.gurobi import gurobi_model_creation

    parameters = {}
    model = SimpleNamespace(setParam=parameters.__setitem__)
    backend = ModuleType("gurobipy")
    backend.GRB = SimpleNamespace(Param=SimpleNamespace(Threads="Threads"))
    backend.Model = lambda name: model
    monkeypatch.setitem(sys.modules, "gurobipy", backend)

    assert gurobi_model_creation(threads=3, memory_limit_gb=8) is model
    assert parameters["Threads"] == 3
    assert parameters["MemLimit"] == 8
    assert parameters["OutputFlag"] == 0
    assert parameters["LogToConsole"] == 0


@pytest.mark.parametrize(
    "name", ["marmote_vi", "marmote_vigs", "marmote_mpi", "marmote_mpigs"]
)
def test_marmote_iteration_budgets_are_forwarded(chain, monkeypatch, name):
    calls = []

    class MDP:
        def __init__(self, *args):
            pass

        def solve(self, *args):
            calls.append(args)
            return SimpleNamespace(getValueIndex=lambda state: [1.8, 2, 0][state])

        ValueIteration = ValueIterationGS = solve
        PolicyIterationModified = PolicyIterationModifiedGS = solve

    package = ModuleType("marmote")
    core = ModuleType("marmote.core")
    core.MarmoteInterval = lambda start, end: (start, end)
    mdp = ModuleType("marmote.mdp")
    mdp.DiscountedMDP = MDP
    mdp.SolutionMDP = mdp.FeedbackSolutionMDP = object
    monkeypatch.setitem(sys.modules, "marmote", package)
    monkeypatch.setitem(sys.modules, "marmote.core", core)
    monkeypatch.setitem(sys.modules, "marmote.mdp", mdp)
    adapter = import_module(f"mdpforge.solvers.{name}")
    monkeypatch.setattr(adapter, "DiscountedMDP", MDP)
    monkeypatch.setattr(adapter, "MarmoteInterval", core.MarmoteInterval)
    monkeypatch.setattr(adapter, "compute_marmote_args", lambda model: ([], []))
    options = {"max_iterations": 17}
    if "mpi" in name:
        options.update(max_evaluation_iterations=3, evaluation_precision=1e-5)
    solver = adapter.Solver(chain, 0.9, final_precision=1e-4, **options)
    solver.run()

    assert calls[-1][1] == 17
    if "mpi" in name:
        assert calls[-1][2:] == (1e-5, 3)
    assert_optimal_solution(solver, chain, 0.9, 1e-4)
