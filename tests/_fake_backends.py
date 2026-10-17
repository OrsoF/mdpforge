"""Small controlled doubles for optional native backends (no PETSc/JAX needed)."""

from types import ModuleType, SimpleNamespace
import sys

import numpy as np


def install_madupite(monkeypatch, values, policy=None):
    """Capture PETSc options and write backend-shaped value/policy artifacts."""
    calls = []
    state = SimpleNamespace(ranks=(0, 1), values=np.asarray(values), policy=policy)
    backend = ModuleType("madupite")
    backend.MatrixCategory = SimpleNamespace(Cost="cost", Dynamics="dynamics")
    backend.MatrixType = SimpleNamespace(Dense="dense", Sparse="sparse")
    backend.mpi_rank_size = lambda: state.ranks
    backend.written = {}

    class Matrix:
        @staticmethod
        def fromFile(*, filename, category, type):
            return filename, category, type

    class MDPBackend:
        def __init__(self):
            self.options = {}
            calls.append(self)

        def setStageCostMatrix(self, matrix):
            self.cost_path = matrix[0]

        def setTransitionProbabilityTensor(self, matrix):
            self.transition_path = matrix[0]

        def setOption(self, key, value):
            self.options[key] = value

        def solve(self):
            np.savetxt(self.options["-file_cost"], state.values)
            actions = (
                np.zeros(len(state.values), dtype=int)
                if state.policy is None
                else state.policy
            )
            np.savetxt(self.options["-file_policy"], actions, fmt="%d")

    def write_petsc(matrix, path):
        backend.written[path] = matrix

    backend.Matrix = Matrix
    backend.MDP = MDPBackend
    backend.writePETScBinary = write_petsc
    monkeypatch.setitem(sys.modules, "madupite", backend)
    # The generic adapter binds md at import time; replace it for every test.
    from mdpforge.solvers import madupite_ipi

    monkeypatch.setattr(madupite_ipi, "md", backend)
    return SimpleNamespace(calls=calls, state=state, backend=backend)


def install_mdpax(monkeypatch, values, policy=None):
    """Capture JAX solver configuration without importing JAX or MDPax."""
    calls = []
    backend = ModuleType("mdpax.solvers")
    root = ModuleType("mdpax")
    root.__path__ = []
    state = SimpleNamespace(values=np.asarray(values), policy=policy)

    class FakeSolver:
        def __init__(self, problem, **options):
            self.problem = problem
            self.options = options
            calls.append(self)

        def solve(self, **options):
            self.run_options = options
            policy = (
                np.zeros((len(state.values), 1), dtype=int)
                if state.policy is None
                else state.policy
            )
            return SimpleNamespace(values=state.values, policy=policy)

    backend.ValueIteration = FakeSolver
    backend.PolicyIteration = FakeSolver
    monkeypatch.setitem(sys.modules, "mdpax", root)
    monkeypatch.setitem(sys.modules, "mdpax.solvers", backend)
    return SimpleNamespace(calls=calls, state=state, backend=backend)
