"""Discounted value references computed by VI at a requested precision."""

from mdpforge.core.mdp import MDP


def compute_value_function(model: MDP, discount: float, precision: float):
    """Solve with VI while preserving the model's matrix representation."""
    from mdpforge.solvers.mdpforge_vi import Solver

    solver = Solver(model, discount, precision)
    return solver.run().value
