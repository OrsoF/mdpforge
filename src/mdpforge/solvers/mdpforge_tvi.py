"""Topological value iteration on strongly connected MDP components."""

from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_names import solver_names
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "Topological VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Solve transition-graph SCCs in reverse topological order using local VI.",
}


def strongly_connected_components(transitions):
    """Return sink-first SCC state groups for the union of action supports."""
    n = transitions[0].shape[0]
    graph = csr_matrix((n, n), dtype=np.int32)
    for transition in transitions:
        support = transition.copy()
        support.eliminate_zeros()
        support.data = np.ones_like(support.data, dtype=np.int32)
        graph = graph + support
    graph.data[:] = 1
    graph.setdiag(0)
    graph.eliminate_zeros()

    count, labels = connected_components(graph, directed=True, connection="strong")
    if count == 1:
        return [np.arange(n, dtype=np.int64)]

    # A sparse condensation graph holds an edge i -> j if i depends on j.
    member = csr_matrix(
        (np.ones(n, dtype=np.int32), (np.arange(n), labels)), shape=(n, count)
    )
    dependency = (member.T @ graph @ member).tocsr()
    dependency.setdiag(0)
    dependency.eliminate_zeros()
    reverse = dependency.transpose().tocsr()
    remaining = np.diff(dependency.indptr).astype(np.int64)
    queue = deque(np.flatnonzero(remaining == 0).tolist())
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for predecessor in reverse.indices[reverse.indptr[node] : reverse.indptr[node + 1]]:
            remaining[predecessor] -= 1
            if remaining[predecessor] == 0:
                queue.append(int(predecessor))
    if len(order) != count:
        raise RuntimeError("TVI condensation graph unexpectedly contains a cycle")
    grouped_states = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=count))]
    return [grouped_states[offsets[label] : offsets[label + 1]] for label in order]


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure SCC-wise topological value iteration."""
        if not 0 < discount < 1:
            raise ValueError("discount must lie strictly between 0 and 1")
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        super().__init__(model, discount, final_precision, max_iterations=max_iterations)
        initial, _ = initial_vectors(initial_value)
        if initial is not None and (
            initial.shape != (model.state_dim,) or not np.isfinite(initial).all()
        ):
            raise ValueError("initial_value must be a finite vector of shape (S,)")
        self.initial_value = (
            np.zeros(model.state_dim, dtype=float) if initial is None else initial
        )
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        transitions = self.model.transition_matrix
        rewards = self.model.reward_matrix
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)
        components = strongly_connected_components(transitions)
        self.n_components = len(components)
        self.component_sizes = np.array([len(c) for c in components], dtype=int)
        self.backup_sweeps = 0

        for states in components:
            # Rows include transitions leaving the SCC; those targets are solved.
            rows = [transition[states, :].tocsr() for transition in transitions]
            local_rewards = rewards[states, :]
            for _ in range(self.max_iterations):
                q_values = np.column_stack(
                    [local_rewards[:, a] + self.discount * (p @ value)
                     for a, p in enumerate(rows)]
                )
                updated = q_values.max(axis=1)
                if not np.isfinite(updated).all():
                    raise RuntimeError("TVI stopped: non_finite values")
                change = np.max(np.abs(updated - value[states]))
                value[states] = updated
                self.backup_sweeps += 1
                # Every successor outside this SCC was already finalized.
                if change <= tolerance:
                    break
            else:
                raise RuntimeError("TVI stopped: max_iterations reached in an SCC")

        full_residual = optimal_bellman_operator(self.model, value, self.discount) - value
        if np.max(np.abs(full_residual)) > tolerance * (1 + 1e-9) + 1e-12:
            raise RuntimeError("TVI stopped: global Bellman certificate failed")
        return SolveResult(value, None)
