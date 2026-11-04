"""Topological Value Iteration (TVI) for finite discounted MDPs.

Dai, Mausam, Weld & Goldsmith, *Topological Value Iteration Algorithms*,
JAIR (2011), Algorithm 2.

The paper uses undiscounted stochastic-shortest-path cost minimization.
This mdpforge implementation instead maximizes expected discounted reward,
solves all states (there is no mandatory initial state), and uses the
repository's full Bellman-residual accuracy criterion.
"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "TVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "SCC decomposition and dependency-ordered local value iteration (Dai et al., 2011).",
}


def _scc_dependency_order(model: MDP, enabled: np.ndarray | None = None) -> list[np.ndarray]:
    """Return SCCs in *reverse* topological order (successors before sources).

    An edge s -> t denotes dependence of V(s) on V(t). With ``enabled``,
    only transitions belonging to non-eliminated actions are retained.
    """
    n, actions = model.state_dim, model.action_dim
    graph = csr_matrix((n, n), dtype=np.float64)
    for action in range(actions):
        transition = model.transition_matrix[action]
        if enabled is not None:
            rows = enabled[:, action]
            if not np.any(rows):
                continue
            transition = diags(rows.astype(np.float64), format="csr") @ transition
        graph = graph + transition
    graph.eliminate_zeros()

    n_scc, labels = connected_components(graph, directed=True, connection="strong")
    order = np.argsort(labels, kind="stable")
    boundaries = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, boundaries)  # groups[id] corresponds to SCC label id

    # Condensation DAG, with edges from dependent SCC to dependency SCC.
    coo = graph.tocoo()
    source, target = labels[coo.row], labels[coo.col]
    different = source != target
    condensed = csr_matrix(
        (np.ones(np.count_nonzero(different), dtype=np.int8),
         (source[different], target[different])),
        shape=(n_scc, n_scc),
    )
    condensed.sum_duplicates()
    condensed.data[:] = 1
    condensed.sort_indices()

    out_degree = np.diff(condensed.indptr).copy()
    incoming = condensed.T.tocsr()
    ready: list[int] = []
    for component in np.flatnonzero(out_degree == 0):
        heappush(ready, int(component))

    ordered: list[np.ndarray] = []
    while ready:
        component = heappop(ready)
        ordered.append(groups[component])
        for pred in incoming.indices[incoming.indptr[component]:incoming.indptr[component + 1]]:
            out_degree[pred] -= 1
            if out_degree[pred] == 0:
                heappush(ready, int(pred))
    if len(ordered) != n_scc:
        raise RuntimeError("TVI: condensation graph is not a DAG")
    return ordered


def _solve_sccs(
    model: MDP,
    discount: float,
    value: np.ndarray,
    sccs: list[np.ndarray],
    tolerance: float,
    max_iterations: int,
    enabled: np.ndarray | None = None,
) -> int:
    """In-place SCC-local Bellman iteration. Returns number of local sweeps.

    Singleton SCCs are solved *exactly*, including a potential self-loop.
    Other components use synchronous restricted VI with fixed outside values.
    No dense S-by-S transition matrix is formed.
    """
    actions = model.action_dim
    sweeps = 0
    for states in sccs:
        if states.size == 1:
            state = int(states[0])
            best = -np.inf
            for action in range(actions):
                if enabled is not None and not enabled[state, action]:
                    continue
                row = model.transition_matrix[action].getrow(state)
                self_prob = float(row[0, state])
                external = float(row.dot(value)[0] - self_prob * value[state])
                candidate = (model.reward_matrix[state, action] + discount * external) / (
                    1.0 - discount * self_prob
                )
                best = max(best, candidate)
            if not np.isfinite(best):
                raise RuntimeError("TVI: nonfinite singleton value / no remaining action")
            value[state] = best
            sweeps += 1
            continue

        # Rows are sliced once per SCC, not once per value-iteration sweep.
        matrices = [matrix[states, :].tocsr() for matrix in model.transition_matrix]
        rewards = model.reward_matrix[states, :]
        active = None if enabled is None else enabled[states, :]
        for _ in range(max_iterations):
            updated = np.full(states.size, -np.inf, dtype=np.float64)
            for action, matrix in enumerate(matrices):
                if active is not None and not np.any(active[:, action]):
                    continue
                candidate = rewards[:, action] + discount * matrix.dot(value)
                if active is not None:
                    candidate[~active[:, action]] = -np.inf
                np.maximum(updated, candidate, out=updated)
            if not np.isfinite(updated).all():
                raise RuntimeError("TVI: nonfinite value / no remaining action")
            variation = float(np.max(np.abs(updated - value[states])))
            value[states] = updated
            sweeps += 1
            # The local Bellman residual at the new vector is <= gamma*variation.
            if variation <= tolerance:
                break
        else:
            raise RuntimeError("TVI: max_iterations reached in an SCC")
    return sweeps


def _greedy_policy(model: MDP, value: np.ndarray, discount: float) -> np.ndarray:
    best = model.reward_matrix[:, 0] + discount * model.transition_matrix[0].dot(value)
    policy = np.zeros(model.state_dim, dtype=np.intp)
    for action in range(1, model.action_dim):
        candidate = model.reward_matrix[:, action] + discount * model.transition_matrix[action].dot(value)
        improvement = candidate > best
        best[improvement] = candidate[improvement]
        policy[improvement] = action
    return policy


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
        """Initialize TVI; max_iterations is the per-SCC VI iteration budget."""
        if not np.isfinite(discount) or not (0.0 < discount < 1.0):
            raise ValueError("discount must lie in (0, 1)")
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        super().__init__(model, discount, final_precision, max_iterations=max_iterations)
        if initial_value is None:
            self.initial_value = np.zeros(model.state_dim, dtype=np.float64)
        else:
            initial_value = np.asarray(initial_value, dtype=np.float64)
            if initial_value.shape != (model.state_dim,) or not np.isfinite(initial_value).all():
                raise ValueError("initial_value must be a finite vector of shape (S,)")
            self.initial_value = initial_value.copy()
        self.name = "mdpforge_tvi"
        self.n_scc = 0
        self.largest_scc = 0
        self.n_sweeps = 0
        self.final_residual = np.inf

    def _solve(self) -> SolveResult:
        value = self.initial_value.copy()
        sccs = _scc_dependency_order(self.model)
        self.n_scc = len(sccs)
        self.largest_scc = max(map(len, sccs))
        tolerance = self.final_precision * (1.0 - self.discount)
        self.n_sweeps = _solve_sccs(
            self.model, self.discount, value, sccs, tolerance,
            self.max_iterations,
        )
        backup = compact_optimal_bellman_operator(self.model, value, self.discount)
        self.final_residual = float(np.max(np.abs(backup - value)))
        if not np.isfinite(self.final_residual) or self.final_residual > tolerance * (1 + 1e-10):
            raise RuntimeError("TVI: global Bellman-residual certificate failed")
        return SolveResult(
            value=value,
            policy=_greedy_policy(self.model, value, self.discount),
            validation_method="residual",
        )
