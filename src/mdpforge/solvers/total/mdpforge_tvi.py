"""Topological value iteration for absorbing total-reward MDPs.

Adapted from Dai, Mausam, Weld and Goldsmith (JAIR 2011), Algorithm 2.
Maximize undiscounted rewards over all states with zero terminal values.
"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total TVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Total-reward SCC decomposition and dependency-ordered local value "
        "iteration (Dai et al., 2011)."
    ),
}


def _scc_dependency_order(
    model: MDP, enabled: np.ndarray | None = None
) -> list[np.ndarray]:
    """Order SCCs with successors first, retaining only enabled actions."""
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
        (
            np.ones(np.count_nonzero(different), dtype=np.int8),
            (source[different], target[different]),
        ),
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
        for pred in incoming.indices[
            incoming.indptr[component] : incoming.indptr[component + 1]
        ]:
            out_degree[pred] -= 1
            if out_degree[pred] == 0:
                heappush(ready, int(pred))
    if len(ordered) != n_scc:
        raise RuntimeError("TVI: condensation graph is not a DAG")
    return ordered


def _solve_sccs(
    model: MDP,
    value: np.ndarray,
    sccs: list[np.ndarray],
    tolerance: float,
    max_iterations: int,
    terminal: set[int],
    enabled: np.ndarray | None = None,
) -> int:
    """Solve transient components with fixed successor values."""
    actions = model.action_dim
    sweeps = 0
    for states in sccs:
        states = np.array(
            [state for state in states if state not in terminal], dtype=np.intp
        )
        if states.size == 0:
            continue
        if states.size == 1:
            state = int(states[0])
            best = -np.inf
            for action in range(actions):
                if enabled is not None and not enabled[state, action]:
                    continue
                row = model.transition_matrix[action].getrow(state)
                self_prob = float(row[0, state])
                external = float(row.dot(value)[0] - self_prob * value[state])
                reward = model.reward_matrix[state, action]
                if self_prob == 1.0:
                    if reward > 0:
                        raise ValueError(
                            "TVI: positive-reward self-loop has infinite value"
                        )
                    candidate = 0.0 if reward == 0 else -np.inf
                else:
                    candidate = (reward + external) / (1.0 - self_prob)
                best = max(best, candidate)
            if not np.isfinite(best):
                raise RuntimeError(
                    "TVI: nonfinite singleton value / no remaining action"
                )
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
                candidate = rewards[:, action] + matrix.dot(value)
                if active is not None:
                    candidate[~active[:, action]] = -np.inf
                np.maximum(updated, candidate, out=updated)
            if not np.isfinite(updated).all():
                raise RuntimeError("TVI: nonfinite value / no remaining action")
            variation = float(np.max(np.abs(updated - value[states])))
            value[states] = updated
            sweeps += 1
            # The undiscounted local operator is nonexpansive in the maximum norm.
            if variation <= tolerance:
                break
        else:
            raise RuntimeError("TVI: max_iterations reached in an SCC")
    return sweeps


def _greedy_policy(model: MDP, value: np.ndarray) -> np.ndarray:
    best = model.reward_matrix[:, 0] + model.transition_matrix[0].dot(value)
    policy = np.zeros(model.state_dim, dtype=np.intp)
    for action in range(1, model.action_dim):
        candidate = model.reward_matrix[:, action] + model.transition_matrix[
            action
        ].dot(value)
        improvement = candidate > best
        best[improvement] = candidate[improvement]
        policy[improvement] = action
    return policy


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        max_iterations: int = 100_000,
    ):
        """Initialize TVI; max_iterations is the per-SCC VI iteration budget."""
        self.terminal = check_options(model, final_precision, max_iterations)
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        if initial_value is None:
            self.initial_value = np.zeros(model.state_dim, dtype=np.float64)
        else:
            initial_value = np.asarray(initial_value, dtype=np.float64)
            if (
                initial_value.shape != (model.state_dim,)
                or not np.isfinite(initial_value).all()
            ):
                raise ValueError("initial_value must be a finite vector of shape (S,)")
            self.initial_value = initial_value.copy()
        self.initial_value[list(self.terminal)] = 0.0
        self.name = "mdpforge_tvi"
        self.n_scc = 0
        self.largest_scc = 0
        self.n_sweeps = 0
        self.final_residual = np.inf

    def run(self) -> SolveResult:
        """Solve components and check the full undiscounted Bellman residual."""
        value = self.initial_value.copy()
        sccs = _scc_dependency_order(self.model)
        self.n_scc = len(sccs)
        self.largest_scc = max(map(len, sccs))
        tolerance = self.final_precision
        self.n_sweeps = _solve_sccs(
            self.model,
            value,
            sccs,
            tolerance,
            self.max_iterations,
            self.terminal,
        )
        backup = compact_optimal_bellman_operator(self.model, value, 1.0)
        self.final_residual = float(np.max(np.abs(backup - value)))
        if not np.isfinite(self.final_residual) or self.final_residual > tolerance * (
            1 + 1e-10
        ):
            raise RuntimeError("TVI: full Bellman residual exceeds the tolerance")
        return SolveResult(
            value=value,
            policy=_greedy_policy(self.model, value),
            validation_method="residual",
        )
