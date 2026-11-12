"""SHARP-style adaptive block value iteration for absorbing total-reward MDPs.

Adapted from Evangelidis, Vázquez and Gerasimou (FSE 2026).
Maximize undiscounted rewards with zero terminal values and frozen boundaries.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total SHARP · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Total-reward hierarchical adaptive refinement with concrete-state local VI, "
        "frozen boundary values and boundary-change-triggered re-solving."
    ),
    "update_scheme": "asynchronous",
}


@dataclass(slots=True)
class _Block:
    states: np.ndarray
    depth: int
    children: list["_Block"] = field(default_factory=list)
    boundary: np.ndarray | None = None
    inside: list[csr_matrix] | None = None
    outside: list[csr_matrix] | None = None
    previous_boundary: np.ndarray | None = None
    local_residual: float = np.inf


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        final_precision: float = 1e-3,
        *,
        initial_blocks: int = 8,
        max_depth: int = 3,
        refinement_threshold: float = 0.05,
        boundary_threshold: float | None = None,
        partition_strategy: str = "index",
        spread_mode: str = "relative",
        max_iterations: int = 10_000,
        max_local_iterations: int = 100_000,
    ):
        """Configure the total-reward SHARP adaptation (index/scc, absolute/relative)."""
        self.terminal = check_options(model, final_precision, max_iterations)
        if initial_blocks < 1 or max_depth < 1:
            raise ValueError("initial_blocks and max_depth must be positive")
        if max_iterations < 1 or max_local_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if not (np.isfinite(refinement_threshold) and refinement_threshold >= 0):
            raise ValueError("refinement_threshold must be finite and nonnegative")
        if boundary_threshold is not None and not (
            np.isfinite(boundary_threshold) and boundary_threshold >= 0
        ):
            raise ValueError("boundary_threshold must be finite and nonnegative")
        if partition_strategy not in {"index", "scc"}:
            raise ValueError("partition_strategy must be 'index' or 'scc'")
        if spread_mode not in {"absolute", "relative"}:
            raise ValueError("spread_mode must be 'absolute' or 'relative'")

        self.model = model
        self.final_precision = float(final_precision)
        self.initial_blocks = int(initial_blocks)
        self.max_depth = int(max_depth)
        self.refinement_threshold = float(refinement_threshold)
        self.partition_strategy = partition_strategy
        self.spread_mode = spread_mode
        self.max_iterations = int(max_iterations)
        self.max_local_iterations = int(max_local_iterations)
        self.name = "mdpforge_sharp"
        self.runtime = 0.0
        self.policy: np.ndarray | None = None
        self.value: np.ndarray | None = None
        self.root: _Block | None = None
        self.leaves: list[_Block] = []
        self.n_local_solves = 0
        self.n_refinements = 0
        self.n_outer_iterations = 0
        self.final_residual = np.inf

        # This is a residual tolerance, not a discounted value-error bound.
        self._target = self.final_precision
        self._local_tolerance = self._target / 2
        if boundary_threshold is None:
            self.boundary_threshold = self._target / 4
        else:
            self.boundary_threshold = float(boundary_threshold)

    def _initial_partition(self) -> list[np.ndarray]:
        n = self.model.state_dim
        target_blocks = min(n, self.initial_blocks)
        if self.partition_strategy == "index":
            return list(np.array_split(np.arange(n, dtype=np.intp), target_blocks))

        # SCCs are kept together where possible; large SCCs are split into
        # consecutive chunks so local blocks remain computationally bounded.
        adjacency = self.model.transition_matrix[0].copy()
        for matrix in self.model.transition_matrix[1:]:
            adjacency = adjacency + matrix
        _, labels = connected_components(adjacency, directed=True, connection="strong")
        order = np.argsort(labels, kind="stable")
        offsets = np.flatnonzero(np.diff(labels[order])) + 1
        target_size = int(np.ceil(n / target_blocks))
        groups: list[np.ndarray] = []
        pending: list[np.ndarray] = []
        pending_size = 0
        for component in np.split(order, offsets):
            if pending and pending_size + len(component) > target_size:
                groups.append(np.concatenate(pending))
                pending = []
                pending_size = 0
            if len(component) > target_size:
                groups.extend(
                    np.array_split(
                        component, int(np.ceil(len(component) / target_size))
                    )
                )
            else:
                pending.append(component)
                pending_size += len(component)
        if pending:
            groups.append(np.concatenate(pending))
        return groups

    def _make_leaf(self, states: np.ndarray, depth: int) -> _Block:
        states = np.sort(np.asarray(states, dtype=np.intp))
        rows = [matrix[states, :].tocsr() for matrix in self.model.transition_matrix]
        successors = np.unique(np.concatenate([matrix.indices for matrix in rows]))
        boundary = np.setdiff1d(successors, states, assume_unique=True)
        return _Block(
            states=states,
            depth=depth,
            boundary=boundary,
            inside=[matrix[:, states].tocsr() for matrix in rows],
            outside=[matrix[:, boundary].tocsr() for matrix in rows],
        )

    def _solve_leaf(self, block: _Block) -> None:
        states = block.states
        boundary = block.boundary
        snapshot = self.value[boundary].copy()
        local = self.value[states].copy()
        rewards = self.model.reward_matrix[states, :]
        terminal = np.isin(states, list(self.terminal))
        local[terminal] = 0.0
        # Boundary contributions remain fixed during the entire local solve.
        offset = np.array(rewards, copy=True)
        for action, matrix in enumerate(block.outside):
            offset[:, action] += matrix.dot(snapshot)

        for _ in range(self.max_local_iterations):
            updated = offset[:, 0] + block.inside[0].dot(local)
            for action in range(1, self.model.action_dim):
                candidate = offset[:, action] + block.inside[action].dot(local)
                np.maximum(updated, candidate, out=updated)
            updated[terminal] = 0.0
            if not np.isfinite(updated).all():
                raise RuntimeError("SHARP: nonfinite local values")
            variation = float(np.max(np.abs(updated - local)))
            local = updated
            if variation <= self._local_tolerance:
                # The undiscounted local operator is nonexpansive after
                # fixing terminal values at zero.
                block.local_residual = variation
                break
        else:
            raise RuntimeError("SHARP: max_local_iterations exhausted")

        best_value = offset[:, 0] + block.inside[0].dot(local)
        best_action = np.zeros(len(states), dtype=np.intp)
        for action in range(1, self.model.action_dim):
            candidate = offset[:, action] + block.inside[action].dot(local)
            better = candidate > best_value
            best_value[better] = candidate[better]
            best_action[better] = action
        self.value[states] = local
        best_action[terminal] = 0
        self.policy[states] = best_action
        block.previous_boundary = snapshot
        self.n_local_solves += 1

    def _boundary_changed(self, block: _Block) -> bool:
        if block.previous_boundary is None:
            return True
        if block.boundary.size == 0:
            return False
        change = np.max(np.abs(self.value[block.boundary] - block.previous_boundary))
        scale = max(1.0, float(np.max(np.abs(block.previous_boundary))))
        return bool(change > self.boundary_threshold * scale)

    def _should_refine(self, block: _Block) -> bool:
        if block.depth >= self.max_depth or len(block.states) < 2:
            return False
        values = self.value[block.states]
        spread = float(np.max(values) - np.min(values))
        if self.spread_mode == "relative":
            spread /= abs(float(np.mean(values))) + 1e-6
        return spread > self.refinement_threshold

    def _refine(self, block: _Block) -> list[_Block]:
        halfway = len(block.states) // 2
        block.children = [
            self._make_leaf(block.states[:halfway], block.depth + 1),
            self._make_leaf(block.states[halfway:], block.depth + 1),
        ]
        # Only leaves own transition submatrices.
        block.boundary = None
        block.inside = None
        block.outside = None
        block.previous_boundary = None
        for child in block.children:
            self._solve_leaf(child)
        self.n_refinements += 1
        return block.children

    def _extract_policy(self) -> None:
        # Greedify once using the final global values: this removes action
        # inconsistencies caused by stale local boundary snapshots.
        best = self.model.reward_matrix[:, 0] + self.model.transition_matrix[0].dot(
            self.value
        )
        self.policy = np.zeros(self.model.state_dim, dtype=np.intp)
        for action in range(1, self.model.action_dim):
            candidate = self.model.reward_matrix[
                :, action
            ] + self.model.transition_matrix[action].dot(self.value)
            better = candidate > best
            best[better] = candidate[better]
            self.policy[better] = action
        self.policy[list(self.terminal)] = 0

    def run(self) -> SolveResult:
        """Refine local solves until the full total-reward residual is small."""
        start = perf_counter()
        n = self.model.state_dim
        self.value = np.zeros(n, dtype=float)
        self.policy = np.zeros(n, dtype=np.intp)
        self.n_local_solves = self.n_refinements = self.n_outer_iterations = 0
        self.final_residual = np.inf
        self.root = _Block(np.arange(n, dtype=np.intp), depth=0)
        self.root.children = [
            self._make_leaf(states, 1) for states in self._initial_partition()
        ]
        self.leaves = list(self.root.children)
        for leaf in self.leaves:
            self._solve_leaf(leaf)

        for iteration in range(1, self.max_iterations + 1):
            self.n_outer_iterations = iteration
            previous = self.value.copy()
            solved = False
            for leaf in self.leaves:
                if self._boundary_changed(leaf):
                    self._solve_leaf(leaf)
                    solved = True

            refined = False
            new_leaves = []
            for leaf in self.leaves:
                if self._should_refine(leaf):
                    new_leaves.extend(self._refine(leaf))
                    refined = True
                else:
                    new_leaves.append(leaf)
            self.leaves = new_leaves

            change = float(np.max(np.abs(self.value - previous)))
            pending = any(self._boundary_changed(leaf) for leaf in self.leaves)
            if not refined and change <= self._target:
                backup = compact_optimal_bellman_operator(self.model, self.value, 1.0)
                self.final_residual = float(np.max(np.abs(backup - self.value)))
                if self.final_residual <= self._target:
                    # Check the full residual even when some boundaries
                    # remain out of date.
                    self._extract_policy()
                    self.runtime = perf_counter() - start
                    return SolveResult.from_solver(self, validation_method="residual")
                if not pending:
                    # A large user-supplied boundary threshold can hide drift.
                    # Force a sweep rather than accept a low-accuracy result.
                    for leaf in self.leaves:
                        self._solve_leaf(leaf)
            elif not solved and not refined and not pending:
                for leaf in self.leaves:
                    self._solve_leaf(leaf)

        raise RuntimeError(
            "SHARP: max_iterations exhausted before the residual tolerance was reached"
        )
