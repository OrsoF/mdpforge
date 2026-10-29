"""Discounted block-refinement adaptation inspired by SHARP (FSE 2026).

Not a reproduction of the original PRISM SHARP solver: the original solves
reachability/SSP objectives and uses domain-specific graph/spatial partitions.
This implementation solves discounted max-reward MDPs with state-index blocks,
local concrete-state Bellman updates, boundary propagation and value-spread
refinement. All returned values have a full-MDP Bellman residual certificate.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.solvers.mdpforge_tvi import strongly_connected_components
from mdpforge.utils.solver_names import solver_names
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "SHARP-inspired block refinement · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Discounted adaptation of SHARP-style local block VI, boundary updates "
        "and hierarchical value-spread refinement; not upstream PRISM SHARP."
    ),
}


@dataclass
class _Leaf:
    states: np.ndarray
    depth: int
    internal: list[csr_matrix]
    external: list[csr_matrix]
    boundary: np.ndarray
    boundary_snapshot: np.ndarray | None = None


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        initial_blocks: int = 8,
        partition_strategy: str = "scc",
        max_depth: int = 2,
        refinement_threshold: float = 0.5,
        boundary_tolerance: float = 0.0,
        min_block_size: int = 2,
        max_local_iterations: int = 100_000,
        *,
        max_iterations: int = 1000,
    ):
        """Configure hierarchical state-index block refinement and local VI."""
        if not 0 < discount < 1:
            raise ValueError("discount must lie strictly between 0 and 1")
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if not isinstance(initial_blocks, int) or initial_blocks < 1:
            raise ValueError("initial_blocks must be a positive integer")
        if partition_strategy not in {"scc", "index"}:
            raise ValueError("partition_strategy must be 'scc' or 'index'")
        if not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if not np.isfinite(refinement_threshold) or refinement_threshold < 0:
            raise ValueError("refinement_threshold must be finite and nonnegative")
        if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be finite and nonnegative")
        if min_block_size < 1 or max_local_iterations < 1 or max_iterations < 1:
            raise ValueError("iteration budgets and min_block_size must be positive")
        super().__init__(model, discount, final_precision, max_iterations=max_iterations)
        initial, _ = initial_vectors(initial_value)
        if initial is not None and (
            initial.shape != (model.state_dim,) or not np.isfinite(initial).all()
        ):
            raise ValueError("initial_value must be a finite vector of shape (S,)")
        self.initial_value = (
            np.zeros(model.state_dim, dtype=float) if initial is None else initial
        )
        self.initial_blocks = initial_blocks
        self.partition_strategy = partition_strategy
        self.max_depth = max_depth
        self.refinement_threshold = refinement_threshold
        self.boundary_tolerance = boundary_tolerance
        self.min_block_size = min_block_size
        self.max_local_iterations = max_local_iterations
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _make_leaf(self, states: np.ndarray, depth: int) -> _Leaf:
        """Cache concrete transitions inside a block and to its outside boundary."""
        n = self.model.state_dim
        outside_mask = np.ones(n, dtype=bool)
        outside_mask[states] = False
        row_matrices = [transition[states, :].tocsr()
                        for transition in self.model.transition_matrix]
        boundary = np.unique(np.concatenate([p.indices for p in row_matrices]))
        boundary = boundary[outside_mask[boundary]]
        internal = [p[:, states].tocsr() for p in row_matrices]
        external = [p[:, boundary].tocsr() for p in row_matrices]
        return _Leaf(states, depth, internal, external, boundary)

    def _boundary_changed(self, leaf: _Leaf, value: np.ndarray) -> bool:
        if leaf.boundary_snapshot is None:
            return True
        if not len(leaf.boundary):
            return False
        change = np.max(np.abs(value[leaf.boundary] - leaf.boundary_snapshot))
        scale = max(1., np.max(np.abs(leaf.boundary_snapshot)))
        return change > self.boundary_tolerance * scale

    def _solve_leaf(self, leaf: _Leaf, value: np.ndarray, tolerance: float) -> None:
        """Run concrete-state VI with outside-of-block values frozen."""
        states = leaf.states
        boundary_values = value[leaf.boundary].copy()
        rewards = self.model.reward_matrix[states]
        boundary_terms = np.column_stack(
            [p @ boundary_values for p in leaf.external]
        )
        current = value[states].copy()
        for _ in range(self.max_local_iterations):
            q = np.column_stack(
                [rewards[:, a] + self.discount *
                 (p @ current + boundary_terms[:, a])
                 for a, p in enumerate(leaf.internal)]
            )
            updated = q.max(axis=1)
            if not np.isfinite(updated).all():
                raise RuntimeError("SHARP adaptation stopped: non_finite values")
            change = np.max(np.abs(updated - current))
            current = updated
            if change <= tolerance:
                break
        else:
            raise RuntimeError("SHARP adaptation: max_local_iterations reached")
        value[states] = current
        leaf.boundary_snapshot = boundary_values
        self.local_solves += 1

    def _split(self, leaf: _Leaf) -> list[_Leaf]:
        # On each partition, bisect the chosen SCC or state-index ordering.
        # There is no spatial information in the generic mdpforge model.
        midpoint = len(leaf.states) // 2
        return [
            self._make_leaf(leaf.states[:midpoint], leaf.depth + 1),
            self._make_leaf(leaf.states[midpoint:], leaf.depth + 1),
        ]

    def _solve(self) -> SolveResult:
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)
        if self.partition_strategy == "scc":
            ordered_states = np.concatenate(
                strongly_connected_components(self.model.transition_matrix)
            )
        else:
            ordered_states = np.arange(self.model.state_dim, dtype=np.int64)
        leaves = [
            self._make_leaf(states, 0)
            for states in np.array_split(
                ordered_states, min(self.initial_blocks, self.model.state_dim)
            )
        ]
        self.refinements = 0
        self.local_solves = 0
        # Ensures local updates are more accurate than the final global residual.
        local_tolerance = tolerance / 4
        force = True
        for _ in range(self.max_iterations):
            any_solved = False
            for leaf in leaves:
                if force or self._boundary_changed(leaf, value):
                    self._solve_leaf(leaf, value, local_tolerance)
                    any_solved = True
            force = False

            # SHARP-style refinement uses spread of *concrete-state* values.
            refined = []
            any_refined = False
            for leaf in leaves:
                if (
                    leaf.depth < self.max_depth
                    and len(leaf.states) >= 2 * self.min_block_size
                    and np.ptp(value[leaf.states]) > self.refinement_threshold
                ):
                    children = self._split(leaf)
                    for child in children:
                        self._solve_leaf(child, value, local_tolerance)
                    refined.extend(children)
                    self.refinements += 1
                    any_refined = True
                else:
                    refined.append(leaf)
            leaves = refined
            if not np.isfinite(value).all():
                raise RuntimeError("SHARP adaptation stopped: non_finite values")

            # Unlike SHARP's SSP objective, mdpforge requires a *global*
            # discounted value-error certificate at the target accuracy.
            bellman = optimal_bellman_operator(self.model, value, self.discount)
            residual = np.max(np.abs(bellman - value))
            if residual <= tolerance and not any_refined:
                self.n_final_blocks = len(leaves)
                q = np.column_stack(
                    [self.model.reward_matrix[:, a] + self.discount * (p @ value)
                     for a, p in enumerate(self.model.transition_matrix)]
                )
                return SolveResult(value, q.argmax(axis=1).astype(int))
            if not any_solved and not any_refined:
                # A nonzero boundary threshold must not silently accept an
                # uncertified result: revisit all blocks instead.
                force = True
        raise RuntimeError("SHARP adaptation stopped: max_iterations reached")
