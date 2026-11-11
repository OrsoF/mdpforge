"""Focused Topological Value Iteration (FTVI) for finite discounted MDPs.

Dai, Mausam, Weld & Goldsmith, *Topological Value Iteration Algorithms*,
JAIR (2011), Algorithm 3.

Adaptation: the paper minimizes undiscounted SSP costs. mdpforge maximizes
infinite-horizon discounted reward, hence the sound pruning test is

    upper_Q(s, a) < lower_V(s)  =>  action a is strictly suboptimal.

Bounds use global reward extrema divided by (1 - discount), followed by
monotone Bellman updates on an optimistic-policy search graph. The final
stage solves *all* states, not just those reachable from initial_state,
because mdpforge validates the entire value vector. No state/action is pruned
unless certified by the bounds. Large-component intra-SCC search (an optional
experiment in the paper) is intentionally omitted, as in its final benchmarks.
"""

from __future__ import annotations

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.solvers.discounted.mdpforge_tvi import (
    _greedy_policy,
    _scc_dependency_order,
    _solve_sccs,
)
from mdpforge.utils.solver_options import validate_discount

METADATA = {
    "display_name": "FTVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Bound-certified action elimination followed by topological value iteration (Dai et al., 2011).",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        initial_state: int = 0,
        search_batch_size: int = 20,
        max_search_batches: int = 3,
        min_relative_improvement: float = 0.03,
        max_iterations: int = 100_000,
    ):
        """Configure discounted FTVI.

        initial_state: root of heuristic DFS (default 0); final stage solves all S.
        search_batch_size: DFS trials per batch (paper: 100; default 20 for
            inexpensive pre-processing on the smaller mdpforge benchmarks).
        max_search_batches: hard search budget, distinct from SCC-VI budget.
        min_relative_improvement: stop searching when root lower-bound change
            over a batch is <= this fraction (paper: 0.03).
        max_iterations: maximum VI iterations *per SCC* in the computation stage.
        """
        validate_discount(discount)
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(initial_state, (int, np.integer)) or not (
            0 <= initial_state < model.state_dim
        ):
            raise ValueError("initial_state must be a valid integer state index")
        if not isinstance(search_batch_size, int) or search_batch_size < 1:
            raise ValueError("search_batch_size must be a positive integer")
        if not isinstance(max_search_batches, int) or max_search_batches < 1:
            raise ValueError("max_search_batches must be a positive integer")
        if not np.isfinite(min_relative_improvement) or not (
            0 <= min_relative_improvement < 1
        ):
            raise ValueError("min_relative_improvement must lie in [0, 1)")
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        if initial_value is not None:
            initial_value = np.asarray(initial_value, dtype=np.float64)
            if (
                initial_value.shape != (model.state_dim,)
                or not np.isfinite(initial_value).all()
            ):
                raise ValueError("initial_value must be a finite vector of shape (S,)")
            initial_value = initial_value.copy()
        self.initial_value = initial_value
        self.initial_state = int(initial_state)
        self.search_batch_size = search_batch_size
        self.max_search_batches = max_search_batches
        self.min_relative_improvement = float(min_relative_improvement)
        self.name = "mdpforge_ftvi"
        self.n_search_trials = 0
        self.n_pruned_actions = 0
        self.n_scc = 0
        self.largest_scc = 0
        self.n_sweeps = 0
        self.final_residual = np.inf
        self.enabled: np.ndarray | None = None

    def _optimistic_action(
        self, state: int, upper: np.ndarray, enabled: np.ndarray
    ) -> int:
        """Greedy upper-bound action for a reward-maximizing search."""
        best = -np.inf
        best_action = -1
        for action in np.flatnonzero(enabled[state]):
            matrix = self.model.transition_matrix[action]
            left, right = matrix.indptr[state], matrix.indptr[state + 1]
            q = self.model.reward_matrix[state, action] + self.discount * np.dot(
                matrix.data[left:right], upper[matrix.indices[left:right]]
            )
            if q > best:
                best, best_action = float(q), int(action)
        if best_action < 0:
            raise RuntimeError("FTVI: no action remains enabled")
        return best_action

    def _bound_backup(
        self, state: int, lower: np.ndarray, upper: np.ndarray, enabled: np.ndarray
    ) -> None:
        """Tighten V bounds and eliminate only provably suboptimal actions."""
        actions = self.model.action_dim
        lower_q = np.full(actions, -np.inf, dtype=np.float64)
        upper_q = np.full(actions, -np.inf, dtype=np.float64)
        active = np.flatnonzero(enabled[state])
        for action in active:
            matrix = self.model.transition_matrix[action]
            left, right = matrix.indptr[state], matrix.indptr[state + 1]
            indices, weights = matrix.indices[left:right], matrix.data[left:right]
            reward = self.model.reward_matrix[state, action]
            lower_q[action] = reward + self.discount * np.dot(weights, lower[indices])
            upper_q[action] = reward + self.discount * np.dot(weights, upper[indices])

        # Any Q lower bound is a value-function lower bound for max-reward MDPs.
        lower_certificate = max(float(lower[state]), float(np.max(lower_q[active])))
        scale = max(1.0, abs(lower_certificate), abs(float(upper[state])))
        slack = 1e-12 * scale  # Conservative guard against roundoff.
        eliminated = active[upper_q[active] < lower_certificate - slack]
        if eliminated.size >= active.size:
            # Defensive safeguard; with valid bounds this should not happen.
            eliminated = eliminated[
                eliminated != active[int(np.argmax(upper_q[active]))]
            ]
        enabled[state, eliminated] = False
        self.n_pruned_actions += int(eliminated.size)
        remaining = np.flatnonzero(enabled[state])
        new_lower = max(float(lower[state]), float(np.max(lower_q[remaining])))
        new_upper = min(float(upper[state]), float(np.max(upper_q[remaining])))
        if new_lower > new_upper + 1e-9 * max(1.0, abs(new_lower), abs(new_upper)):
            raise RuntimeError("FTVI: violated value-bound invariants")
        lower[state] = new_lower
        upper[state] = max(new_lower, new_upper)  # numerical roundoff only

    def _search_trial(
        self, lower: np.ndarray, upper: np.ndarray, enabled: np.ndarray
    ) -> int:
        """Iterative depth-first search, with postorder bound backups (Algorithm 3)."""
        n = self.model.state_dim
        visited = np.zeros(n, dtype=np.bool_)
        stack: list[tuple[int, bool]] = [(self.initial_state, False)]
        touched = 0
        while stack:
            state, exiting = stack.pop()
            if exiting:
                self._bound_backup(state, lower, upper, enabled)
                continue
            if visited[state]:
                continue
            visited[state] = True
            touched += 1
            action = self._optimistic_action(state, upper, enabled)
            stack.append((state, True))
            matrix = self.model.transition_matrix[action]
            left, right = matrix.indptr[state], matrix.indptr[state + 1]
            successors = matrix.indices[left:right]
            probabilities = matrix.data[left:right]
            for successor in successors[probabilities > 0][::-1]:
                successor = int(successor)
                if not visited[successor]:
                    stack.append((successor, False))
        return touched

    def _solve(self) -> SolveResult:
        n, actions = self.model.state_dim, self.model.action_dim
        reward = np.asarray(self.model.reward_matrix, dtype=np.float64)
        if not np.isfinite(reward).all():
            raise ValueError("FTVI requires finite rewards")
        lo = float(np.min(reward)) / (1.0 - self.discount)
        hi = float(np.max(reward)) / (1.0 - self.discount)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("FTVI: nonfinite initial bounds")
        lower = np.full(n, lo, dtype=np.float64)
        upper = np.full(n, hi, dtype=np.float64)
        enabled = np.ones((n, actions), dtype=np.bool_)
        self.n_pruned_actions = 0
        self.n_search_trials = 0

        # Heuristic search + certified action elimination, Algorithm 3 step 1.
        for _ in range(self.max_search_batches):
            previous_root = float(lower[self.initial_state])
            for _ in range(self.search_batch_size):
                self._search_trial(lower, upper, enabled)
                self.n_search_trials += 1
            root = float(lower[self.initial_state])
            improvement = abs(root - previous_root) / max(
                1.0, abs(root), abs(previous_root)
            )
            if improvement <= self.min_relative_improvement:
                break

        self.enabled = enabled
        sccs = _scc_dependency_order(self.model, enabled)
        self.n_scc = len(sccs)
        self.largest_scc = max(map(len, sccs))
        base = np.zeros(n) if self.initial_value is None else self.initial_value.copy()
        value = np.clip(base, lower, upper)
        target = self.final_precision * (1.0 - self.discount)
        # Algorithm 3 step 2: TVI on the pruned MDP, with a stricter internal
        # threshold because the public certificate uses the *full* Bellman T.
        local_tolerance = target / 4.0
        self.n_sweeps = 0
        for _ in range(7):
            self.n_sweeps += _solve_sccs(
                self.model,
                self.discount,
                value,
                sccs,
                local_tolerance,
                self.max_iterations,
                enabled,
            )
            full_backup = compact_optimal_bellman_operator(
                self.model, value, self.discount
            )
            self.final_residual = float(np.max(np.abs(full_backup - value)))
            if np.isfinite(self.final_residual) and self.final_residual <= target * (
                1 + 1e-10
            ):
                return SolveResult(
                    value=value,
                    policy=_greedy_policy(self.model, value, self.discount),
                    validation_method="residual",
                )
            local_tolerance /= 4
        raise RuntimeError(
            "FTVI: full Bellman-residual certificate failed after pruning"
        )
