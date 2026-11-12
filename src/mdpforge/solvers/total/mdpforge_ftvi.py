"""Focused topological value iteration for absorbing total-reward MDPs.

Adapted from Dai, Mausam, Weld and Goldsmith (JAIR 2011), Algorithm 3.
Maximize undiscounted rewards over all states with zero terminal values.
Prune using uniform absorption bounds when available; otherwise use TVI.
"""

from __future__ import annotations

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compact_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options
from mdpforge.solvers.total.mdpforge_tvi import (
    _greedy_policy,
    _scc_dependency_order,
    _solve_sccs,
)

METADATA = {
    "display_name": "Total FTVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Total-reward bound-certified action elimination and topological value "
        "iteration, falling back to TVI without uniform absorption bounds."
    ),
}


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        initial_state: int = 0,
        search_batch_size: int = 20,
        max_search_batches: int = 3,
        min_relative_improvement: float = 0.03,
        max_iterations: int = 100_000,
    ):
        """Configure total FTVI with a full Bellman-residual tolerance.

        initial_state: valid state index; root of heuristic search.
        search_batch_size: positive int; search trials per batch.
        max_search_batches: positive int; search batch budget.
        min_relative_improvement: float in [0, 1); search stopping threshold.
        max_iterations: positive int; local VI and absorption-bound budgets.
        """
        self.terminal = check_options(model, final_precision, max_iterations)
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
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
            q = self.model.reward_matrix[state, action] + np.dot(
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
            lower_q[action] = reward + np.dot(weights, lower[indices])
            upper_q[action] = reward + np.dot(weights, upper[indices])

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
            if visited[state] or state in self.terminal:
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

    def _initial_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return total-reward bounds when uniform absorption is established."""
        reward = np.asarray(self.model.reward_matrix, dtype=np.float64)
        if not np.isfinite(reward).all():
            raise ValueError("FTVI requires finite rewards")
        survival = np.ones(self.model.state_dim)
        survival[list(self.terminal)] = 0.0
        # The maximal probability of surviving k steps bounds every policy.
        # If it is <= q < 1, the expected absorption time is <= k / (1-q).
        for steps in range(1, min(self.model.state_dim, self.max_iterations) + 1):
            next_survival = np.zeros(self.model.state_dim)
            for matrix in self.model.transition_matrix:
                np.maximum(next_survival, matrix.dot(survival), out=next_survival)
            next_survival[list(self.terminal)] = 0.0
            contraction = float(np.max(next_survival)) + 1e-12
            if contraction < 1.0:
                horizon = steps / (1.0 - contraction)
                break
            if np.array_equal(next_survival, survival):
                return None
            survival = next_survival
        else:
            return None
        lower = np.full(self.model.state_dim, min(0.0, float(reward.min())) * horizon)
        upper = np.full(self.model.state_dim, max(0.0, float(reward.max())) * horizon)
        lower[list(self.terminal)] = upper[list(self.terminal)] = 0.0
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            return None
        return lower, upper

    def run(self) -> SolveResult:
        """Prune actions with total-reward bounds, then solve the components."""
        n, actions = self.model.state_dim, self.model.action_dim
        bounds = self._initial_bounds()
        enabled = np.ones((n, actions), dtype=np.bool_)
        self.n_pruned_actions = 0
        self.n_search_trials = 0

        value = np.zeros(n) if self.initial_value is None else self.initial_value.copy()
        value[list(self.terminal)] = 0.0
        if bounds is not None:
            lower, upper = bounds
            # Heuristic search and certified elimination, Algorithm 3 step 1.
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
            value = np.clip(value, lower, upper)

        self.enabled = enabled
        sccs = _scc_dependency_order(self.model, enabled)
        self.n_scc = len(sccs)
        self.largest_scc = max(map(len, sccs))
        target = self.final_precision
        # Algorithm 3 step 2: TVI on the pruned MDP, with a stricter internal
        # threshold because the final check uses the *full* Bellman T.
        local_tolerance = target / 4.0
        self.n_sweeps = 0
        for _ in range(7):
            self.n_sweeps += _solve_sccs(
                self.model,
                value,
                sccs,
                local_tolerance,
                self.max_iterations,
                self.terminal,
                enabled,
            )
            full_backup = compact_optimal_bellman_operator(self.model, value, 1.0)
            self.final_residual = float(np.max(np.abs(full_backup - value)))
            if np.isfinite(self.final_residual) and self.final_residual <= target * (
                1 + 1e-10
            ):
                return SolveResult(
                    value=value,
                    policy=_greedy_policy(self.model, value),
                    validation_method="residual",
                )
            local_tolerance /= 4
        raise RuntimeError(
            "FTVI: full Bellman residual exceeds the tolerance after pruning"
        )
