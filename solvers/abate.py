"""
Simple implementation of Abate, Ceska and Kwiatkowska (2016),
"Approximate Policy Iteration for Markov Decision Processes via
Quantitative Adaptive Aggregations".

The implementation follows the same solver formalism as aggregated_qvi.py:
- GenericSolver interface;
- sparse model representation;
- a run() method storing value, policy and runtime;
- exact ground-state fallback when the abstraction becomes too large.

It follows Algorithm 1 of Abate et al.: policy improvement is performed on the
original MDP, while policy evaluation is performed on an aggregated closed-loop
Markov chain. Regions whose evaluation error becomes too large are refined and
policy evaluation is restarted.

The paper does not fully specify initAggregation/reAggregation. Here we use a
simple deterministic value-guided bisection: offending regions are split at the
median of their concrete one-step policy Bellman targets. The policy-evaluation
error uses the paper's quantities X, Y and Z and the region-wise matrix bound of
Equation (6).
"""

from time import time

import numpy as np

from core.conversion import SPARSE
from core.model import GenericModel
from core.operators import bellman_operator, iterative_policy_evaluation
from core.partition import Partition
from core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-2,
        verbose: bool = False,
        max_policy_iterations: int = 100,
        max_value_iterations: int = int(1e3),
        max_error_matrix_entries: int = 1_000,
        direct_threshold: int = 1_000,
    ):
        if not 0 <= discount < 1:
            raise ValueError(
                "abate is a discounted solver; discount must be in [0, 1)."
            )

        self.model = model
        self.discount = discount
        self.epsilon = final_precision / 10
        self.verbose = verbose
        self.max_policy_iterations = max_policy_iterations
        self.max_value_iterations = max_value_iterations
        self.max_error_matrix_entries = max_error_matrix_entries
        self.direct_threshold = direct_threshold
        self.name = "Abate"
        self.policy = None
        self.model._convert_model(SPARSE)

        self.n_states = self.model.state_dim
        raw_reward = self.model.reward_matrix
        # Abate et al. assume a bounded non-negative reward. A constant shift does
        # not change the optimal policy and lets the paper's bounds apply directly.
        self.reward_shift = max(0.0, -float(np.min(raw_reward)))
        self.reward = raw_reward + self.reward_shift

        # Theorem 2: ||J^pi - J*||_inf <= 2 gamma delta / (1-gamma).
        # Hence delta <= epsilon (1-gamma)/(2 gamma) targets final_precision.
        if self.discount > 0:
            self.theta = self.epsilon * (1 - self.discount) / (2 * self.discount)
        else:
            self.theta = self.epsilon

        self.partition = Partition(self.model)

    def run(self):
        start_time = time()

        # Abate et al. allow an arbitrary initial policy. Greedy immediate reward
        # is deterministic and usually a better warm start than a random policy.
        policy = self.reward.argmax(axis=1)
        value = self.reward[np.arange(self.n_states), policy].copy()

        if self.n_states <= self.direct_threshold:
            value, policy = self._exact_policy_iteration(value, policy)
            self._finish(value, policy, start_time)
            return

        for _ in range(self.max_policy_iterations):
            if self.partition.n_regions > 0.9 * self.n_states:
                value, policy = self._exact_policy_iteration(value, policy)
                self._finish(value, policy, start_time)
                return

            value = self._approximate_policy_evaluation(policy, value)
            new_policy = self._greedy_policy(value, policy)

            if np.array_equal(new_policy, policy):
                self._finish(value, policy, start_time)
                return

            policy = new_policy

        # Conservative fallback if approximate PI has not stabilized.
        value, policy = self._exact_policy_iteration(value, policy)
        self._finish(value, policy, start_time)

    def _approximate_policy_evaluation(
        self, policy: np.ndarray, initial_value: np.ndarray
    ) -> np.ndarray:
        """Adaptive aggregated evaluation of a fixed policy."""
        while True:
            if self.partition.n_regions > 0.9 * self.n_states:
                return self._exact_policy_evaluation(policy, initial_value)

            # Section 3.3 uses an m x m matrix Z. Avoid accidental quadratic
            # memory blow-ups in this simple implementation.
            if self.partition.n_regions**2 > self.max_error_matrix_entries:
                return self._exact_policy_evaluation(policy, initial_value)

            (
                phi,
                p_policy,
                reward_policy,
                p_bar,
                reward_bar,
                value_bar,
                x_error,
                y_error,
                z_error,
            ) = self._aggregated_policy_model(policy, initial_value)

            # Equation (6):
            # E_m <= sum gamma^i Y + gamma^m Pbar^m X
            #        + sum gamma^(m-i) Pbar^(m-i-1) Z Jbar_i.
            # These recurrences evaluate the three terms without matrix powers.
            error_y = y_error.copy()
            error_x = x_error.copy()
            error_z = np.zeros(self.partition.n_regions)

            # Coarse Theorem-1 bound B(m), used only as a global PE certificate.
            x_global = float(np.max(x_error))
            y_global = float(np.max(y_error))
            z_global = np.max(z_error, axis=0)
            b_y = y_global
            b_x = x_global
            b_z = float(z_global.dot(value_bar))
            b_previous = b_y + b_x + b_z

            for _ in range(self.max_value_iterations):
                old_value_bar = value_bar
                value_bar = reward_bar + self.discount * p_bar.dot(old_value_bar)

                error_y = y_error + self.discount * error_y
                error_x = self.discount * p_bar.dot(error_x)
                error_z = self.discount * p_bar.dot(
                    error_z
                ) + self.discount * z_error.dot(old_value_bar)
                aggregation_error = error_y + error_x + error_z

                b_y = y_global + self.discount * b_y
                b_x = self.discount * b_x
                b_z = float(z_global.dot(value_bar)) + self.discount * b_z
                b_current = b_y + b_x + b_z

                variation = float(np.max(np.abs(value_bar - old_value_bar)))
                evaluation_bound = 2 * b_current + b_previous + variation

                if evaluation_bound <= self.theta:
                    return phi.dot(value_bar)

                # Algorithm 1 re-aggregates when the aggregation error becomes
                # excessive and restarts evaluation from the previous policy step.
                bad_regions = np.flatnonzero(aggregation_error > self.theta)
                if bad_regions.size:
                    policy_target = reward_policy + self.discount * p_policy.dot(
                        phi.dot(value_bar)
                    )
                    changed = self._refine(bad_regions, policy_target)
                    if changed:
                        break
                    return self._exact_policy_evaluation(policy, initial_value)

                # If the abstract iteration itself has converged while the global
                # bound is still too large, refine the region with largest bound.
                if variation <= self.theta:
                    worst = np.array([int(np.argmax(aggregation_error))])
                    policy_target = reward_policy + self.discount * p_policy.dot(
                        phi.dot(value_bar)
                    )
                    if self._refine(worst, policy_target):
                        break
                    return self._exact_policy_evaluation(policy, initial_value)

                b_previous = b_current
            else:
                return self._exact_policy_evaluation(policy, initial_value)

    def _aggregated_policy_model(self, policy, initial_value):
        """Build the closed-loop aggregate model using uniform region weights."""
        phi, omega = self.partition.phi, self.partition.weights
        p_policy, reward_policy = self.model.transition_reward_policy(policy)
        reward_policy += self.reward_shift
        p_bar, reward_bar = self.partition.compute_agg_trans_reward_pi(
            p_policy, reward_policy
        )
        value_bar = np.asarray(omega @ initial_value).ravel()

        lifted_initial = phi.dot(value_bar)
        lifted_reward = phi.dot(reward_bar)
        x_error = self._region_max(np.abs(initial_value - lifted_initial))
        y_error = self._region_max(np.abs(reward_policy - lifted_reward))
        z_error = self._transition_error_matrix(p_policy, phi, p_bar)

        return (
            phi,
            p_policy,
            reward_policy,
            p_bar,
            reward_bar,
            value_bar,
            x_error,
            y_error,
            z_error,
        )

    def _transition_error_matrix(self, p_policy, phi, p_bar) -> np.ndarray:
        """Compute Z_ij = max_{s in S_i}|P_pi(s,S_j)-Pbar(i,j)|."""
        probabilities = (p_policy.dot(phi)).tocsr()
        k = self.partition.n_regions
        z_error = np.zeros((k, k))

        for region, states in enumerate(self.partition.states_in_region):
            idx = np.asarray(states)
            block = probabilities[idx]
            pbar_row = p_bar.getrow(region).toarray().ravel()

            # For a fixed target region j, the largest deviation from Pbar_ij
            # occurs at the minimum or maximum concrete probability.
            pmax = np.asarray(block.max(axis=0).toarray()).ravel()
            pmin = np.asarray(block.min(axis=0).toarray()).ravel()
            z_error[region] = np.maximum(
                np.abs(pmax - pbar_row), np.abs(pmin - pbar_row)
            )

        return z_error

    def _refine(self, bad_regions: np.ndarray, score: np.ndarray) -> bool:
        """Split each offending non-singleton region by the median score."""
        changed = False
        for region in bad_regions:
            idx = np.asarray(self.partition.states_in_region[region])
            if idx.size <= 1:
                continue

            values = score[idx]
            order = np.argsort(values, kind="stable")
            cut = idx.size // 2

            # If the score is completely flat, the stable state-index split still
            # guarantees progress. The paper leaves this clustering heuristic open.
            self.partition.refine_region(
                int(region), [idx[order[:cut]], idx[order[cut:]]]
            )
            changed = True

        return changed

    def _greedy_policy(
        self, value: np.ndarray, policy: np.ndarray | None = None
    ) -> np.ndarray:
        q_value = bellman_operator(self.model, value, self.discount)
        greedy = q_value.argmax(axis=1)
        if policy is None:
            return greedy

        states = np.arange(self.n_states)
        improve = q_value[states, greedy] > q_value[states, policy] + 1e-12
        return np.where(improve, greedy, policy)

    def _exact_policy_evaluation(
        self, policy: np.ndarray, value: np.ndarray
    ) -> np.ndarray:
        p_policy, reward_policy = self.model.transition_reward_policy(policy)
        reward_policy += self.reward_shift
        return iterative_policy_evaluation(
            p_policy,
            reward_policy,
            self.discount,
            self.epsilon * (1 - self.discount),
            self.max_value_iterations,
            value,
        )

    def _exact_policy_iteration(self, value, policy):
        for _ in range(self.max_policy_iterations):
            value = self._exact_policy_evaluation(policy, value)
            new_policy = self._greedy_policy(value, policy)
            if np.array_equal(new_policy, policy):
                return value, policy
            policy = new_policy
        return value, policy

    def _region_max(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.partition.n_regions, -np.inf)
        np.maximum.at(out, self.partition.state_to_region, np.asarray(values).ravel())
        return out

    def _finish(self, value: np.ndarray, policy: np.ndarray, start_time: float):
        value = np.asarray(value).ravel()
        if self.reward_shift and self.discount < 1:
            value = value - self.reward_shift / (1 - self.discount)
        self.value = value
        self.policy = np.asarray(policy, dtype=int)
        self.runtime = time() - start_time
