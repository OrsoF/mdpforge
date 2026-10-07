import time

import numpy as np

from core.model import SPARSE, GenericModel
from core.operators import compute_transition_reward_policy
from utils.bellman import bellman_no_max


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-2,
    ):
        # Class arguments
        self.model = model
        self.discount = discount

        self.name = "PIM"
        self.model._convert_model(SPARSE)

        self.max_iter_eval = int(1e2)
        self.precision_policy_eval = final_precision

    def run(self):
        start_time = time.perf_counter()

        policy = np.zeros(self.model.state_dim, dtype=int)
        value = np.zeros(self.model.state_dim)

        while True:
            value = self._policy_evaluation(
                policy,
                self.precision_policy_eval,
                self.max_iter_eval,
                value,
            )

            new_policy = bellman_no_max(self.model, value, self.discount).argmax(axis=1)
            policy_update_condition = np.all(new_policy == policy)

            if policy_update_condition:
                self.value = value
                self.policy = policy
                self.runtime = time.perf_counter() - start_time
                break
            else:
                policy = new_policy

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsi_eval: float,
        max_iteration_evaluation: int,
        value: np.ndarray,
    ) -> np.ndarray:
        """Policy evaluation step of the modified policy iteration algorithm."""
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)

        tolerance = (
            (1 - self.discount) * epsi_eval if self.discount < 1.0 else epsi_eval
        )

        while True:
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy.dot(value)
            variation = np.absolute(new_value - value).max()
            if variation < tolerance or eval_iter == max_iteration_evaluation:
                break
            value = new_value

        return value

    def _compute_transition_reward_pi(self, policy):
        return compute_transition_reward_policy(self.model, policy)
