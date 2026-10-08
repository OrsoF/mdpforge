import time

import numpy as np

from mdpforge.core.model import GenericModel


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-3,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision

        self.name = "QVI"

        # print("Change the product in at in Personal Value Iteration")

    def q_optimal_bellman_operator(self, q_value: np.ndarray) -> np.ndarray:
        value = q_value.max(axis=1)

        new_q_value = np.empty((self.model.state_dim, self.model.action_dim))
        for aa in range(self.model.action_dim):
            new_q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return new_q_value

    def run(self):
        start_time = time.time()
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        while True:
            new_q_value = self.q_optimal_bellman_operator(q_value)
            bellman_residual = np.abs(new_q_value - q_value).max()
            if bellman_residual < self.epsilon * (1 - self.discount):
                self.value = new_q_value.max(axis=1)
                self.policy = new_q_value.argmax(axis=1)
                break
            q_value = new_q_value

        self.runtime = time.time() - start_time
