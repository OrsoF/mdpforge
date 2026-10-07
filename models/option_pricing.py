# Source: finite-horizon American put option as an optimal-stopping MDP.

import itertools

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        grid = int(np.floor(np.sqrt(max(16, state_dim))))
        self.price_points = max(4, grid)
        self.time_points = max(4, grid)
        self.state_dim = self.price_points * self.time_points + 1
        self.action_dim = 2

        self.exercise_state = self.state_dim - 1
        self.strike = 1.0
        self.up_factor = 1.08
        self.down_factor = 0.94
        self.up_probability = 0.5

        self.name = "{}_{}_option_pricing_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.price_points,
            self.time_points,
        )

    def _build_model(self):
        self.prices = np.linspace(0.25, 2.0, self.price_points)
        self.states = list(
            itertools.product(range(self.time_points), range(self.price_points))
        )
        self.state_to_index = {state: index for index, state in enumerate(self.states)}

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        continue_action = 0
        exercise_action = 1

        for state_index, (time_index, price_index) in enumerate(self.states):
            price = self.prices[price_index]
            payoff = max(self.strike - price, 0.0)

            self.reward_matrix[state_index, exercise_action] = payoff
            self.transition_matrix[exercise_action][
                state_index, self.exercise_state
            ] = 1.0

            if time_index == self.time_points - 1:
                self.reward_matrix[state_index, continue_action] = payoff
                self.transition_matrix[continue_action][
                    state_index, self.exercise_state
                ] = 1.0
            else:
                up_index = self._closest_price_index(price * self.up_factor)
                down_index = self._closest_price_index(price * self.down_factor)
                self.transition_matrix[continue_action][
                    state_index,
                    self.state_to_index[(time_index + 1, up_index)],
                ] += self.up_probability
                self.transition_matrix[continue_action][
                    state_index,
                    self.state_to_index[(time_index + 1, down_index)],
                ] += 1.0 - self.up_probability

        for action in range(self.action_dim):
            self.transition_matrix[action][self.exercise_state, self.exercise_state] = (
                1.0
            )

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _closest_price_index(self, price: float) -> int:
        return int(np.argmin(np.abs(self.prices - price)))
