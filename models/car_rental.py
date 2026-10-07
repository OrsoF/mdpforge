# Source:
# Sutton, R. S. and Barto, A. G. (2018).
# Reinforcement Learning: An Introduction, 2nd ed., Example 4.2.
#
# Jack's Car Rental. State is the number of cars at two locations. An action
# moves cars overnight from location 1 to location 2 (positive) or conversely.
# Rental requests and returns are independent Poisson random variables.
#
# For tractability, Poisson distributions are truncated and their residual
# mass is accumulated at the truncation point.

import math

import numpy as np
from scipy.sparse import dok_array

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        state_dim = int(np.sqrt(state_dim))
        requested_states = max(25, int(state_dim))
        side = max(5, int(round(np.sqrt(requested_states))))

        self.max_cars = side - 1
        self.max_move = max(1, min(5, (max(3, int(action_dim)) - 1) // 2))
        self.action_values = np.arange(-self.max_move, self.max_move + 1)

        self.state_dim = (self.max_cars + 1) ** 2
        self.action_dim = len(self.action_values)

        self.request_rates = (3.0, 4.0)
        self.return_rates = (3.0, 2.0)
        self.rental_reward = 10.0
        self.movement_cost = 2.0
        self.poisson_cutoff = max(8, self.max_cars + 1)

        self.name = f"{self.state_dim}_{self.action_dim}_jacks_car_rental"

    def _encode(self, cars_1: int, cars_2: int) -> int:
        return cars_1 * (self.max_cars + 1) + cars_2

    @staticmethod
    def _poisson_probabilities(rate: float, cutoff: int) -> np.ndarray:
        probabilities = np.zeros(cutoff + 1, dtype=np.float64)
        probabilities[0] = math.exp(-rate)
        for k in range(1, cutoff):
            probabilities[k] = probabilities[k - 1] * rate / k
        probabilities[cutoff] = max(0.0, 1.0 - probabilities[:cutoff].sum())
        probabilities /= probabilities.sum()
        return probabilities

    def _location_outcomes(
        self,
        cars: int,
        request_rate: float,
        return_rate: float,
    ) -> dict[int, tuple[float, float]]:
        request_probs = self._poisson_probabilities(request_rate, self.poisson_cutoff)
        return_probs = self._poisson_probabilities(return_rate, self.poisson_cutoff)

        # next_cars -> [probability, probability-weighted rentals]
        outcomes: dict[int, list[float]] = {}

        for requests, p_request in enumerate(request_probs):
            rentals = min(cars, requests)
            remaining = cars - rentals

            for returns, p_return in enumerate(return_probs):
                probability = p_request * p_return
                next_cars = min(self.max_cars, remaining + returns)
                if next_cars not in outcomes:
                    outcomes[next_cars] = [0.0, 0.0]
                outcomes[next_cars][0] += probability
                outcomes[next_cars][1] += probability * rentals

        return {
            next_cars: (values[0], values[1]) for next_cars, values in outcomes.items()
        }

    def _build_model(self) -> None:
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )

        location_cache: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}

        for cars_1 in range(self.max_cars + 1):
            for cars_2 in range(self.max_cars + 1):
                state = self._encode(cars_1, cars_2)

                for action_index, requested_move in enumerate(self.action_values):
                    actual_move = int(np.clip(requested_move, -cars_2, cars_1))
                    actual_move = int(
                        np.clip(
                            actual_move,
                            -(self.max_cars - cars_1),
                            self.max_cars - cars_2,
                        )
                    )

                    after_move_1 = cars_1 - actual_move
                    after_move_2 = cars_2 + actual_move

                    key_1 = (0, after_move_1)
                    key_2 = (1, after_move_2)
                    if key_1 not in location_cache:
                        location_cache[key_1] = self._location_outcomes(
                            after_move_1,
                            self.request_rates[0],
                            self.return_rates[0],
                        )
                    if key_2 not in location_cache:
                        location_cache[key_2] = self._location_outcomes(
                            after_move_2,
                            self.request_rates[1],
                            self.return_rates[1],
                        )

                    expected_rentals = 0.0
                    for next_1, (p_1, weighted_rentals_1) in location_cache[
                        key_1
                    ].items():
                        for next_2, (p_2, weighted_rentals_2) in location_cache[
                            key_2
                        ].items():
                            next_state = self._encode(next_1, next_2)
                            probability = p_1 * p_2
                            matrices[action_index][state, next_state] += probability

                            expected_rentals += (
                                weighted_rentals_1 * p_2 + weighted_rentals_2 * p_1
                            )

                    self.reward_matrix[state, action_index] = (
                        self.rental_reward * expected_rentals
                        - self.movement_cost * abs(actual_move)
                    )

        self.transition_matrix = [matrix.tocsr() for matrix in matrices]
