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
from scipy.sparse import csr_matrix, kron

from mdpforge.core.mdp import MDP

METADATA = {
    "sizes": {
        "small": {
            "requested_state_dim": 625,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 15000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 50000,
            "parameters": {},
        },
    },
    "category": "asset_management",
    "description": "Overnight car transfers between two locations with rental demand.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Example 4.2."
    ),
}


class Model(MDP):
    CACHE_VERSION = 1

    def __init__(self, state_dim: int = 625, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
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

        self.name = f"{self.state_dim}_{self.action_dim}_car_rental"

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
        request_probs: np.ndarray,
        return_probs: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        rentals = np.minimum(cars, np.arange(len(request_probs)))
        next_cars = np.minimum(
            self.max_cars,
            (cars - rentals)[:, None] + np.arange(len(return_probs)),
        )
        joint = request_probs[:, None] * return_probs
        probabilities = np.zeros(self.max_cars + 1)
        weighted_rentals = np.zeros(self.max_cars + 1)
        # Unbuffered accumulation preserves the scalar request/return order.
        np.add.at(probabilities, next_cars.ravel(), joint.ravel())
        np.add.at(
            weighted_rentals, next_cars.ravel(), (joint * rentals[:, None]).ravel()
        )
        destinations, first = np.unique(next_cars, return_index=True)
        order = destinations[np.argsort(first)]
        return {
            int(target): (probabilities[target], weighted_rentals[target])
            for target in order
        }

    def _build_model(self) -> None:
        side = self.max_cars + 1
        location_transitions = []
        location_outcomes = []
        for location in range(2):
            request_probs = self._poisson_probabilities(
                self.request_rates[location], self.poisson_cutoff
            )
            return_probs = self._poisson_probabilities(
                self.return_rates[location], self.poisson_cutoff
            )
            probabilities = np.zeros((side, side))
            outcomes = []
            for cars in range(side):
                distribution = self._location_outcomes(cars, request_probs, return_probs)
                for next_cars, (probability, _) in distribution.items():
                    probabilities[cars, next_cars] = probability
                # Keep the original outcome order for reward accumulation.
                outcomes.append(np.array(list(distribution.values())))
            location_transitions.append(csr_matrix(probabilities))
            location_outcomes.append(outcomes)

        joint_transition = kron(*location_transitions, format="csr")
        joint_transition.eliminate_zeros()
        expected_rentals = np.empty((side, side))
        for cars_1, outcomes_1 in enumerate(location_outcomes[0]):
            p_1, rentals_1 = outcomes_1.T
            for cars_2, outcomes_2 in enumerate(location_outcomes[1]):
                p_2, rentals_2 = outcomes_2.T
                weighted_rentals = (
                    rentals_1[:, None] * p_2[None, :]
                    + rentals_2[None, :] * p_1[:, None]
                )
                # cumsum preserves the scalar loop's floating-point sum order.
                expected_rentals[cars_1, cars_2] = np.cumsum(weighted_rentals)[-1]

        cars_1, cars_2 = np.divmod(np.arange(self.state_dim), side)
        self.reward_matrix = np.empty((self.state_dim, self.action_dim))
        self.transition_matrix = []
        for action_index, requested_move in enumerate(self.action_values):
            actual_move = np.clip(requested_move, -cars_2, cars_1)
            actual_move = np.clip(
                actual_move, -(self.max_cars - cars_1), self.max_cars - cars_2
            )
            after_move_1 = cars_1 - actual_move
            after_move_2 = cars_2 + actual_move
            after_move = after_move_1 * side + after_move_2
            self.transition_matrix.append(
                joint_transition
                if requested_move == 0
                else joint_transition[after_move, :]
            )
            self.reward_matrix[:, action_index] = (
                self.rental_reward * expected_rentals[after_move_1, after_move_2]
                - self.movement_cost * np.abs(actual_move)
            )
