# Source: Howard, R. A. (1960). Dynamic Programming and Markov Processes, p.54, The Replacement Problem.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

param_list = [{}]

METADATA = {
    "category": "resource_management",
    "description": "Replacement decisions for equipment with aging-related costs.",
    "reference": (
        "Howard, R. A. (1960). Dynamic Programming and Markov Processes, p.54, The "
        "Replacement Problem."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.current_car_value = np.linspace(1600, 80, self.state_dim)
        self.new_car_cost = np.linspace(2000, 130, self.action_dim)
        self.car_maintain_cost = np.linspace(50, 250, self.state_dim)
        self.survival_proba = np.linspace(1.0, 0.0, self.state_dim)

        self.name = "{}_{}_replacement".format(self.state_dim, self.action_dim)

    def _build_model(self):

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for s1 in range(self.state_dim):
            for aa in range(self.action_dim):
                if aa == 0 and s1 < self.state_dim - 1:
                    # Continuer avec la voiture actuelle coute l'entretien, elle cassera avec proba p
                    self.reward_matrix[s1, aa] = -self.car_maintain_cost[s1]
                    self.transition_matrix[aa][s1, s1 + 1] += self.survival_proba[s1]
                    self.transition_matrix[aa][s1, -1] += 1 - self.survival_proba[s1]
                else:
                    # Sinon
                    self.reward_matrix[s1, aa] = (
                        self.current_car_value[s1]
                        - self.new_car_cost[aa - 1]
                        - self.car_maintain_cost[s1 - 1]
                    )
                    self.transition_matrix[aa][s1, s1 - 1] += self.survival_proba[
                        aa - 2
                    ]
                    self.transition_matrix[aa][s1, -1] += (
                        1 - self.survival_proba[aa - 2]
                    )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
