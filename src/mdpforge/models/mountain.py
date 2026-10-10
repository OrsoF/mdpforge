# Source: Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. PhD thesis, University of Cambridge.

import itertools
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


def index_closest_value(value: float, list_of_values) -> Tuple:
    distances = [abs(elem - value) for elem in list_of_values]
    index = np.argmin(distances)
    return index, list_of_values[index]


METADATA = {
    "tags": ["control", "real-world"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 100},
            "source": "measured",
        },
        "medium": {
            "state_dim": 961,
            "parameters": {"state_dim": 1000},
            "source": "measured",
        },
        "large": {
            "state_dim": 7569,
            "parameters": {"state_dim": 7665},
            "source": "inferred",
        },
    },
    "category": "control",
    "description": "Discretized MountainCar position and velocity control.",
    "reference": (
        "Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. "
        "PhD thesis, University of Cambridge."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10) -> None:
        """
        params variable should contain :
        - x_axis_states_number
        - speed_states_number
        """

        state_dim = max(state_dim, 100)
        self.x_axis_states_number = int(np.floor(np.sqrt(state_dim)))
        self.speed_states_number = int(np.floor(np.sqrt(state_dim)))

        self.state_dim = self.x_axis_states_number * self.speed_states_number
        self.action_dim = 3

        self.name = "{}_{}_mountain_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.x_axis_states_number,
            self.speed_states_number,
        )

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.x_axis_state_list = np.linspace(-1.2, 0.6, self.x_axis_states_number)
        self.speed_state_list = np.linspace(-0.07, 0.07, self.speed_states_number)

        self.state_space = {
            (position, speed): ss
            for ss, (position, speed) in enumerate(
                itertools.product(self.x_axis_state_list, self.speed_state_list)
            )
        }

        self.state_list = list(self.state_space.keys())

        _, speed_min = index_closest_value(0.0, self.speed_state_list)

        self._start_states = [
            (pos, speed)
            for (pos, speed) in self.state_list
            if (speed == speed_min and -0.6 <= pos <= -0.4)
        ]
        assert len(self._start_states), (
            "There is no start states possibles in those conditions. {}".format(
                self.state_dim
            )
        )

        FORCE = 1e-3
        GRAVITY = 2.5e-3

        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                position, speed = self.state_list[ss1]
                if position > 0.5:
                    self.reward_matrix[ss1, aa] = 0.0
                    for pos, speed in self._start_states:
                        ss2 = self.state_space[(pos, speed)]
                        self.transition_matrix[aa][ss1, ss2] = 1 / len(
                            self._start_states
                        )

                else:
                    next_speed = (
                        speed + (aa - 1) * FORCE - np.cos(3 * position) * GRAVITY
                    )
                    next_position = position + next_speed

                    self.reward_matrix[ss1, aa] = -1.0

                    _, next_speed_state_space = index_closest_value(
                        next_speed, self.speed_state_list
                    )
                    _, next_position_state_space = index_closest_value(
                        next_position, self.x_axis_state_list
                    )
                    ss2 = self.state_space[
                        (next_position_state_space, next_speed_state_space)
                    ]
                    self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]
