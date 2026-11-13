# Source: Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. PhD thesis, University of Cambridge.

import itertools
from typing import Tuple

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

FORCE = 1e-3
GRAVITY = 2.5e-3
GOAL_POSITION = 0.5


def index_closest_value(value: float, list_of_values) -> Tuple:
    distances = [abs(elem - value) for elem in list_of_values]
    index = np.argmin(distances)
    return index, list_of_values[index]


METADATA = {
    "sizes": {
        "tiny": {
            "requested_state_dim": 1080,
            "parameters": {},
        },
        "small": {
            "requested_state_dim": 10800,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 43200,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 172800,
            "parameters": {},
        },
    },
    "category": "navigation",
    "description": "Reach an absorbing MountainCar goal with reward on entry.",
    "reference": (
        "Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. "
        "PhD thesis, University of Cambridge."
    ),
}


TEST_PARAMETERS = {"state_dim": 100}


class Model(MDP):
    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100, action_dim: int = 10) -> None:
        """state_dim: target, clamped to 100; action_dim: ignored, three actions."""
        self.requested_state_dim = int(state_dim)

        state_dim = max(state_dim, 100)
        self.x_axis_states_number = int(np.floor(np.sqrt(state_dim)))
        self.speed_states_number = int(np.floor(np.sqrt(state_dim)))

        self.state_dim = self.x_axis_states_number * self.speed_states_number
        self.action_dim = 3

        self.name = "{}_{}_mountain_total_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.x_axis_states_number,
            self.speed_states_number,
        )
        self.x_axis_state_list = np.linspace(-1.2, 0.6, self.x_axis_states_number)
        self.speed_state_list = np.linspace(-0.07, 0.07, self.speed_states_number)
        self.terminal_states = tuple(
            position * self.speed_states_number + speed
            for position, x in enumerate(self.x_axis_state_list)
            if x > GOAL_POSITION
            for speed in range(self.speed_states_number)
        )

    def create_model(self, check_transition: bool = False, *, cache: str = "readwrite"):
        """Build or load matrices, then declare zero-reward absorbing terminals."""
        super().create_model(check_transition, cache=cache)
        terminal = np.all(self.reward_matrix == 0.0, axis=1)
        for transition in self.transition_matrix:
            terminal &= transition.diagonal() == 1.0
        self.terminal_states = tuple(np.flatnonzero(terminal).tolist())

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.state_space = {
            (position, speed): state
            for state, (position, speed) in enumerate(
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

        for action in range(self.action_dim):
            for state, (position, speed) in enumerate(self.state_list):
                if position > GOAL_POSITION:
                    # The goal stays absorbing in the total-reward version.
                    self.transition_matrix[action][state, state] = 1.0
                    continue

                next_position, next_speed = self._next_state(position, speed, action)
                next_state = self.state_space[(next_position, next_speed)]
                self.transition_matrix[action][state, next_state] = 1.0
                self.reward_matrix[state, action] = float(next_position > GOAL_POSITION)

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _next_state(self, position, speed, action):
        """Apply the dynamics and select the nearest position and speed grid points."""
        next_speed = speed + (action - 1) * FORCE - np.cos(3 * position) * GRAVITY
        next_position = position + next_speed
        _, next_speed = index_closest_value(next_speed, self.speed_state_list)
        _, next_position = index_closest_value(next_position, self.x_axis_state_list)
        return next_position, next_speed
