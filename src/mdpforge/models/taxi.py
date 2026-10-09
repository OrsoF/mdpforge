# Source: Dietterich, T. G. (2000). Hierarchical reinforcement learning
# with the MAXQ value function decomposition. JAIR 13:227-303.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

WEST, EAST, NORTH, SOUTH, PICKUP, DROPOFF = range(6)
IN_TAXI = 4

METADATA = {
    "category": "navigation",
    "description": "Taxi navigation with passenger pickup and drop-off decisions.",
    "reference": (
        "Dietterich, T. G. (2000). Hierarchical reinforcement learning with the "
        "MAXQ value function decomposition. JAIR 13:227-303."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 500, action_dim: int = 10) -> None:

        self.grid_size = max(int(np.sqrt(state_dim // 20)), 1)

        self.base_state_dim = self.grid_size**2 * 20
        self.terminal_state = self.base_state_dim

        # +1 for terminal state
        self.state_dim = self.base_state_dim + 1
        self.action_dim = 6

        self.name = f"{self.state_dim}_{self.action_dim}_terminal_taxi_benchmark"

    def build_state_space(self) -> None:
        self.state_list = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for passenger_position in range(5):
                    for destination in range(4):
                        self.state_list.append((x, y, passenger_position, destination))

        self.state_list.append(("terminal",))

        self.state_space = {
            state: state_index for state_index, state in enumerate(self.state_list)
        }

    def landmark_positions(self) -> list[tuple[int, int]]:
        return [
            (0, 0),
            (self.grid_size - 1, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, self.grid_size - 1),
        ]

    def move(self, x: int, y: int, action: int) -> tuple[int, int]:
        if action == WEST:
            return x, max(0, y - 1)
        if action == EAST:
            return x, min(self.grid_size - 1, y + 1)
        if action == NORTH:
            return max(0, x - 1), y
        if action == SOUTH:
            return min(self.grid_size - 1, x + 1), y

        return x, y

    def _build_model(self) -> None:
        self.build_state_space()

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        self.reward_matrix = -1.0 * np.ones((self.state_dim, self.action_dim))

        self.positions_on_map = self.landmark_positions()

        for ss1 in range(self.state_dim):
            for aa in range(self.action_dim):
                # Terminal state: absorbing, zero reward.
                if ss1 == self.terminal_state:
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    self.reward_matrix[ss1, aa] = 0.0
                    continue

                x, y, passenger_position, destination = self.state_list[ss1]

                if aa == PICKUP:
                    valid_pickup = (
                        passenger_position != IN_TAXI
                        and (x, y) == self.positions_on_map[passenger_position]
                    )

                    if valid_pickup:
                        next_state = (x, y, IN_TAXI, destination)
                        ss2 = self.state_space[next_state]
                        self.transition_matrix[aa][ss1, ss2] = 1.0
                    else:
                        self.reward_matrix[ss1, aa] = -10.0
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                elif aa == DROPOFF:
                    valid_dropoff = (
                        passenger_position == IN_TAXI
                        and (x, y) == self.positions_on_map[destination]
                    )

                    if valid_dropoff:
                        self.reward_matrix[ss1, aa] = 20.0
                        self.transition_matrix[aa][ss1, self.terminal_state] = 1.0
                    else:
                        self.reward_matrix[ss1, aa] = -10.0
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                else:
                    next_x, next_y = self.move(x, y, aa)
                    next_state = (
                        next_x,
                        next_y,
                        passenger_position,
                        destination,
                    )
                    ss2 = self.state_space[next_state]
                    self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

        reward = self.reward_matrix
        reward.fill(0.0)
        for s, state in enumerate(self.state_list[:-1]):
            x, y, passenger_position, destination = state
            if (
                passenger_position != IN_TAXI
                and (x, y) == self.positions_on_map[passenger_position]
            ):
                reward[s, PICKUP] = 0.2
            if (
                passenger_position == IN_TAXI
                and (x, y) == self.positions_on_map[destination]
            ):
                reward[s, DROPOFF] = 1.0

        reward[self.terminal_state, :] = 1.0
