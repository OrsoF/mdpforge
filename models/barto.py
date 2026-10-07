# Source: Barto, A. G., Bradtke, S. J. and Singh, S. P. (1995). Learning to act using real-time dynamic programming. Artificial Intelligence 72(1-2):81-138.

import numpy as np
from scipy.sparse import dok_matrix

from core.model import GenericModel


def build_track(state_dim: int, max_speed: int) -> np.ndarray:
    speed_count = 2 * max_speed + 1
    size = int(np.sqrt(state_dim / speed_count / speed_count))
    track = np.ones((size, size))
    track[0, :] = 3
    track[:, 0] = 2
    track[0, 0] = 0
    return track


class Model(GenericModel):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        custom_track: np.ndarray | None = None,
    ):
        self._proba_fail_action: float = 0.1
        self._max_speed = 3

        state_dim = max(300, state_dim)
        self._track = (
            np.asarray(custom_track)
            if custom_track is not None
            else build_track(state_dim, self._max_speed)
        )

        self._X, self._Y = self._track.shape

        self._build_state_space()
        self._build_action_space()

        self.name = "{}_{}_barto_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self._max_speed,
            self._proba_fail_action,
        )

    def _update_transition_matrix(
        self, aa: int, ax: int, ay: int, ss1: int, x: int, y: int, sx: int, sy: int
    ):
        expected_reward = 0.0
        for probability, state, reward in self.transition_from(x, y, sx, sy, ax, ay):
            ss2 = self.state_space_index[state]
            self.transition_matrix[aa][ss1, ss2] += probability
            expected_reward += probability * reward
        self.reward_matrix[ss1, aa] = expected_reward

    def _build_model(self):
        self._init_matrices()

        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                self._update_transition_matrix(a, ax, ay, s1, x, y, sx, sy)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
        self._validate_model()

    def _build_state_space(self):
        # A state is made of coordinate and speed : (x, y, sx, sy)
        self.state_space = [
            (x, y, sx, sy)
            for x in range(self._X)
            for y in range(self._Y)
            for sx in range(-self._max_speed, self._max_speed + 1)
            for sy in range(-self._max_speed, self._max_speed + 1)
            if self.check_on_track(x, y)
            and (
                (sx, sy) != (0, 0)
                or self.is_start_cell(x, y)
                or self.is_finish_cell(x, y)
            )
        ]

        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

        # state_dim is an integer
        self.state_dim = len(self.state_space)

    def _build_action_space(self):
        # An action is made of two increments (ax, ay) between -1 and +1
        self.action_space = [(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)]

        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }

        self.action_dim = len(self.action_space)

    def _init_matrices(self):
        self._build_state_space()
        self._build_action_space()
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = -1 * np.ones(
            (self.state_dim, self.action_dim), dtype=np.float64
        )

        self.start_states = [
            state
            for state in range(self.state_dim)
            if (
                self.is_start_cell(*self.state_space[state][:2])
                and self.state_space[state][2:] == (0, 0)
            )
        ]
        assert self.start_states, "Racetrack must contain at least one start cell."
        for start_state in self.start_states:
            assert self.state_space[start_state][2:] == (0, 0)
        self.proba_start_states = 1 / len(self.start_states)

    def is_start_cell(self, x, y):
        return 0 <= x < self._X and 0 <= y < self._Y and self._track[x, y] == 2

    def is_finish_cell(self, x, y):
        return 0 <= x < self._X and 0 <= y < self._Y and self._track[x, y] == 3

    def check_on_track(self, x, y):
        """
        Checks if the point of coordinates (x, y) is on the track
        """
        return (0 <= x < self._X and 0 <= y < self._Y) and bool(self._track[x, y])

    def compute_new_speed(self, sx, sy, ax, ay):
        """
        From the speed (sx, sy), compute the next
        speed using the action (ax, ay)
        """
        new_sx = max(-self._max_speed, min(self._max_speed, sx + ax))
        new_sy = max(-self._max_speed, min(self._max_speed, sy + ay))
        return new_sx, new_sy

    def trace_path(self, x, y, sx, sy):
        assert self.check_on_track(x, y)
        steps = abs(sx) + abs(sy)
        if steps == 0:
            return "valid", x, y

        previous_cell = (x, y)
        for step in range(1, steps + 1):
            next_x = int(round(x + sx * step / steps))
            next_y = int(round(y + sy * step / steps))
            if (next_x, next_y) == previous_cell:
                continue
            previous_cell = (next_x, next_y)

            if self.is_finish_cell(next_x, next_y):
                return "finish", next_x, next_y
            if not self.check_on_track(next_x, next_y):
                return "crash", x, y

        return "valid", x + sx, y + sy

    def compute_new_position(self, x, y, sx, sy):
        status, new_x, new_y = self.trace_path(x, y, sx, sy)
        return new_x, new_y, status != "crash"

    def transition_from(self, x, y, sx, sy, ax, ay):
        if self.is_finish_cell(x, y):
            # Finish states are absorbing terminals.
            return [(1.0, (x, y, sx, sy), 0.0)]

        new_sx, new_sy = self.compute_new_speed(sx, sy, ax, ay)
        if (new_sx, new_sy) == (0, 0) and not self.is_start_cell(x, y):
            new_sx, new_sy = sx, sy

        # Acceleration fails with p: move with the old velocity.
        failed = self._successors_for_velocity(self._proba_fail_action, x, y, sx, sy)
        # Acceleration succeeds with 1-p: move with the updated velocity.
        succeeded = self._successors_for_velocity(
            1 - self._proba_fail_action, x, y, new_sx, new_sy
        )
        return failed + succeeded

    def _successors_for_velocity(self, probability, x, y, next_sx, next_sy):
        if probability == 0:
            return []

        status, next_x, next_y = self.trace_path(x, y, next_sx, next_sy)
        if status == "crash":
            # Crashes reset uniformly to start states with zero velocity.
            return [
                (probability * self.proba_start_states, self.state_space[start], -1.0)
                for start in self.start_states
            ]
        if status == "finish":
            state = (next_x, next_y, next_sx, next_sy)
            assert state in self.state_space_index
            return [(probability, state, 0.0)]

        state = (next_x, next_y, next_sx, next_sy)
        assert state in self.state_space_index
        return [(probability, state, -1.0)]

    def _validate_model(self):
        for state in self.start_states:
            assert self.state_space[state][2:] == (0, 0)
            assert self.is_start_cell(*self.state_space[state][:2])

        for action, transition in enumerate(self.transition_matrix):
            row_sums = np.asarray(transition.sum(axis=1)).ravel()
            assert np.allclose(row_sums, 1.0), (
                f"Transition rows for action {action} do not sum to 1."
            )


# fmt: off

track_L = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

track_R = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3],
])

# fmt: on
