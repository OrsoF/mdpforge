# Source: Barto, A. G., Bradtke, S. J. and Singh, S. P. (1995).
# Learning to act using real-time dynamic programming.
# Artificial Intelligence 72(1-2):81-138.

import numpy as np
from scipy.sparse import dok_matrix

from mdpforge.core.model import GenericModel


def build_track(state_dim: int, max_speed: int) -> np.ndarray:
    """
    Build a simple square racetrack.

    - track[:, 0] is the start line;
    - track[0, :] is the finish line;
    - track[0, 0] is excluded to avoid overlap between start and finish.

    The spatial size is chosen so that the resulting state space is at least
    of the requested order of magnitude.
    """
    n_speed = 2 * max_speed + 1
    size = max(2, int(np.ceil(np.sqrt(state_dim / (n_speed * n_speed)))))

    track = np.ones((size, size))
    track[0, :] = 3  # Finish line
    track[:, 0] = 2  # Start line
    track[0, 0] = 0  # Do not let start and finish overlap
    return track


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self._proba_fail_action: float = 0.1
        self._max_speed = 3

        state_dim = max(300, state_dim)
        self._track = build_track(state_dim, self._max_speed)

        self._X, self._Y = self._track.shape

        self._build_state_space()
        self._build_action_space()

        self.name = "{}_{}_barto_total_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self._max_speed,
            self._proba_fail_action,
        )

    def _update_transition_matrix(
        self,
        aa: int,
        ax: int,
        ay: int,
        ss1: int,
        x: int,
        y: int,
        sx: int,
        sy: int,
    ):
        """
        Fill one row P(. | s, a) and the corresponding expected one-step reward.

        With probability p, the acceleration fails and the current speed is
        preserved. With probability 1-p, the acceleration is applied.

        The resulting speed is used immediately to move the car.

        - reaching the finish gives reward 1;
        - crashing resets uniformly to a start state with zero speed;
        - finish states are absorbing and give no further reward.
        """
        if self._track[x, y] == 3:  # Absorbing finish state
            self.transition_matrix[aa][ss1, ss1] = 1.0
            return

        outcomes = [
            (self._proba_fail_action, sx, sy),
            (
                1.0 - self._proba_fail_action,
                *self.compute_new_speed(sx, sy, ax, ay),
            ),
        ]

        for probability, next_sx, next_sy in outcomes:
            new_x, new_y, status = self.compute_new_position(x, y, next_sx, next_sy)

            if status == "crash":
                reset_probability = probability * self.proba_start_states
                for start_state in self.start_states:
                    self.transition_matrix[aa][ss1, start_state] += reset_probability
                continue

            ss2 = self.state_space_index[(new_x, new_y, next_sx, next_sy)]
            self.transition_matrix[aa][ss1, ss2] += probability

            if status == "finish":
                # reward_matrix stores E[r_{t+1} | s_t=s, a_t=a]
                self.reward_matrix[ss1, aa] += probability

    def _build_model(self):
        self._init_matrices()

        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                self._update_transition_matrix(a, ax, ay, s1, x, y, sx, sy)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _build_state_space(self):
        # A state is made of position and speed: (x, y, sx, sy).
        # Signed speeds are required by the chosen geometry because the
        # finish line is x = 0 while start states satisfy x > 0.
        self.state_space = [
            (x, y, sx, sy)
            for x in range(self._X)
            for y in range(self._Y)
            for sx in range(-self._max_speed, self._max_speed + 1)
            for sy in range(-self._max_speed, self._max_speed + 1)
            if self.check_on_track(x, y)
        ]

        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

        self.state_dim = len(self.state_space)

    def _build_action_space(self):
        # An action is made of two acceleration increments in {-1, 0, +1}.
        self.action_space = [(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)]

        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }

        self.action_dim = len(self.action_space)

    def _init_matrices(self):
        self._build_state_space()
        self._build_action_space()

        self.transition_matrix = [
            dok_matrix(
                (self.state_dim, self.state_dim),
                dtype=np.float64,
            )
            for _ in range(self.action_dim)
        ]

        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )

        # Start states: start line with zero velocity.
        self.start_states = [
            state_index
            for state_index, state in enumerate(self.state_space)
            if (abs(self._track[state[:2]] - 2.0) <= 0.01 and state[2:] == (0, 0))
        ]

        if not self.start_states:
            raise RuntimeError("The track contains no valid start state.")

        # Uniform reset distribution over the start line.
        self.proba_start_states = 1.0 / len(self.start_states)

    def check_on_track(self, x: int, y: int) -> bool:
        """
        Check whether (x, y) is a valid track cell.
        """
        return 0 <= x < self._X and 0 <= y < self._Y and bool(self._track[x, y])

    def compute_new_speed(
        self,
        sx: int,
        sy: int,
        ax: int,
        ay: int,
    ):
        """
        Apply the acceleration and clip both signed speed components.
        """
        new_sx = max(
            -self._max_speed,
            min(self._max_speed, sx + ax),
        )
        new_sy = max(
            -self._max_speed,
            min(self._max_speed, sy + ay),
        )
        return new_sx, new_sy

    def compute_new_position(
        self,
        x: int,
        y: int,
        sx: int,
        sy: int,
    ):
        """
        Move according to the supplied speed while checking intermediate cells.

        Returns
        -------
        new_x, new_y, status
            status is one of:
            - "ok": valid non-terminal transition;
            - "finish": the finish line has been reached;
            - "crash": the trajectory has left the track.

        The movement is decomposed into unit grid moves so that crossing the
        finish line or leaving the track is detected before the final position.
        """
        assert self.check_on_track(x, y)

        remaining_x = sx
        remaining_y = sy

        n_steps = max(abs(sx), abs(sy))

        for _ in range(n_steps):
            if remaining_x != 0:
                step_x = 1 if remaining_x > 0 else -1
                candidate_x = x + step_x

                if not self.check_on_track(candidate_x, y):
                    return x, y, "crash"

                x = candidate_x
                remaining_x -= step_x

                if self._track[x, y] == 3:
                    return x, y, "finish"

            if remaining_y != 0:
                step_y = 1 if remaining_y > 0 else -1
                candidate_y = y + step_y

                if not self.check_on_track(x, candidate_y):
                    return x, y, "crash"

                y = candidate_y
                remaining_y -= step_y

                if self._track[x, y] == 3:
                    return x, y, "finish"

        return x, y, "ok"
