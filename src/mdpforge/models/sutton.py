# Source: Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Exercise 5.12.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

# fmt: off

track_L = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])


track_R = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
])


# fmt: on


TERMINAL_STATE = ("terminal",)
TEST_PARAMETERS = {
    "state_dim": 7,
    "action_dim": 9,
    "custom_track": np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]]),
}


def build_track(state_dim: int) -> np.ndarray:
    size = int(np.sqrt(state_dim / 35))
    track = np.ones((size, size))
    track[-1, -1] = 0
    return track


METADATA = {
    "category": "navigation",
    "description": "Grid racetrack navigation with position and velocity states.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Exercise 5.12."
    ),
}


class Model(MDP):
    def __init__(
        self,
        state_dim: int = 500,
        action_dim: int | None = 10,
        custom_track: np.ndarray = track_L,
    ):
        state_dim = int(np.sqrt(state_dim))
        state_dim = max(state_dim, 500)
        self.track = build_track(state_dim) if custom_track is None else custom_track
        self.p_fail = 0.1

        self.Y, self.X = self.track.shape
        self.start_states = self._find_start_states()
        self.terminal_state = TERMINAL_STATE

        # action_dim is kept for constructor compatibility; racetrack always has
        # the 3 x 3 acceleration action set.
        self.action_dim = 9
        self.state_dim = self._count_states()
        self.name = "{}_{}_sutton".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self._build_state_space()
        self._build_state_index()
        self._build_action_space()

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]
        self.reward_matrix = -1 * np.ones(
            (self.state_dim, self.action_dim), dtype=np.float64
        )

        terminal_index = self.state_space_index[self.terminal_state]
        self.reward_matrix[terminal_index, :] = 0.0

        for a, action in enumerate(self.action_space):
            self.transition_matrix[a][terminal_index, terminal_index] = 1.0

            for s1, state in enumerate(self.state_space):
                if state == self.terminal_state:
                    continue
                self._add_action_transitions(s1, state, a, action)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
        self._validate_transitions()

    def _build_state_space(self):
        self.state_space = []
        for x in range(self.X):
            for y in range(self.Y):
                if not self._check_on_track(x, y):
                    continue
                for vx in range(6):
                    for vy in range(6):
                        if vx == 0 and vy == 0 and (x, y) not in self.start_states:
                            continue
                        self.state_space.append((x, y, vx, vy))
        self.state_space.append(self.terminal_state)
        self.state_dim = len(self.state_space)

    def _build_state_index(self):
        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

    def _build_action_space(self):
        self.action_space = [(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)]
        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }
        self.action_dim = len(self.action_space)

    def _add_action_transitions(self, s1: int, state: tuple, a: int, action: tuple):
        x, y, vx, vy = state

        # Sutton & Barto stochasticity: with p_fail the chosen acceleration is
        # ignored; otherwise the acceleration is applied before movement.
        outcomes = [(self.p_fail, (0, 0)), (1 - self.p_fail, action)]
        for probability, (ax, ay) in outcomes:
            new_vx, new_vy = self._compute_new_speed(vx, vy, ax, ay, x, y)
            next_state = self._next_state_after_move(x, y, new_vx, new_vy)
            successors = self._state_indices(next_state)
            transition_probability = probability / len(successors)
            for s2 in successors:
                self.transition_matrix[a][s1, s2] += transition_probability

    def _next_state_after_move(self, x: int, y: int, vx: int, vy: int):
        status, new_x, new_y = self._compute_new_position(x, y, vx, vy)

        if status == "finish":
            return self.terminal_state

        if status == "crash":
            # Crashing sends the car back to the start line with velocity reset.
            return [(sx, sy, 0, 0) for sx, sy in self.start_states]

        return (new_x, new_y, vx, vy)

    def _state_indices(self, state):
        if isinstance(state, list):
            return [self._single_state_index(item) for item in state]
        return [self._single_state_index(state)]

    def _single_state_index(self, state):
        assert state in self.state_space_index, f"Unknown successor state: {state}"
        return self.state_space_index[state]

    def _find_start_states(self) -> list[tuple[int, int]]:
        starts = [(x, 0) for x in range(self.X) if self._check_on_track(x, 0)]
        if not starts:
            raise ValueError("Track must contain at least one valid start cell")
        return starts

    def _count_states(self) -> int:
        on_track = int(np.count_nonzero(self.track))
        return 35 * on_track + len(self.start_states) + 1

    def _check_on_track(self, x: int, y: int) -> bool:
        if not (0 <= x < self.X and 0 <= y < self.Y):
            return False
        return bool(self.track[self.Y - y - 1, x])

    def _compute_new_speed(
        self, vx: int, vy: int, ax: int, ay: int, x: int, y: int
    ) -> tuple[int, int]:
        new_vx = int(np.clip(vx + ax, 0, 5))
        new_vy = int(np.clip(vy + ay, 0, 5))

        # Zero velocity is only represented on the start line.  Away from the
        # start line, reject an acceleration that would stop the car entirely.
        if new_vx == 0 and new_vy == 0 and (x, y) not in self.start_states:
            return vx, vy
        return new_vx, new_vy

    def _compute_new_position(
        self, x: int, y: int, vx: int, vy: int
    ) -> tuple[str, int, int]:
        """Trace the full movement path and return ok, crash, or finish."""
        assert self._check_on_track(x, y)

        new_x, new_y = x, y
        for new_x, new_y in self._line_cells(x, y, x + vx, y + vy):
            # Crossing the right boundary is the finish line for these tracks.
            if new_x >= self.X:
                return "finish", new_x, new_y
            if not self._check_on_track(new_x, new_y):
                return "crash", new_x, new_y

        return "ok", new_x, new_y

    def _line_cells(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            return [(x0, y0)]

        cells = []
        previous = None
        for step in range(1, steps + 1):
            t = step / steps
            cell = (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
            if cell != previous:
                cells.append(cell)
                previous = cell
        return cells

    def _validate_transitions(self):
        all_states = set(range(self.state_dim))
        for action, matrix in enumerate(self.transition_matrix):
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()
            assert np.allclose(row_sums, 1.0), (
                "Transition rows for action {} do not sum to 1".format(action)
            )
            assert set(matrix.indices).issubset(all_states), (
                "Transition matrix contains an unknown successor state"
            )
