# Source: Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Section 1.5.
import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

EMPTY = 0
X = 1
O = 2


def get_grid_size(action_dim: int) -> int:
    size = int(np.sqrt(action_dim))
    if size >= 3 and size * size == action_dim:
        return size
    return 3


def get_win_lines(size: int) -> list:
    lines = []
    for row in range(size):
        lines.append(tuple(row * size + col for col in range(size)))
    for col in range(size):
        lines.append(tuple(row * size + col for row in range(size)))
    lines.append(tuple(index * size + index for index in range(size)))
    lines.append(tuple(index * size + size - index - 1 for index in range(size)))
    return lines


def empty_cells(board: tuple) -> list:
    return [index for index, value in enumerate(board) if value == EMPTY]


def has_won(board: tuple, player: int) -> bool:
    size = int(np.sqrt(len(board)))
    for line in get_win_lines(size):
        if all(board[index] == player for index in line):
            return True
    return False


def is_tie(board: tuple) -> bool:
    return not empty_cells(board) and not has_won(board, X) and not has_won(board, O)


def is_terminal(board: tuple) -> bool:
    return has_won(board, X) or has_won(board, O) or is_tie(board)


def play(board: tuple, action: int, player: int) -> tuple:
    next_board = list(board)
    next_board[action] = player
    return tuple(next_board)


def transition_after_x_action(board: tuple, action: int) -> dict:
    if is_terminal(board) or board[action] != EMPTY:
        return {board: 1.0}

    board_after_x = play(board, action, X)
    if is_terminal(board_after_x):
        return {board_after_x: 1.0}

    o_actions = empty_cells(board_after_x)
    probability = 1.0 / len(o_actions)
    transitions = {}
    for o_action in o_actions:
        next_board = play(board_after_x, o_action, O)
        transitions[next_board] = transitions.get(next_board, 0.0) + probability
    return transitions


def reward(board: tuple, action: int, transitions: dict) -> float:
    if is_terminal(board):
        if has_won(board, X):
            return 1.0
        if has_won(board, O):
            return -1.0
        return 0.0

    if board[action] != EMPTY:
        return -1.0

    result = 0.0
    for next_board, probability in transitions.items():
        if has_won(next_board, X):
            result += probability
        elif has_won(next_board, O):
            result -= probability
    return result


def render_grid(board: tuple):
    size = int(np.sqrt(len(board)))
    symbols = {EMPTY: ".", X: "x", O: "o"}
    rows = []
    for row in range(size):
        rows.append([symbols[board[size * row + col]] for col in range(size)])
    print(np.array(rows))


METADATA = {
    "category": "games",
    "description": "Tic-tac-toe board decisions against an opponent policy.",
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Section 1.5."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10) -> None:
        self.size = get_grid_size(action_dim)
        self.action_dim = self.size * self.size
        self.build_state_space()
        self.name = "{}_{}_tictactoe".format(self.state_dim, self.action_dim)

    def build_state_space(self):
        start_board = tuple([EMPTY for _ in range(self.action_dim)])
        self.state_list = [start_board]

        index = 0
        while index < len(self.state_list):
            board = self.state_list[index]
            index += 1

            for action in range(self.action_dim):
                for next_board in transition_after_x_action(board, action):
                    if next_board not in self.state_list:
                        self.state_list.append(next_board)

        self.state_space = {
            board: state_index for state_index, board in enumerate(self.state_list)
        }
        self.state_dim = len(self.state_list)

    def _build_model(self):
        self.build_state_space()

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index, board in enumerate(self.state_list):
            for action in range(self.action_dim):
                transitions = transition_after_x_action(board, action)
                self.reward_matrix[state_index, action] = reward(
                    board, action, transitions
                )

                for next_board, probability in transitions.items():
                    next_state_index = self.state_space[next_board]
                    self.transition_matrix[action][state_index, next_state_index] = (
                        probability
                    )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
