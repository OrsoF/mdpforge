"""Shared setup and matrix conversion for optional solver backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from mdpforge.core.mdp import MDP

if TYPE_CHECKING:
    from marmote.core import FullMatrix


#### Shared ####


def _nonzero_entries(matrix: csr_matrix):
    """Traverse stored CSR entries, preserving coordinate sums for duplicates."""
    entries = matrix.tocoo()
    canonical = matrix.has_canonical_format
    for row, column, value in zip(entries.row, entries.col, entries.data):
        if value != 0:
            if not canonical:
                value = matrix[row, column]
            yield int(row), int(column), float(value)


#### Marmote ####


def build_marmote_reward_matrix(
    state_dim: int, action_dim: int, reward_matrix: np.ndarray
) -> FullMatrix:
    from marmote.core import FullMatrix

    marmote_reward_matrix = FullMatrix(state_dim, action_dim)
    for a in range(action_dim):
        for s in range(state_dim):
            marmote_reward_matrix.setEntry(s, a, float(reward_matrix[s, a]))
    return marmote_reward_matrix


def build_marmote_transition_list(
    state_dim: int, action_dim: int, transition_matrix: list[csr_matrix]
) -> list:
    from marmote.core import SparseMatrix

    marmote_transitions_list = []
    for action in range(action_dim):
        transition = SparseMatrix(state_dim)
        for state_from, state_to, value in _nonzero_entries(transition_matrix[action]):
            transition.addToEntry(state_from, state_to, value)
        marmote_transitions_list.append(transition)
    return marmote_transitions_list


def compute_marmote_args(model: "MDP") -> tuple:
    """Build native Marmote matrices without changing the CSR model."""
    return (
        build_marmote_transition_list(
            model.state_dim, model.action_dim, model.transition_matrix
        ),
        build_marmote_reward_matrix(
            model.state_dim, model.action_dim, model.reward_matrix
        ),
    )


#### Madupite ####


def madupite_matrices(model: MDP):
    """Return (S,A) rewards and (S*A,S) CSR transitions in state-major order."""
    # Columns of the horizontal stack are ordered (action, next_state).
    # C-order reshape then maps a row to state * n_actions + action.
    transitions = hstack(model.transition_matrix, format="csr")
    transitions = transitions.reshape(
        (model.state_dim * model.action_dim, model.state_dim), order="C"
    ).tocsr()
    return np.asarray(model.reward_matrix, dtype=np.float64), transitions


#### MDPax ####


def padded_transitions(model: MDP, max_event_entries: int):
    """Pack variable-length CSR rows as state/action/event arrays."""
    s, a = model.state_dim, model.action_dim
    width = max(int(np.diff(p.indptr).max(initial=0)) for p in model.transition_matrix)
    if s * a * width > max_event_entries:
        raise ValueError(
            f"MDPax conversion needs {s * a * width:,} event slots, "
            f"above max_event_entries={max_event_entries:,}"
        )
    successors = np.zeros((s, a, width), dtype=np.int32)
    probabilities = np.zeros((s, a, width), dtype=np.float64)
    for action, transition in enumerate(model.transition_matrix):
        # Do not materialize dense S-by-S transition matrices.
        for state in range(s):
            lo, hi = transition.indptr[state : state + 2]
            count = hi - lo
            successors[state, action, :count] = transition.indices[lo:hi]
            probabilities[state, action, :count] = transition.data[lo:hi]
    return successors, probabilities


def make_mdpax_problem(model: MDP, max_event_entries: int):
    """Construct a JAX-compatible MDPax Problem from explicit transitions."""
    import jax
    import jax.numpy as jnp

    # The external config enables x64 in its constructor, but that is too late
    # for matrices created while constructing the Problem.
    jax.config.update("jax_enable_x64", True)
    from mdpax.core.problem import Problem

    successor, probability = padded_transitions(model, max_event_entries)
    n_states, n_actions, n_events = successor.shape
    rewards = np.asarray(model.reward_matrix, dtype=np.float64)

    class MatrixProblem(Problem):
        """Explicit MDP with one random event per padded CSR row slot."""

        def __init__(self):
            self.successors = jnp.asarray(successor)
            self.probabilities = jnp.asarray(probability)
            self.rewards = jnp.asarray(rewards)
            super().__init__()

        @property
        def name(self) -> str:
            return model.name

        def _construct_state_space(self):
            return jnp.arange(n_states, dtype=jnp.int32).reshape(-1, 1)

        def state_to_index(self, state):
            return state[0]

        def _construct_action_space(self):
            return jnp.arange(n_actions, dtype=jnp.int32).reshape(-1, 1)

        def _construct_random_event_space(self):
            return jnp.arange(n_events, dtype=jnp.int32).reshape(-1, 1)

        def random_event_probability(self, state, action, random_event):
            return self.probabilities[state[0], action[0], random_event[0]]

        def transition(self, state, action, random_event):
            nxt = self.successors[state[0], action[0], random_event[0]]
            return jnp.expand_dims(nxt, 0), self.rewards[state[0], action[0]]

    return MatrixProblem()


#### MDPSolver ####


def compute_mdpsolver_args(model: "MDP") -> tuple:
    """Convert transition and reward matrices to MDPSolver input lists."""
    transitions = []
    for action in range(model.action_dim):
        for state_from, state_to, value in _nonzero_entries(
            model.transition_matrix[action]
        ):
            transitions.append([state_from, action, state_to, value])

    rewards = [
        [
            float(model.reward_matrix[state, action])
            for action in range(model.action_dim)
        ]
        for state in range(model.state_dim)
    ]

    return transitions, rewards


def create_mdpsolver(model: MDP, discount: float, backend):
    """Build a backend model without changing the source matrices."""
    transitions, rewards = compute_mdpsolver_args(model)
    mdl = backend.model()
    mdl.mdp(
        discount=discount,
        rewards=rewards,
        tranMatElementwise=transitions,
    )
    return mdl


def solve_mdpsolver(
    mdl,
    *,
    algorithm: str,
    tolerance: float,
    update: str,
    parallel: bool,
    initial_value=None,
    initial_policy=None,
    **options,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a discounted MDP and return NumPy values and policy."""
    mdl.solve(
        algorithm=algorithm,
        tolerance=tolerance,
        update=update,
        criterion="discounted",
        parallel=parallel,
        initValueVector=[] if initial_value is None else initial_value.tolist(),
        initPolicy=[] if initial_policy is None else initial_policy.tolist(),
        **options,
    )
    return (
        np.asarray(mdl.getValueVector(), dtype=float),
        np.asarray(mdl.getPolicy(), dtype=int),
    )


#### QuantEcon ####


def create_quantecon(model: MDP, discount: float, *, dense: bool = False):
    """Build DiscreteDP without changing the model's rewards or CSR matrices."""
    from quantecon.markov import DiscreteDP

    if dense:
        transitions = np.empty(
            (model.state_dim, model.action_dim, model.state_dim), dtype=float
        )
        for action, matrix in enumerate(model.transition_matrix):
            transitions[:, action, :] = matrix.toarray()
        return DiscreteDP(model.reward_matrix.copy(), transitions, discount)

    transitions = vstack(model.transition_matrix, format="csr")
    states = np.tile(np.arange(model.state_dim), model.action_dim)
    actions = np.repeat(np.arange(model.action_dim), model.state_dim)
    rewards = model.reward_matrix.T.flatten()
    return DiscreteDP(
        rewards, transitions, discount, s_indices=states, a_indices=actions
    )


#### Gurobi ####


def gurobi_model_creation(*, threads: int = 1, memory_limit_gb: float = 16.0):
    from gurobipy import GRB, Model

    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, threads)
    model.setParam("LogToConsole", 0)
    model.setParam("MemLimit", memory_limit_gb)
    return model
