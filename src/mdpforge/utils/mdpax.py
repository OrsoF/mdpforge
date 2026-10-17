"""Convert mdpforge CSR transitions into an optional MDPax Problem."""

import numpy as np

from mdpforge.core.model import MDPProtocol


def padded_transitions(model: MDPProtocol, max_event_entries: int):
    """Pack variable-length CSR rows as state/action/event arrays."""
    if max_event_entries < 1:
        raise ValueError("max_event_entries must be positive")
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


def make_mdpax_problem(model: MDPProtocol, max_event_entries: int):
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
