"""Shared checks for undiscounted total-reward solvers."""

import numpy as np


def check_options(model, final_precision, max_iterations):
    """Validate the accuracy target, iteration budget and terminal states."""
    if not np.isfinite(final_precision) or final_precision <= 0:
        raise ValueError("final_precision must be finite and positive")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    terminal = set(getattr(model, "terminal_states", ()))
    if not terminal or any(
        type(s) is not int or not 0 <= s < model.state_dim for s in terminal
    ):
        raise ValueError("total-reward model must declare valid terminal_states")
    return terminal


def proper_initial_policy(model, terminal, *, require_all_states=True):
    """Choose terminal-directed actions, requiring all states by default."""
    predecessors = [[] for _ in range(model.state_dim)]
    for action, transition in enumerate(model.transition_matrix):
        for state in range(model.state_dim):
            lo, hi = transition.indptr[state : state + 2]
            for successor, probability in zip(
                transition.indices[lo:hi], transition.data[lo:hi]
            ):
                if probability > 0:
                    predecessors[successor].append((state, action))
    policy = np.zeros(model.state_dim, dtype=int)
    reached = set(terminal)
    frontier = list(terminal)
    for successor in frontier:
        for state, action in predecessors[successor]:
            if state not in reached:
                reached.add(state)
                policy[state] = action
                frontier.append(state)
    if require_all_states and len(reached) != model.state_dim:
        raise ValueError("every state must be able to reach a terminal state")
    return policy
