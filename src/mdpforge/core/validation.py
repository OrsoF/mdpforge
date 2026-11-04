from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.operators import (
    bellman_policy_operator,
    compact_optimal_bellman_operator,
    compute_transition_reward_policy,
)
from mdpforge.core.solve_result import SolveResult

if TYPE_CHECKING:
    from mdpforge.core.mdp import MDP


def validate_result(
    model: "MDP",
    result: SolveResult,
    *,
    discount: float,
    precision: float,
) -> float:
    """Certify absolute value error and, for 'policy', policy quality, or raise ValueError.

    discount: finite real in (0, 1).
    precision: finite real > 0; maximum accepted error bound.
    """
    if (
        not isinstance(discount, Real)
        or not np.isfinite(discount)
        or not 0 < discount < 1
    ):
        raise ValueError("discount must be finite and strictly between 0 and 1")
    if (
        isinstance(precision, bool)
        or not isinstance(precision, Real)
        or not np.isfinite(precision)
        or precision <= 0
    ):
        raise ValueError("precision must be positive and finite")
    validate_model(model)
    if not isinstance(result, SolveResult):
        raise ValueError("result must be a SolveResult")
    method = result.validation_method
    if not isinstance(method, str) or method not in (
        "residual",
        "span",
        "policy",
        "bounds",
    ):
        raise ValueError(f"Unsupported validation_method: {method!r}")

    vectors = {"value": result.value}
    if method == "bounds":
        vectors.update(lower_value=result.lower_value, upper_value=result.upper_value)
    for name, vector in vectors.items():
        if (
            not isinstance(vector, np.ndarray)
            or vector.shape != (model.state_dim,)
            or vector.dtype.kind not in "fiu"
            or not np.isfinite(vector).all()
        ):
            raise ValueError(
                f"result.{name} must be a finite real NumPy vector "
                f"of shape ({model.state_dim},)"
            )
    policy = result.policy
    if method == "policy" and policy is None:
        raise ValueError("policy validation requires result.policy")
    if policy is not None and (
        not isinstance(policy, np.ndarray)
        or policy.shape != (model.state_dim,)
        or policy.dtype.kind not in "iu"
        or np.any(policy < 0)
        or np.any(policy >= model.action_dim)
    ):
        raise ValueError(
            "result.policy must contain one valid integer action per state"
        )

    value = np.asarray(result.value, dtype=float)
    # Account for row-sum roundoff admitted by validate_model().
    contraction = discount * max(
        float(matrix.sum(axis=1).max()) for matrix in model.transition_matrix
    )
    if contraction >= 1:
        raise ValueError("The supplied model does not define a contracting Bellman map")
    denominator = 1 - contraction
    if method == "bounds":
        lower = np.asarray(result.lower_value, dtype=float)
        upper = np.asarray(result.upper_value, dtype=float)
        if np.any(lower > upper):
            raise ValueError("result.lower_value must not exceed result.upper_value")
        lower_backup = compact_optimal_bellman_operator(model, lower, discount)
        upper_backup = compact_optimal_bellman_operator(model, upper, discount)
        if not np.isfinite(lower_backup).all() or not np.isfinite(upper_backup).all():
            raise ValueError("bounds validation produced nonfinite Bellman values")
        # Bellman inequality violations widen the supplied interval to certify it.
        lower_slack = max(0.0, float(np.max(lower - lower_backup))) / denominator
        upper_slack = max(0.0, float(np.max(upper_backup - upper))) / denominator
        certified_lower = lower - lower_slack
        certified_upper = upper + upper_slack
        error_bound = float(
            np.max(
                np.maximum(
                    np.abs(value - certified_lower), np.abs(certified_upper - value)
                )
            )
        )
    else:
        backup = compact_optimal_bellman_operator(model, value, discount)
        residual = backup - value
        if method == "span":
            # Width alone cannot detect a constant error in the returned values.
            minimum, maximum = float(residual.min()), float(residual.max())
            half_span = maximum / 2 - minimum / 2
            midpoint = minimum / 2 + maximum / 2
            error_bound = (half_span + abs(midpoint)) / denominator
        else:
            error_bound = float(np.max(np.abs(residual))) / denominator
            if method == "policy":
                transition, reward = compute_transition_reward_policy(model, policy)
                policy_backup = bellman_policy_operator(
                    value, discount, transition, reward
                )
                # Also bound the optimality gap of the supplied policy.
                error_bound += (
                    float(np.max(np.abs(policy_backup - value))) / denominator
                )

    if not np.isfinite(error_bound) or error_bound > precision:
        raise ValueError(
            f"{method} validation failed: error bound {error_bound:g} "
            f"exceeds precision {precision:g} or is nonfinite"
        )
    return float(error_bound)


def validate_model(
    model: "MDP",
    tol: float = 1e-6,
    check_stochastic: bool = True,
    check_finite: bool = True,
) -> None:
    validate_model_attributes(model)
    validate_model_shapes(model)

    if check_finite:
        validate_model_finite_values(model)

    if check_stochastic:
        validate_transition_stochasticity(model, tol)


def validate_model_attributes(model: "MDP") -> None:
    required_attributes = [
        "state_dim",
        "action_dim",
        "transition_matrix",
        "reward_matrix",
        "name",
    ]

    missing_attributes = [
        attribute for attribute in required_attributes if not hasattr(model, attribute)
    ]

    if missing_attributes:
        raise ValueError(
            "Model is missing required attributes: {}".format(
                ", ".join(missing_attributes)
            )
        )

    if not isinstance(model.state_dim, int) or model.state_dim <= 0:
        raise ValueError("model.state_dim must be a positive integer.")

    if not isinstance(model.action_dim, int) or model.action_dim <= 0:
        raise ValueError("model.action_dim must be a positive integer.")
    if not isinstance(model.transition_matrix, list):
        raise ValueError("transition_matrix must be a list of CSR matrices.")
    if not isinstance(model.reward_matrix, np.ndarray):
        raise ValueError("reward_matrix must be a NumPy array.")


def validate_model_shapes(model: "MDP") -> None:
    expected_reward_shape = (model.state_dim, model.action_dim)
    if model.reward_matrix.shape != expected_reward_shape:
        raise ValueError(
            "Reward matrix has shape {}, expected {}.".format(
                model.reward_matrix.shape,
                expected_reward_shape,
            )
        )

    if len(model.transition_matrix) != model.action_dim:
        raise ValueError(
            "Transition matrix contains {} actions, expected {}.".format(
                len(model.transition_matrix),
                model.action_dim,
            )
        )

    expected_transition_shape = (model.state_dim, model.state_dim)
    for action, transition in enumerate(model.transition_matrix):
        if not isinstance(transition, csr_matrix):
            raise ValueError(
                f"Transition matrix for action {action} must be a CSR matrix."
            )
        if transition.shape != expected_transition_shape:
            raise ValueError(
                "Transition matrix for action {} has shape {}, expected {}.".format(
                    action,
                    transition.shape,
                    expected_transition_shape,
                )
            )


def validate_model_finite_values(model: "MDP") -> None:
    if not np.all(np.isfinite(model.reward_matrix)):
        raise ValueError("Reward matrix contains NaN or infinite values.")

    for action, transition in enumerate(model.transition_matrix):
        values = transition.data
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Transition matrix for action {} contains NaN or infinite values.".format(
                    action
                )
            )


def validate_transition_stochasticity(model: "MDP", tol: float = 1e-6) -> None:
    for action, transition in enumerate(model.transition_matrix):
        values = transition.data
        if np.any(values < 0):
            raise ValueError(
                f"Transition matrix for action {action} contains negative probabilities."
            )
        row_sums = np.asarray(transition.sum(axis=1)).reshape(-1)
        bad_rows = np.flatnonzero(np.abs(row_sums - 1.0) > tol)

        if bad_rows.size:
            first_bad_row = int(bad_rows[0])
            raise ValueError(
                "Transition matrix for action {} is not stochastic: row {} sums to {}.".format(
                    action,
                    first_bad_row,
                    row_sums[first_bad_row],
                )
            )
