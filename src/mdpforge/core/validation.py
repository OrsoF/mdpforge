import numpy as np
from scipy.sparse import csr_matrix


def validate_model(
    model,
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


def validate_model_attributes(model) -> None:
    required_attributes = [
        "state_dim",
        "action_dim",
        "transition_matrix",
        "reward_matrix",
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


def validate_model_shapes(model) -> None:
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


def validate_model_finite_values(model) -> None:
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


def validate_transition_stochasticity(model, tol: float = 1e-6) -> None:
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
