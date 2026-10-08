from typing import Callable, Optional

import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.utils.paths import SAVED_VALUE_FUNCTIONS_PATH
from mdpforge.utils.persistence import get_cached_object

# from mdpforge.utils.calculus import norminf


def norminf(value: np.ndarray) -> float:
    """
    Infinite norm of a vector.
    """
    return np.absolute(value).max()


av_str, dis_str, tot_str = "average", "discounted", "total"
all_method_set = {av_str, dis_str, tot_str}


def build_value_function_name(
    model: GenericModel, method: str, discount: float = 1.0
) -> str:
    """
    The name of the file we are going to use to save the optimal value function.
    """

    assert method in all_method_set, "Method {} not recognized.".format(method)
    return "{}_{}_{}".format(method, discount, model.name)


def compute_exact_value(
    model: GenericModel,
    method: str,
    discount: Optional[float] = None,
    warning: bool = True,
    epsilon_exact_value: float = 1e-6,
    max_iter_exact_value: int = int(1e8),
    skip_check: bool = True,
) -> np.ndarray:
    """
    Given a model, we compute the exact value function
    up to a given precision.
    """
    assert not warning, (
        "The function should not be used in itself, use get_exact_value function instead."
    )

    from mdptoolbox.mdp import RelativeValueIteration, ValueIteration

    if method == av_str:
        solver = RelativeValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
            skip_check=skip_check,
        )
    elif method == dis_str:
        solver = ValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            discount=discount,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
            skip_check=skip_check,
        )
    elif method == tot_str:
        solver = ValueIteration(
            transitions=model.transition_matrix,
            reward=model.reward_matrix,
            discount=1.0,
            epsilon=epsilon_exact_value,
            max_iter=max_iter_exact_value,
            skip_check=skip_check,
        )
    solver.run()
    return np.array(solver.V)


def check_model_method_discount(model: GenericModel, method: str, discount: float):
    """
    Check if given model, method and discount are logical inputs.
    """
    assert 0 < discount <= 1.0, "The discount should be between 0 and 1."
    assert hasattr(model, "transition_matrix") and hasattr(model, "reward_matrix"), (
        "Model should have been created using model.create_model()."
    )

    if method == tot_str or method == av_str:
        assert abs(discount - 1.0) < 1e-9, (
            "The discount should be equal to 1 in the total or average case."
        )
    elif method == dis_str:
        assert discount < 1.0, "The discount should be smaller than 1."


def get_exact_value(model: GenericModel, discount: float) -> np.ndarray:
    """
    For a given model, either we fetch the pickle
    saved optimal value or we compute it.
    Parameters :
    criterion : total, discounted or average
    """
    if discount < 0:
        criterion = "average"
    elif discount < 1.0:
        criterion = "discounted"
    else:
        criterion = "total"

    check_model_method_discount(model, criterion, discount)

    saving_folder = SAVED_VALUE_FUNCTIONS_PATH
    saving_folder.mkdir(parents=True, exist_ok=True)

    exact_value_function_pickle_file = "{}.pkl".format(
        build_value_function_name(model, criterion, discount)
    )

    solving_function = lambda: compute_exact_value(
        model, criterion, discount, warning=False
    )

    return get_saved_object(
        saving_folder, solving_function, exact_value_function_pickle_file
    )


def get_saved_object(
    folder_path: str, function_to_compute_it: Callable, file_name: str
):
    """
    If file_name already exists, return the pickle load of it.
    Else, compute a result with function_to_compute_it and save it using pickle.
    """
    return get_cached_object(folder_path, file_name, function_to_compute_it)


def distance_to_optimal(
    value: np.ndarray,
    model: GenericModel,
    discount: float,
    norm_method: float = np.inf,
) -> float:
    if value.ndim == 2:
        value = value.max(axis=1)
    exact_value = get_exact_value(model, discount)
    exact_value -= exact_value.mean()
    value -= value.mean()
    if norm_method == np.inf:
        return norminf(value - exact_value)
    else:
        return np.linalg.norm(value - exact_value, ord=norm_method)


def distance_to_optimal_q(
    q_value: np.ndarray,
    model: GenericModel,
    discount: float,
    norm_method: float = np.inf,
):
    return distance_to_optimal(q_value.max(axis=1), model, discount, norm_method)


def bellman_no_max(
    model: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    For a given value V, returns (R + gamma * T @ V)
    """
    q_value = np.empty((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value[:, aa] = (
            model.reward_matrix[:, aa] + discount * model.transition_matrix[aa] @ value
        )
    return q_value


def get_optimal_policy(model: GenericModel, discount: float) -> np.ndarray:
    """
    Get the optimal policy for a given model and discount.
    """
    value = get_exact_value(model, discount)
    q_value = bellman_no_max(model, value, discount)
    return q_value.argmax(axis=1).astype(np.int32)
