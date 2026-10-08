import pickle as pkl
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mdpforge.utils.paths import SAVED_MODELS_PATH, SAVED_VALUE_FUNCTIONS_PATH

if TYPE_CHECKING:
    from mdpforge.core.model import GenericModel


def save_pickle(object_instance: object, file_path: str | Path) -> None:
    """Save an object as a pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as handle:
        pkl.dump(object_instance, handle, protocol=pkl.HIGHEST_PROTOCOL)


def load_pickle(file_path: str | Path) -> Any:
    """Load an object from a pickle file."""
    with Path(file_path).open("rb") as handle:
        return pkl.load(handle)


def ensure_pickle_suffix(file_name: str | Path) -> Path:
    """Return a path with a .pkl suffix."""
    file_path = Path(file_name)
    if file_path.suffix == ".pkl":
        return file_path
    file_path = file_path.with_name(f"{file_path.name}.pkl")
    return file_path


def get_cached_object(
    folder_path: str | Path,
    file_name: str | Path,
    compute: Callable[[], Any],
) -> Any:
    """Load a cached object if present, otherwise compute, save, and return it."""
    folder_path = Path(folder_path)
    file_path = folder_path / ensure_pickle_suffix(file_name)

    if file_path.exists():
        return load_pickle(file_path)

    object_instance = compute()
    save_pickle(object_instance, file_path)
    return object_instance


def model_cache_path(model_name: str) -> Path:
    """Return the cache path for a built model."""
    return SAVED_MODELS_PATH / ensure_pickle_suffix(model_name)


def pack_model(model: "GenericModel") -> tuple:
    """Pack the persisted part of a model."""
    return (
        model.state_dim,
        model.action_dim,
        model.transition_matrix,
        model.reward_matrix,
    )


def restore_model(model: "GenericModel", payload: tuple) -> None:
    """Restore persisted model fields in place."""
    (
        model.state_dim,
        model.action_dim,
        model.transition_matrix,
        model.reward_matrix,
    ) = payload


def save_model(model: "GenericModel") -> None:
    """Save a built model to its standard cache path."""
    save_pickle(pack_model(model), model_cache_path(model.name))


def load_model(model: "GenericModel") -> bool:
    """Load a model from its standard cache path. Return True if found."""
    file_path = model_cache_path(model.name)
    if not file_path.exists():
        return False

    restore_model(model, load_pickle(file_path))
    return True


def value_function_cache_path(name: str) -> Path:
    """Return the cache path for a saved value function."""
    return SAVED_VALUE_FUNCTIONS_PATH / ensure_pickle_suffix(name)


def get_cached_value_function(name: str, compute: Callable[[], Any]) -> Any:
    """Load or compute a value function using the standard value cache."""
    return get_cached_object(SAVED_VALUE_FUNCTIONS_PATH, name, compute)
