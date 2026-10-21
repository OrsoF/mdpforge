"""Versioned model and value caches; legacy pickle helpers are opt-in only."""

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mdpforge.utils.cache_keys import (
    CacheIdentityError,
    recipe_identity,
    value_identity,
)
from mdpforge.utils.cache_store import (
    CacheCorruptError,
    load_model_entry,
    load_value_entry,
    save_model_entry,
    save_value_entry,
)
from mdpforge.utils.paths import SAVED_MODELS_PATH, SAVED_VALUE_FUNCTIONS_PATH

if TYPE_CHECKING:
    from mdpforge.core.model import MDPProtocol

CACHE_MODES = frozenset({"readwrite", "readonly", "refresh", "off"})


def validate_cache_mode(cache):
    if cache not in CACHE_MODES:
        raise ValueError(f"cache must be one of {sorted(CACHE_MODES)}")
    return cache


def _entry_path(directory, key):
    return Path(directory) / "v2" / key[:2] / f"{key}.npz"


def model_cache_path(model, *, normalize_reward=False):
    """Return a v2 path for a cacheable model recipe, or None."""
    if isinstance(model, str):
        raise TypeError("Provide a model instance; names are not cache identities")
    if hasattr(model, "_cache_recipe_key"):
        key = model._cache_recipe_key
    elif hasattr(model, "transition_matrix") and hasattr(model, "reward_matrix"):
        # There is no trustworthy pre-build recipe for a pre-populated model.
        return None
    else:
        key = recipe_identity(model, normalize_reward=normalize_reward)
    return _entry_path(SAVED_MODELS_PATH, key) if key else None


def value_function_cache_path(model, discount, precision=1e-3):
    """Return the v2 path for a built model's numerical reference."""
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before accessing its value cache")
    return _entry_path(
        SAVED_VALUE_FUNCTIONS_PATH, value_identity(model, discount, precision)
    )


def pack_model(model: "MDPProtocol") -> tuple:
    """Pack numerical fields for an explicitly trusted legacy export."""
    return (
        model.state_dim,
        model.action_dim,
        model.transition_matrix,
        model.reward_matrix,
    )


def restore_model(model: "MDPProtocol", payload: tuple) -> None:
    """Restore numerical fields from an explicitly trusted legacy export."""
    (
        model.state_dim,
        model.action_dim,
        model.transition_matrix,
        model.reward_matrix,
    ) = payload


def load_model(model, *, key=None, normalize_reward=False, strict=False):
    """Load v2 matrices if their recipe and checksums match."""
    key = key or recipe_identity(model, normalize_reward=normalize_reward)
    if key is None:
        return False
    path = _entry_path(SAVED_MODELS_PATH, key)
    if not path.exists():
        return False
    try:
        candidate = load_model_entry(path, key)
        if candidate.action_dim != model.action_dim:
            raise CacheCorruptError("Cached action count differs from requested model")
    except CacheCorruptError:
        if strict:
            raise
        return False
    restore_model(model, pack_model(candidate))
    return True


def save_model(model, *, key=None, normalize_reward=False):
    """Save matrices under a previously captured recipe identity."""
    key = key or getattr(model, "_cache_recipe_key", None)
    if key is None:
        # A post-build snapshot is not a safe substitute for the original recipe.
        raise CacheIdentityError(
            "No pre-build recipe; use create_model(cache='readwrite')"
        )
    save_model_entry(_entry_path(SAVED_MODELS_PATH, key), key, model)


def get_cached_value_function(
    model, discount, precision, compute: Callable[[], Any], *, cache="readwrite"
):
    """Fetch or calculate a numerical reference, checking the Bellman residual."""
    from mdpforge.utils.cache_store import verify_value

    validate_cache_mode(cache)
    if not (0 < discount < 1) or not np.isfinite(discount):
        raise ValueError("discount must be strictly between 0 and 1")
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be positive and finite")
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before computing or loading reference values")
    if cache == "off":
        return verify_value(model, compute(), discount, precision)
    key = value_identity(model, discount, precision)
    path = _entry_path(SAVED_VALUE_FUNCTIONS_PATH, key)
    if cache != "refresh" and path.exists():
        try:
            return load_value_entry(path, key, model, discount, precision)
        except CacheCorruptError:
            if cache == "readonly":
                raise
    value = verify_value(model, compute(), discount, precision)
    if cache in ("refresh", "readwrite"):
        return save_value_entry(
            path,
            key,
            model,
            discount,
            precision,
            value,
            replace=cache == "refresh",
        )
    return value


# Old pickle files are not read by any automatic v2 cache path. Only callers who
# explicitly trust an old artifact should use these legacy helpers.
def ensure_pickle_suffix(file_name: str | Path) -> Path:
    path = Path(file_name)
    return path if path.suffix == ".pkl" else path.with_name(path.name + ".pkl")


def save_pickle(object_instance: object, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(object_instance, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: str | Path) -> Any:
    """Unpickle only an explicitly trusted legacy artifact."""
    with Path(file_path).open("rb") as handle:
        return pickle.load(handle)


def legacy_model_cache_path(model_name: str) -> Path:
    return Path(SAVED_MODELS_PATH) / ensure_pickle_suffix(model_name)


def legacy_value_function_cache_path(name: str) -> Path:
    return Path(SAVED_VALUE_FUNCTIONS_PATH) / ensure_pickle_suffix(name)


def import_legacy_model_cache(model, *, trusted=False, normalize_reward=False):
    """Import a trusted legacy pickle only when explicitly authorized."""
    from scipy.sparse import csr_matrix

    from mdpforge.core.validation import validate_model

    if not trusted:
        raise ValueError("Legacy pickle import requires trusted=True")
    key = recipe_identity(model, normalize_reward=normalize_reward)
    if key is None:
        raise CacheIdentityError("Model has no declared cacheable recipe")
    payload = load_pickle(legacy_model_cache_path(model.name))
    if not isinstance(payload, tuple) or len(payload) != 4:
        raise ValueError("Invalid legacy model payload")
    from types import SimpleNamespace

    candidate = SimpleNamespace(
        name=model.name,
        state_dim=payload[0],
        action_dim=payload[1],
        transition_matrix=[csr_matrix(matrix) for matrix in payload[2]],
        reward_matrix=np.asarray(payload[3], dtype=float),
    )
    if candidate.action_dim != model.action_dim:
        raise ValueError("Legacy model action count mismatch")
    validate_model(candidate)
    save_model_entry(_entry_path(SAVED_MODELS_PATH, key), key, candidate)
    return True
