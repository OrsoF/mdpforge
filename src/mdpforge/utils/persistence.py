"""Load and save model matrices and discounted value references."""

from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import numpy as np

from mdpforge.utils.cache_keys import recipe_identity
from mdpforge.utils.cache_store import (
    CacheCorruptError,
    load_model_entry,
    load_value_entry,
    save_model_entry,
    save_value_entry,
    verify_value,
)
from mdpforge.utils.paths import SAVED_MODELS_PATH, SAVED_VALUE_FUNCTIONS_PATH

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


def value_function_cache_path(model, discount):
    """Return the named path for a built model's discounted value reference."""
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before accessing its value cache")
    name = quote(f"{discount}_{model.name}", safe="")
    return Path(SAVED_VALUE_FUNCTIONS_PATH) / f"{name}.npz"


def load_model(model, *, key, strict=False):
    """Load v2 matrices if their recipe and checksums match."""
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
    model.state_dim = candidate.state_dim
    model.action_dim = candidate.action_dim
    model.transition_matrix = candidate.transition_matrix
    model.reward_matrix = candidate.reward_matrix
    return True


def save_model(model, *, key):
    """Save matrices under a previously captured recipe identity."""
    save_model_entry(_entry_path(SAVED_MODELS_PATH, key), key, model)


def get_cached_value_function(
    model, discount, precision, compute: Callable[[], np.ndarray], *, cache="readwrite"
):
    """Fetch or calculate a numerical reference, checking the Bellman residual."""
    validate_cache_mode(cache)
    if not (0 < discount < 1) or not np.isfinite(discount):
        raise ValueError("discount must be strictly between 0 and 1")
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be positive and finite")
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before computing or loading reference values")
    if cache == "off":
        return verify_value(model, compute(), discount, precision)
    path = value_function_cache_path(model, discount)
    key = f"{discount}_{model.name}"
    if cache != "refresh" and path.exists():
        try:
            cached = load_value_entry(path, key, model, discount, precision)
        except CacheCorruptError:
            if cache == "readonly":
                raise
        else:
            if cached is not None:
                return cached
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
