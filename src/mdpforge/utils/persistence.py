"""Load and save model matrices and discounted value references."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.utils.cache_store import (
    CacheCorruptError,
    load_value_entry,
    save_value_entry,
    write_npz,
)
from mdpforge.utils.paths import SAVED_MODELS_PATH, SAVED_VALUE_FUNCTIONS_PATH

CACHE_MODES = frozenset({"readwrite", "readonly", "refresh", "off"})


def validate_cache_mode(cache):
    if cache not in CACHE_MODES:
        raise ValueError(f"cache must be one of {sorted(CACHE_MODES)}")
    return cache


def model_cache_path(model):
    """Return the NPZ path identified by the model's unique name."""
    return Path(SAVED_MODELS_PATH) / f"{quote(model.name, safe='')}.npz"


def value_function_cache_path(model, discount):
    """Return the named path for a built model's discounted value reference."""
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before accessing its value cache")
    name = quote(f"{discount}_{model.name}", safe="")
    return Path(SAVED_VALUE_FUNCTIONS_PATH) / f"{name}.npz"


def load_model(model, *, strict=False):
    """Load and validate named matrices before updating the model."""
    from mdpforge.core.validation import validate_model

    path = model_cache_path(model)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            rewards = arrays["rewards"]
            states, actions = rewards.shape
            if actions != model.action_dim:
                raise ValueError("Cached action count differs from requested model")
            matrices = [
                csr_matrix(
                    (
                        arrays[f"t{i}_data"],
                        arrays[f"t{i}_indices"],
                        arrays[f"t{i}_indptr"],
                    ),
                    shape=(states, states),
                )
                for i in range(actions)
            ]
        for matrix in matrices:
            matrix.check_format(full_check=True)
        candidate = SimpleNamespace(
            name=model.name,
            state_dim=states,
            action_dim=actions,
            transition_matrix=matrices,
            reward_matrix=rewards,
        )
        validate_model(candidate)
    except (
        OSError,
        ValueError,
        KeyError,
        BadZipFile,
        EOFError,
        TypeError,
        IndexError,
    ) as exc:
        if strict:
            raise CacheCorruptError(f"Invalid model cache {path}: {exc}") from exc
        return False
    model.state_dim = candidate.state_dim
    model.action_dim = candidate.action_dim
    model.transition_matrix = candidate.transition_matrix
    model.reward_matrix = candidate.reward_matrix
    return True


def save_model(model):
    """Atomically save CSR transitions and rewards under the model's name."""
    arrays = {"rewards": model.reward_matrix}
    for i, matrix in enumerate(model.transition_matrix):
        arrays[f"t{i}_data"] = matrix.data
        arrays[f"t{i}_indices"] = matrix.indices
        arrays[f"t{i}_indptr"] = matrix.indptr
    write_npz(model_cache_path(model), **arrays)


def get_cached_value_function(
    model, discount, precision, compute: Callable[[], np.ndarray], *, cache="readwrite"
):
    """Fetch or calculate a discounted value reference."""
    validate_cache_mode(cache)
    if not (0 < discount < 1) or not np.isfinite(discount):
        raise ValueError("discount must be strictly between 0 and 1")
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be positive and finite")
    if not hasattr(model, "transition_matrix") or not hasattr(model, "reward_matrix"):
        raise ValueError("Build the MDP before computing or loading reference values")
    if cache == "off":
        return np.asarray(compute(), dtype=float)
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
    value = np.asarray(compute(), dtype=float)
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
