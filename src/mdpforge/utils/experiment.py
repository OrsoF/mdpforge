"""Capture experiment provenance without changing solver inputs or RNG state."""

import ast
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
from threadpoolctl import threadpool_info


def _qualified_name(value):
    target = value if inspect.isclass(value) or inspect.isroutine(value) else type(value)
    return f"{target.__module__}.{target.__qualname__}"


def _json_value(value):
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            return {"dtype": str(value.dtype), "value": str(value)}
        return _json_value(scalar)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _json_value(value.tolist()),
        }
    if issparse(value):
        return {"shape": list(value.shape), "sha256": _sparse_hash(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        # Non-string keys occur in model state encodings; retain their types.
        if all(isinstance(key, str) for key in value):
            return {key: _json_value(item) for key, item in value.items()}
        return {
            "entries": [
                [_json_value(key), _json_value(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if callable(value):
        return {"callable": _qualified_name(value)}
    return {"unserialized_type": _qualified_name(value)}


def model_config(model):
    """Serialize public attributes; model recipes may override MDP.get_config()."""
    excluded = {"name", "transition_matrix", "reward_matrix"}
    attributes = dict(getattr(model, "__dict__", {}))
    for cls in type(model).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if hasattr(model, name):
                attributes[name] = getattr(model, name)
    return {
        name: _json_value(value)
        for name, value in attributes.items()
        if not name.startswith("_") and name not in excluded
    }


def _update_array(digest, value):
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    array = np.ascontiguousarray(array, dtype=dtype)
    digest.update(json.dumps([array.shape, dtype.str]).encode("ascii"))
    digest.update(array.tobytes())


def _array_hash(value):
    digest = hashlib.sha256()
    _update_array(digest, value)
    return digest.hexdigest()


def _sparse_hash(value):
    matrix = csr_matrix(value, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    digest = hashlib.sha256()
    _update_array(digest, np.asarray(matrix.shape, dtype="<i8"))
    _update_array(digest, matrix.indptr.astype("<i8"))
    _update_array(digest, matrix.indices.astype("<i8"))
    _update_array(digest, matrix.data)
    return digest.hexdigest()


def _source_hash(implementation):
    try:
        source = inspect.getsource(implementation).encode("utf-8")
        return hashlib.sha256(source).hexdigest()
    except (OSError, TypeError):
        return None


def _solver_metadata(constructor, options):
    try:
        parameters = inspect.signature(constructor).parameters
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # Match Benchmark's calling convention, regardless of argument names.
        input_count = 2 if inspect.isclass(constructor) else 4
        controlled = set(positional[:input_count])
        if inspect.isclass(constructor):
            controlled.add("final_precision")
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if name not in controlled
            and parameter.default is not inspect.Parameter.empty
        }
    except (TypeError, ValueError):
        defaults = {}
    return {
        "callable": _qualified_name(constructor),
        "source_sha256": _source_hash(constructor),
        "options": _json_value(options),
        "parameters": _json_value(defaults | options),
    }


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _backend_versions(implementations):
    # Read imports, including lazy imports, without loading any optional backend.
    roots = set()
    for implementation in implementations:
        roots.add(getattr(implementation, "__module__", "").split(".")[0])
        module = inspect.getmodule(implementation)
        try:
            tree = ast.parse(inspect.getsource(module))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                roots.add(node.module.split(".")[0])
    aliases = {"mdptoolbox": "pymdptoolbox", "stable_baselines3": "stable-baselines3"}
    roots -= sys.stdlib_module_names | {"", "__main__", "mdpforge", "numpy", "scipy"}
    backends = {}
    for root in sorted(roots):
        distribution = aliases.get(root, root)
        backends[root] = {
            "distribution": distribution,
            "version": _package_version(distribution),
        }
    return backends


def _git_metadata():
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *arguments],
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout

    try:
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(git("status", "--porcelain")),
            "diff_sha256": hashlib.sha256(
                git("diff", "HEAD", "--binary")
            ).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def _machine_metadata():
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    if sys.platform == "win32":
        import winreg

        try:
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    processor = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": processor or None,
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
        "thread_pools": threadpool_info(),
        "capture_scope": "benchmark process before solver execution",
        "numpy_build": _json_value(getattr(np.__config__, "CONFIG", None)),
    }


def capture_experiment(
    models, solvers, discount, precision, repeats, seed, *, timeout=None
):
    """Freeze metadata once per run, before any solver is constructed."""
    model_metadata = {}
    for model in models:
        transitions = [_sparse_hash(matrix) for matrix in model.transition_matrix]
        rewards = _array_hash(model.reward_matrix)
        get_config = getattr(model, "get_config", None)
        model_metadata[model.name] = {
            "class": _qualified_name(model),
            "source_sha256": _source_hash(type(model)),
            "state_dim": int(model.state_dim),
            "action_dim": int(model.action_dim),
            "config": _json_value(
                get_config() if callable(get_config) else model_config(model)
            ),
            "transition_sha256": transitions,
            "reward_sha256": rewards,
            "data_sha256": hashlib.sha256(
                "".join([*transitions, rewards]).encode("ascii")
            ).hexdigest(),
        }
    implementations = [type(model) for model in models] + [
        constructor for constructor, _ in solvers.values()
    ]
    return {
        "schema_version": 1,
        "experiment_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "discount": float(discount),
            "precision": float(precision),
            "repeats": repeats,
            "seed": seed,
            "trial_seeds": list(range(seed, seed + repeats)),
            "seed_scope": "solver trials; model generation is not reseeded",
            "timeout": None if timeout is None else float(timeout),
            "execution_mode": "in_process" if timeout is None else "spawn",
        },
        "models": model_metadata,
        "solvers": {
            name: _solver_metadata(constructor, options)
            for name, (constructor, options) in solvers.items()
        },
        "environment": {
            "python": platform.python_version(),
            "mdpforge": _package_version("mdpforge"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threadpoolctl": _package_version("threadpoolctl"),
            "backends": _backend_versions(implementations),
        },
        "machine": _machine_metadata(),
        "code": {"git": _git_metadata()},
    }


def export_experiment(manifest, csv_path):
    """Write the captured manifest next to the CSV, using the same basename."""
    path = Path(csv_path).with_suffix(".experiment.json")
    payload = manifest | {"results_file": Path(csv_path).name}
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path
