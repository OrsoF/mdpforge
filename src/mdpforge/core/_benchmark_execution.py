"""Execute independent solver trials with optional process isolation."""

import inspect
import multiprocessing
import pickle
import random
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from mdpforge.utils.solver_names import is_catalogue_solver


def _solver_trial(model, constructor, options, discount, precision, seed, solver_name):
    """Return values, solver runtime, status and error for one independent trial."""
    runtime = np.nan
    start = None
    try:
        trial_model = deepcopy(model)
        np.random.seed(seed)
        random.seed(seed)
        precision_options = {}
        if isinstance(constructor, type):
            parameters = inspect.signature(constructor).parameters
            precision_options = (
                {"final_precision": precision}
                if "final_precision" in parameters
                or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
                else {}
            )
        start = perf_counter()
        if isinstance(constructor, type):
            solver = constructor(trial_model, discount, **precision_options, **options)
            if is_catalogue_solver(constructor):
                solver.name = solver_name
            value = solver.run().value
        else:
            value = constructor(
                trial_model.transition_matrix,
                trial_model.reward_matrix,
                discount,
                precision,
                **options,
            )
        runtime = perf_counter() - start
        return np.asarray(value), runtime, "success", ""
    except Exception as exc:
        if start is not None and np.isnan(runtime):
            runtime = perf_counter() - start
        return None, runtime, "error", f"{type(exc).__name__}: {exc}"


def _solver_worker(input_path, result_path):
    # Files avoid blocking on a full IPC pipe during dispatch or worker exit.
    with input_path.open("rb") as handle:
        args = pickle.load(handle)
    with result_path.open("wb") as handle:
        pickle.dump(_solver_trial(*args), handle)


def _isolated_solver_trial(
    model, constructor, options, discount, precision, seed, solver_name, timeout
):
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="mdpforge-trial-") as directory:
        # Serialize before spawning so unpickleable inputs cannot leave a worker
        # behind if process startup fails halfway through transmitting its inputs.
        input_path = Path(directory) / "input.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(
                (model, constructor, options, discount, precision, seed, solver_name),
                handle,
            )
        result_path = Path(directory) / "result.pkl"
        process = context.Process(
            target=_solver_worker,
            args=(input_path, result_path),
        )
        try:
            start = perf_counter()
            process.start()
            process.join(max(0, timeout - (perf_counter() - start)))
            if process.is_alive():
                return (
                    None,
                    np.nan,
                    "timeout",
                    f"Trial exceeded timeout of {timeout:g} s",
                )
            if process.exitcode != 0:
                return (
                    None,
                    np.nan,
                    "error",
                    f"Solver process exited with code {process.exitcode}",
                )
            with result_path.open("rb") as handle:
                return pickle.load(handle)
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join(1)
                if not process.is_alive():
                    process.close()
