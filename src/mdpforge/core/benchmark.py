"""Compare solvers at a common discounted value precision."""

import csv
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np

from mdpforge.core.model import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model


def benchmark(model: MDP, solvers, *, discount, epsilon=1e-3, repeats=3, seed=0):
    """Return one measurement per solver and repeat on an already built model.

    ``solvers`` maps labels to constructors accepting model, discount and
    final_precision. Runtime includes construction and run(), excluding model
    copying and independent precision verification. Each repeat uses seed+i for
    NumPy's global RNG and Python random; both states are restored afterward.
    Independent generators must be seeded through the solver's own options.
    """
    assert 0 < discount < 1, "discount must be strictly between 0 and 1"
    assert np.isfinite(epsilon) and epsilon > 0, "epsilon must be finite and positive"
    assert isinstance(repeats, int) and repeats > 0, (
        "repeats must be a positive integer"
    )
    validate_model(model)
    numpy_state, random_state = np.random.get_state(), random.getstate()
    results = []
    try:
        for name, constructor in solvers.items():
            for repeat in range(repeats):
                runtime = residual = error_bound = np.nan
                status, error = "error", ""
                start = None
                try:
                    trial_model = deepcopy(model)
                    np.random.seed(seed + repeat)
                    random.seed(seed + repeat)
                    start = perf_counter()
                    solver = constructor(trial_model, discount, final_precision=epsilon)
                    solver.run()
                    runtime = perf_counter() - start
                    value = np.asarray(solver.value)
                    if value.shape != (model.state_dim,) or not np.all(
                        np.isfinite(value)
                    ):
                        raise ValueError(
                            "solver.value must be a finite vector of state_dim"
                        )
                    residual = float(
                        np.abs(
                            optimal_bellman_operator(model, value, discount) - value
                        ).max()
                    )
                    error_bound = residual / (1 - discount)
                    status = "success" if error_bound <= epsilon else "imprecise"
                    if status == "imprecise":
                        error = f"Value error bound {error_bound:g} exceeds epsilon {epsilon:g}"
                except Exception as exc:
                    if start is not None and np.isnan(runtime):
                        runtime = perf_counter() - start
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    {
                        "model": model.name,
                        "state_dim": model.state_dim,
                        "action_dim": model.action_dim,
                        "solver": name,
                        "repeat": repeat + 1,
                        "seed": seed + repeat,
                        "discount": discount,
                        "epsilon": epsilon,
                        "runtime": runtime,
                        "residual": residual,
                        "error_bound": error_bound,
                        "status": status,
                        "error": error,
                    }
                )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)
    return results


def export_csv(results, path):
    """Write benchmark measurements as CSV and return the output path."""
    if not results:
        raise ValueError("No benchmark measurements to export")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return path
