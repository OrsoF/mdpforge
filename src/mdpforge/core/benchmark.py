"""Benchmark finite discounted MDP solvers at a fixed value precision."""

import csv
import random
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from time import perf_counter

import numpy as np

from mdpforge import models, solvers
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model


class Benchmark:
    """Compare catalogue, matrix-defined or already-built finite discounted MDPs.

    VI is registered by default; use default_solvers=False to select
    only your own solvers.

    Models must expose ``name``, ``state_dim``, ``action_dim``,
    ``transition_matrix`` and ``reward_matrix``. Solver functions accept
    ``(transitions, rewards, discount, precision)`` and return a value vector.
    Solver classes accept ``(model, discount, final_precision=...)`` and provide
    ``run()`` and a ``value`` vector.

    Runtime includes function execution or class construction and ``run()``,
    but excludes model copying, validation and independent precision verification.
    """

    def __init__(self, *, default_solvers=True):
        self._mdps: list[MDPProtocol] = []
        self._solvers = {}  # label -> (solver class/function, options)
        self.results = []
        if default_solvers:
            from mdpforge.solvers.personal_vi import Solver as VI

            self.add_solver("VI", solve_function=VI)

    def add_mdp(
        self, model: MDPProtocol | str, *, transitions=None, reward=None, rewards=None
    ):
        """Register a catalogue name, custom matrices or a built MDP.

        Catalogue models use Model() and create_model() with their defaults.
        Unknown names require transitions and reward (or rewards), converted to
        CSR and validated with dimensions inferred from rewards of shape (S, A).
        Each built MDP name must be unique. Construction is outside solver timing.
        """
        if reward is not None:
            if rewards is not None:
                raise ValueError("Provide either reward or rewards, not both.")
            rewards = reward
        if isinstance(model, str):
            if not model.strip():
                raise ValueError("Each MDP must have a nonempty string 'name'.")
            catalogue = {
                module.name
                for module in iter_modules(models.__path__)
                if not module.ispkg and not module.name.startswith("_")
            }
            if model in catalogue:
                if transitions is not None or rewards is not None:
                    raise ValueError(
                        f"Catalogue model {model!r} uses its generator; "
                        "choose a custom name when supplying matrices."
                    )
                model = import_module(f"{models.__name__}.{model}").Model()
                model.create_model()
            else:
                if transitions is None or rewards is None:
                    raise ValueError(
                        f"Unknown model {model!r}. Provide both transitions and rewards "
                        "for a custom MDP. Available models: "
                        + ", ".join(sorted(catalogue))
                    )
                model = MDP.from_matrices(model, transitions, rewards)
        elif transitions is not None or rewards is not None:
            raise ValueError("Pass a name when providing transitions and rewards.")
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each MDP must have a nonempty string 'name'.")
        if any(existing.name == name for existing in self._mdps):
            raise ValueError(f"An MDP named {name!r} is already registered.")
        self._mdps.append(model)
        self.results = []
        return self

    def add_solver(self, solver_name: str, *, solve_function=None, **options):
        """Register a catalogue solver or a named custom function/class.

        Without solve_function, solver_name identifies a module in solvers/.
        Otherwise it labels the supplied callable. Functions receive
        (CSR transitions, NumPy rewards, discount, precision) and return values.
        Classes receive (model, discount, final_precision=...) and expose run()
        and value. Options are forwarded; problem data, discount and precision
        are supplied by run() and cannot be overridden per solver.
        """
        if not isinstance(solver_name, str) or not solver_name.strip():
            raise ValueError("A solver must have a nonempty string name.")
        if solver_name in self._solvers:
            raise ValueError(f"A solver named {solver_name!r} is already registered.")
        reserved = {
            "model",
            "transitions",
            "rewards",
            "discount",
            "precision",
            "final_precision",
        } & options.keys()
        if reserved:
            raise ValueError(
                f"Solver parameters controlled by run(): {sorted(reserved)}"
            )
        if solve_function is None:
            catalogue = {
                module.name
                for module in iter_modules(solvers.__path__)
                if not module.ispkg and not module.name.startswith("_")
            }
            if solver_name not in catalogue:
                raise ValueError(
                    f"Unknown solver {solver_name!r}. Available solvers: "
                    + ", ".join(sorted(catalogue))
                )
            solve_function = import_module(f"{solvers.__name__}.{solver_name}").Solver
        if not callable(solve_function):
            raise TypeError("solve_function must be a callable function or class.")
        self._solvers[solver_name] = (solve_function, options)
        self.results = []
        return self

    def run(self, discount, precision=1e-3, *, repeats=3, seed=0):
        """Run every registered MDP/solver pair at a common value precision.

        ``precision`` is an upper bound on ||V - V*||_infinity, certified by
        ||T V - V||_infinity / (1 - discount). All failures are recorded rather
        than interrupting the remaining comparisons.

        Each solver receives an independent copy of its MDP. Seeds are paired
        across solvers and MDPs for each repeat. Global NumPy and Python RNG
        states are restored; solvers using private RNGs must seed those
        generators themselves.
        """
        if not np.isfinite(discount) or not 0 < discount < 1:
            raise ValueError("discount must be strictly between 0 and 1")
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be finite and positive")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32 - repeats:
            raise ValueError("seed must allow valid NumPy seeds for every repeat")
        if not self._mdps:
            raise ValueError("No MDP registered; call add_mdp() first")
        if not self._solvers:
            raise ValueError("No solver registered; call add_solver() first")

        for model in self._mdps:
            validate_model(model)

        self.results = []
        numpy_state, random_state = np.random.get_state(), random.getstate()
        try:
            for model in self._mdps:
                for solver_name, (constructor, kwargs) in self._solvers.items():
                    for repeat in range(repeats):
                        runtime = residual = error_bound = np.nan
                        status, error = "error", ""
                        start = None
                        try:
                            trial_model = deepcopy(model)
                            np.random.seed(seed + repeat)
                            random.seed(seed + repeat)
                            start = perf_counter()
                            if isinstance(constructor, type):
                                solver = constructor(
                                    trial_model,
                                    discount,
                                    final_precision=precision,
                                    **kwargs,
                                )
                                solver.run()
                                value = solver.value
                            else:
                                value = constructor(
                                    trial_model.transition_matrix,
                                    trial_model.reward_matrix,
                                    discount,
                                    precision,
                                    **kwargs,
                                )
                            runtime = perf_counter() - start

                            value = np.asarray(value)
                            if value.shape != (model.state_dim,) or not np.all(
                                np.isfinite(value)
                            ):
                                raise ValueError(
                                    "Solver values must be a finite vector of state_dim"
                                )

                            residual = float(
                                np.max(
                                    np.abs(
                                        optimal_bellman_operator(
                                            model, value, discount
                                        )
                                        - value
                                    )
                                )
                            )
                            error_bound = residual / (1 - discount)
                            status = (
                                "success" if error_bound <= precision else "imprecise"
                            )
                            if status == "imprecise":
                                error = (
                                    f"Value error bound {error_bound:g} "
                                    f"exceeds precision {precision:g}"
                                )
                        except Exception as exc:
                            if start is not None and np.isnan(runtime):
                                runtime = perf_counter() - start
                            error = f"{type(exc).__name__}: {exc}"
                        self.results.append(
                            {
                                "model": model.name,
                                "state_dim": model.state_dim,
                                "action_dim": model.action_dim,
                                "solver": solver_name,
                                "repeat": repeat + 1,
                                "seed": seed + repeat,
                                "discount": discount,
                                "epsilon": precision,  # Keep the original CSV schema.
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
        return self.results

    def plot_heat(self, *, reference=None, metric="speedup", ax=None, show=True):
        """Plot a solver-by-MDP heatmap from the last run.

        ``metric='speedup'`` shows median reference runtime / median solver
        runtime; ``metric='runtime'`` shows median runtime in seconds.
        A cell is only plotted when *all* repeats met the target precision.
        Failed/imprecise trials are marked FAIL, not silently discarded.

        Returns ``(figure, axes)``. Matplotlib is an optional dependency.
        """
        if not self.results:
            raise ValueError("No results available; call run() before plot_heat()")
        if metric not in {"speedup", "runtime"}:
            raise ValueError("metric must be 'speedup' or 'runtime'")
        if reference is None:
            reference = next(iter(self._solvers))
        if metric == "speedup" and reference not in self._solvers:
            raise ValueError(f"Unknown reference solver: {reference!r}")

        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        models = [model.name for model in self._mdps]
        solvers = list(self._solvers)
        times = np.full((len(solvers), len(models)), np.nan)
        failures = np.zeros_like(times, dtype=bool)
        for i, solver in enumerate(solvers):
            for j, model in enumerate(models):
                rows = [
                    row for row in self.results
                    if row["solver"] == solver and row["model"] == model
                ]
                if not rows or any(row["status"] != "success" for row in rows):
                    failures[i, j] = True
                    continue
                times[i, j] = float(np.median([row["runtime"] for row in rows]))

        if metric == "speedup":
            ref_times = times[solvers.index(reference)]
            with np.errstate(divide="ignore", invalid="ignore"):
                values = ref_times[np.newaxis, :] / times
                colors = np.log2(values)
            colors[~np.isfinite(colors)] = np.nan
            finite = np.abs(colors[np.isfinite(colors)])
            limit = max(1.0, float(finite.max())) if finite.size else 1.0
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            cmap = plt.get_cmap("RdYlGn").copy()
            color_label = f"log₂ speedup vs {reference}"
        else:
            values = times
            colors = times
            norm = None
            cmap = plt.get_cmap("viridis_r").copy()
            color_label = "Runtime (s)"

        cmap.set_bad("#e5e7eb")
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(max(6, 1.4 * len(models) + 2), max(3, 0.55 * len(solvers) + 2))
            )
        else:
            fig = ax.figure
        im = ax.imshow(
            np.ma.masked_invalid(colors), cmap=cmap, norm=norm, aspect="auto"
        )
        ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
        ax.set_yticks(range(len(solvers)), solvers)
        ax.set_xlabel("MDP")
        ax.set_ylabel("Solver")
        settings = self.results[0]
        ax.set_title(
            f"Runtime at fixed precision — γ={settings['discount']:g}, "
            f"ε={settings['epsilon']:g}"
        )
        for i in range(len(solvers)):
            for j in range(len(models)):
                if failures[i, j]:
                    label = "FAIL"
                elif not np.isfinite(values[i, j]):
                    label = "N/A"
                elif metric == "speedup":
                    label = f"{values[i, j]:.2f}×"
                else:
                    label = f"{values[i, j]:.3g} s"
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=color_label)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def export_csv(self, path):
        """Export the latest run without rerunning any solver."""
        return export_csv(self.results, path)


def benchmark(
    model: MDPProtocol, solvers, *, discount, epsilon=1e-3, repeats=3, seed=0
):
    """Backward-compatible single-MDP benchmark, returning result rows."""
    bench = Benchmark(default_solvers=False).add_mdp(model)
    for name, constructor in solvers.items():
        bench.add_solver(name, solve_function=constructor)
    return bench.run(discount, precision=epsilon, repeats=repeats, seed=seed)


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
