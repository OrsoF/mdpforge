"""Run total-reward planning benchmarks and write runtime tables."""

import argparse
import csv
from importlib import import_module
from pathlib import Path

import numpy as np

from utils.data_management import import_models_from_file
from utils.paths import RESULTS_PATH

MODELS = {
    "rooms": ("total/rooms_total", 100_000, 4, "Rooms total"),
    "mountain": ("total/mountain_total", 50_000, 3, "Mountain car total"),
    "taxi": ("total/taxi_total", 100_000, 6, "Taxi total"),
    "barto": ("total/barto_total", 50_000, 9, "Barto total"),
}

ALIASES = {"mountain_car": "mountain", "rooms_total": "rooms", "taxi_total": "taxi"}

SOLVERS = {
    "pdvi": ("aggregated_vi_total", "PDVI"),
    "pdqvi": ("aggregated_qvi_total", "PDQVI"),
    "vi": ("personal_vi_total", "VI"),
    "pdpi": ("aggregated_pim_total", "PDPI"),
    "mpi": ("personal_pim_total", "MPI"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", nargs="+", required=True, choices=sorted(MODELS | ALIASES)
    )
    p.add_argument(
        "--solvers", "--solver", nargs="+", choices=SOLVERS, default=list(SOLVERS)
    )
    p.add_argument("--state", type=int)
    p.add_argument("--action", type=int)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output-dir", type=Path, default=RESULTS_PATH / "article_runtimes_total"
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--projected-steps", type=int, default=20)
    p.add_argument("--pdvi-bellman-updates", type=int, default=20)
    p.add_argument("--pdvi-projected-steps", type=int, default=100)
    p.add_argument("--pdqvi-bellman-updates", type=int, default=20)
    p.add_argument("--pdpi-bellman-updates", type=int, default=15)
    p.add_argument("--pdpi-projected-steps", type=int, default=100)
    p.add_argument("--mpi-max-eval-iterations", type=int, default=100_000_000)
    return p.parse_args()


def solver_options(name, a):
    if name == "pdvi":
        return {
            "verbose": a.verbose,
            "bellman_updates": a.pdvi_bellman_updates,
            "projected_steps": a.pdvi_projected_steps,
        }
    if name == "pdqvi":
        return {
            "verbose": a.verbose,
            "bellman_updates": a.pdqvi_bellman_updates,
            "projected_steps": a.projected_steps,
        }
    if name == "pdpi":
        return {
            "verbose": a.verbose,
            "bellman_updates": a.pdpi_bellman_updates,
            "projected_steps": a.pdpi_projected_steps,
        }
    return {}


def make_solver(name, model, a):
    solver = import_module(f"solvers.total.{SOLVERS[name][0]}").Solver(
        model, a.discount, **solver_options(name, a)
    )
    if name == "mpi":
        solver.max_iter_eval = a.mpi_max_eval_iterations
    return solver


def write_results(out, model_name, title, model, a, runs, summary):
    out.mkdir(parents=True, exist_ok=True)
    stem = out / model_name
    with stem.with_name(f"{model_name}_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.DictWriter(f, fieldnames=runs[0])
        w.writeheader()
        w.writerows(runs)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0])
        w.writeheader()
        w.writerows(summary)
    lines = ["\\begin{tabular}{c|c}", "Method & Runtime (s) \\\\ \\toprule"]
    lines += [
        f"{row['method']} & {row['mean_runtime']:.2f} $\\pm$ {row['std_runtime']:.2f} \\\\"
        for row in summary
    ]
    lines += [
        "\\end{tabular}",
        f"% {title}: |S|={model.state_dim}, |A|={model.action_dim}, "
        f"runs={a.repeat}, total reward.",
    ]
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model(requested_model, a):
    model_name = ALIASES.get(requested_model, requested_model)
    module, default_state, default_action, title = MODELS[model_name]
    state = default_state if a.state is None else a.state
    action = default_action if a.action is None else a.action
    np.random.seed(a.seed)
    model = import_models_from_file(module)(state, action)
    model.create_model()
    print(f"{title}: |S|={model.state_dim}, |A|={model.action_dim}, runs={a.repeat}")
    if a.verbose:
        print(
            f"requested_state={state}, requested_action={action}, "
            f"seed={a.seed}, output={a.output_dir}"
        )
        print("solvers=" + ", ".join(SOLVERS[name][1] for name in a.solvers))
        for name in a.solvers:
            options = solver_options(name, a)
            if name == "mpi":
                options["max_iter_eval"] = a.mpi_max_eval_iterations
            print(f"  {SOLVERS[name][1]}: {options}")

    runs, summary = [], []
    for name in a.solvers:
        method = SOLVERS[name][1]
        runtimes, regions = [], []
        for run in range(a.repeat):
            np.random.seed(a.seed + run)
            solver = make_solver(name, model, a)
            solver.run()
            runtimes.append(float(solver.runtime))
            n_regions = getattr(getattr(solver, "partition", None), "n_regions", None)
            regions.append(n_regions)
            runs.append(
                {
                    "model": model_name,
                    "solver": name,
                    "run": run + 1,
                    "runtime": runtimes[-1],
                    "regions": n_regions,
                }
            )
            print(f"  {method} {run + 1}/{a.repeat}: {runtimes[-1]:.3f}s")
        summary.append(
            {
                "model": model_name,
                "solver": name,
                "method": method,
                "runs": a.repeat,
                "mean_runtime": float(np.mean(runtimes)),
                "std_runtime": float(np.std(runtimes)),
                "mean_regions": float(np.mean([k for k in regions if k is not None]))
                if any(k is not None for k in regions)
                else "",
            }
        )
        write_results(a.output_dir, model_name, title, model, a, runs, summary)
    print(f"Wrote results to {a.output_dir / model_name}.[csv,tex]")
    if len(a.solvers) == len(SOLVERS) and set(a.solvers) == set(SOLVERS):
        fastest = sorted(summary, key=lambda row: row["mean_runtime"])[:2]
        print(
            f"Fastest: {fastest[0]['method']} ({fastest[0]['mean_runtime']:.3f}s mean)"
        )
        print(
            f"Second fastest: {fastest[1]['method']} "
            f"({fastest[1]['mean_runtime']:.3f}s mean)"
        )


def main():
    a = parse_args()
    if a.repeat < 2:
        raise ValueError(
            "--repeat must be at least 2 to estimate runtime standard deviation"
        )
    if a.discount != 1.0:
        raise ValueError("total-reward solvers require --discount 1")
    for model_name in a.model:
        run_model(model_name, a)


if __name__ == "__main__":
    main()
