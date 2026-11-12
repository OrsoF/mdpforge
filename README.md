<h1 align="center">mdpforge</h1>

<p align="center"><strong>Benchmark finite MDPs. Compare solvers at a shared accuracy.</strong></p>

<p align="center">
  <a href="https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OrsoF/mdpforge/ci.yml?branch=main&amp;label=Tests" alt="Tests"></a>
  <a href="https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OrsoF/mdpforge/ci.yml?branch=main&amp;label=Code%20Quality" alt="Code Quality"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python_versions-3.11%2B-blue" alt="Python versions: 3.11 or later"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-008c87" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> · <a href="#models-and-solvers">Models &amp; solvers</a> · <a href="#visualizations">Visualizations</a> · <a href="#examples-and-documentation">Documentation</a> · <a href="#contributing">Contributing</a>
</p>

![Paired speedups against value iteration on Garnet, Chain Walk and Rooms; the fastest solver varies by model.](artifacts/figures/readme_speedups.png)

*Saved speedups at discount 0.99 and target value error 0.001; timings depend on hardware.*

## Quick start

### Install

Requires **Python 3.11+**. `Benchmark()` uses discounted reward (`0 < discount < 1`); `criterion="total"` uses absorbing total reward, and `criterion="average"` uses the discounted model catalogue with custom average-reward solvers.

Install mdpforge with its built-in models, solvers and plotting:

```sh
python -m pip install "git+https://github.com/OrsoF/mdpforge.git"
```

From an existing checkout, run this at the repository root instead:

```sh
python -m pip install -e .
```

The only extras are `[solvers]` for optional pip solver backends and `[models]` for optional model generators. From a checkout, use `python -m pip install ".[solvers]"` or `python -m pip install ".[models]"`. Marmote and madupite require separate native installations; see [Optional backends](#optional-backends).

### Run a benchmark

```python
from mdpforge import Benchmark

bench = Benchmark()
bench.add_mdp("rooms")
bench.add_solver("mdpforge_vi")
bench.add_solver("mdpforge_mpi")

result = bench.run(discount=0.9, precision=1e-3, repeats=3, seed=0)
result.export_csv("results.csv")
```

Each trial records runtime. Pass `validate=True` to certify supported results after timing; average-reward validation is not yet available. `result.export_csv(path)` saves results and settings; use `BenchmarkResult.from_csv(path)` to reopen them.

For total reward, use `Benchmark(criterion="total")`, `bench.add_mdp("progress_chain")`, `bench.add_solver("marmote_vi")`, then `bench.run(precision=1e-3)`. The Marmote backend is installed separately as described below.

Plot the results:

```python
result.plot_performance_profile()
result.plot_heat(reference="mdpforge_vi")
```

Paired speedups use models solved by both solvers on every repeat. Set the reference explicitly when comparing runs.

## Visualizations

### How often is a solver close to the fastest?

![Performance profiles showing the share of models solved within a runtime factor of the fastest eligible solver, grouped by backend.](artifacts/figures/gallery_performance_profiles.png)

At 2×, a curve shows the share of models solved within twice the fastest eligible median runtime; failed or missing pairs remain in the denominator. See the [historical export notes](docs/visualization-roadmap.md).

### How do discount and precision affect runtime?

![Runtime sensitivity on a fixed Garnet instance, varying discount at fixed precision and target value error at fixed discount.](artifacts/figures/readme_sensitivity.png)

The curves use one Garnet instance; bands show the range over three repeats.

## Models and solvers

Top-level `list_models()` and `list_solvers()` browse the discounted catalogues; total catalogues are in `mdpforge.models.total` and `mdpforge.solvers.total`. Select models by `size` and `category`, and solvers by `family` and `backend`. Use `ScalingBenchmark(criterion=...)` for size sweeps.

`Benchmark()` starts empty. Call `bench.add_default_mdps()` for small Rooms, Mountain Car and Garnet models, and `bench.add_default_solvers()` for `mdpforge_vi` and `mdpforge_mpi`.

### Bring your own model

Register transition matrices, one per action, and expected rewards of shape `(states, actions)`. Dense and sparse inputs are accepted:

```python
bench.add_mdp(
    "my_mdp",
    transitions=[[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
    reward=[[1, 0], [0, 2]],
)
```

### Bring your own solver

Register a callable with `bench.add_solver("my_solver", solve_function=my_solver)`. Discounted functions take `(transitions, rewards, discount, precision)`; total and average functions take `(model, precision)`. Return a NumPy value vector or `SolveResult`; the benchmark handles copies and timing. For a matrix-defined total model, pass `terminal_states` to `add_mdp()` if you want residual validation.

### Optional backends

The `solvers` extra installs MDPax, MDPSolver, MDPToolbox, Pyomo with HiGHS, QuantEcon and CMA. The `models` extra installs mazelib. Version requirements are listed in [pyproject.toml](pyproject.toml).

Install Marmote separately in the same active Conda environment using its [official installation instructions](https://marmote.gitlabpages.inria.fr/marmote/instructions/linux/install_linux.html):

```sh
conda install -c marmote -c conda-forge marmote
```

madupite requires MPI/PETSc and a separate installation following its upstream instructions.

## Examples and documentation

| Start here | What you will find |
| :--- | :--- |
| [Benchmark API](src/mdpforge/core/benchmark.py) · [Results API](src/mdpforge/core/benchmark_result.py) · [Scaling API](src/mdpforge/core/scaling_benchmark.py) | Inspect public interfaces and options. |
| [Cache and reproducibility](docs/adding-a-model.md#3-reproducibility-and-caching) | Understand model identities, cache modes and reproducibility. |

## Contributing

Extend the catalogue by [adding a model](docs/adding-a-model.md) or [adding a solver](docs/adding-a-solver.md). See the [repository guidelines](AGENTS.md) for the scientific contract and the [open questions](docs/roadmap.md) for planned work.

Experimental: APIs and the catalogue may change. Released under the [MIT license](LICENSE).
