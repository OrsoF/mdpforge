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

*Saved speedups at discount 0.99 and target value error 0.001; timings depend on hardware. See the [plotting notebook](notebooks/visualize_results.ipynb) for the method.*

## Quick start

### Install

Requires **Python 3.11+** and supports finite discounted problems with `0 < discount < 1`.

```sh
python -m pip install "mdpforge @ git+https://github.com/OrsoF/mdpforge.git"
```

For tables and plots, install the `notebooks` extra:

```sh
python -m pip install "mdpforge[notebooks] @ git+https://github.com/OrsoF/mdpforge.git"
```

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

Each trial records runtime and an independently checked value-error bound. `result.export_csv(path)` saves the results and their experiment settings; use `BenchmarkResult.from_csv(path)` to reopen them.

With the `notebooks` extra installed:

```python
result.solver_summary(reference="mdpforge_vi")
result.plot_performance_profile()
result.plot_heat(reference="mdpforge_vi")
```

Paired speedups use models solved by both solvers on every repeat. Set the reference explicitly when comparing runs.

## Visualizations

### How often is a solver close to the fastest?

![Performance profiles showing the share of models solved within a runtime factor of the fastest eligible solver, grouped by backend.](artifacts/figures/gallery_performance_profiles.png)

At 2×, a curve shows the share of models solved within twice the fastest eligible median runtime; failed or missing pairs remain in the denominator. See the [visualization notebook](notebooks/visualize_results.ipynb) and [historical export notes](docs/visualization-roadmap.md).

### How do discount and precision affect runtime?

![Runtime sensitivity on a fixed Garnet instance, varying discount at fixed precision and target value error at fixed discount.](artifacts/figures/readme_sensitivity.png)

The curves use one Garnet instance; bands show the range over three repeats. See the [sensitivity notebook](notebooks/solver_sensitivity.ipynb) for the study design.

## Models and solvers

Browse the catalogue with `list_models()` and `list_solvers()` from `mdpforge`. Select models by `size` and `category`, and solvers by `family` and `backend`. Use `ScalingBenchmark` for size sweeps over seeded model families.

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

Register a callable with `bench.add_solver("my_solver", solve_function=my_solver)`. It takes `(transitions, rewards, discount, precision)` and returns a NumPy value vector. Inputs are SciPy CSR transitions and NumPy rewards; the benchmark handles copies, timing and accuracy checks.

### Optional backends

Optional solver integrations and their dependencies are listed in [installation extras](pyproject.toml); see the [madupite and MDPax guide](docs/madupite-mdpax.md) for those backends.

## Examples and documentation

| Start here | What you will find |
| :--- | :--- |
| [Backend comparison](notebooks/benchmark_marmote.ipynb) | Set up and compare solver backends. |
| [Results and plots](notebooks/visualize_results.ipynb) | Explore speedups, performance profiles, outcomes and accuracy. |
| [Discount and precision sensitivity](notebooks/solver_sensitivity.ipynb) | Compare runtimes while varying discount and target accuracy. |
| [Benchmark API](src/mdpforge/core/benchmark.py) · [Results API](src/mdpforge/core/result.py) · [Scaling API](src/mdpforge/core/scaling_benchmark.py) | Inspect public interfaces and options. |
| [Cache and reproducibility](docs/cache.md) | Understand cache identities, validation and migration. |

## Contributing

Extend the catalogue by [adding a model](docs/adding-a-model.md) or [adding a solver](docs/adding-a-solver.md). See the [repository guidelines](AGENTS.md) for the scientific contract and the [open questions](docs/roadmap.md) for planned work.

Experimental: APIs and the catalogue may change. Released under the [MIT license](LICENSE).
