# mdpforge

Benchmark finite discounted MDPs and compare solvers at a shared target accuracy. Use the catalogue or bring your own models and solvers.

[![CI](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

Python 3.11+:

```sh
python -m pip install "mdpforge @ git+https://github.com/OrsoF/mdpforge.git"
```

Use `mdpforge[notebooks]` in the same command for tables and plots. Optional backends include `mdptoolbox`, `mdpsolver`, `marmote` and `gurobi`; see [all extras](pyproject.toml).

## Run a benchmark

```python
from mdpforge import Benchmark

bench = Benchmark(default_solvers=False)
bench.add_mdp("rooms")
bench.add_solver("mdpforge_vi")
bench.add_solver("mdpforge_mpi")

result = bench.run(discount=0.9, precision=1e-3, repeats=3, seed=0)
result.export_csv("results.csv")
```

Each trial records runtime, status and an independently checked value-error bound. Only finite discounted problems (`0 < discount < 1`) are supported. The export includes an experiment manifest with settings, seeds, matrix fingerprints, versions and machine information.

With the `notebooks` extra installed:

```python
result.solver_summary(reference="mdpforge_vi")
result.plot_performance_profile()
result.plot_heat(reference="mdpforge_vi")
```

Comparisons require every repeat of an MDP/solver pair to succeed. Failed or missing pairs count as unsolved; paired speedups use models completed by both solvers. Set the reference explicitly when comparing runs.

## Use your own models and solvers

Register transition matrices, one per action, and expected rewards of shape `(states, actions)`. Dense and sparse inputs are accepted:

```python
bench.add_mdp(
    "my_mdp",
    transitions=[[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
    reward=[[1, 0], [0, 2]],
)
```

Register a callable with `bench.add_solver("my_solver", solve_function=my_solver)`. It takes `(transitions, rewards, discount, precision)` and returns a NumPy value vector. Inputs are SciPy CSR transitions and NumPy rewards; the benchmark handles copies, timing and accuracy checks.

Browse the catalogue with `list_models()` and `list_solvers()` from `mdpforge`. Select models by `size` and `category`, and solvers by `family`, `backend` and `update_scheme`. Use `ScalingBenchmark` for size sweeps over seeded model families.

## Examples and documentation

- [Backend comparison](notebooks/benchmark_marmote.ipynb)
- [Results and plots](notebooks/visualize_results.ipynb)
- [Discount and precision sensitivity](notebooks/solver_sensitivity.ipynb)
- [Contribute a model](docs/adding-a-model.md)
- [Experimental CMA-ES policy evaluation](docs/cmaes-policy-evaluation.md)
- [Benchmark API](src/mdpforge/core/benchmark.py) · [Results API](src/mdpforge/core/result.py) · [Scaling API](src/mdpforge/core/scaling_benchmark.py)

Experimental: APIs and the catalogue may change. External backends require their optional dependencies; MDPSolver requires version 0.10.2 or later.

### Optional high-performance backends

Adapters for **madupite VI, PI, OPI and iPI (GMRES, BiCGStab, TFQMR)** and **MDPax VI/PI** are available via `list_solvers(backend=["madupite", "mdpax"])`. For installation, usage, conversion costs and accuracy limits, see [docs/madupite-mdpax.md](docs/madupite-mdpax.md).

## Cache et reproductibilité

Le cache v2 utilise des clés par recette pour les modèles et par contenu numérique pour les références VI ; voir [la documentation du cache](docs/cache.md). Les modèles aléatoires non explicitement reproductibles ne sont pas mis en cache automatiquement.
