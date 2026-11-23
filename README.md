<h1 align="center">MDPForge</h1>

<p align="center"><strong>Finite Markov decision processes and reproducible solver benchmarks in Python.</strong></p>

<p align="center">
  <a href="https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OrsoF/mdpforge/ci.yml?branch=main&amp;label=Tests" alt="Tests"></a>
  <a href="https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/OrsoF/mdpforge/ci.yml?branch=main&amp;label=Code%20Quality" alt="Code Quality"></a>
  <a href="https://github.com/OrsoF/mdpforge/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python_versions-3.11%2B-blue" alt="Python versions: 3.11 or later"></a>
  <a href="https://github.com/OrsoF/mdpforge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-008c87" alt="MIT license"></a>
</p>

MDPForge lets you build finite Markov decision processes (MDPs) from a model catalogue or your own transition and reward matrices. Compare built-in solvers and optional backends, then save the trial results and settings needed to repeat an experiment.

<p align="center">
  <a href="#quick-start">Quick start</a> · <a href="#comparing-solution-quality">Solution quality</a> · <a href="#visualizations">Visualizations</a> · <a href="#models-and-solvers">Models &amp; solvers</a> · <a href="#examples-and-documentation">Documentation</a> · <a href="#citation">Citation</a>
</p>

## Quick start

With **Python 3.11+**, install MDPForge from GitHub:

```sh
python -m pip install "git+https://github.com/OrsoF/mdpforge.git"
```

Compare value iteration (VI) and modified policy iteration (MPI) on a small Rooms instance:

```python
from mdpforge import Benchmark

bench = Benchmark()
bench.add_mdp("rooms", size="tiny")
bench.add_solver("mdpforge_vi")
bench.add_solver("mdpforge_mpi")

result = bench.run(discount=0.9, precision=1e-3, repeats=3, seed=0, validate=True)
result.export_csv("results.csv")
result.plot_performance_profile()
```

The export writes trial results to CSV and experiment settings to a companion JSON manifest. Keep both files together to reopen the results with `BenchmarkResult.from_csv()`.

## Comparing solution quality

`validate=True` checks solution quality after solver timing and records the outcome in `validation_status`, separately from execution status. Discounted checks can bound value error, and the total-reward check covers absorbing models with no cycles among nonterminal states. Average-reward checks report a residual diagnostic. See the [validation guide](https://github.com/OrsoF/mdpforge/blob/main/docs/solvers.md#precision-stopping-and-validation).

## Visualizations

![Performance profiles showing the share of models solved within a runtime factor of the fastest eligible solver, grouped by backend.](https://raw.githubusercontent.com/OrsoF/mdpforge/main/artifacts/figures/gallery_performance_profiles.png)

The November 2026 campaign compared 17 solvers on 102 catalogue recipes. The figure uses only the first repeat and counts runs that passed validation. See the [campaign manifest](https://github.com/OrsoF/mdpforge/blob/main/artifacts/results/backend_profiles_small_20261107/campaign.json) and the [data and plotting instructions](https://github.com/OrsoF/mdpforge/tree/main/artifacts/results/backend_profiles_small_20261107).

## Models and solvers

Use `list_models()` and `list_solvers()` to browse the discounted catalogues. `Benchmark()` uses discounted reward by default. Use `criterion="total"` for finite total reward or `criterion="average"` for average reward.

- [Model catalogue](https://github.com/OrsoF/mdpforge/blob/main/docs/models.md) for model families, sizes and sources.
- [Solver catalogue](https://github.com/OrsoF/mdpforge/blob/main/docs/solvers.md) for methods, backends and supported reward criteria.

## Examples and documentation

From a checkout, install notebook dependencies with `python -m pip install -e ".[notebooks]"`, then start `python -m jupyterlab`.

- [Compare VI, PI and MPI](https://github.com/OrsoF/mdpforge/blob/main/notebooks/compare_vi_pi_mpi.ipynb) on three small models and reopen the saved results.
- [Benchmark external solver libraries](https://github.com/OrsoF/mdpforge/blob/main/notebooks/benchmark_industrial_solvers.ipynb) or [measure how runtime changes with model size](https://github.com/OrsoF/mdpforge/blob/main/notebooks/benchmark_scaling.ipynb).
- [Benchmark API](https://github.com/OrsoF/mdpforge/blob/main/src/mdpforge/core/benchmark.py) · [Results API](https://github.com/OrsoF/mdpforge/blob/main/src/mdpforge/core/benchmark_result.py) · [Scaling API](https://github.com/OrsoF/mdpforge/blob/main/src/mdpforge/core/scaling_benchmark.py)

## Citation

If you use MDPForge in your research, please cite the software. The BibTeX entry is also available in [CITATION.bib](CITATION.bib).

```bibtex
@software{Forghieri_MDPForge_2026,
  author = {Forghieri, Orso},
  title = {{MDPForge: Finite Markov Decision Processes and Reproducible Solver Benchmarks}},
  year = {2026},
  version = {0.2.1},
  url = {https://github.com/OrsoF/mdpforge},
  note = {Python software}
}
```

## Contributing

See the guides for [adding a model](https://github.com/OrsoF/mdpforge/blob/main/docs/adding-a-model.md) or [adding a solver](https://github.com/OrsoF/mdpforge/blob/main/docs/adding-a-solver.md).
