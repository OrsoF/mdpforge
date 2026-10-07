# mdpforge

[![CI](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml)

A research framework for building, solving, and benchmarking finite Markov decision
processes with NumPy and SciPy. It includes tabular planning methods, adaptive state
aggregation, and discounted, total-reward, and average-reward model/solver variants.

## Quick start

Use Python 3.11 or newer and Git. The following commands use an isolated environment
and run a small discounted planning example; no commercial solver or reference
results are needed.

```sh
git clone https://github.com/OrsoF/mdpforge.git
cd mdpforge
python -m venv .venv
```

Activate it on Linux/macOS:

```sh
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then, from the repository root:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python main.py --model rooms --state 100 --solvers vi pdvi --discount 0.9 --repeat 2 --seed 0 --output-dir artifacts/tmp/demo_discounted
```

The example constructs a 100-state, four-action Rooms MDP and runs VI and PDVI
twice each. It writes `rooms_runs.csv` (individual runtimes), `rooms.csv` (runtime
summary), and `rooms.tex` under `artifacts/tmp/demo_discounted/`. These CLI tables
measure runtime; they do not report policy-quality metrics. Model construction may
also create a cache under `artifacts/saved_models/`.

The tests check numerical invariants and small end-to-end workflows, including both
benchmark entry points. Seeds are explicit; measured runtimes depend on the machine.
The install, test, and demo commands have been validated locally on Windows with
Python 3.11 and 3.12 in isolated environments. CI checks the installation, lint,
format, and tests on Ubuntu for both versions.

## Tests

Run `python -m pytest` from the repository root. The suite covers:

- Model validation: dense/sparse transitions, matrix dimensions, finite values,
  nonnegative probabilities, and stochastic row sums. Negative probabilities are
  rejected even when a row sums to one.
- Partitions: complete state coverage, normalized projection weights, stochastic
  aggregation, and refinement that preserves existing region boundaries.
- A shared model contract for Rooms, Forest, and RandomWalk, including each
  environment's actual dimensions and construction with `save=False`.
- Model-to-solver integration for VI and PDVI on those three environments, plus
  both benchmark CLIs. Solver checks cover finite values, Bellman residuals, and
  agreement with independently evaluated greedy policies.
- Core imports and a model-to-VI workflow with optional dependencies explicitly
  blocked in a fresh interpreter.
- Cached optimal values: computation on cache miss using the existing VI solver,
  dense/sparse model representations, discounted and finite total rewards, cache
  reuse, and separate entries for different discounts.

The tests isolate model caches and CLI outputs from existing research artifacts.
They use small deterministic environments and require only the `dev` installation.
`.gitignore` excludes `artifacts/tmp/`, generated model/value caches, Python/tool
caches, virtual environments, and local editor settings. Reference results and
curated figures under `artifacts/results/`, `artifacts/figures/`, and
`artifacts/exps/` remain versionable; keep experimental outputs in `artifacts/tmp/`.
The solver checks use discount 0.9: a Bellman residual bounded by `1e-4` bounds
the value error by `1e-3` through the discounted Bellman contraction.

## Development checks

After installing the `dev` extra, run these commands from the repository root:

```sh
ruff check .
ruff format --check .
pytest
```

To apply formatting, use `ruff format .`. Ruff's version is pinned in
`pyproject.toml` so developers use the same formatter. Lint starts with common
Python errors (`E4`, `E7`, `E9`, `F`) and import sorting (`I`). The two style rules
`E731` and `E741` are disabled to retain concise numerical callbacks and existing
coordinate/board symbols. Research notebooks under `studies/` and generated or
scratch material under `artifacts/` are excluded; all Python source packages,
entry points, and tests are checked.

Type annotations are adopted incrementally. No MyPy or strict typing gate is
required. For VS Code, install the Ruff extension and select it as the Python
formatter; this checkout does not include workspace editor settings.

For the existing Conda research environment, use:

```sh
conda run -n benchmark python -m pip install -e ".[dev]"
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
conda run -n benchmark python -m pytest
```

## Continuous integration

[CI](.github/workflows/ci.yml) runs on push and pull request with a small Ubuntu
matrix for Python 3.11 and 3.12: install `.[dev]`, then run `ruff check .`,
`ruff format --check .`, and `pytest` using `pyproject.toml`. The pip download
cache is keyed by `pyproject.toml`. The badge links to the workflow and reports
the latest push result on `main`.
External solver/RL extras and heavy research benchmarks need separate validation.

## Cached optimal values

For a built model, `model.optimal_value_function(discount)` returns its cached
value vector or computes it with the existing VI solver at precision `1e-3`.
This works with the core installation, preserves the model's dense/sparse format,
and stores values under `artifacts/saved_value_functions/` using the existing
`<discount>_<model.name>.pkl` key. A matching cache can be reused without rebuilding
the model. Rebuild or remove the corresponding cache after changing model
parameters; the key identifies the model by name, not by matrix contents.
For total reward (`discount=1`), value iteration must converge for the chosen model.

## Total-reward example

```sh
python main_total.py --model rooms --state 100 --solvers vi pdvi --repeat 2 --seed 0 --output-dir artifacts/tmp/demo_total
```

Total-reward solvers use `discount=1`. Discounted benchmarks require a discount in
`[0, 1)`. Both commands require at least two repeats. Some models adjust requested
state/action counts, so use the dimensions printed by the command. Defaults target
larger research benchmarks; keep the explicit small sizes above for a quick demo.

To inspect the available models, solvers, and parameters:

```sh
python main.py --help
python main_total.py --help
```

## Dependencies and optional workflows

`pyproject.toml` is the source of dependency declarations. Install the core with
`python -m pip install -e .`: its only runtime dependencies are NumPy and SciPy.
Choose extras for the workflows you need:

| Extra | Dependencies and purpose |
| --- | --- |
| `dev` | pytest and Ruff for tests and development checks |
| `mdptoolbox` | MDPtoolbox fork pinned to a Git commit for reference values and adapters; requires Git |
| `gurobi` | gurobipy for linear-programming adapters; solver use requires a suitable license |
| `mdpsolver` | mdpsolver for its external planning adapters |
| `deep` | Stable-Baselines3 and Gymnasium for external RL experiments and the Gymnasium conversion utility; includes PyTorch transitively |
| `maze` | mazelib for `models/maze_*` and their total-reward variants |
| `progress` | tqdm for the Ambulance and Impatience models |
| `plot` | Matplotlib for plotting |
| `notebooks` | plotting tools, IPython, pandas, seaborn, and openpyxl for research analysis and tables |

For example:

```sh
python -m pip install -e ".[dev]"
python -m pip install -e ".[deep]"
python -m pip install -e ".[mdptoolbox,plot]"
python -m pip install -e ".[gurobi]"
```

Plots comparing solver values to reference values need both `plot` and
`mdptoolbox`. The existing `reference` extra selects `mdptoolbox`; `legacy`
selects all historical research integrations and analysis tools, without `dev`.
For that broad environment, explicitly install `.[dev,legacy]`.

`python -m pip install -r requirements.txt`, run from the repository root, now
installs only the core. It no longer installs development or external solver/RL
dependencies. Existing users of the broad requirements file should use
`python -m pip install -e ".[dev,legacy]"` instead.

Marmote requires separate platform-specific installation and is not supplied by an
extra. Installing an extra supplies its dependencies; it does not validate every
historical workflow. Several legacy RL modules reference a `solvers_agg` package
that is absent from this checkout; installing `deep` does not restore it. Those
workflows and the absent `agg_*.py` scripts are outside the supported quick start.

## Repository layout

```text
core/            Model/solver interfaces, Bellman operators, partitions, validation
models/          Finite MDP environments, including models/total/
solvers/         Planning methods and optional external/RL integrations
utils/           Persistence, simulation, and numerical utilities
tests/           Deterministic unit and integration tests
studies/         Research notebooks
artifacts/       Reference results, caches, and temporary outputs
main.py          Discounted planning benchmark CLI
main_total.py    Total-reward planning benchmark CLI
pyproject.toml   Packaging and development-tool configuration
.github/workflows/ci.yml  Push/PR checks on Python 3.11 and 3.12
```

See [STRUCTURE.md](STRUCTURE.md) for module responsibilities,
[MODELS.md](MODELS.md) for the model inventory, and [AGENTS.md](AGENTS.md) for
development conventions. The project uses the [MIT license](LICENSE).
