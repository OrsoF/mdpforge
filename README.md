# mdpforge

[![CI](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml)

**Benchmark a new MDP with reference solvers, or a new solver with reference MDPs.**

mdpforge is a Python benchmark platform for finite Markov decision processes with
explicit transition and reward matrices. It brings together NumPy/SciPy model
representations, planning algorithms, numerical validation, and experiment scripts.

| You bring | Use mdpforge to |
| --- | --- |
| A new MDP | Validate its matrices, solve it with reference methods, and study how performance changes with problem size and parameters. |
| A new MDP solver | Run it on benchmark environments, check solution quality, and compare runtime with reference methods. |

The starting point is **discounted planning with known dynamics**. Total and
average reward require separate convergence assumptions and evaluation protocols.
A fast runtime is useful only alongside a verified solution.

## Quick start

The package requires Python 3.11 or newer; the development workflow targets Python
3.11 and 3.12. Core dependencies are NumPy and SciPy. From a checkout containing
`src/mdpforge/` and `pyproject.toml`, create an environment:

```sh
python -m venv .venv
```

Activate it on Linux/macOS with `source .venv/bin/activate`, or in Windows
PowerShell with `.venv\Scripts\Activate.ps1`. Then install from the repository root:

```sh
python -m pip install -e ".[dev]"
```

Run a small discounted benchmark:

```sh
python main.py --model rooms --state 100 --solvers vi mpi --discount 0.9 --repeat 2 --seed 0 --output-dir artifacts/tmp/demo_discounted
```

This constructs a 100-state, four-action Rooms MDP and runs value iteration (VI)
and modified policy iteration (MPI) twice each. Outputs are:

- `rooms_runs.csv`: individual runtimes and region counts when available.
- `rooms.csv`: mean runtime, standard deviation, and mean region count.
- `rooms.tex`: a LaTeX runtime table.

The CLI currently exports runtime statistics. The Python operators below provide
solution-quality checks; these metrics are not yet included in the CSVs.
Use explicit small sizes for a first run: CLI defaults target larger experiments.
Some models adjust requested dimensions; use the actual dimensions printed.

For runtime-only installation, use `python -m pip install .` or a built wheel.
`python -m pip install -r requirements.txt` selects the core editable installation.
The library is installed under `mdpforge`; `main.py` and `main_total.py` are
repository scripts and are not included in the wheel. Former imports starting
with `core`, `models`, `solvers`, or `utils` now start with `mdpforge.`.

## Solve and check a reference MDP

```python
from mdpforge.core.validation import validate_model
from mdpforge.models.rooms import Model
from mdpforge.solvers.personal_vi import Solver

discount = 0.9
model = Model(100, 4)
model.create_model(save=False)
validate_model(model)

solver = Solver(model, discount=discount, final_precision=1e-4)
solver.run()
print("Dimensions:", model.state_dim, model.action_dim)
print("Runtime (s):", solver.runtime)
print("Bellman residual:", solver.bellman_residual())
```

For a discounted problem, independently evaluate the policy greedy with respect
to the returned value. Continuing the example above:

```python
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)

policy = bellman_operator(model, solver.value, discount).argmax(axis=1)
transition, reward = compute_transition_reward_policy(model, policy)
policy_value = spsolve(
    eye(model.state_dim, format="csr") - discount * transition, reward
)
residual = np.linalg.norm(
    optimal_bellman_operator(model, solver.value, discount) - solver.value,
    ord=np.inf,
)
discrepancy = np.max(np.abs(solver.value - policy_value))
print("Value error bound:", residual / (1 - discount))
print("Value/policy evaluation discrepancy:", discrepancy)
```

For `0 <= discount < 1`, the Bellman residual divided by `1 - discount` bounds
the infinity-norm error of the returned value relative to the optimal value.
For example, a residual of `1e-4` at discount `0.9` bounds that error by `1e-3`.
The policy evaluation discrepancy checks consistency with the greedy policy;
it is not itself a measure of policy suboptimality. These formulas do not provide
a total-reward or average-reward evaluation protocol.

## Bring a new MDP

Implement a model using the shared matrix contract:

| Field | Convention |
| --- | --- |
| `state_dim`, `action_dim` | Actual numbers of states and actions. |
| `transition_matrix[action]` | A `(state_dim, state_dim)` matrix whose rows are probability distributions over next states. |
| `reward_matrix[state, action]` | Expected immediate reward, in an array of shape `(state_dim, action_dim)`. |
| `name` | Model identifier used for caches; distinguish parameter settings and instances. |

Transitions can be a dense array of shape `(action_dim, state_dim, state_dim)`
or a list of SciPy sparse matrices. Models inheriting from `GenericModel` implement
`_build_model()` and use `create_model()` to build or load their matrices.

Here is a complete two-state, two-action example:

```python
import numpy as np

from mdpforge.core.model import GenericModel
from mdpforge.core.validation import validate_model
from mdpforge.solvers.personal_vi import Solver


class TwoStateModel(GenericModel):
    def _build_model(self):
        self.transition_matrix = np.array(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
        )
        self.reward_matrix = np.array([[0.0, 1.0], [0.0, 0.0]])


model = TwoStateModel(2, 2)
model.create_model(save=False)
validate_model(model)
solver = Solver(model, discount=0.9, final_precision=1e-4)
solver.run()
print(solver.value)
```

Start with a small instance. Check dimensions, finite values, nonnegative
probabilities, and stochastic row sums, then verify a reference solution before
increasing the problem size. Preserve reward scales and terminal-state conventions
when comparing solvers.

For inclusion in the discounted CLI, expose `Model(state_dim, action_dim)` in
`src/mdpforge/models/<name>.py` and add an explicit entry to `MODELS` in `main.py`.
The registry is maintained in source; it does not discover user plugins.
Document model parameters and provenance, and add small deterministic tests.

## Bring a new solver

Use explicit constructor options followed by an option-free `run()`:

- Accept a built model and the reward criterion's parameters, including
  `discount` for discounted planning.
- Expose a final `value` vector of shape `(state_dim,)` for value-based checks.
- Expose `policy` or `q_value` when available, and `runtime` in seconds.
- Document initialization, numerical tolerance, stopping criterion, and what
  the reported runtime includes.

Use [VI](src/mdpforge/solvers/personal_vi.py) and
[MPI](src/mdpforge/solvers/personal_pim.py) as reference implementations.
`GenericSolver` provides value diagnostics, but existing solvers also use the
contract without inheriting from it.

To integrate with the discounted CLI, expose a `Solver` class in
`src/mdpforge/solvers/<name>.py`, add an entry to `SOLVERS` in `main.py`, and pass
its constructor settings through `solver_options()` as needed. The current runner
records `runtime` and optional `partition.n_regions`; verify values and greedy
policies separately before interpreting the runtime comparison.

Compare methods at a stated solution quality: identical constructor tolerances
may have different meanings across algorithms. Add tests on small instances
before running scaling experiments.

## Benchmark protocol and reproducibility

The discounted runner requires `0 <= discount < 1` and at least two repeats.
It seeds model construction with `--seed`, constructs or loads one model, then
uses `seed + run_index` for each solver repetition. Repeats measure performance
on the same instance; they do not sample independently generated MDPs. Both
runners use NumPy's global random seed.

Keep experiments in separate output directories: a run overwrites the same
model's output files there. For a reproducible comparison, record:

- Actual dimensions, model parameters, instance seed, and dense/sparse format.
- Reward criterion, discount, solver settings, and initialization.
- Runtime alongside Bellman residuals and independently evaluated policy values.
- Run count, machine, Python/dependency versions, and code revision.

The current CSVs do not contain a complete experiment manifest or standardized
timing boundaries. They report each solver's own `runtime`. For VI, that timer
covers `run()` and excludes model construction and constructor-time conversion.
Check each method's timing boundary before comparing results.

### Caches

Artifact paths are anchored to the working directory when the package is imported.
Model caches live in `artifacts/saved_models/`. `create_model(save=False)` prevents
a new cache write but can still load an existing matching cache.

`model.optimal_value_function(discount)` loads a cached value vector or computes
one with VI at `final_precision=1e-3`, preserving the model's dense/sparse format.
Values are stored in `artifacts/saved_value_functions/` under
`<discount>_<model.name>.pkl`. This is a numerical reference at that precision.
For `discount=1`, the model must have convergent value iteration.

Cache keys use model names, not matrix contents. A cached model can bypass seeded
construction. Use distinct names or remove the corresponding stale cache when
changing parameters, generation seeds, or reward conventions. Keep scratch outputs
under `artifacts/tmp/`; reference results and curated artifacts remain versionable.

## Current scope and next steps

The examples above use Rooms and the VI/MPI sources present in this checkout.
Schoolboy and RiverSwim model sources are also present. The research catalogue
is described in [MODELS.md](MODELS.md), but several listed sources and CLI
registrations are currently absent, including PDVI and the total-reward solver
modules. `main_total.py` exists, but its registered solver sources must be available
before it can run a total-reward benchmark.

Inspect configured CLI choices with:

```sh
python main.py --help
python main_total.py --help
```

Help lists registry entries; it does not verify module availability. The test
suite also references absent modules and fixtures; these gaps must be resolved
before the full suite can pass.

The next benchmark milestones are integrated quality metrics, saved experiment
metadata, and broader validated model/solver coverage. Total and average reward
need separate protocols. Historical RL workflows are outside this starting scope;
the former `agg_*.py` entry points and `solvers_agg/` package are absent.

## Optional dependencies

Dependencies and tool configuration are declared in [pyproject.toml](pyproject.toml).
Select extras only for the relevant workflow:

| Extra | Dependencies/purpose |
| --- | --- |
| `dev` | pytest and pinned Ruff. |
| `mdptoolbox` | MDPtoolbox fork pinned to a Git commit; requires Git. |
| `gurobi` | gurobipy; solver use requires a suitable license. |
| `mdpsolver` | External mdpsolver backend. |
| `deep` | Stable-Baselines3 and Gymnasium; includes PyTorch transitively. |
| `maze` | mazelib for maze generation. |
| `progress` | tqdm for progress reporting. |
| `plot` | Matplotlib. |
| `notebooks` | Plotting, IPython, pandas, seaborn, and openpyxl. |

For example, `python -m pip install -e ".[dev,notebooks]"` installs development
and analysis dependencies. `reference` selects `mdptoolbox`; `legacy` selects all
historical integration and analysis extras, without `dev`. Marmote needs a separate
platform-specific installation. Extras install dependencies; they do not restore
missing source modules or establish that historical workflows run.

## Development and validation

After installing `.[dev]`, run from the repository root:

```sh
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

For the project's Conda environment:

```sh
conda run -n benchmark python -m pip install -e ".[dev]"
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
conda run -n benchmark python -m pytest
```

Tests cover model validation, Bellman operators, partitions, optional dependency
isolation, cached values, package imports, and model-to-solver/CLI workflows,
subject to the missing-source limitations above. Use deterministic small models,
explicit seeds, and justified numerical tolerances for new tests.

Ruff checks source, entry points, and tests; `studies/` and `artifacts/` are excluded.
Lint uses `E4`, `E7`, `E9`, `F`, and `I`, with `E731` and `E741` disabled to preserve
the numerical style. No strict typing gate is configured.

The [CI badge](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml) links to
hosted push results on `main`, independently of local checks. The workflow file
is currently absent from this checkout. Optional backends and heavy experiments
require their own validation.

## Repository map

```text
src/mdpforge/
    core/        Model contract, Bellman operators, partitions, validation
    models/      Finite MDP model implementations
    solvers/     Planning algorithm implementations
    utils/       Persistence, simulation, and numerical utilities
tests/           Unit and integration tests
studies/         Research notebooks
artifacts/       Reference results, caches, and scratch outputs
main.py          Discounted runtime benchmark CLI
main_total.py    Total-reward runner (solver sources currently missing)
pyproject.toml   Packaging, dependencies, and tool configuration
```

See [STRUCTURE.md](STRUCTURE.md) for the module map,
[MODELS.md](MODELS.md) for the research model catalogue,
[studies/README.md](studies/README.md) for notebooks, and
[AGENTS.md](AGENTS.md) for engineering conventions. Some catalogue/module-map
entries describe sources absent from this checkout, as noted above.

Licensed under [MIT](LICENSE).
