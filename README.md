# mdpforge

[![CI](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/OrsoF/mdpforge/actions/workflows/ci.yml)

**Benchmark a new MDP with reference solvers, or a new solver with reference MDPs.**

mdpforge provides finite Markov decision process models, planning solvers, and
numerical validation with NumPy and SciPy. Experiments use the Python API.

| You bring | Use mdpforge to |
| --- | --- |
| A new MDP | Validate its matrices and study reference solvers across sizes and parameters. |
| A new solver | Compare solution quality and runtime on reference MDPs. |

The current starting scope is discounted planning with known dynamics: Rooms,
Schoolboy, and RiverSwim, with value iteration (VI) and modified policy iteration
(MPI). Total-reward model sources remain available, but the current integration
tests focus on discounted workflows. Total and average reward need separate
convergence assumptions and evaluation protocols.

## Quick start

Use Python 3.11 or newer. Core dependencies are NumPy and SciPy. From the
repository root, install the package in your Python environment:

```sh
python -m pip install -e ".[dev]"
```

For runtime-only installation, use `python -m pip install .` or a built wheel.
`requirements.txt` selects the core editable installation. Imports use the
`mdpforge` namespace.

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

Some model constructors adjust requested dimensions; inspect the resulting
`state_dim` and `action_dim`. MPI is available in
[personal_pim.py](src/mdpforge/solvers/personal_pim.py). Its policy evaluation budget
is controlled by `max_iter_eval`; the default budget is 10 iterations. Increase
that budget when testing value accuracy, and check the final residual.

## Check solution quality

Continuing the example, independently evaluate the greedy policy:

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
    eye(model.state_dim, format="csr") - discount * transition,
    reward.astype(float),
)
residual = np.linalg.norm(
    optimal_bellman_operator(model, solver.value, discount) - solver.value,
    ord=np.inf,
)
print("Value error bound:", residual / (1 - discount))
print("Value/policy discrepancy:", np.max(np.abs(solver.value - policy_value)))
```

For `0 <= discount < 1`, the Bellman residual divided by `1 - discount` bounds
the infinity-norm error relative to the optimal value. The policy discrepancy
checks consistency, not policy suboptimality. These formulas are specific to
discounted problems.

## Add a model or solver

Models inheriting from `GenericModel` implement `_build_model()` and use
`create_model()` to build or load their matrices:

| Field | Convention |
| --- | --- |
| `state_dim`, `action_dim` | Actual numbers of states and actions. |
| `transition_matrix[action]` | A `(state_dim, state_dim)` matrix with stochastic rows. |
| `reward_matrix[state, action]` | Expected immediate reward, shape `(state_dim, action_dim)`. |
| `name` | Cache identifier; distinguish parameters and instances. |

Transitions can be a dense `(action_dim, state_dim, state_dim)` array or a list
of SciPy sparse matrices. Start with a small instance and validate dimensions,
finite values, nonnegative probabilities, and row sums.

Solvers take explicit constructor options, then expose an option-free `run()`.
Useful outputs are `value`, `runtime`, and optionally `policy` or `q_value`.
Use [VI](src/mdpforge/solvers/personal_vi.py) and
[MPI](src/mdpforge/solvers/personal_pim.py) as references. Document stopping criteria
and timing boundaries; compare methods at verified solution quality. Add small
deterministic tests for new behavior. No registry or CLI is required.

## Reproducible experiments and caches

Create a fresh solver for each repetition. Separate instance seeds from solver
seeds, and record actual dimensions, parameters, reward criterion, accuracy,
run count, machine, dependency versions, and code revision. A solver's `runtime`
may exclude construction and conversion; VI times `run()` only.

Artifact paths are anchored to the working directory when the package is imported.
Keep experiments under `artifacts/tmp/` and preserve curated reference results.

- Model caches: `artifacts/saved_models/`. `create_model(save=False)` prevents a
  new write but can still load an existing matching cache.
- Value caches: `artifacts/saved_value_functions/`. On a miss,
  `model.optimal_value_function(discount)` uses VI at `final_precision=1e-3`,
  preserves dense/sparse representation, and uses `<discount>_<model.name>.pkl`.
  Total reward requires convergent value iteration.

Keys use model names, not matrix contents. Distinguish instance names or remove
stale caches after changing parameters or generation seeds.

## Development

Run checks after installing `.[dev]`:

```sh
conda run -n benchmark python -m pytest
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
```

Outside Conda, use the corresponding `python -m ...` commands. Tests cover matrix
validation, Bellman operators, partitions, caches, installed imports, optional
dependency isolation, and VI/MPI workflows on the three discounted models.
Ruff excludes `studies/` and `artifacts/`; no strict typing gate is configured.

[CI](.github/workflows/ci.yml) runs these checks on Ubuntu/Python 3.11 and 3.12.
Local results do not establish the status of a hosted run.

Optional dependencies are declared in [pyproject.toml](pyproject.toml). Use `maze`
for maze models, `plot` for Matplotlib, `notebooks` for research analysis, and
`mdptoolbox` for external reference-value utilities. Historical backend extras
remain available for future integrations; installing them does not restore
removed source modules.

## Repository

- `src/mdpforge/core/`: model contract, Bellman operators, partitions, validation.
- `src/mdpforge/models/`: finite MDP implementations.
- `src/mdpforge/solvers/`: VI and MPI.
- `src/mdpforge/utils/`: persistence, simulation, and numerical utilities.
- `tests/`: deterministic unit and integration tests.
- [studies/](studies/README.md): historical notebooks; some cells need removed
  modules or adaptations before they can run.

See [AGENTS.md](AGENTS.md) for concise engineering guidelines. Licensed under
[MIT](LICENSE).
