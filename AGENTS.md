# Repository guidelines

mdpforge benchmarks new finite MDPs and new MDP solvers. Prefer small, useful
changes that preserve scientific behavior and keep the project easy to extend.

## Keep work focused

- Inspect the working-tree status, affected files, and relevant callers. Read the
  whole repository or run a baseline only when needed to understand the change.
- Make the smallest robust change. Avoid unrelated refactors, wrappers,
  compatibility layers, new hierarchies, and repository-wide formatting.
- Preserve user changes and deliberate deletions. Do not restore removed models
  or solvers, or modify baseline algorithms, unless the task calls for it.
- Verify paths and interfaces before documenting them. Update only documentation
  affected by the change.

## Preserve the scientific contract

- Keep numerical conventions, stopping criteria, reward scales, and RNG behavior
  unless the requested change requires otherwise.
- Models expose `state_dim`, `action_dim`, `transition_matrix[action]`, and
  `reward_matrix[state, action]`; construct them with `create_model()`.
  Solvers take explicit constructor options followed by an option-free `run()`.
  Transitions are a list of SciPy `csr_matrix` objects; rewards and values are
  NumPy arrays. Finalize CSR in `create_model()`; keep backend conversions local.
- Support only discounted criteria (`0 < discount < 1`, asserted by solvers).
  VI precision bounds absolute value error using the full Bellman residual.
  Check solution quality, not just runtime.
- Preserve the VI-based optimal-value cache, its precision and representation,
  and the `<discount>_<model.name>` key. Account for stale caches when changing
  model parameters.

## Scale validation to the change

| Change | Checks |
| --- | --- |
| Documentation | Edited text, affected paths, links, and changed commands. No test suite. |
| Small code change | Relevant tests and Ruff on changed Python files. |
| Model or solver | Small deterministic instances, explicit seeds, numerical invariants, and value/policy quality. |
| Migration, packaging, dependencies, or shared core | Full tests and Ruff; installation/import checks when relevant. |

Add tests for meaningful behavior changes. Run final checks once after the change
stabilizes; repeat only after a failure or further edits. Do not run heavy
benchmarks unless needed for the task. Respect requests to skip checks.

## Environment and reporting

Use Python 3.11+ in Conda: `conda run -n benchmark python ...`. If needed, use
`C:\Users\orsof\anaconda3\condabin\conda.bat` instead of `conda`.
Install development tools only when needed with `python -m pip install -e ".[dev]"`
through Conda. Full checks from the repository root are:

```text
conda run -n benchmark python -m pytest
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
```

Declare dependencies in `pyproject.toml`; keep NumPy/SciPy core imports independent
of optional integrations. Put scratch scripts and outputs in `artifacts/tmp/`;
preserve reference results and existing tool configuration.

Finish with a short report: changes, checks and outcomes, remaining limitations.
Distinguish existing failures from regressions and local checks from hosted CI.
Do not claim checks that were skipped, unavailable, or collected no tests.
