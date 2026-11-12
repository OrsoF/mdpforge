# Adding a solver

Follow [AGENTS.md](../AGENTS.md); existing solver algorithms and backend internals must remain unchanged.

## 1. Interface

Create `src/mdpforge/solvers/<criterion>/<name>.py` and expose `Solver`. Use lowercase letters and underscores without a leading underscore; `<criterion>` is `discounted`, `total` or `average`. Discovery is automatic.

- Discounted constructors accept `(model, discount, ...)`, with `0 < discount < 1`; total/average constructors accept `(model, ...)`. Use explicit options and finite positive `final_precision` where needed; `run()` takes no options.
- Models supply `state_dim=S`, `action_dim=A`, SciPy CSR `transition_matrix[a]` `(S, S)` and NumPy `reward_matrix` `(S, A)`. Preserve inputs and their representation; keep backend conversions local.
- Return [`SolveResult`](../src/mdpforge/core/solve_result.py) with finite NumPy `value` `(S,)` and `policy=None` or integer actions `(S,)` in `[0, A)`. Expose a readable solver `name`.

For discounted solvers, prefer [`GenericSolver`](../src/mdpforge/core/solver.py): call its constructor, set `self.name` and implement `_solve()`. See [`mdpforge_vi.py`](../src/mdpforge/solvers/discounted/mdpforge_vi.py). Custom `run()` methods return the same result type.

The benchmark times construction, conversions and `run()`. Durations belong in benchmark results, not `SolveResult`.

## 2. Catalogue

Declare a literal `METADATA` dictionary readable without backend imports:

- `display_name` and `description`: nonempty label and one-sentence description.
- `family`: `vi`, `pi`, `mpi`, `lp` or `experimental`.
- `backend`: nonempty implementation-library name, such as `mdpforge`, `scipy` or `mdpsolver`.

For discounted VI, set `solver_type = "vi"`. Check `list_solvers()` in the selected catalogue and `Benchmark(criterion=...).add_solver("<name>")`. Declare dependencies in `pyproject.toml`; optional integrations must not block core imports or discovery.

## 3. Precision and behavior

- Document the reference, adaptations, initialization, stopping rule and option domains. Reproduce random behavior after resetting NumPy/Python seeds, including private RNGs.
- Use public backend APIs and propagate errors. Preserve reward scales, discount conventions, stopping rules and the VI reference cache.
- `Benchmark.run(validate=True)` checks results after timing. Execution and validation statuses are separate; `success` alone does not certify accuracy.
- Discounted residual validation bounds value error by `||T(V) - V||∞ / (1 - discount)`. Total certification covers absorbing acyclic models; average certification is unavailable. Declare the appropriate `validation_method` in `SolveResult`.

Keep option names consistent with existing constructors: `initial_value`, `initial_policy`, `refinement`, positive integer budgets `max_iterations`, `max_evaluation_iterations`, `max_projected_iterations`, `max_fallback_iterations`, and batch counts `bellman_iterations`/`aggregation_iterations`. Preserve backend defaults and registered profiles; avoid historical aliases. Constructors do not expose `verbose`; benchmark output uses `Benchmark.run(verbose=...)`.

## 4. Tests

See the [test guide](../tests/README.md) for shared interface and Rooms coverage. Add tests or run pytest/Ruff only when explicitly requested; follow [AGENTS.md](../AGENTS.md) for execution and reporting rules.
