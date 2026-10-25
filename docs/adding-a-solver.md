# Adding a solver

Catalogue integration requirements for mdpforge; see also [AGENTS.md](../AGENTS.md).

## 1. Interface

Create `src/mdpforge/solvers/<name>.py` using lowercase letters and underscores, without a leading underscore, and expose `Solver`. Discovery is automatic: `<name>` is the identifier used by `list_solvers()` and `bench.add_solver("<name>")`.

- API: `Solver(model, discount, ...)` with explicit options, `assert 0 < discount < 1`, then argument-free `run()`. Iterative solvers accept a finite, positive `final_precision`; **omit** it when unused (`mdpforge_pi`, `mdptoolbox_pi`, `gurobi_primal`, `gurobi_dual`). The benchmark passes it only when accepted.
- Available data: `state_dim=S`, `action_dim=A`, `transition_matrix[a]` as SciPy CSR `(S, S)`, and `reward_matrix` as NumPy `(S, A)`. No additional model methods are required. **Preserve inputs**, including their representation; keep backend conversions local and avoid unnecessary dense matrices.
- `run()` returns a [`SolveResult`](../src/mdpforge/core/solve_result.py): finite NumPy `value` `(S,)`, `policy` as `None` or integer actions `(S,)` in `[0, A)`, `iterations` as an integer or `None`, and `stop_reason` as text or `None`. Expose a readable solver `name`; callers consume results through `SolveResult`.

**Prefer [`GenericSolver`](../src/mdpforge/core/solver.py):** call its constructor, set `self.name`, and implement `_solve()` to return `SolveResult` explicitly. `run()` forwards that result. A custom `run()` must return the same result type. Example: [`mdpforge_vi.py`](../src/mdpforge/solvers/mdpforge_vi.py).

Timing belongs to the benchmark, which measures construction and `run()`, including conversions. `SolveResult` has no `runtime` field; durations remain in `BenchmarkResult` and benchmark CSV exports.

## 2. Catalogue

Declare a **literal** `METADATA` dictionary, readable without importing the backend, with four nonempty strings:

- `display_name` (label) and `description` (one sentence).
- `family`: `vi`, `pi`, `mpi`, `lp`, `experimental`.
- `backend`: `mdpforge`, `mdptoolbox`, `mdpsolver`, `marmote`, `madupite`, `mdpax`, `gurobi`, `cma`.

For VI, add `solver_type = "vi"` for discovery by VI-specific fixtures. Check `list_solvers()` and `bench.add_solver("<name>")`. Declare dependencies in `pyproject.toml`; optional backends use extras and must not block the NumPy/SciPy core or the catalogue.

## 3. Precision and behavior

- Document the algorithm/reference, adaptations, initialization, stopping rule and accepted option values. Random behavior must reproduce after resetting NumPy/Python seeds, including private RNGs.
- The benchmark currently assesses absolute value precision using `||T(V) - V||∞ / (1 - discount)`, with the **full optimal Bellman operator**. Iteration differences or span alone do not establish this bound. The output-precision check and its replacement test are tracked in the [roadmap](roadmap.md).
- Report failures explicitly and preserve algorithm-specific stopping rules. The benchmark records unsuccessful or imprecise runs separately from successful solves.
- Use **public backend APIs** without bypassing initialization, checks or bounds; propagate native errors. Preserve reward scales, discount conventions, numerical behavior, existing solver algorithms and the VI reference cache.

**Specific options:** positive integer budgets are `max_iterations`, `max_evaluation_iterations`, `max_projected_iterations` and `max_fallback_iterations`; batch update counts are `bellman_iterations` and `aggregation_iterations`. Respect existing signatures and profiles. Current names are `initial_value`, `initial_policy`, `seed` (CMA-ES) and `refinement`, with **no historical aliases**. Constructors do not expose `verbose`; benchmark output is controlled by `Benchmark.run(verbose=...)`.

**Existing backend behavior:** `mdptoolbox_vi`/`mdptoolbox_vigs` pass `max_iterations` as `max_iter` without overriding native limits; `mdptoolbox_vi` calls `ValueIteration`. MDPSolver budgets cover internal policy evaluation. MPIGS/MPISOR default to `max_evaluation_iterations=10`, and the relevant `*_init` profiles retain fixed zero initialization. Native mdpforge VI/MPI solvers handle convergence, nonfinite values, stagnation and iteration limits locally; failures raise `RuntimeError`, and convergence on the last allowed iteration is accepted.

## 4. Validation

[`tests/test_workflows.py`](../tests/test_workflows.py) currently checks solver interfaces and input preservation on the cases configured in [`tests/conftest.py`](../tests/conftest.py). The previous value/policy accuracy checks have been removed; the new shared value-function check is pending. See the [test guide](../tests/README.md).

Use Python 3.11+ in Conda `benchmark`. **Add tests and run pytest/Ruff only when explicitly requested.** When pytest is requested, run the full suite, including installed backends; skip only missing dependencies. Do not run heavy benchmarks automatically. Report changes, checks actually performed, results and limitations; commands are in [AGENTS.md](../AGENTS.md).
