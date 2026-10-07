# Repository engineering guidelines

Improve software engineering maturity for a public GitHub release and technical recruiting demo while preserving scientific and algorithmic behavior. Prefer small, incremental, reviewable changes that improve reproducibility, maintainability, testing, and developer experience.

Write concise, research-grade, NumPy-centric Python. Avoid wrappers, compatibility shims, defensive boilerplate, and unrelated refactors. Preserve user changes and do not modify baseline solvers unless asked. Preserve public behavior, numerical conventions, stopping criteria, and random-number behavior unless the requested change requires otherwise. Do not rewrite working code or add abstractions solely for style.

## Before editing

1. Inspect the full repository structure and working-tree status, including untracked files.
2. Read the README, configuration files, core abstractions, models, solvers, utilities, and existing tests. Verify documentation against the actual checkout; do not assume referenced files, interfaces, or commands exist.
3. Identify module dependencies and callers before refactoring. Check dynamic imports, notebook workflows, optional integrations, and import cycles where relevant.
4. Run the existing test suite and available validation commands to establish a baseline, unless the user asks to skip them. For documentation-only work, verify referenced paths and affected commands. Record missing tooling, absent tests, and existing failures explicitly.

## Change workflow

- Identify the root cause first, then make the smallest robust change that fixes the requested issue.
- Keep changes within the requested scope and avoid breaking unrelated components.
- Add or update tests for meaningful behavior changes. Update documentation when installation, commands, user-facing behavior, or project structure changes.
- Use clear module boundaries and explicit responsibilities. Reduce unnecessary coupling without broad rewrites.
- Use inheritance only for a genuine shared contract; otherwise prefer composition. Align implementations with existing interfaces where useful without forcing every solver into a new hierarchy.
- Preserve model and solver conventions: `state_dim`, `action_dim`, `transition_matrix[action]`, `reward_matrix[state, action]`, model construction through `create_model()`, and explicit solver constructor options followed by `run()`.
- `GenericModel.optimal_value_function()` reuses `solvers.personal_vi.Solver` on a cache miss. Keep the existing precision, model representation, and `<discount>_<model.name>` cache key; account for stale caches when changing model parameters. Total reward requires convergent value iteration.

## Tests and validation

- Prefer deterministic, reasonably fast pytest tests of behavior and invariants over tests of implementation details or coverage targets.
- When adding test coverage, include unit tests for core components and a few integration tests for representative model-to-solver workflows. Use small models, explicit seeds, and justified numerical tolerances.
- Relevant invariants include matrix dimensions, finite values, transition stochasticity, partition projection/extension, Bellman residuals, and greedy-policy quality. Treat discounted, total, and average reward criteria separately.
- After changes, run relevant tests and configured lint/format checks, plus compilation or import checks where useful. Verify documented commands actually work.
- For documentation-only changes, check the edited text and affected commands; do not add artificial tests or run expensive benchmarks solely for coverage.
- Report failures and limitations explicitly. No collected tests is not a passing suite, and successful compilation does not establish runtime or numerical correctness. Do not claim verification for checks that were skipped or unavailable.

## Dependencies and tooling

- Prefer conventional Python structure and `pyproject.toml` for packaging and tool configuration when adding them.
- Separate core runtime dependencies from development and optional dependencies. Make clean-environment installation straightforward and verify it when changing packaging or dependencies.
- Declare dependencies in `pyproject.toml`; keep `requirements.txt` limited to the core installation. Use explicit extras for external backends and research tools, and verify core imports without optional integrations.
- The core requires NumPy/SciPy. Use `.[dev]` for pytest/Ruff; select `mdptoolbox`, `gurobi`, `mdpsolver`, `deep`, `maze`, `progress`, `plot`, or `notebooks` only for the relevant workflow. Keep the pinned MDPtoolbox fork and existing `reference`/`legacy` selections consistent with these groups.
- Optional integrations must not prevent core imports or use. Keep heavy or optional imports local to the functionality that needs them where practical; avoid compatibility layers.
- Prefer pytest and Ruff for new test, lint, and format tooling unless an existing alternative has a concrete justification. Introduce tooling incrementally; avoid repository-wide formatting churn during unrelated changes.
- Keep CI simple, fast, and reproducible. Add containers only when they solve a concrete reproducibility need.

## Python environment

The package supports Python 3.11 or newer; the core/dev workflow has been validated on 3.11 and 3.12. Keep `requires-python`, Ruff's target version, documented compatibility, and any CI matrix consistent. Revalidate supported versions when changing dependencies or Python syntax.

Run Python in the Conda environment. Install the development tools with:

```text
conda run -n benchmark python -m pip install -e ".[dev]"
```

Run the standard checks from the repository root:

```text
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
conda run -n benchmark python -m pytest
```

If `conda` is unavailable, use `C:\Users\orsof\anaconda3\condabin\conda.bat run -n benchmark ...`. Respect requests to skip tests or benchmarks. Put scratch scripts and outputs in `artifacts/tmp/`; do not overwrite reference results while tuning.

Keep `.gitignore` aligned with actual output paths: ignore scratch outputs, generated model/value caches, environments, tool caches, and local editor files. Keep reference results and curated artifacts versionable; avoid blanket rules for `artifacts/`, CSV, LaTeX, or pickle files. Verify ignore rules with `git check-ignore`, including paths that must remain versionable.

Use `conda run -n benchmark python ...` for other Python commands. For runtime-only installation, use `conda run -n benchmark python -m pip install -e .`; `requirements.txt` also selects only the core. Report missing tools rather than assuming checks work. Ruff excludes `artifacts/` and `studies/`; retain the configured incremental lint scope and numerical style exceptions unless the requested work requires changing them. No strict typing gate is configured.

## CI and release documentation

- `.github/workflows/ci.yml` runs the core/dev checks on Ubuntu with Python 3.11 and 3.12 for pushes and pull requests. The README badge follows push results on `main`; verify the run for the published commit before reporting it green.
- When adding or updating CI, use one workflow with installation of `.[dev]`, Ruff lint, Ruff format check, and pytest. Keep the Python 3.11/3.12 matrix small and share local configuration. Optional backends and heavy research benchmarks need separate justification.
- Verify that workflow files and editor settings actually exist before documenting them. Distinguish configured checks, successful local runs, and successful hosted CI runs; never infer a green GitHub status from local tests.
- Check local Markdown links and documented paths when editing the README. Do not advertise absent scripts or packages as runnable entry points.
- Before claiming a public clone works, verify the published checkout includes source packages, tests, packaging metadata, and any documented workflow. Untracked local files are not included by `git clone`.

## Aggregation benchmarks

The `agg_*.py` and `solvers_agg` conventions below apply when those components are present or explicitly requested. Verify their existence before using them; do not create them solely to satisfy these conventions. Existing planning benchmark entry points are `main.py` and `main_total.py`.

- Keep one independent top-level `agg_*.py` per environment, limited to `BenchmarkConfig`, explicit local hyperparameters, and `main()`.
- Put shared utilities in `solvers_agg`; use explicit solver registries, not glob discovery.
- Uniformize separate solvers without merging them or adding inheritance/compatibility layers unless asked.
- Keep environment and solver-specific parameters explicit.
- Judge policies using greedy-policy evaluation reward, distinct from training reward. Replace plots rather than adding extras unless asked.

## Legacy `solvers/aggregated_*.py`

- Keep `run(self)` option-free and constructor options explicit; default projected budgets to `projected_steps=100`.
- Keep transient state local and expose only useful final outputs (`value`, `q_value`, `policy`, `runtime`, contracted values).
- Prefer `self.partition.span(value)`, small `_bellman_steps(...)` methods, and `_finish(...)` where useful.
- Put shared projected-Bellman logic in `utils.projected_bellman`; keep discounted and total variants separate.

## Performance and tuning

- Inspect the model, solver, and configuration first; verify actual dimensions and spatial `state_shape`.
- For sparse policy matrices, use `core.operators.compute_transition_reward_policy`; never assemble CSR matrices row by row.
- Prefer batched sparse operations and compact Bellman operators. Detect action-independent rewards once when useful.
- Near discount one, consider uniform residual-shift acceleration in projected Bellman iterations, then verify the final Bellman residual.
- Never apply acceleration containing `1 / (1-discount)` to total-reward problems with `discount=1`.
- Benchmark multiple runs in `artifacts/tmp/` and validate values or policies, not runtime alone.
- For AggQL comparisons, include tabular QL. Diagnose QL through coverage, exploration, tie-breaking, episode budget, and greedy rather than training reward.
- Report changed files, experiments, seeds/runs, episode budget when applicable, and unrelated test-suite failures.

## Completion report

Give a concise summary of changed files, important architectural decisions, tests added or modified, commands executed and their outcomes, remaining issues or risks, and anything deliberately left unchanged to avoid unnecessary work or over-engineering. For experiments, include dimensions, solver settings, seeds, runs, and episode budgets when applicable. Distinguish pre-existing failures from regressions.
