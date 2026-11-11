# Repository guidelines

mdpforge benchmarks new finite MDPs and new MDP solvers. Prefer small, useful changes that preserve scientific behavior and keep the project easy to extend.

## Keep work focused

- Treat requests for advice or proposals (for example, "How would you modify the README?") as discussion only: describe the suggested changes in chat without editing files or executing the proposed actions. Apply them only when the user asks you to do so.
- Inspect the working-tree status, affected files, and relevant callers. Read the whole repository or run a baseline only when needed to understand the change.
- Make the smallest robust change. Avoid unrelated refactors, wrappers, compatibility layers, new hierarchies, and repository-wide formatting.
- Do not create, move, or edit repository-root scripts as a side effect of model, solver, catalogue, or benchmark changes. Change a root script only when the user explicitly requests that script; keep scratch scripts in `artifacts/tmp/`. When deleting a script, remove its references in the same change.
- Organizational refactors move or rename code and update references without changing runtime behavior. Do not introduce new computations, validation, or benchmark protocol changes unless explicitly requested.
- Preserve user changes and deliberate deletions. Do not restore removed models or solvers, or modify baseline algorithms, unless the task calls for it.
- Do not create or modify documentation unless the user explicitly asks for it. When requested, write concisely and compactly, and verify paths and interfaces before documenting them.
- In all Markdown (`.md`) files, keep each prose paragraph on a single source line, including paragraphs within list items; do not insert manual line wrapping. Preserve structural line breaks in lists, tables, code blocks, and other Markdown constructs.
- Keep function and method docstrings very short: one sentence by default. For public options, list each keyword, its accepted values and a brief meaning. Include only essential constraints; put longer explanations in dedicated docs.
- Keep individual catalogue model modules directly in `src/mdpforge/models/`, preserving their public names. Put JANI readers and shared constructors in `_jani/`, PRISM utilities in `_prism/`, and each collection's sources, manifests and licenses in its own `_data/` (`_jani/_data/`, `_qvbs/_data/`, `_prism/_data/`). QVBS models reuse the JANI utilities; general helpers belong in `src/mdpforge/utils/`. Keep utility code and source data out of the `models/` root, and update imports, resource paths and packaging when moving them.
- Keep domain objects and benchmark orchestration in `core/`; put catalogue loading, trial execution, statistics and plotting helpers in `utils/`. Use descriptive module names without a leading underscore in `core/`, except `__init__.py`, and preserve the package's public imports when reorganizing files.

## Preserve the scientific contract

- **Absolute golden rule: never modify solver implementations or their internal functions, including backend internals.** Limit changes to constructor interfaces, parameters passed to public backend APIs, metadata, documentation and tests. If a request requires changing computations, loops or internal stopping criteria, flag the conflict before making any changes.
- When adding a catalogue model, follow [docs/adding-a-model.md](docs/adding-a-model.md). Declare size presets and one task category, justify unavailable sizes, and add a small test instance with scientific validation only when tests are explicitly requested. Preserve the validation execution rules below.
- Keep numerical conventions, stopping criteria, reward scales, and RNG behavior unless the requested change requires otherwise.
- Models expose `state_dim`, `action_dim`, `transition_matrix[action]`, and `reward_matrix[state, action]`; construct them with `create_model()`. Solvers take explicit constructor options followed by an option-free `run()`. Transitions are a list of SciPy `csr_matrix` objects; rewards and values are NumPy arrays. Finalize CSR in `create_model()`; keep backend conversions local.
- Support only discounted criteria (`0 < discount < 1`, asserted by solvers). VI precision bounds absolute value error using the full Bellman residual. Check solution quality in dedicated tests, explicit validation calls or the opt-in benchmark validation mode, always outside solver timing.
- Preserve the VI-based optimal-value cache, its precision and representation, and the `<discount>_<model.name>` key. Account for stale caches when changing model parameters.

## Benchmark execution

- Benchmark runs orchestrate, time and record solver execution without numerical validation by default. `Benchmark.run(validate=True)` and `ScalingBenchmark.run(validate=True)` may explicitly certify each successful result with `validate_result()` after solver timing, in the parent process and outside any trial timeout. Record only scalar diagnostics and release the solution before the next trial; do not add reference solves, repeated solver executions or validation inside timed trial helpers. Validation increases total experiment time and can affect subsequent cache and allocation behavior, so record the mode in experiment metadata and distinguish campaigns with and without validation.
- Runtime includes solver construction, backend conversions and `run()`. Keep model copying, seed setup, catalogue loading, experiment metadata, logging and exports outside solver timing. Preserve paired seeds and restore global RNG state.
- A benchmark status of `success` means execution completed without an exception; it does not certify accuracy. `validation_method` identifies the requested certificate, not a completed check. Record validation status, errors and runtime separately; certification failure must not overwrite solver execution status or runtime. Keep unavailable residuals and error bounds as `NaN`, including legacy CSV fields, rather than implying they were checked.
- Do not add implicit warm-up runs, extra repetitions, or cache clearing/disabling. Changes to JIT compilation, cache reuse, process isolation or timing boundaries require an explicit request; distinguish cold and warm measurements when interpreting results.

## Validation

Do not add new tests unless the user explicitly requests them. Do not run tests or Ruff automatically; run them only when explicitly requested. When pytest is requested, run the full suite without a separate backend mode. Tests run all installed backends automatically; skip only missing dependencies. When adding tests is requested, cover meaningful behavior changes using small deterministic instances, numerical invariants, and value/policy quality. Run requested checks once after the change stabilizes; repeat only after a failure or further edits. Do not run heavy benchmarks unless needed for the task. Respect requests to skip checks.

## Checkpush

When the user says `checkpush`, run Ruff check and format check, then the full pytest suite. If they pass, run `git add -A`, create a commit, push the current branch, and check the hosted CI result. `git add -A` intentionally stages every change, including deletions and untracked files. Stop before committing or pushing if a check fails, and report the failure.

## Environment and reporting

Use Python 3.11+ in Conda: `conda run -n benchmark python ...`. If needed, use `C:\Users\orsof\anaconda3\condabin\conda.bat` instead of `conda`. Install development tools only when needed with `python -m pip install -e . "pytest>=8" "ruff==0.16.10"` through Conda. Full checks from the repository root are:

```text
conda run -n benchmark python -m pytest
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
```

Declare dependencies in `pyproject.toml`; keep NumPy/SciPy core imports independent of optional integrations. Put scratch scripts and outputs in `artifacts/tmp/`; preserve reference results and existing tool configuration.

Finish with a short report: changes, checks and outcomes, remaining limitations. Distinguish existing failures from regressions and local checks from hosted CI. Do not claim checks that were skipped, unavailable, or collected no tests.
