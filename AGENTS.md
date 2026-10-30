# Repository guidelines

mdpforge benchmarks new finite MDPs and new MDP solvers. Prefer small, useful changes that preserve scientific behavior and keep the project easy to extend.

## Keep work focused

- Inspect the working-tree status, affected files, and relevant callers. Read the whole repository or run a baseline only when needed to understand the change.
- Make the smallest robust change. Avoid unrelated refactors, wrappers, compatibility layers, new hierarchies, and repository-wide formatting.
- Preserve user changes and deliberate deletions. Do not restore removed models or solvers, or modify baseline algorithms, unless the task calls for it.
- Do not create or modify documentation unless the user explicitly asks for it. When requested, write concisely and compactly, and verify paths and interfaces before documenting them.
- In all Markdown (`.md`) files, keep each prose paragraph on a single source line, including paragraphs within list items; do not insert manual line wrapping. Preserve structural line breaks in lists, tables, code blocks, and other Markdown constructs.
- Keep function and method docstrings very short: one sentence by default. For public options, list each keyword, its accepted values and a brief meaning. Include only essential constraints; put longer explanations in dedicated docs.
- Keep individual catalogue model modules directly in `src/mdpforge/models/`, preserving their public names. Put JANI readers and shared constructors in `_jani/`, PRISM utilities in `_prism/`, and each collection's sources, manifests and licenses in its own `_data/` (`_jani/_data/`, `_qvbs/_data/`, `_prism/_data/`). QVBS models reuse the JANI utilities; general helpers belong in `src/mdpforge/utils/`. Keep utility code and source data out of the `models/` root, and update imports, resource paths and packaging when moving them.

## Preserve the scientific contract

- **Absolute golden rule: never modify solver implementations or their internal functions, including backend internals.** Limit changes to constructor interfaces, parameters passed to public backend APIs, metadata, documentation and tests. If a request requires changing computations, loops or internal stopping criteria, flag the conflict before making any changes.
- When adding a catalogue model, follow [docs/adding-a-model.md](docs/adding-a-model.md). Declare size presets and one task category, justify unavailable sizes, and add a small test instance with scientific validation only when tests are explicitly requested. Preserve the validation execution rules below.
- Keep numerical conventions, stopping criteria, reward scales, and RNG behavior unless the requested change requires otherwise.
- Models expose `state_dim`, `action_dim`, `transition_matrix[action]`, and `reward_matrix[state, action]`; construct them with `create_model()`. Solvers take explicit constructor options followed by an option-free `run()`. Transitions are a list of SciPy `csr_matrix` objects; rewards and values are NumPy arrays. Finalize CSR in `create_model()`; keep backend conversions local.
- Support only discounted criteria (`0 < discount < 1`, asserted by solvers). VI precision bounds absolute value error using the full Bellman residual. Check solution quality, not just runtime.
- Preserve the VI-based optimal-value cache, its precision and representation, and the `<discount>_<model.name>` key. Account for stale caches when changing model parameters.

## Validation

Do not add new tests unless the user explicitly requests them. Do not run tests or Ruff automatically; run them only when explicitly requested. When pytest is requested, run the full suite without a separate backend mode. Tests run all installed backends automatically; skip only missing dependencies. When adding tests is requested, cover meaningful behavior changes using small deterministic instances, numerical invariants, and value/policy quality. Run requested checks once after the change stabilizes; repeat only after a failure or further edits. Do not run heavy benchmarks unless needed for the task. Respect requests to skip checks.

## Checkpush

When the user says `checkpush`, run Ruff check and format check, then the full pytest suite. If they pass, run `git add -A`, create a commit, push the current branch, and check the hosted CI result. `git add -A` intentionally stages every change, including deletions and untracked files. Stop before committing or pushing if a check fails, and report the failure.

## Environment and reporting

Use Python 3.11+ in Conda: `conda run -n benchmark python ...`. If needed, use `C:\Users\orsof\anaconda3\condabin\conda.bat` instead of `conda`. Install development tools only when needed with `python -m pip install -e ".[dev]"` through Conda. Full checks from the repository root are:

```text
conda run -n benchmark python -m pytest
conda run -n benchmark python -m ruff check .
conda run -n benchmark python -m ruff format --check .
```

Declare dependencies in `pyproject.toml`; keep NumPy/SciPy core imports independent of optional integrations. Put scratch scripts and outputs in `artifacts/tmp/`; preserve reference results and existing tool configuration.

Finish with a short report: changes, checks and outcomes, remaining limitations. Distinguish existing failures from regressions and local checks from hosted CI. Do not claim checks that were skipped, unavailable, or collected no tests.
