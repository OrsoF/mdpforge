# Adding an MDP model

Catalogue integration requirements for mdpforge; read alongside [AGENTS.md](../AGENTS.md). **MUST** means required; **SHOULD** means recommended, with any exception justified.

## 1. Definition and interface

Create `src/mdpforge/models/<name>.py` using lowercase letters and underscores, without a leading underscore. Discovery is automatic. Provide:

- `Model(MDP)` with `__init__(state_dim, ...)` and `_build_model()`; let `create_model()` handle construction, caching and CSR conversion.
- By the end of the constructor: `state_dim` and `action_dim` are the **actual dimensions**, as positive Python integers; `requested_state_dim` preserves the original target before adjustment; invalid parameters raise an explicit error. Document dimension adjustments and keep the constructor lightweight.
- `name = "<state_dim>_<action_dim>_<name>[_parameters]"`: include **every** option that changes the matrices (dynamics, rewards, seed, geometry), and version the name when their definition changes. Prefer `str(value)` for floats, without extra rounding.
- A literal `TEST_PARAMETERS` dictionary describing a small, fast instance.

After `create_model()`, the data must satisfy:

| Attribute | Type and meaning |
| --- | --- |
| `transition_matrix` | List of `A` SciPy **CSR** matrices `(S, S)`; `[a][s, t] = P(t \| s, a)` |
| `reward_matrix` | NumPy array `(S, A)`; `[s, a] = E[r \| s, a]` |

`S = state_dim`, `A = action_dim`. All values are finite; `P >= 0` and each row sums to 1 (current tolerance: `1e-6`). Compute the expected reward if it depends on the next state. Prefer sparse structures; do not convert the model for a specific backend.

**Required scientific documentation:** state encoding, valid/invalid actions, transitions, rewards (sign/scale), terminal/absorbing states, parameters (defaults/domains), references and adaptations. For an original model, use `reference=None` and state its purpose and assumptions. The MDP must be finite, discounted (`0 < discount < 1`), independent of solvers, and free of implicit reward normalization.

## 2. Metadata and sizes

Declare `METADATA` as a **literal dictionary** (read with `ast.literal_eval`, so no computations or variable references), containing:

- `category`: **one** of `abstract`, `navigation`, `physical_control`, `service_operations`, `asset_management`, `games_and_puzzles`; choose by the main task, rather than the setting or a secondary mechanism.
- `description`: the model's task and distinctive features.
- `reference`: a bibliographic source, or `None` for an original model.
- `sizes`: **exactly** the entries `small`, `medium`, `large`.

For each available size, use `{"requested_state_dim": <positive integer>, "parameters": {<other arguments>}}`; `parameters` may be `{}` and never contains `state_dim`. Actual dimensions come **only** from the constructor: do not repeat them in presets. For an unavailable size, use `{"requested_state_dim": None, "parameters": None, "reason": "<reason>"}`. Do not present one configuration as several sizes; adjustments must preserve the model's structure. Provide at least one small usable test instance, even if some presets are unavailable.

`list_models()` must read metadata without importing or constructing the model. Check `list_models(size="small", category="abstract")` (filters intersect) and `bench.add_mdp("<name>", size="small")`. Explicitly requesting an unavailable preset must fail, **without** silent fallback. Collective additions warn and skip missing optional dependencies; other errors propagate. Importing the package must not require files from `artifacts/`.

## 3. Reproducibility and caching

- Random generation: document the seed and reproduce the same matrices with the same parameters; prefer a local `np.random.default_rng(seed)` generator.
- Model caches use `model.name`; VI reference caches use `<discount>_<model.name>`. Preserve the reference precision of `1e-3` used by `model.optimal_value_function()`. Change the model name or use `cache="refresh"` when dynamics or rewards change.
- Isolate caches in tests (`isolated_model_cache`): `create_model(cache="readonly")` prevents writes, **not** reads from an existing cache. Use `cache="off"` to disable both.

## 4. Tests and calibration

**Validation requirements:**

1. Validate matrices with `mdpforge.core.validation.validate_model(model)` and verify catalogue discovery. Supply the small instance through `TEST_PARAMETERS`.
2. When tests are explicitly requested, cover at least one **scientific** property beyond array shapes (known transition/reward, terminal behavior, value or policy). Include relevant invalid parameters and size adjustments; for random models, check reproducibility with an isolated cache.
3. When using a solver for validation, check solution quality. The full optimal Bellman residual gives the bound `||T(V) - V||∞ / (1 - discount)` on absolute value error. A failure or timeout never counts as a successful solve time.

**Calibration:** indicative `small / medium / large` targets are approximately `1 / 5 / 30 s` per model/solver pair, with `discount=0.99`, absolute error `1e-3` and one thread. Measure solver preparation and execution; report model construction and memory separately. Keep CSVs and provenance: parameters, requested/actual dimensions, seeds, solvers/versions, precision, machine, repeats, timeouts and any extrapolations (method, assumptions, covered solvers). Sizes may remain unavailable with a reason.

From the repository root, in Conda `benchmark` (Python 3.11+):

```bash
# Check dimensions, construction and solving in sequence
conda run -n benchmark python check_model_sizes.py --dimensions-only --models <name>
conda run -n benchmark python check_model_sizes.py --build-only --sizes small medium large --models <name>
conda run -n benchmark python check_model_sizes.py --sizes large --models <name>

# Calibrate only the new model when requested
conda run -n benchmark python benchmark_sizes.py --models <name> --repeats 3 --output artifacts/results/<name>_sizes.csv
```

`benchmark_sizes.py` measures only `small` and `medium` using its own `STATE_DIM_REQUESTS`, rather than `METADATA` presets. `check_model_sizes.py` runs each case with a timeout, resumes existing results and returns a nonzero exit code for unavailable configurations, failures or budget overruns; change `--output` to force a fresh measurement. Put scratch scripts in `artifacts/tmp/` and reference results in `artifacts/results/`. Do not run heavy measurements automatically.

Declare dependencies in `pyproject.toml`; optional dependencies belong in extras and must not block core imports or catalogue access. **Run pytest/Ruff only when explicitly requested**; see [AGENTS.md](../AGENTS.md). Report checks actually performed, results and remaining limitations.

## Integration checklist

- [ ] Scientific contract, dimensions and matrices are valid.
- [ ] `METADATA`, three sizes (or reasons), `TEST_PARAMETERS` and discovery work.
- [ ] Seeds and configuration identities support reproducibility and safe caching.
- [ ] Scientific tests and numerical validation are planned/performed as requested.
- [ ] Calibration provenance is retained when measured; dependencies are declared.
