# Adding an MDP model

Follow [AGENTS.md](../AGENTS.md) and the catalogue requirements below.

## 1. Definition and interface

Create `src/mdpforge/models/discounted/<name>.py` or `src/mdpforge/models/total/<name>.py`. Use lowercase letters and underscores without a leading underscore; discovery is automatic. Provide:

- `Model(MDP)` with `__init__(state_dim, ...)` and `_build_model()`; `create_model()` handles construction, caching and CSR conversion.
- Positive Python integers `state_dim` and `action_dim` giving actual dimensions after construction; `requested_state_dim` retains the original target. Reject invalid arguments and document adjustments; keep constructors lightweight.
- `name = "<state_dim>_<action_dim>_<nameoneword>[_<param1>_<param2>_...]"` for discounted models; total models use `"<state_dim>_<action_dim>_<nameoneword>_total[_<param1>_<param2>_...]"`. Use actual state and action counts, one lowercase model name without underscores, then numeric parameter values only. Include every option affecting dynamics or rewards, such as seeds and geometry; use unrounded `str(value)` for floats. Encode categorical options with stable numeric identifiers; retain source/configuration fingerprints as decimal integers when needed for cache identity. Do not append reward descriptions or text versions such as `entry_reward` or `v1`; use a numeric version parameter when definitions change. Examples: `2282_5_treasure_1_18_0.15` and `2282_5_treasure_total_1_18_0.15`.
- Literal `TEST_PARAMETERS` for a small, fast instance.

After `create_model()`, the data must satisfy:

| Attribute | Type and meaning |
| --- | --- |
| `transition_matrix` | List of `A` SciPy **CSR** matrices `(S, S)`; `[a][s, t] = P(t \| s, a)` |
| `reward_matrix` | NumPy array `(S, A)`; `[s, a] = E[r \| s, a]` |

`S = state_dim`, `A = action_dim`. Values must be finite, transitions nonnegative and row sums within `1e-6` of 1. Average next-state-dependent rewards into expected rewards. Prefer sparse construction and keep backend conversions out of models.

Document state encoding, actions, transitions, reward sign/scale, terminal states, parameter defaults/domains, references and adaptations. Original models use `reference=None` and explain their purpose and assumptions. Total models declare `supported_criteria={"total"}` and `terminal_states` and require finite return; average benchmarks reuse discounted models.

For a goal-based total adaptation, preserve states, actions and non-goal transitions without adding a timeout. Make goals absorbing if needed, documenting the change. Reward is 1 for each nonterminal action with positive probability of entering a goal, otherwise 0, including at goals. Declare goals as `terminal_states` and use a distinct cache name.

## 2. Metadata and sizes

Declare literal `METADATA`, readable with `ast.literal_eval` without importing the model:

- `category`: one main task from `abstract`, `navigation`, `physical_control`, `service_operations`, `asset_management`, `games_and_puzzles`.
- `description`: task and distinctive features.
- `reference`: bibliographic source or `None`.
- `sizes`: exactly `small`, `medium`, `large`.

Available sizes use `{"requested_state_dim": <positive integer>, "parameters": {<other arguments>}}`; parameters may be empty and exclude `state_dim`. Actual dimensions come from the constructor. Unavailable sizes use `{"requested_state_dim": None, "parameters": None, "reason": "<reason>"}`. Keep sizes distinct and preserve model structure; always supply a small test instance.

Check the criterion catalogue's `list_models()` filters and `Benchmark(criterion=...).add_mdp("<name>", size="small")`. Unavailable explicit sizes must fail; groups warn and skip missing optional dependencies, while other errors propagate. Discovery and core imports must not load models, optional dependencies or `artifacts/` files. Declare dependencies in `pyproject.toml`, using extras for optional integrations.

## 3. Reproducibility and caching

- Document seeds and reproduce matrices for identical parameters; prefer local `np.random.default_rng(seed)`.
- Model caches use `model.name`; VI caches use `<discount>_<model.name>`. Preserve the `1e-3` reference precision of `optimal_value_function()`. Rename changed models or refresh stale caches; `cache="refresh"` requires a fresh instance.
- `create_model(cache="readonly")` allows reads but prevents writes; `cache="off"` disables both. Isolate test caches with `isolated_model_cache`.

## 4. Tests and calibration

Use `mdpforge.core.validation.validate_model(model)` for matrix checks. When tests are requested, cover a scientific property beyond shapes, relevant invalid arguments, dimension adjustments and seeded reproducibility. See the [test guide](../tests/README.md).

Solver-based checks must assess solution quality: discounted residual bounds use the full Bellman operator; total certification covers absorbing acyclic models. Failures and timeouts are not successful solve times.

Discounted size targets are indicative: approximately `1 / 5 / 30 s` for `small / medium / large`, at `discount=0.99`, absolute error `1e-3` and one thread. Time solver preparation and execution; report model construction and memory separately. Retain CSVs and manifests with parameters, requested/actual dimensions, seeds, solver versions/settings, machine, repeats, timeouts, validation mode and any extrapolation assumptions.

Keep scratch work in `artifacts/tmp/` and reference results in `artifacts/results/`. Add tests or run pytest/Ruff only when explicitly requested; do not run heavy measurements automatically.
