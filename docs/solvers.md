# Solver catalogue

This page lists the public solver entries in MDPForge: 50 discounted, 19 total-reward and eight average-reward entries. Names are the identifiers accepted by `Benchmark.add_solver()` in the corresponding reward catalogue. Entries include implementations, adapters and named parameter profiles; their count is not a count of independent algorithms or a performance ranking. Catalogue membership does not establish that an optional library is installed.

[Sources](#sources-and-integrations) · [Discounted solvers](#discounted-solvers) · [Total-reward solvers](#total-reward-solvers) · [Average-reward solvers](#average-reward-solvers) · [Precision and validation](#precision-stopping-and-validation) · [Python catalogue](#browse-from-python)

## Reading the tables

**Library** identifies the implementation used, while section headings follow the catalogue's algorithm families: value iteration (`vi`), policy iteration (`pi`), modified policy iteration (`mpi`), Q-value iteration (`qvi`), linear programming (`lp`) and `experimental`. QVI here means repeated Bellman updates of state-action values, not sampled Q-learning. Experimental entries retain that metadata label even when their main operation is a form of VI or PI.

**Method and differences** describes updates, initialization or adaptations that distinguish entries. **Stopping rule** describes the local check or the tolerance passed to an external library; it is separate from validation after execution. Below, ε denotes `final_precision`, γ denotes `discount`, and `T` is the optimal Bellman operator: the best expected immediate reward plus continuation value. A full residual is the maximum absolute component of `T(V) - V`; the span of a vector is its maximum minus its minimum. GS denotes sequential Gauss–Seidel updates; SOR adds a relaxation factor, defaulting to 1 in the exposed SOR profiles.

**Native** means initialization or termination is delegated to the library's public API, including its iteration limits and postprocessing; the adapter does not add an independent convergence certificate. A source-module link gives the exact adapter, options and defaults. Source attributions below follow those modules; named local adaptations do not imply exact reproduction of a published algorithm. Full signatures remain in the linked constructors.

## Sources and integrations

| Implementation or integration | Dependency | Representation and scope |
| --- | --- | --- |
| MDPForge | Core NumPy/SciPy dependencies | Local algorithms over CSR transitions and NumPy rewards; aggregation uses the shared [partition representation](../src/mdpforge/core/partition.py). |
| SciPy/HiGHS | Core SciPy dependency | Sparse value-function or occupancy-flow linear programs through `scipy.optimize.linprog`; `method` selects `highs`, `highs-ds` or `highs-ipm`. |
| Pyomo/HiGHS | Optional `pyomo` and `highspy` | Explicit Pyomo formulations through its HiGHS interface; `threads=1` by default. |
| QuantEcon | Optional `quantecon` | `DiscreteDP` with sparse state-action rows for VI/PI/MPI; the LP adapter constructs a dense transition tensor and calls the native simplex method. |
| MDPToolbox | Optional `pymdptoolbox` (import name `mdptoolbox`) | Native Python methods called with the model matrices and the adapter's explicit options. |
| MDPSolver | Optional `mdpsolver` | CSR entries and rewards become native input lists; standard, GS and SOR profiles use the public `solve()` API. `parallel=False` by default. |
| Marmote | Separate native installation | CSR transitions become Marmote sparse matrices and rewards a native full matrix; discounted, total and average adapters select different native MDP classes. |
| MDPax | Optional `mdpax`, using JAX | CSR successors become padded state/action/event arrays, with double precision requested. Conversion is limited by `max_event_entries`; batching uses `max_batch_size`. |
| madupite | Separate madupite/PETSc installation | Single MPI rank only; transitions and rewards are written to temporary PETSc files and values/policy read back. Named profiles fix the native policy-evaluation method. |

From a checkout, `python -m pip install ".[solvers]"` installs the declared optional Python integrations; Marmote and madupite require separate installations. Exact package requirements are in [pyproject.toml](../pyproject.toml). Shared matrix conversions are in [solver_backends.py](../src/mdpforge/utils/solver_backends.py).

The metadata value `backend="linear"` groups all LP entries, including SciPy, Pyomo and QuantEcon LP. It names a catalogue group rather than one implementation library; QuantEcon VI/PI/MPI instead use `backend="quantecon"`.

The discounted MDPax constructors currently disable JAX compilation caching and clear JAX caches. Those existing setup operations, matrix conversions and any compilation during the trial are included in solver timing. Native representations, initialization, cache behavior and iteration budgets differ across integrations; matching an algorithm-family label does not make their timings or accuracy targets equivalent.

## Discounted solvers

Use `Benchmark()` or `Benchmark(criterion="discounted")`, with `0 < discount < 1`. Models come from the [discounted catalogue](models.md#discounted-recipes). Constructors accept `(model, discount, ...)`; all public solver runs are option-free `run()` calls.

### Value iteration (`vi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_vi`](../src/mdpforge/solvers/discounted/mdpforge_vi.py) | MDPForge | Synchronous Bellman updates; zero values by default, optional initial vector. | Full residual < ε(1−γ). | Local implementation |
| [`mdpforge_gsvi`](../src/mdpforge/solvers/discounted/mdpforge_gsvi.py) | MDPForge | In-place, state-ordered GS sweeps; zero values by default. | Full residual ≤ ε(1−γ) after a sweep. | Local implementation |
| [`mdpforge_visor`](../src/mdpforge/solvers/discounted/mdpforge_visor.py) | MDPForge | Sequential relaxed Bellman updates; configurable `relaxation`. | Full residual < ε(1−γ) after a sweep. | Local implementation |
| [`mdptoolbox_vi`](../src/mdpforge/solvers/discounted/mdptoolbox_vi.py) | MDPToolbox | Native synchronous VI and initialization; the backend may recompute the requested iteration bound. | Native VI; `epsilon=ε`. | MDPToolbox API |
| [`mdptoolbox_vigs`](../src/mdpforge/solvers/discounted/mdptoolbox_vigs.py) | MDPToolbox | Native GS VI and initialization; the backend may recompute the requested iteration bound. | Native GS VI; `epsilon=ε`. | MDPToolbox API |
| [`mdpsolver_vi`](../src/mdpforge/solvers/discounted/mdpsolver_vi.py) | MDPSolver | Standard sweeps; native initialization unless vectors are supplied. | Native; `tolerance=ε(1−γ)`. | MDPSolver API |
| [`mdpsolver_vigs`](../src/mdpforge/solvers/discounted/mdpsolver_vigs.py) | MDPSolver | GS sweeps; optional initial value and policy. | Native; `tolerance=ε(1−γ)`. | MDPSolver API |
| [`mdpsolver_visor`](../src/mdpforge/solvers/discounted/mdpsolver_visor.py) | MDPSolver | SOR sweeps; optional initialization and relaxation. | Native; `tolerance=ε(1−γ)`. | MDPSolver API |
| [`marmote_vi`](../src/mdpforge/solvers/discounted/marmote_vi.py) | Marmote | Native `DiscountedMDP.ValueIteration`; returns values without a policy. | Native VI with tolerance ε and outer budget. | Marmote API |
| [`marmote_vigs`](../src/mdpforge/solvers/discounted/marmote_vigs.py) | Marmote | Native `ValueIterationGS`; returns values without a policy. | Native GS VI with tolerance ε and outer budget. | Marmote API |
| [`mdpax_vi`](../src/mdpforge/solvers/discounted/mdpax_vi.py) | MDPax | JAX VI over padded CSR successors; native initialization. | Native VI; `epsilon=ε`. | MDPax API |
| [`madupite_vi`](../src/mdpforge/solvers/discounted/madupite_vi.py) | madupite | One unpreconditioned Richardson evaluation step per outer iteration. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`quantecon_vi`](../src/mdpforge/solvers/discounted/quantecon_vi.py) | QuantEcon | Sparse `DiscreteDP` VI with native initialization and norm-based stopping. | Native VI; `epsilon=2ε`. | QuantEcon API |

### Policy iteration (`pi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_pi`](../src/mdpforge/solvers/discounted/mdpforge_pi.py) | MDPForge | Sparse linear-system evaluation; action-zero policy by default, optional initial policy. | Policy stability. | Local implementation |
| [`mdptoolbox_pi`](../src/mdpforge/solvers/discounted/mdptoolbox_pi.py) | MDPToolbox | Native default matrix-based policy evaluation and improvement. | Native PI with outer budget; no ε option. | MDPToolbox API |
| [`mdpsolver_pi`](../src/mdpforge/solvers/discounted/mdpsolver_pi.py) | MDPSolver | Standard updates; values and policy explicitly zero by default, optional replacements. | Native PI; `tolerance=ε`. | MDPSolver API |
| [`mdpax_pi`](../src/mdpforge/solvers/discounted/mdpax_pi.py) | MDPax | JAX iterative policy evaluation and improvement; separate evaluation budget. | Native `max_diff` test with `epsilon=ε(1−γ)`. | MDPax API |
| [`madupite_ipi`](../src/mdpforge/solvers/discounted/madupite_ipi.py) | madupite | Inexact PI; configurable Krylov solver/preconditioner, GMRES/none by default. | Native outer tolerance ε(1−γ), separate evaluation budget. | madupite/PETSc API |
| [`madupite_ipi_gmres`](../src/mdpforge/solvers/discounted/madupite_ipi_gmres.py) | madupite | Named GMRES profile of inexact PI. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`madupite_ipi_bicgstab`](../src/mdpforge/solvers/discounted/madupite_ipi_bicgstab.py) | madupite | Inexact PI with PETSc BiCGStab (`bcgs`) evaluation. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`madupite_ipi_tfqmr`](../src/mdpforge/solvers/discounted/madupite_ipi_tfqmr.py) | madupite | Inexact PI with transpose-free QMR evaluation. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`madupite_pi`](../src/mdpforge/solvers/discounted/madupite_pi.py) | madupite | One Richardson evaluation step with an LU preconditioner. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`quantecon_pi`](../src/mdpforge/solvers/discounted/quantecon_pi.py) | QuantEcon | Sparse PI with exact policy evaluation and native initialization. | Native PI with outer budget; no ε option. | QuantEcon API |

### Modified policy iteration (`mpi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_mpi`](../src/mdpforge/solvers/discounted/mdpforge_mpi.py) | MDPForge | Synchronous truncated evaluation; zero values, random initial policy; ten inner updates by default. | Full residual < ε(1−γ); separate inner tolerance. | Local implementation |
| [`mdptoolbox_mpi`](../src/mdpforge/solvers/discounted/mdptoolbox_mpi.py) | MDPToolbox | Native truncated evaluation; ten inner updates by default. | Native MPI; `epsilon=ε`. | MDPToolbox API |
| [`mdpsolver_mpi`](../src/mdpforge/solvers/discounted/mdpsolver_mpi.py) | MDPSolver | Standard partial evaluation; native initialization unless supplied. | Native; `tolerance=ε`, `parIterLim=10` by default. | MDPSolver API |
| [`mdpsolver_mpigs`](../src/mdpforge/solvers/discounted/mdpsolver_mpigs.py) | MDPSolver | GS partial evaluation; native initialization unless supplied. | Native; tolerance ε and configurable partial-evaluation limit. | MDPSolver API |
| [`mdpsolver_mpigs_init`](../src/mdpforge/solvers/discounted/mdpsolver_mpigs_init.py) | MDPSolver | GS profile with explicitly zero-initialized values and policy. | Native; tolerance ε and configurable partial-evaluation limit. | MDPSolver profile |
| [`mdpsolver_mpisor`](../src/mdpforge/solvers/discounted/mdpsolver_mpisor.py) | MDPSolver | SOR partial evaluation; configurable relaxation and initialization. | Native; tolerance ε and configurable partial-evaluation limit. | MDPSolver API |
| [`mdpsolver_mpisor_init`](../src/mdpforge/solvers/discounted/mdpsolver_mpisor_init.py) | MDPSolver | SOR profile with explicitly zero-initialized values and policy. | Native; tolerance ε and configurable partial-evaluation limit. | MDPSolver profile |
| [`marmote_mpi`](../src/mdpforge/solvers/discounted/marmote_mpi.py) | Marmote | Native `PolicyIterationModified`; ten inner updates by default; returns values without a policy. | Native outer/inner tolerances; inner ε defaults to outer ε. | Marmote API |
| [`marmote_mpigs`](../src/mdpforge/solvers/discounted/marmote_mpigs.py) | Marmote | Native GS modified PI; ten inner updates by default; returns values without a policy. | Native outer/inner tolerances and iteration budgets. | Marmote API |
| [`madupite_opi`](../src/mdpforge/solvers/discounted/madupite_opi.py) | madupite | Optimistic PI with a fixed Richardson evaluation budget, default 20. | Native outer tolerance ε(1−γ). | madupite/PETSc profile |
| [`quantecon_mpi`](../src/mdpforge/solvers/discounted/quantecon_mpi.py) | QuantEcon | Sparse MPI with native span stopping and value shift; 20 evaluation steps by default. | Native MPI; `epsilon=2ε`. | QuantEcon API |

### Q-value iteration (`qvi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_qvi`](../src/mdpforge/solvers/discounted/mdpforge_qvi.py) | MDPForge | Synchronous state-action Bellman updates; zero Q-values by default, optional initial array. | Full Q-Bellman residual < ε(1−γ). | Local implementation |

### Linear programming (`lp`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`scipy_primal`](../src/mdpforge/solvers/discounted/scipy_primal.py) | SciPy/HiGHS | Value-function Bellman inequalities; returns values without a policy. | HiGHS success status and native tolerances. | SciPy `linprog` API |
| [`scipy_dual`](../src/mdpforge/solvers/discounted/scipy_dual.py) | SciPy/HiGHS | Discounted occupancy flows; extracts a deterministic policy and evaluates it by a sparse linear solve. | HiGHS success status and native tolerances. | SciPy `linprog` API |
| [`pyomo_primal`](../src/mdpforge/solvers/discounted/pyomo_primal.py) | Pyomo/HiGHS | Explicit value-function LP; returns values without a policy. | Native optimal termination required. | Pyomo HiGHS API |
| [`pyomo_dual`](../src/mdpforge/solvers/discounted/pyomo_dual.py) | Pyomo/HiGHS | Explicit occupancy LP; extracts and evaluates a deterministic policy. | Native optimal termination required. | Pyomo HiGHS API |
| [`quantecon_lp`](../src/mdpforge/solvers/discounted/quantecon_lp.py) | QuantEcon | Dense transition tensor and native simplex LP; returns values and policy. | Native simplex termination and pivot budget. | QuantEcon API |

### Experimental methods (`experimental`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_vi_reward_init`](../src/mdpforge/solvers/discounted/mdpforge_vi_reward_init.py) | MDPForge | Synchronous VI initialized with maximum immediate rewards divided by 1−γ. | Full residual < ε(1−γ). | Local initialization variant |
| [`mdpforge_vi_span`](../src/mdpforge/solvers/discounted/mdpforge_vi_span.py) | MDPForge | Synchronous VI with a constant-shift correction; zero values by default. | Residual span < 2ε(1−γ)/γ before correction. | Local span variant |
| [`mdpforge_mpi_reward_init`](../src/mdpforge/solvers/discounted/mdpforge_mpi_reward_init.py) | MDPForge | MPI initialized with maximum immediate rewards and their greedy policy; different default budgets from `mdpforge_mpi`. | Full residual < ε(1−γ); separate inner tolerance. | Local initialization variant |
| [`mdpforge_aggvi`](../src/mdpforge/solvers/discounted/mdpforge_aggvi.py) | MDPForge | Progressively refined aggregation, projected backups, full-state updates and VI fallback. | Projected/refinement checks plus full residual ≤ ε(1−γ); fallback VI tolerance. | Local aggregation implementation |
| [`mdpforge_aggqvi`](../src/mdpforge/solvers/discounted/mdpforge_aggqvi.py) | MDPForge | Progressive Q-value aggregation with full-state Bellman batches and VI fallback. | Projected/refinement checks plus full value residual ≤ ε(1−γ); fallback VI tolerance. | Local aggregation implementation |
| [`mdpforge_aggmpi`](../src/mdpforge/solvers/discounted/mdpforge_aggmpi.py) | MDPForge | Aggregate policy evaluation with progressive refinement and full-state evaluation fallback. | Full residual < ε(1−γ) at policy improvement. | Local aggregation implementation |
| [`mdpforge_envvi`](../src/mdpforge/solvers/discounted/mdpforge_envvi.py) | MDPForge | Progressive lower/upper Bellman envelopes using compressed transitions. | Default residual polishing; optional `envelope` stop at half-width ≤ ε. | Local envelope implementation |
| [`mdpforge_bertsekas`](../src/mdpforge/solvers/discounted/mdpforge_bertsekas.py) | MDPForge | PI with adaptive aggregate linear-system corrections and final full-state policy evaluation. | Policy stability; inner span and final evaluation tolerances. | Bertsekas and Castanon, *Adaptive Aggregation Methods for Infinite Horizon Dynamic Programming*, module adaptation |
| [`mdpforge_abate`](../src/mdpforge/solvers/discounted/mdpforge_abate.py) | MDPForge | Adaptive aggregate policy evaluation, deterministic median refinement and full-state PI fallback; reward shift undone on return. | Policy stability or fallback iteration budget; aggregate evaluation error bounds. | Abate, Ceska and Kwiatkowska (2016), module adaptation |
| [`mdpforge_chen`](../src/mdpforge/solvers/discounted/mdpforge_chen.py) | MDPForge | Alternates full-state Bellman sweeps and sampled aggregate stochastic-approximation updates. | Full-state Bellman variation < ε(1−γ). | Chen-named local variant; no paper cited in module |

## Total-reward solvers

Use `Benchmark(criterion="total")` and the [total-reward model catalogue](models.md#total-reward-recipes). Constructors take `(model, ...)`, without a discount. These methods require a finite-return problem under their respective assumptions; setting γ to 1 in a discounted problem does not establish those assumptions. Local total-reward formulations use declared `terminal_states`, with absorbing zero-reward terminals. A small undiscounted residual alone is not a general bound on value error.

### Value iteration (`vi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_vi`](../src/mdpforge/solvers/total/mdpforge_vi.py) | MDPForge | Synchronous undiscounted VI from zero values. | Bellman variation ≤ ε. | Local implementation |
| [`mdpforge_gsvi`](../src/mdpforge/solvers/total/mdpforge_gsvi.py) | MDPForge | In-place GS sweeps from zero values. | Full residual ≤ ε. | Local implementation |
| [`mdpforge_tvi`](../src/mdpforge/solvers/total/mdpforge_tvi.py) | MDPForge | Strongly connected components solved in dependency order; singleton self-loops handled directly. | Local VI tolerances and final full residual check. | Dai, Mausam, Weld and Goldsmith (JAIR 2011), Algorithm 2 adaptation |
| [`mdpforge_ftvi`](../src/mdpforge/solvers/total/mdpforge_ftvi.py) | MDPForge | Bound-certified action elimination then TVI over all states; falls back to TVI without uniform absorption bounds. | Final full residual check after pruning. | Dai et al. (JAIR 2011), Algorithm 3 adaptation |
| [`marmote_vi`](../src/mdpforge/solvers/total/marmote_vi.py) | Marmote | Native `TotalRewardMDP.ValueIteration`; returns values and policy. | Native VI with tolerance ε and outer budget. | Marmote API |
| [`marmote_vigs`](../src/mdpforge/solvers/total/marmote_vigs.py) | Marmote | Native total-reward `ValueIterationGS`; returns values and policy. | Native GS VI with tolerance ε and outer budget. | Marmote API |

### Policy and modified policy iteration (`pi`, `mpi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_pi`](../src/mdpforge/solvers/total/mdpforge_pi.py) | MDPForge | Sparse evaluation on nonterminal states; terminal-directed initial policy, requiring terminal reachability from every state. | Policy stability, retaining ties within 1e−12. | Local PI implementation |
| [`mdpforge_mpi`](../src/mdpforge/solvers/total/mdpforge_mpi.py) | MDPForge | Terminal-directed initialization where available; zero values and ten truncated evaluation updates by default. | Full residual ≤ ε. | Local MPI implementation |
| [`marmote_mpi`](../src/mdpforge/solvers/total/marmote_mpi.py) | Marmote | Native total-reward `PolicyIterationModified`; ten inner updates by default. | Native outer/inner tolerances and iteration budgets. | Marmote API |
| [`marmote_mpigs`](../src/mdpforge/solvers/total/marmote_mpigs.py) | Marmote | Native total-reward `PolicyIterationModifiedGS`. | Native outer/inner tolerances and iteration budgets. | Marmote API |

### Q-value iteration (`qvi`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_qvi`](../src/mdpforge/solvers/total/mdpforge_qvi.py) | MDPForge | Synchronous undiscounted Q-value updates from zeros. | Full Q-Bellman variation ≤ ε. | Local implementation |

### Linear programming (`lp`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`scipy_primal`](../src/mdpforge/solvers/total/scipy_primal.py) | SciPy/HiGHS | Bellman inequalities with terminal values fixed at zero; returns values without a policy. | HiGHS success status and native tolerances. | SciPy `linprog` API |
| [`scipy_dual`](../src/mdpforge/solvers/total/scipy_dual.py) | SciPy/HiGHS | Transient-state occupancy flows; values from equality-constraint marginals and policy from occupancies. | HiGHS success status and native tolerances. | SciPy `linprog` API |
| [`pyomo_primal`](../src/mdpforge/solvers/total/pyomo_primal.py) | Pyomo/HiGHS | Bellman inequalities with zero terminal values; returns values without a policy. | Native optimal termination required. | Pyomo HiGHS API |
| [`pyomo_dual`](../src/mdpforge/solvers/total/pyomo_dual.py) | Pyomo/HiGHS | Transient-state occupancy LP, then deterministic policy extraction and sparse evaluation. | Native optimal termination and finite policy evaluation required. | Pyomo HiGHS API |

### Experimental methods (`experimental`)

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`mdpforge_aggvi`](../src/mdpforge/solvers/total/mdpforge_aggvi.py) | MDPForge | Progressive value aggregation with full-state VI fallback; terminal values kept at zero. | Full residual ≤ ε, including the fallback. | Local total-reward adaptation |
| [`mdpforge_aggqvi`](../src/mdpforge/solvers/total/mdpforge_aggqvi.py) | MDPForge | Progressive Q-value aggregation and full-state VI fallback. | Full value residual ≤ ε, including the fallback. | Local total-reward adaptation |
| [`mdpforge_aggmpi`](../src/mdpforge/solvers/total/mdpforge_aggmpi.py) | MDPForge | Progressively refined aggregate policy evaluation with full-state evaluation fallback. | Full residual ≤ ε at policy improvement. | Local total-reward adaptation |
| [`mdpforge_sharp`](../src/mdpforge/solvers/total/mdpforge_sharp.py) | MDPForge | Hierarchical adaptive blocks, concrete-state local VI and frozen boundaries; boundary changes trigger re-solving. | Global change check followed by full residual ≤ ε. | Evangelidis, Vázquez and Gerasimou (FSE 2026), module adaptation |

## Average-reward solvers

Use `Benchmark(criterion="average")`; models come from the discounted model catalogue, but no discount is passed to these solver constructors. Native methods have their own constant-gain, unichain or span-convergence assumptions; catalogue discovery does not verify them for a selected model. Relative value iteration (RVI) works with potentials rather than discounted returns.

The [shared result adapter](../src/mdpforge/solvers/average/_common.py) subtracts the potential at state zero from every returned potential. `SolveResult.value` is this relative vector; `gain` is the scalar average reward when exposed by the backend. MDPSolver adapters return no gain. The scalar-gain LP assumes state-independent optimal average gain and returns a feasible Bellman potential, which need not identify optimal transient actions; it deliberately returns no policy. Average-reward value-error certification is unavailable; a residual diagnostic is described below.

| Solver | Library | Method and differences | Stopping rule | Source or reference |
| --- | --- | --- | --- | --- |
| [`marmote_rvi`](../src/mdpforge/solvers/average/marmote_rvi.py) | Marmote | `AverageMDP.RelativeValueIteration` for unichain problems; returns potential, policy and gain. | Native RVI tolerance ε and outer budget. | Marmote API; family `vi` |
| [`mdptoolbox_rvi`](../src/mdpforge/solvers/average/mdptoolbox_rvi.py) | MDPToolbox | Native RVI; returns potential, policy and average reward estimate. | Native span tolerance ε and outer budget. | MDPToolbox API; family `vi` |
| [`mdpax_rvi`](../src/mdpforge/solvers/average/mdpax_rvi.py) | MDPax | JAX RVI over padded successors; returns potential, policy and reference-state gain. | Native span tolerance ε and outer budget. | MDPax API; family `vi` |
| [`mdpsolver_vi`](../src/mdpforge/solvers/average/mdpsolver_vi.py) | MDPSolver | Standard average-reward VI for constant-gain problems; returns potential and policy. | Native span tolerance ε. | MDPSolver API; family `vi` |
| [`mdpsolver_pi`](../src/mdpforge/solvers/average/mdpsolver_pi.py) | MDPSolver | Standard average-reward policy evaluation and improvement for constant-gain problems. | Native PI; tolerance ε. | MDPSolver API; family `pi` |
| [`marmote_mpi`](../src/mdpforge/solvers/average/marmote_mpi.py) | Marmote | Native unichain MPI; returns potential, policy and gain; ten inner updates by default. | Native outer/inner tolerances and iteration budgets. | Marmote API; family `mpi` |
| [`mdpsolver_mpi`](../src/mdpforge/solvers/average/mdpsolver_mpi.py) | MDPSolver | Standard partial evaluation for constant-gain problems; ten inner updates by default. | Native span tolerance ε and partial-evaluation limit. | MDPSolver API; family `mpi` |
| [`scipy_primal`](../src/mdpforge/solvers/average/scipy_primal.py) | SciPy/HiGHS | Minimize scalar gain subject to Bellman inequalities, fixing the reference potential to zero; no policy. | HiGHS success status and native tolerances. | SciPy `linprog` API; family `lp` |

## Precision, stopping and validation

`Benchmark.run(precision=...)` supplies `final_precision` to constructors that expose it. Constructors without that option, including local exact PI and LP adapters, keep their own stopping rules; the requested precision still controls any requested post-run validation. Configure solver-specific options through `add_solver()` and benchmark precision through `run()`. In direct use, pass options to `Solver(...)`, then call `run()` without arguments.

An identical ε does not imply an identical stopping test. Full residuals, residual spans, policy stability, interval widths and native optimization tolerances measure different things. In particular, discounted MDPToolbox receives `epsilon=ε`, QuantEcon VI/MPI receive `epsilon=2ε`, and MDPSolver VI profiles receive `tolerance=ε(1−γ)` while its PI/MPI profiles receive ε. LP adapters do not translate the benchmark precision into HiGHS tolerances. The linked modules define the actual transformations and budgets.

For discounted reward, the classical value-error bound is `max(abs(T(V) - V)) / (1 - discount)`. Post-run methods additionally support span-and-offset checks, returned-policy checks and lower/upper value intervals. `mdpforge_envvi` selects `residual` validation by default and `bounds` when its `stop_criterion="envelope"` option is used. The [validation implementation](../src/mdpforge/core/validation.py) defines these checks; `validation_method` names the requested check, not evidence that it ran or passed.

For total reward, the current value-error certificate covers absorbing acyclic models, using declared zero-reward absorbing terminals; a cyclic finite-return model may lie outside that certificate even when a solver can handle it. For average reward, the current benchmark can check `max(abs(T(h) - h - gain))`, or the span of `T(h) - h` when no gain is returned, against `precision`. Passing this diagnostic does not certify relative value error; an unavailable error bound remains `NaN`.

Set `validate=True` explicitly when requesting checks, or `validate=False` to omit them. Validation runs after solver timing, in the parent process and outside the trial timeout. `status="success"` means execution completed without an exception; inspect `validation_status`, `validation_error`, `validation_runtime` and any available error bound separately. A failed validation does not overwrite execution status or solver runtime.

Timing includes solver construction, backend conversions and `run()`; model construction, copying, seed setup, logging and validation are outside that interval. Trial seeds are paired and global NumPy/Python RNG states restored, but initialization and private backend RNGs can differ. Compare recorded configurations and distinguish cold/warm compilation and cache conditions; no universal ranking follows from algorithm names alone.

## Browse from Python

```python
from mdpforge import list_solvers
from mdpforge.solvers.total import list_solvers as list_total_solvers
from mdpforge.solvers.average import list_solvers as list_average_solvers

discounted = list_solvers()
local_vi = list_solvers(family="vi", backend="mdpforge")
policy_methods = list_solvers(family=["pi", "mpi"])
linear_programs = list_solvers(backend="linear")
total = list_total_solvers()
average = list_average_solvers()

for entry in local_vi:
    print(entry["name"], entry["family"], entry["backend"], entry["description"])
```

Listing reads literal metadata without importing optional backend modules or executing solvers. Filters combine different fields with AND and multiple values in one field with OR; unknown values and empty intersections raise `ValueError`. The top-level `list_solvers()` lists discounted entries only; total and average catalogues are separate. There is no installed-availability filter on `list_solvers()`.

```python
from mdpforge import Benchmark

bench = Benchmark(criterion="discounted")
bench.add_solver("mdpforge_vi", max_iterations=100_000)
bench.add_solver(
    "mdpforge_mpi",
    max_evaluation_iterations=10,
    evaluation_precision=1e-3,
)

# Select a group using metadata; options must suit every selected constructor.
bench.add_solver(family="pi", backend="mdpforge")

total_bench = Benchmark(criterion="total")
total_bench.add_solver("mdpforge_vi")
average_bench = Benchmark(criterion="average")
average_bench.add_solver("scipy_primal")
```

Registration selects the catalogue from the benchmark criterion; an identifier such as `mdpforge_vi` can therefore name different implementations in different catalogues. Explicit selection raises an error for a missing dependency; group selection warns and skips missing dependencies. These examples register configurations without running them. Add models from the [model catalogue](models.md) before an experiment; see [adding a solver](adding-a-solver.md) for custom interfaces. When changing a catalogue entry, update its row here with its metadata.
