# Model catalogue

This page lists the public model recipes in MDPForge: 102 available discounted recipes, 32 total-reward recipes and five imported sources awaiting adaptation. Available model names are the identifiers accepted by `Benchmark.add_mdp()` in the corresponding reward catalogue; they are distinct from generated instance names. Several recipes can represent the same problem family with different dynamics, source instances or objectives.

[Sources](#sources-and-adaptations) · [Discounted recipes](#discounted-recipes) · [Total-reward recipes](#total-reward-recipes) · [Unavailable sources](#imported-sources-not-yet-available) · [Python catalogue](#browse-from-python)

## Reading the tables

**Structure** describes the state variables and actions. **Dynamics and rewards** describes the implemented transitions and objective, including reductions of published problems. A local implementation inspired by a benchmark is distinct from an import of its original source. The linked Python modules contain constructor options and full metadata references.

**Presets:** T = `tiny`, S = `small`, M = `medium`, L = `large`. Only declared available presets are listed. A preset selects parameters or a bundled source configuration; it is not a fixed number of states, an ordering guarantee or a runtime measurement. Requested dimensions can be rounded, and several presets can select the same source instance. Inspect actual `state_dim` and `action_dim` on the constructed model; missing presets have a `reason` in `METADATA`. The three generated maze families require the optional `mazelib` dependency (`pip install ".[models]"` from a checkout).

After `create_model()`, available recipes expose one SciPy CSR transition matrix per action and a NumPy array of expected rewards indexed by state and action. State counts include any explicit absorbing states. Aliased observations in some navigation tasks do not replace their full Markov states.

## Sources and adaptations

| Implementation or collection | Provenance | Interpretation |
| --- | --- | --- |
| Local generators and adaptations | Bibliographic references in each module; some are explicitly family references, general references or inspirations. | The equations and rewards in the local implementation define the task. Incomplete attribution is retained explicitly, as for `admission`. |
| jani-models | [Pinned source manifest][jani] and [upstream model overview](../src/mdpforge/models/discounted/_jani/_data/README.md). | Bundled JANI sources, with explicit discounted reward objectives; includes one-action Markov chains. |
| QVBS | [Pinned source manifest][qvbs], original references and author/conversion records; Hartmanns et al., TACAS 2019. | JANI imports and selected scalable constructions from original PPDDL rules; bundled sources carry [CC-BY-4.0](../src/mdpforge/models/discounted/_qvbs/_data/LICENSE). |
| PRISM benchmarks and examples | [Pinned source manifest][prism], source authors and per-model adaptations. | Native command semantics reconstructed in Python. The benchmark-suite sources are CC-BY-4.0; the Rabin and philosopher example sources are GPL-2.0-only, as recorded in the manifest. |

For JANI/QVBS imports, states are reachable variable valuations and automaton locations; actions are enabled local choices, with unused action slots repeating the first enabled choice. Native maximization rewards retain their scale; minimization costs are negated. Reachability properties become signed unit rewards on first reaching the selected target, stored as expected immediate rewards; target and stopped states then absorb at zero reward. These are discounted objectives, not the original unbounded reachability probabilities or full verification assertions. Finite property horizons are replaced by discounting; explicit source timers and counters remain part of the state when present.

The PRISM `consensus`, `csma`, `firewire`, `wlan` and `zeroconf` recipes retain native transition steps and zero their costs at completion without removing terminal transitions. `prism_rabin` and `prism_dining_philosophers` instead add step costs to a designated participant's first target entry and make the target absorbing. No fairness constraint is added. Source variants and default objectives differ across collections: for example, `jani_dining_philosophers` defaults to a negative first-hit reward, `qvbs_philosophers_mdp` to a positive first-hit reward, and `prism_dining_philosophers` to negative steps until a designated philosopher eats.

## Discounted recipes

Use `Benchmark()` or `Benchmark(criterion="discounted")`, with `0 < discount < 1`. Average-reward benchmarks reuse this catalogue; average-reward certification is unavailable. The imported rows below describe their default objective; use the linked constructor and manifest for alternatives.

### Abstract

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`block`](../src/mdpforge/models/discounted/block.py) | State in one of ten blocks; configurable action count | Random dense blocks with sparse connections, normalized by row; random rewards scaled by state count. | Meuleau et al. (1998). Solving very large weakly coupled Markov decision processes. AAAI. | T/S/M/L |
| [`chain_walk`](../src/mdpforge/models/discounted/chain_walk.py) | Chain position; left/right | Intended direction with probability 0.9, opposite otherwise; expected reward for reaching either boundary. | Lagoudakis, M. G. and Parr, R. (2003). Least-Squares Policy Iteration. JMLR 4:1107-1149. | T/S/M/L |
| [`garnet`](../src/mdpforge/models/discounted/garnet.py) | Abstract state; configurable action count | Normalized random transitions and Gaussian rewards; a synthetic MDP generator. | Bhatnagar, S. et al. (2009). Natural actor-critic algorithms. Automatica 45(11):2471-2482. | T/S/M/L |
| [`jani_approx_pi`](../src/mdpforge/models/discounted/jani_approx_pi.py) | Sampling locations, counters and hit count; enabled source choices | Monte Carlo sampling; native accumulated hit reward, without normalization into an estimate of pi. | [jani-models][jani] | T/S |
| [`jani_coupon_collector`](../src/mdpforge/models/discounted/jani_coupon_collector.py) | Collection state and draw count; enabled source choices | Coupon sampling; native numberDraws cost is negated by the default draws objective. | [jani-models][jani] | T/S |
| [`jani_random_walk`](../src/mdpforge/models/discounted/jani_random_walk.py) | Line position and source location; enabled source choices | Source random walk; default position reward is the native state variable, rather than the local random_walk entry reward. | [jani-models][jani] | T/S |
| [`qvbs_random_predicates`](../src/mdpforge/models/discounted/qvbs_random_predicates.py) | Finite-domain planning predicates; enabled planning choices | Configurable object count constructed from original PPDDL rules; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S |
| [`random_walk`](../src/mdpforge/models/discounted/random_walk.py) | Line position; left/right bias | Directional movement succeeds with probability 0.8; absorbing boundaries, expected -1/+1 entry rewards at left/right. | Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed. Random-walk prediction examples. | T/S/M/L |
| [`unichain`](../src/mdpforge/models/discounted/unichain.py) | Chain position; left/right | Deterministic movement clipped at boundaries; -1 except 0 in state zero. | Puterman, M. L. (2014). Markov Decision Processes: Discrete Stochastic Dynamic Programming, p.353. | T/S/M/L |

### Navigation and control

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`acrobot`](../src/mdpforge/models/discounted/acrobot.py) | Two joint angles and velocities plus goal; three torques | Deterministic discretized pendulum dynamics; -1 until the height goal, with 0 on goal-reaching actions and at the goal. | Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding. Advances in Neural Information Processing Systems 8. | T/S/M/L |
| [`barto`](../src/mdpforge/models/discounted/barto.py) | Track position and signed velocity; nine accelerations | Acceleration can fail; crashes reset to the start; -1 for nonfinishing outcomes, 0 on finish. | Barto, A. G., Bradtke, S. J. and Singh, S. P. (1995). Learning to act using real-time dynamic programming. Artificial Intelligence 72(1-2):81-138. | T/S/M/L |
| [`cartpole`](../src/mdpforge/models/discounted/cartpole.py) | Cart position/velocity and pole angle/angular velocity; left/right force | Discretized deterministic dynamics, reset after threshold violations; -1 within thresholds and 0 in reset states. | inspired by the classical CartPole model Barto, Sutton and Anderson (1983), and by discretized control benchmarks. | T/S/M/L |
| [`cliff`](../src/mdpforge/models/discounted/cliff.py) | Grid cell; four directions | Deterministic motion and cliff resets; -1 per move, -100 for unsafe moves, 0 at the absorbing goal. | Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Example 6.6. | T/S/M/L |
| [`deep_sea_treasure`](../src/mdpforge/models/discounted/deep_sea_treasure.py) | Depth, column, oxygen and terminals; five diving/collection actions | Reduced scalar-reward model with currents and drowning risk; depth-dependent treasure payoff and movement/failure costs. | ElSayed-Aly, Parker & Feng, NFM 2024, DS Treasure example: [source](https://www.prismmodelchecker.org/files/nfm24dpmc/) | T/S/M/L |
| [`drone_energy`](../src/mdpforge/models/discounted/drone_energy.py) | Location, battery, wind and terminals; fly forward/backward, recharge, wait | Resampled wind and delayed flights consume battery; landing reward and movement/crash costs. | ElSayed-Aly, Parker & Feng, Distributional Probabilistic Model Checking (NFM 2024), Energy drone example: [source](https://www.prismmodelchecker.org/files/nfm24dpmc/) | T/S/M/L |
| [`frozen`](../src/mdpforge/models/discounted/frozen.py) | Lake cell; four directions | Slippage among three directions; holes and goal absorb; expected unit reward on goal entry. | Gymnasium FrozenLake-v1 / OpenAI Gym FrozenLake. | T/S/M/L |
| [`local_perception_grid_world`](../src/mdpforge/models/discounted/local_perception_grid_world.py) | True grid cell; four directions | Deterministic walls and absorbing goal; movement cost and goal bonus. Separate local-wall observations can alias different states. | McCallum (1996), Reinforcement Learning with Selective Perception and Hidden State. | T/S/M/L |
| [`lqregulator`](../src/mdpforge/models/discounted/lqregulator.py) | Scalar state grid; scalar control grid | Linear drift projected to grid with neighboring-state noise; negative quadratic state and control costs. | Family reference: Kalman (1960). Contributions to the Theory of Optimal Control. | T/S/M/L |
| [`m_maze`](../src/mdpforge/models/discounted/m_maze.py) | True cell in an M-shaped maze; four directions | Intended movement with probability 0.9, otherwise stay; goal-directed action bonus and movement cost; aliased observations are separate from states. | McCallum (1996), U-Tree / UDM aliased maze variants. | T/S/M/L |
| [`maze_backtrack`](../src/mdpforge/models/discounted/maze_backtrack.py) | Maze cell; four directions | Random backtracking-generated corridors, blocked moves stay; -1 per step and 0 at the exit. | Buck, J. (2015). Mazes for Programmers: Code Your Own Twisty Little Passages. | T/S/M/L |
| [`maze_prims`](../src/mdpforge/models/discounted/maze_prims.py) | Maze cell; four directions | Random Prim-generated corridors, blocked moves stay; -1 per step and 0 at the exit. | Prim, R. C. (1957). Shortest connection networks and some generalizations. Bell System Technical Journal 36(6):1389-1401. | T/S/M/L |
| [`maze_wilson`](../src/mdpforge/models/discounted/maze_wilson.py) | Maze cell; four directions | Random Wilson-generated corridors, blocked moves stay; -1 per step and 0 at the exit. | Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover time. STOC, pp. 296-303. | T/S/M/L |
| [`mountain`](../src/mdpforge/models/discounted/mountain.py) | Car position and velocity; three forces | Deterministic discretized MountainCar; -1 before the hilltop and 0 in goal states, which reset to the start. | Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. PhD thesis, University of Cambridge. | T/S/M/L |
| [`ninerooms`](../src/mdpforge/models/discounted/ninerooms.py) | Cell in nine connected rooms; four directions | Uncertain movement constrained by walls and doors; -1 per step and 0 at the absorbing exit. | Inspired by Hengst, B. (2012). Hierarchical approaches. In Reinforcement Learning: State-of-the-Art, pp. 293-323. | T/S/M/L |
| [`office_world`](../src/mdpforge/models/discounted/office_world.py) | Cell plus mail/coffee flags; four directions | Deterministic navigation and item collection; movement/plant penalties and delivery bonus; completed office goal absorbs. | Office World, adapted from Icarte et al. (2018) reward-machine benchmark; used as a tabular/factored benchmark in CAT+RL (Dadvar et al., 2023). | T/S/M/L |
| [`qvbs_cdrive`](../src/mdpforge/models/discounted/qvbs_cdrive.py) | Finite-domain driving configuration; enabled planning choices | City Driving source instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_rectangle_tireworld`](../src/mdpforge/models/discounted/qvbs_rectangle_tireworld.py) | Finite-domain vehicle/tire configuration; planning choices | Rectangular Tireworld instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S/M |
| [`qvbs_resource_gathering`](../src/mdpforge/models/discounted/qvbs_resource_gathering.py) | Grid position, resources, quotas and attack flags; local choices | Resource collection with attacks; default expgold retains native collected-gold reward. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_tireworld`](../src/mdpforge/models/discounted/qvbs_tireworld.py) | Finite-domain vehicle/tire configuration; planning choices | Published Tireworld instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S/M |
| [`qvbs_triangle_tireworld`](../src/mdpforge/models/discounted/qvbs_triangle_tireworld.py) | Finite-domain vehicle/tire configuration; planning choices | Triangular Tireworld instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S |
| [`rooms`](../src/mdpforge/models/discounted/rooms.py) | Cell in four connected rooms; four directions | Intended move with probability 0.8, otherwise stay; walls and doors; -1 per step and 0 at the absorbing exit. | Hengst, B. (2012). Hierarchical approaches. In Reinforcement Learning: State-of-the-Art, pp. 293-323. | T/S/M/L |
| [`schoolboy`](../src/mdpforge/models/discounted/schoolboy.py) | Progress toward school; walk, cycle, bus | Mode-specific random travel increments; -15 per nonterminal step and 0 at school. | Hyon, E. (2020). Apprehender le hasard. Tangente Magazine. | T/S/M/L |
| [`search_and_rescue`](../src/mdpforge/models/discounted/search_and_rescue.py) | Zone, local site knowledge, onboard flag, urgency and terminals; five flight/rescue actions | Reduced Markov model with uncertain survey/pickup and an existing urgency budget; rescue payoff and flight/loss costs. | IPPC 2008 Search-and-Rescue (Teichteil-Konigsbuch): [source](https://ipc08.icaps-conference.org/probabilistic/wiki/images/0/03/Results.pdf) | T/S/M/L |
| [`sutton`](../src/mdpforge/models/discounted/sutton.py) | Track position and velocity; nine accelerations | Grid racetrack with acceleration failure, crash resets and absorbing finish; step costs until finish. | Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Exercise 5.12. | T/S/M/L |
| [`swim`](../src/mdpforge/models/discounted/swim.py) | River position; swim left/right | Left is deterministic, right is opposed by the current; small boundary and larger interior rewards for right-swimming actions. | Strehl, A. L. and Littman, M. L. (2008). An analysis of model-based interval estimation for Markov decision processes. JCSS 74(8):1309-1331. | T/S/M/L |
| [`tireworld`](../src/mdpforge/models/discounted/tireworld.py) | Road column, lane, spare/flat flags and terminals; five route/load/repair actions | Reduced two-lane model with punctures and renewable spare depots; route costs, destination bonus and stranding penalty. | IPPC 2004 Tireworld (Younes et al., JAIR 2005): [source](https://ipc04.icaps-conference.org/probabilistic/) | T/S/M/L |
| [`windgrid`](../src/mdpforge/models/discounted/windgrid.py) | Grid cell; four directions | Deterministic column-dependent upward wind; -1 per step and 0 at the absorbing goal. | Sutton & Barto, Reinforcement Learning: An Introduction, Example 6.5, Windy Gridworld. | T/S/M/L |
| [`wumpus_world`](../src/mdpforge/models/discounted/wumpus_world.py) | Cell, orientation, gold/Wumpus/arrow/terminal flags; move, turn, grab, shoot, climb | Fully observed tabular world; deterministic hazards, item/kill/exit rewards and action costs. | Wumpus World, Russell & Norvig, Artificial Intelligence: A Modern Approach; used as a discrete-state benchmark in CAT+RL (Dadvar et al., 2023). | T/S/M/L |

### Distributed systems

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`consensus`](../src/mdpforge/models/discounted/consensus.py) | Shared counter and process phases; local scheduling choices | Native PRISM shared-coin consensus; negative step cost until completion. | [PRISM benchmarks][prism] | T/S/M/L |
| [`csma`](../src/mdpforge/models/discounted/csma.py) | Bus, station phases, clocks and backoff counters; local choices | Native PRISM synchronization and exponential backoff; negative transmission-time cost. | [PRISM benchmarks][prism] | T/S/M/L |
| [`firewire`](../src/mdpforge/models/discounted/firewire.py) | Wire/node phases and digital clocks; local choices | Native PRISM root contention; negative time or sending-time cost until completion. | [PRISM benchmarks][prism] | T/S/M |
| [`jani_beb_modest`](../src/mdpforge/models/discounted/jani_beb_modest.py) | Clock and host phases, line status; enabled source choices | Bounded exponential backoff; default +1 first-hit bonus for LineSeized. | [jani-models][jani] | T/S |
| [`jani_brp`](../src/mdpforge/models/discounted/jani_brp.py) | Sender/receiver phases, retransmissions and channel flags; enabled source choices | Bounded retransmission; default +1 first-hit bonus for Property_brp_0. | [jani-models][jani] | T/S |
| [`jani_dining_cryptographers`](../src/mdpforge/models/discounted/jani_dining_cryptographers.py) | Payment, coin, protocol and agreement variables; enabled source choices | Dining cryptographers; default property_0 uses a negative first-hit bonus (Pmin). | [jani-models][jani] | T/S |
| [`jani_dining_philosophers`](../src/mdpforge/models/discounted/jani_dining_philosophers.py) | Philosopher phases; scheduling choices | Randomized resource sharing; default property_0 penalizes first entry into an eating state (Pmin). | [jani-models][jani] | T/S |
| [`jani_mutual_exclusion`](../src/mdpforge/models/discounted/jani_mutual_exclusion.py) | Process phases; scheduling choices | Pnueli-Zuck mutual exclusion; default property_1 uses a negative first-hit bonus for its source predicate. | [jani-models][jani] | T/S |
| [`jani_rabin`](../src/mdpforge/models/discounted/jani_rabin.py) | Shared counters and local process/draw variables; scheduling choices | Rabin mutual exclusion; default property_1 penalizes first critical-section entry by any process (Pmin). | [jani-models][jani] | T/S |
| [`prism_dining_philosophers`](../src/mdpforge/models/discounted/prism_dining_philosophers.py) | Reachable philosopher phases; local scheduling choices | Native PRISM protocol with added -1 per step until a designated philosopher eats, then zero-reward absorption; no fairness imposed. | [PRISM examples][prism] | T/S/M/L |
| [`prism_rabin`](../src/mdpforge/models/discounted/prism_rabin.py) | Reachable shared/local protocol variables; local scheduling choices | Native PRISM Rabin protocol with added -1 per step until a designated process enters its critical section, then zero-reward absorption. | [PRISM examples][prism] | T/S/M |
| [`qvbs_beb`](../src/mdpforge/models/discounted/qvbs_beb.py) | Clock and host phases, line status; enabled source choices | Bounded exponential backoff; default +1 first-hit bonus for LineSeized. | [QVBS][qvbs] | T/S |
| [`qvbs_consensus`](../src/mdpforge/models/discounted/qvbs_consensus.py) | Shared counter, process phases and coin values; local choices | Shared-coin consensus; default steps_min negates native step cost. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_csma`](../src/mdpforge/models/discounted/qvbs_csma.py) | Bus/station phases, clocks and backoff; local choices | CSMA/CD protocol; default time_min negates native transmission-time cost. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_echoring`](../src/mdpforge/models/discounted/qvbs_echoring.py) | Stations, messages, failure flags and iteration counters; local choices | EchoRing source protocol; default MinFailed penalizes first reaching the selected failure predicate. | [QVBS][qvbs] | T/S |
| [`qvbs_firewire`](../src/mdpforge/models/discounted/qvbs_firewire.py) | Wire/node phases and clocks; local choices | IEEE 1394 root contention; default time_min negates native election-time cost. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_firewire_abst`](../src/mdpforge/models/discounted/qvbs_firewire_abst.py) | Abstract contention phase and clock; local choices | Abstract IEEE 1394 model; default rounds negates native round cost. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_firewire_dl`](../src/mdpforge/models/discounted/qvbs_firewire_dl.py) | Abstract contention phase and timer; local choices | Timed FireWire source; default deadline uses a negative first-hit bonus (Pmin). | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_ij`](../src/mdpforge/models/discounted/qvbs_ij.py) | Token ownership on a process ring; scheduling choices | Israeli-Jalfon self-stabilization; default +1 first-hit bonus for the single-token stable state. | [QVBS][qvbs] | T/S/M |
| [`qvbs_philosophers_mdp`](../src/mdpforge/models/discounted/qvbs_philosophers_mdp.py) | Philosopher phases; scheduling choices | Dining philosophers; default eat gives +1 on first reaching any eating state. | [QVBS][qvbs] | T/S |
| [`qvbs_pnueli_zuck`](../src/mdpforge/models/discounted/qvbs_pnueli_zuck.py) | Process phases; scheduling choices | Pnueli-Zuck mutual exclusion; default live gives +1 on first reaching its selected process predicate. | [QVBS][qvbs] | T/S/M |
| [`qvbs_rabin`](../src/mdpforge/models/discounted/qvbs_rabin.py) | Shared counters and local process/draw variables; scheduling choices | Rabin mutual exclusion; default live gives +1 on first critical-section entry by any process. | [QVBS][qvbs] | T/S/M |
| [`qvbs_wlan`](../src/mdpforge/models/discounted/qvbs_wlan.py) | Medium/station phases, clocks and backoff; local choices | IEEE 802.11; default cost_min negates native transmission cost, while other objectives select time/collisions. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_wlan_dl`](../src/mdpforge/models/discounted/qvbs_wlan_dl.py) | Station phases, clocks, backoff and timer; local choices | Timed WLAN source; default deadline penalizes first reaching its transmission-completion predicate (Pmin). | [QVBS][qvbs] | T/S/M |
| [`qvbs_zeroconf`](../src/mdpforge/models/discounted/qvbs_zeroconf.py) | Host/environment phases, buffers, probes and clocks; local choices | IPv4 address configuration; default correct_max gives +1 on first reaching correct configuration. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_zeroconf_dl`](../src/mdpforge/models/discounted/qvbs_zeroconf_dl.py) | Address-configuration variables plus timer; local choices | Timed Zeroconf source; default deadline_max rewards reaching the deadline under the source continuation condition. | [QVBS][qvbs] | T/S/M/L |
| [`wlan`](../src/mdpforge/models/discounted/wlan.py) | Medium/station phases, clocks and backoff counters; local choices | Native PRISM wireless LAN; negative time, cost or collision count until completion. | [PRISM benchmarks][prism] | T/S/M/L |
| [`zeroconf`](../src/mdpforge/models/discounted/zeroconf.py) | Host/environment phases, buffers and clocks; local choices | Native PRISM address configuration; added unit time cost, negated until completion. | [PRISM benchmarks][prism] | T/S/M/L |

### Service operations

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`access_control`](../src/mdpforge/models/discounted/access_control.py) | Free servers and incoming priority; accept/reject | Independent service completions and fresh priorities; reward for accepted requests. | Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Example 10.2. | T/S/M/L |
| [`admission`](../src/mdpforge/models/discounted/admission.py) | Occupancy, pending class, service phase; accept/reject | Continuous-time rates converted to discrete-time probabilities by uniformization; negative rejection and holding costs. | Wieczorek, Busic, and Hyon: admission-control / critical-level-policy model, adapted from pyAdmissionControlOF.py (incomplete attribution). | T/S/M/L |
| [`ambulance`](../src/mdpforge/models/discounted/ambulance.py) | Location bin; positioning target | Identical beta-distributed next-location law for all states/actions; reward 1 minus normalized location distance. | Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. PNAS 109(3):764-769. | T/S/M/L |
| [`ambulance_relocation`](../src/mdpforge/models/discounted/ambulance_relocation.py) | Ambulance location; relocation target | Next state is a request location from a mixture distribution; negative expected response and relocation costs. | Sinclair, Banerjee & Yu (2019), Adaptive Discretization for Episodic RL. | T/S/M/L |
| [`boxworld`](../src/mdpforge/models/discounted/boxworld.py) | Truck position, crate position, loaded flag and terminals; five driving/loading actions | Reduced one-crate road model with traffic delays and cargo damage; movement costs and delivery reward. | Younes et al., JAIR 24 (2005), IPPC 2004 Boxworld: [source](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/) | T/S/M/L |
| [`elevator`](../src/mdpforge/models/discounted/elevator.py) | Floor, direction and request masks; move up/down, pick up either direction, wait | Finite request set with no new calls; waiting/invalid-action costs. Truncated spaces replace excluded outcomes with penalized self-loops. | Tartan, E. O. and Ciflikli, C. (2023). Sequential Decision Making for Elevator Control. | T/S/M/L |
| [`impatience`](../src/mdpforge/models/discounted/impatience.py) | Queue length; batch admitted to service | Binomial retention plus Poisson arrivals with capped, tail-trimmed kernel; negative batch-service and remaining-customer costs. | Hyon & Jean-Marie (2012). Scheduling Services in a Queuing System with Impatience and Setup Costs. The Computer Journal 55(5). | T/S/M/L |
| [`queue_network`](../src/mdpforge/models/discounted/queue_network.py) | Three bounded queue lengths; choose a queue to serve | Random arrivals, service and routing; negative weighted queue lengths and service-choice cost. | General reference: Puterman (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. | T/S/M/L |
| [`qvbs_boxworld`](../src/mdpforge/models/discounted/qvbs_boxworld.py) | Box/vehicle configuration; enabled planning choices | Configurable boxes and vehicles constructed from original PPDDL rules; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_eajs`](../src/mdpforge/models/discounted/qvbs_eajs.py) | Process progress, resources and battery load; local choices | Energy-aware job scheduling; default ExpUtil retains native on-time task utility. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_elevators`](../src/mdpforge/models/discounted/qvbs_elevators.py) | Finite-domain elevator/passenger configuration; planning choices | Published elevator instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S/M/L |
| [`qvbs_zenotravel`](../src/mdpforge/models/discounted/qvbs_zenotravel.py) | Finite-domain aircraft/passenger/fuel configuration; planning choices | Published Zeno-Travel source; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S |
| [`tandem`](../src/mdpforge/models/discounted/tandem.py) | Two queue lengths and two service capacities; nine capacity adjustments | Uniformized arrival/transfer/departure events; negative holding, staffing, switching and rejection costs. | Ohno & Ichiki (1987). Computing Optimal Policies for Controlled Tandem Queueing Systems. Operations Research 35(1), 121-126. [source](https://doi.org/10.1287/opre.35.1.121) | T/S/M/L |
| [`tandem_choice`](../src/mdpforge/models/discounted/tandem_choice.py) | Two queue lengths and two service capacities; choose absolute capacity pair | Uniformized tandem queues with absolute capacity choices; negative holding, staffing, switching and rejection costs. | Family reference: Ohno & Ichiki (1987). Computing Optimal Policies for Controlled Tandem Queueing Systems. Operations Research 35(1), 121-126. [source](https://doi.org/10.1287/opre.35.1.121) | T/S/M/L |
| [`taxi`](../src/mdpforge/models/discounted/taxi.py) | Grid position, passenger position, destination and terminal; four moves, pickup, dropoff | Open-grid corner landmarks and deterministic transitions; rewards are 0.2 for valid pickup, 1 for delivery and 1 at the absorbing delivered state, 0 otherwise. | Dietterich, T. G. (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition. JAIR 13:227-303. | T/S/M/L |
| [`traffic_control`](../src/mdpforge/models/discounted/traffic_control.py) | Two queue lengths and light phase; maintain/switch green | Independent random arrivals; switching causes one all-red slot; negative queue lengths and switching cost. | IPPC 2011 Traffic Control RDDL benchmark (inspiration): [source](https://users.cecs.anu.edu.au/~ssanner/IPPC_2011/) | T/S/M/L |
| [`zenotravel`](../src/mdpforge/models/discounted/zenotravel.py) | City, fuel, onboard flag and terminals; six flight/boarding/refuelling/delivery actions | Reduced one-aircraft/passenger model with delays and accidents; delivery payoff minus fuel/action/failure costs. | Younes et al., JAIR 24 (2005), IPPC 2004 Zeno Travel: [source](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/) | T/S/M/L |

### Asset management

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`car_rental`](../src/mdpforge/models/discounted/car_rental.py) | Car counts at two locations; overnight transfer quantity | Poisson rental requests and returns; expected rental income minus transfer cost. | Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Example 4.2. | T/S/M/L |
| [`dam`](../src/mdpforge/models/discounted/dam.py) | Reservoir volumes and outgoing flows; release vector | Coupled dams with random natural inflow; turbined-water reward multiplied by seeded random coefficients. | Carpentier, P., Chancelier, J.-P., Leclere, V. and Pacaud, F. (2018). Stochastic decomposition applied to large-scale hydro valleys management. EJOR 270(3):1086-1098. | T/S/M/L |
| [`forest`](../src/mdpforge/models/discounted/forest.py) | Forest age; wait/cut | Aging, fire risk and reset after harvest; waiting and cutting rewards. | Family reference: Chadès et al. (2014). MDPtoolbox: a multi-platform toolbox to solve stochastic dynamic programming problems. | T/S/M/L |
| [`inventory`](../src/mdpforge/models/discounted/inventory.py) | Stock index; order quantity | Binomial demand and capped replenishment; positive holding/shortage/order cost values are stored as rewards in this implementation. | Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed. | T/S/M/L |
| [`inventory_leadtime`](../src/mdpforge/models/discounted/inventory_leadtime.py) | Stock and pending delivery; order quantity | One-period delivery delay and discrete random demand; negative ordering, holding and shortage costs. | General reference: Puterman (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. | T/S/M/L |
| [`maintenance`](../src/mdpforge/models/discounted/maintenance.py) | Machine condition; do nothing, inspect, repair, replace | Stochastic deterioration, partial repair and replacement reset; operating income minus intervention/failure costs. | General reference: Puterman (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. | T/S/M/L |
| [`oil`](../src/mdpforge/models/discounted/oil.py) | Survey-location index; exploration location | Action index becomes the next state; nonnegative survey value minus normalized movement distance. | Mason, W. A. and Watts, D. J. (2012). Collaborative learning in networks. PNAS 109(3): 764–769. | T/S/M/L |
| [`oil_discovery`](../src/mdpforge/models/discounted/oil_discovery.py) | Reservoir-location grid; drilling location | Truncated Gaussian transition kernel drifting toward the selected location; discovery payoff minus drilling/movement costs. | Sinclair, Banerjee & Yu (2019), Adaptive Discretization for Episodic Reinforcement Learning. | T/S/M/L |
| [`option_pricing`](../src/mdpforge/models/discounted/option_pricing.py) | Time and price grid plus exercise terminal; hold/exercise | Binomial price moves over a finite horizon; American put payoff on exercise or maturity. | Family reference: Cox, Ross & Rubinstein (1979). Option Pricing: A Simplified Approach. | T/S/M/L |
| [`replacement`](../src/mdpforge/models/discounted/replacement.py) | Equipment age/condition index; keep or replacement option | Age- and action-dependent survival/failure transitions; resale value minus purchase and maintenance costs. | Howard, R. A. (1960). Dynamic Programming and Markov Processes, p.54, The Replacement Problem. | T/S/M/L |
| [`sysadmin`](../src/mdpforge/models/discounted/sysadmin.py) | Machine-availability bit mask on a ring; choose a machine to reboot | Neighbor-dependent failure/repair weights, at most one flip per step; operational-machine count minus reboot cost. | Family reference: Guestrin, Koller, Parr & Venkataraman (2003). Efficient Solution Algorithms for Factored MDPs. | T/S/M/L |

### Games and puzzles

| Model | Structure: states; actions | Dynamics and rewards | Source or reference | Presets |
| --- | --- | --- | --- | --- |
| [`blocks_world`](../src/mdpforge/models/discounted/blocks_world.py) | Focus, covers, held block, completion flag and puzzle level; six manipulation actions | Deterministic puzzles connected sequentially; intermediate manipulation rewards and level-completion bonuses. Scaling adds levels, not blocks per puzzle. | McCallum (1996), Whitehead's Blocks World revisited. | T/S/M/L |
| [`gambler`](../src/mdpforge/models/discounted/gambler.py) | Capital; stake amount | Win/loss with clipped stakes, absorbing ruin and target; expected unit reward on reaching the target. | Sutton & Barto, Reinforcement Learning: An Introduction, Example 4.3, Gambler's Problem. | T/S/M/L |
| [`hexagonal_grid_soccer`](../src/mdpforge/models/discounted/hexagonal_grid_soccer.py) | Player, ball and defender cells; six move/kick directions | Deterministic greedy defender in a single-agent reduction; movement, capture and scoring rewards. | Continuous U-Tree, Uther & Veloso (1998), enlarged hexagonal-grid soccer; inspired by Littman (1994) Markov games.  Single-agent tabular reduction. | T/S/M/L |
| [`pacman`](../src/mdpforge/models/discounted/pacman.py) | Player/ghost cells, two-pellet mask and terminals; four moves | Reduced model with random neighboring ghost moves; pellet/win rewards, movement cost and capture penalty. | QVBS Pacman benchmark: [source](https://qcomp.org/benchmarks/) | T/S/M/L |
| [`peg_solitaire`](../src/mdpforge/models/discounted/peg_solitaire.py) | Indexed subset of board bit masks; starting cell and jump direction | Deterministic peg-removing jumps, invalid/excluded moves stay; unit state reward at the designated single-peg configuration. | Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed. | T/S/M/L |
| [`qvbs_blocksworld`](../src/mdpforge/models/discounted/qvbs_blocksworld.py) | Finite-domain block configuration; enabled planning choices | Published Blocksworld instance; default +1 first-hit bonus for goal. | [QVBS][qvbs] | T/S |
| [`qvbs_exploding_blocksworld`](../src/mdpforge/models/discounted/qvbs_exploding_blocksworld.py) | Finite-domain block/explosion configuration; planning choices | Published Exploding Blocksworld instance; default +1 first-hit goal bonus. | [QVBS][qvbs] | T/S |
| [`qvbs_pacman`](../src/mdpforge/models/discounted/qvbs_pacman.py) | Player/ghost positions and directions, arbiter state; local choices | Published Pacman source; default crash penalizes first collision (Pmin), unlike the reduced local pellet task. | [QVBS][qvbs] | T/S |

## Total-reward recipes

Use `Benchmark(criterion="total")` to select these separate implementations. Except for `progress_chain`, they adapt a local discounted model above: state/action structure and non-goal dynamics are retained, while success goals absorb where needed. The module links specify each adaptation; source references are inherited from the corresponding local task.

For goal-based recipes, a nonterminal action receives reward **1 if it has positive probability of entering a success goal**, otherwise 0. All terminal states receive 0. This is an action reward, not the probability of success; stochastic goal-access actions can be rewarded more than once before absorption. Failure terminals receive no success reward. No new time limit is added; `search_and_rescue` retains its existing urgency budget. `progress_chain` instead uses negative step rewards.

Finite total reward does not imply that the current validator can certify the model: the certificate covers absorbing models with no cycles among nonterminal states. Many recipes here contain cycles or self-loops. See [solution-quality checks](../README.md#comparing-solution-quality).

### Abstract (total)

| Model | Success goal and adaptation | Source or discounted structure | Presets |
| --- | --- | --- | --- |
| [`progress_chain`](../src/mdpforge/models/total/progress_chain.py) | Original model: line position, advance by one/two; -1 per nonterminal action, 0 at the absorbing end; every policy finishes in at most S-1 steps. | Original MDPForge model; no external reference. | T/S/M/L |
| [`random_walk`](../src/mdpforge/models/total/random_walk.py) | Right boundary is success; left boundary remains absorbing failure; goal-access reward replaces signed entry rewards. | [`random_walk`](../src/mdpforge/models/discounted/random_walk.py) | T/S/M/L |

### Navigation and control (total)

| Model | Success goal and adaptation | Source or discounted structure | Presets |
| --- | --- | --- | --- |
| [`acrobot`](../src/mdpforge/models/total/acrobot.py) | Absorbing height target; pendulum dynamics unchanged. | [`acrobot`](../src/mdpforge/models/discounted/acrobot.py) | T/S/M/L |
| [`barto`](../src/mdpforge/models/total/barto.py) | Absorbing finish cells; acceleration failures and crash resets retained. | [`barto`](../src/mdpforge/models/discounted/barto.py) | T/S/M/L |
| [`cliff`](../src/mdpforge/models/total/cliff.py) | Absorbing exit; unsafe moves still reset to the start. | [`cliff`](../src/mdpforge/models/discounted/cliff.py) | T/S/M/L |
| [`deep_sea_treasure`](../src/mdpforge/models/total/deep_sea_treasure.py) | Treasure collected is success; drowning remains absorbing failure. | [`deep_sea_treasure`](../src/mdpforge/models/discounted/deep_sea_treasure.py) | T/S/M/L |
| [`drone_energy`](../src/mdpforge/models/total/drone_energy.py) | Landed destination is success; crash remains absorbing failure. | [`drone_energy`](../src/mdpforge/models/discounted/drone_energy.py) | T/S/M/L |
| [`frozen`](../src/mdpforge/models/total/frozen.py) | Lake goal is success; holes remain absorbing failures. | [`frozen`](../src/mdpforge/models/discounted/frozen.py) | T/S/M/L |
| [`local_perception_grid_world`](../src/mdpforge/models/total/local_perception_grid_world.py) | Absorbing maze goal; local-wall observations retained. | [`local_perception_grid_world`](../src/mdpforge/models/discounted/local_perception_grid_world.py) | T/S/M/L |
| [`m_maze`](../src/mdpforge/models/total/m_maze.py) | Absorbing maze goal; uncertain movement and observation aliasing retained. | [`m_maze`](../src/mdpforge/models/discounted/m_maze.py) | T/S/M/L |
| [`maze_backtrack`](../src/mdpforge/models/total/maze_backtrack.py) | Exit becomes absorbing; backtracking-generated corridors retained. | [`maze_backtrack`](../src/mdpforge/models/discounted/maze_backtrack.py) | T/S/M/L |
| [`maze_prims`](../src/mdpforge/models/total/maze_prims.py) | Exit becomes absorbing; Prim-generated corridors retained. | [`maze_prims`](../src/mdpforge/models/discounted/maze_prims.py) | T/S/M/L |
| [`maze_wilson`](../src/mdpforge/models/total/maze_wilson.py) | Exit becomes absorbing; Wilson-generated corridors retained. | [`maze_wilson`](../src/mdpforge/models/discounted/maze_wilson.py) | T/S/M/L |
| [`mountain`](../src/mdpforge/models/total/mountain.py) | Hilltop states become absorbing instead of resetting to the start. | [`mountain`](../src/mdpforge/models/discounted/mountain.py) | T/S/M/L |
| [`ninerooms`](../src/mdpforge/models/total/ninerooms.py) | Absorbing nine-room exit; movement uncertainty retained. | [`ninerooms`](../src/mdpforge/models/discounted/ninerooms.py) | T/S/M/L |
| [`office_world`](../src/mdpforge/models/total/office_world.py) | Absorbing office with mail and coffee collected; intermediate penalties/bonuses removed. | [`office_world`](../src/mdpforge/models/discounted/office_world.py) | T/S/M/L |
| [`rooms`](../src/mdpforge/models/total/rooms.py) | Absorbing four-room exit; movement uncertainty retained. | [`rooms`](../src/mdpforge/models/discounted/rooms.py) | T/S/M/L |
| [`schoolboy`](../src/mdpforge/models/total/schoolboy.py) | Absorbing school; walking/cycling/bus progress distributions retained. | [`schoolboy`](../src/mdpforge/models/discounted/schoolboy.py) | T/S/M/L |
| [`search_and_rescue`](../src/mdpforge/models/total/search_and_rescue.py) | Rescue is success; loss remains absorbing failure; the existing urgency budget is retained. | [`search_and_rescue`](../src/mdpforge/models/discounted/search_and_rescue.py) | T/S/M/L |
| [`sutton`](../src/mdpforge/models/total/sutton.py) | Absorbing finish; acceleration failures and crash resets retained. | [`sutton`](../src/mdpforge/models/discounted/sutton.py) | T/S/M/L |
| [`tireworld`](../src/mdpforge/models/total/tireworld.py) | Destination is success; stranding remains absorbing failure. | [`tireworld`](../src/mdpforge/models/discounted/tireworld.py) | T/S/M/L |
| [`windgrid`](../src/mdpforge/models/total/windgrid.py) | Absorbing grid goal; column wind retained. | [`windgrid`](../src/mdpforge/models/discounted/windgrid.py) | T/S/M/L |
| [`wumpus_world`](../src/mdpforge/models/total/wumpus_world.py) | Exit with gold is success; death and empty-handed exit remain terminal failures. | [`wumpus_world`](../src/mdpforge/models/discounted/wumpus_world.py) | T/S/M/L |

### Service operations (total)

| Model | Success goal and adaptation | Source or discounted structure | Presets |
| --- | --- | --- | --- |
| [`boxworld`](../src/mdpforge/models/total/boxworld.py) | Delivered crate is success; cargo damage remains an absorbing failure. | [`boxworld`](../src/mdpforge/models/discounted/boxworld.py) | T/S/M/L |
| [`taxi`](../src/mdpforge/models/total/taxi.py) | Final delivery is rewarded; pickup bonus and continuing terminal reward removed. | [`taxi`](../src/mdpforge/models/discounted/taxi.py) | T/S/M/L |
| [`zenotravel`](../src/mdpforge/models/total/zenotravel.py) | Delivered passenger is success; accident remains absorbing failure. | [`zenotravel`](../src/mdpforge/models/discounted/zenotravel.py) | T/S/M/L |

### Asset management (total)

| Model | Success goal and adaptation | Source or discounted structure | Presets |
| --- | --- | --- | --- |
| [`sysadmin`](../src/mdpforge/models/total/sysadmin.py) | All-up machine state becomes absorbing instead of allowing further failures. | [`sysadmin`](../src/mdpforge/models/discounted/sysadmin.py) | T/S/M/L |

### Games and puzzles (total)

| Model | Success goal and adaptation | Source or discounted structure | Presets |
| --- | --- | --- | --- |
| [`blocks_world`](../src/mdpforge/models/total/blocks_world.py) | Only final puzzle completion is rewarded; intermediate manipulation bonuses removed. | [`blocks_world`](../src/mdpforge/models/discounted/blocks_world.py) | T/S/M/L |
| [`gambler`](../src/mdpforge/models/total/gambler.py) | Target capital is success; ruin remains absorbing failure; goal-access reward is 1 rather than p_heads. | [`gambler`](../src/mdpforge/models/discounted/gambler.py) | T/S/M/L |
| [`hexagonal_grid_soccer`](../src/mdpforge/models/total/hexagonal_grid_soccer.py) | Right-goal ball configurations are absorbing successes; defender dynamics retained. | [`hexagonal_grid_soccer`](../src/mdpforge/models/discounted/hexagonal_grid_soccer.py) | T/S/M/L |
| [`pacman`](../src/mdpforge/models/total/pacman.py) | Both pellets collected is success; capture remains absorbing failure; goal-access reward is not a win probability. | [`pacman`](../src/mdpforge/models/discounted/pacman.py) | T/S/M/L |
| [`peg_solitaire`](../src/mdpforge/models/total/peg_solitaire.py) | Designated single-peg goal; reward moves from the goal state to actions entering it. | [`peg_solitaire`](../src/mdpforge/models/discounted/peg_solitaire.py) | T/S/M/L |

## Imported sources not yet available

These entries have `available=False` and cannot be benchmarked. Their sources are bundled, but their stochastic timed automata (STA) or Markov automata (MA) require an explicit finite discrete-time MDP adaptation. They have no available size preset.

| Entry | Source structure | Why unavailable | Source |
| --- | --- | --- | --- |
| [`jani_3_tandem_queue`](../src/mdpforge/models/discounted/jani_3_tandem_queue.py) | Three sequential queues (STA) | Source type sta requires a finite discrete-time MDP adaptation before benchmarking. | [jani-models][jani] |
| [`jani_database`](../src/mdpforge/models/discounted/jani_database.py) | Database repairman model (STA) | Source type sta requires a finite discrete-time MDP adaptation before benchmarking. | [jani-models][jani] |
| [`jani_queue_with_breakdowns`](../src/mdpforge/models/discounted/jani_queue_with_breakdowns.py) | Queue with failures and repairs (STA) | Source type sta requires a finite discrete-time MDP adaptation before benchmarking. | [jani-models][jani] |
| [`jani_queues_and_workflows`](../src/mdpforge/models/discounted/jani_queues_and_workflows.py) | Two parallel servers (MA) | Source type ma requires a finite discrete-time MDP adaptation before benchmarking. | [jani-models][jani] |
| [`jani_tandem_queue`](../src/mdpforge/models/discounted/jani_tandem_queue.py) | Two sequential queues (STA) | Source type sta requires a finite discrete-time MDP adaptation before benchmarking. | [jani-models][jani] |

## Browse from Python

```python
from mdpforge import list_models
from mdpforge.models.total import list_models as list_total_models

discounted = list_models(available=True)
small_navigation = list_models(category="navigation", size="small", available=True)
total = list_total_models(available=True)
unavailable = list_models(available=False)

for entry in discounted:
    print(entry["name"], entry["description"], entry["reference"])
```

Listing reads metadata without constructing transition matrices. To select a recipe, use `bench.add_mdp("rooms", size="small")` on a benchmark configured for the desired reward criterion. See [adding a model](adding-a-model.md) for the interface and metadata requirements. When adding or changing a recipe, update its row here together with its metadata.

[jani]: ../src/mdpforge/models/discounted/_jani/_data/model_manifest.json
[qvbs]: ../src/mdpforge/models/discounted/_qvbs/_data/model_manifest.json
[prism]: ../src/mdpforge/models/discounted/_prism/_data/manifest.json
