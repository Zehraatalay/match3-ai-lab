# Stochastic open-loop MCTS

`StochasticMCTSAgent` implements the same `select_action(observation)` interface
as Random and Greedy. Its `MCTSSimulator` must reference the current, reset live
environment. The benchmark handles this with lazy, environment-aware binding;
the shared episode runner does not need a special MCTS execution path.

## Search and stochastic futures

The tree stores action histories, not one supposedly deterministic board for
each action. Every simulation deep-copies the current environment and replaces
the copied engine RNG with a seed drawn from the agent's independent RNG.
Different simulations can therefore encounter different boards and available
actions at the same history. Selection and expansion consider only actions
legal in that simulation. Newly encountered legal actions get new children;
unavailable children are ignored. There is no transposition table or cross-state
cache. A new tree is built for each real decision.

After trying unexpanded actions, selection maximizes
`mean_value + c * sqrt(log(parent_visits) / child_visits)`, with `c=sqrt(2)`.
Unvisited nodes have infinite UCT. Ties during selection follow ascending legal
action order. The final root decision maximizes visits, then mean value, then
prefers the lowest action ID. Each simulation backs up its value once per node
on its selected path. Values are never negated: this is a single-player task.

The actual engine handles matches, specials, blockers, cascades, recovery, and
move limits. Rollouts stop on termination or truncation; a simulated state with
no legal actions receives a loss value. The real engine RNG is not consumed or
used to generate the simulated future.
The benchmark checks complete state fingerprints around each MCTS decision,
including observations, objective state, moves, and both engine/Gym RNG states.

The tree pools values from stochastic histories. UCT is not availability-adjusted
and is not a convergence guarantee for this implementation. Finite budgets can
leave root actions unexplored. The agent assumes the supplied observation and
simulator describe the same current state.

## Configuration and integration

The constructor accepts the following parameters:

| Parameter | Agent default | Supported values |
|---|---|---|
| `simulator` | Required | `MCTSSimulator` bound to the current reset environment |
| `simulations` | 100 | Positive integer, excluding booleans |
| `exploration_constant` | `sqrt(2)` | Finite non-negative real number, excluding booleans |
| `rollout_policy` | `greedy` | `random`, `greedy`, `objective` |
| `value_mode` | `terminal` | `terminal`, `outcome` |
| `seed` | `None` | Seed for the independent agent RNG |

The benchmark defaults differ: 20 episodes per agent per level and 30
simulations per decision, with Random, Greedy, RT, and GO selected. Its
`AGENT_REGISTRY` additionally exposes RO and GT through `--agents`. There is
no project-wide registry that automatically makes MCTS available in every CLI;
use the MCTS benchmark entry point below.

The observation supplies `action_mask`, whose nonzero indices are legal swap
actions. The simulator copies the full environment, including board, specials,
blockers, objectives, moves, and RNG state. Real and simulated moves use the
same environment `step` implementation; search does not approximate game rules.
The agent reads each returned simulated observation for the next legal actions.
`last_search_stats` exposes simulation count, root visits, maximum tree depth,
node count, and per-root-action visits and mean values.

The earlier `MonteCarloAgent` in `agents/mcts_agent.py` is a separate root-action
rollout baseline. It evaluates each initial action using cloned environments;
it does not implement this open-loop UCT tree or resample future engine seeds.
The benchmark configurations discussed here all use `StochasticMCTSAgent`.

Random and Greedy need no search or training. DDQN and PPO learn network-based
policies in advance; MCTS instead spends computation during each real decision
and requires a simulator of the current environment. All share the
`select_action(observation)` interface.

## Rollouts and values

| Configuration | Rollout | Value |
|---|---|---|
| MCTS-RT / `MCTS_random_terminal` | Uniform legal action | Terminal |
| MCTS-RO / `MCTS_random_outcome` | Uniform legal action | Outcome |
| MCTS-GT / `MCTS_greedy_terminal` | Existing GreedyAgent | Terminal |
| MCTS-GO / `MCTS_greedy_outcome` | Existing GreedyAgent | Outcome |

Greedy scores immediate horizontal/vertical matches and breaks ties by action
ID. It intentionally does not score blockers, special effects, or objectives.
The existing objective rollout remains available to direct agent users, but is
outside this four-way ablation. No new rollout policy was added.

Terminal values are +1 for a win, -1 for a loss, and 0 for zero terminal reward.
With default weights, outcome values are:

- win: `1 + 0.2 * moves_remaining / move_limit`, bounded by [1, 1.2]
- loss: `-1 + 0.2 * mean_objective_completion`, bounded by [-1, -0.8]
- zero terminal reward: 0

Completion ratios are clipped to [0, 1]. UCT's exploration constant remains
fixed across value modes; the small value-scale difference is part of the
comparison. Simulation budgets must be positive integers excluding booleans;
exploration constants must be finite, non-negative real numbers excluding
booleans. The objective rollout uses one-step objective lookahead with up to five
candidates and is more costly; it is not the standalone Greedy policy.

## Reproducibility and persistence

All configurations receive identical environment seeds within a level. Each
configuration has an independent deterministic agent seed. Legacy Random,
Greedy, RT, and GO retain their four-slot seed mapping regardless of which
configurations are selected. RO/GT use disjoint additional slots. Selection
order does not change seeds. RNG draw counts can diverge between configurations
and budgets; common environment seeds do not imply common simulated futures.

The benchmark remains sequential. It writes `episodes.csv` incrementally with
flush/fsync, atomically updates `summary.csv` and `manifest.json`, and rejects
existing output directories. An interrupted in-flight episode has no returned
record; its identity remains in the manifest. Previously persisted records
survive interruption. A hard process kill can leave the last manifest status
as running; this is not automatic checkpoint/resume support.

Win rates and Wilson 95% intervals use all returned records, including errors
and time limits. Objective means use available values, with sample counts.
Winning moves use wins only; no wins means unavailable, not zero. Decision
latency is total agent selection time divided by actual calls, including calls
that raise. It excludes lazy construction and integrity checks. Episode time
includes reset and checks, but excludes export and closing the environment.

## Historical compatibility and timing versions

The completed Sprint 3 run is `results/mcts_baselines_main_100`: 100 boards per
level, 30 simulations, environment seeds 10001000–10001099, agent seed start
20010000. Observed win rates were:

| Level | Random | Greedy | RT | GO |
|---|---:|---:|---:|---:|
| double-trouble-v2 | 52% | 55% | 91% | 93% |
| wooden-trouble-v2 | 41% | 37% | 95% | 93% |
| frozen-garden-v2 | 52% | 64% | 94% | 95% |

These are development results on three levels, with one seeded agent stream
per board. RT versus GO changes both rollout and value; that comparison alone
cannot isolate either effect. A few percentage points are not conclusive
evidence of superiority. RL tables elsewhere use different evaluation settings
and should not be combined with this table as a common head-to-head study.

Sprint 4's `mcts_provenance.py` pins the audited legacy package/CLI hashes and
the current critical source hashes. The original package matched the legacy
manifest before changes. Pre-change reference traces are saved in
`tests/fixtures/mcts_sprint3_reference.json`; regression tests compare selected
actions and complete search statistics on all three levels and four modes.

Observation reuse avoids reconstructing a simulated observation after every
transition and again for Greedy. It reuses only the observation returned by
that same simulated `step`; it is not cross-state caching. All reference traces
remained equivalent, with no live-state mutation. Local three-sample timings
are recorded in `mcts_sprint4_validation.json`: median speedups were roughly
1.13–1.18x for random rollouts and 1.26–1.32x for greedy rollouts. These are small,
fixed-order measurements, not a hardware-controlled performance experiment.

The semantic version remains `open-loop-uct-v1`. Timings are labeled
`observations-recomputed-v1` versus `step-observation-reuse-v1`. Reuse validates
complete coverage, configuration options, critical source compatibility, library
versions, level hashes, agent/environment seeds, and reconstructed initial
fingerprints. It rejects duplicate, incomplete, or incompatible records before
new gameplay starts. Imported rows retain origin IDs and source-file hashes;
historical files are never edited. Further critical-source changes require an
explicit compatibility audit rather than blindly accepting a version label.

Outcome comparisons can reuse the audited RT/GO results. Timing contrasts
between the old RT/GO implementation and optimized RO/GT are withheld. Per-agent
timings remain visible with version labels. A controlled optimized timing study
would require fresh RT/GO timing runs after separate approval.

## Completed four-way development ablation

The saved [summary](../results/mcts_ablation_100/summary.csv) contains 100 paired
boards per level at 30 simulations, using the same seed starts as Sprint 3:

| Level | RT | RO | GT | GO |
|---|---:|---:|---:|---:|
| double-trouble-v2 | 91% | 93% | 91% | 93% |
| wooden-trouble-v2 | 95% | 92% | 94% | 93% |
| frozen-garden-v2 | 94% | 97% | 98% | 95% |

The completed manifest records 1,200 episodes: 600 audited historical RT/GO
records and 600 new RO/GT records, with no recorded errors or time limits.
Wilson intervals, objective completion, winning moves, and version-labeled
timings are in the summary. Paired contrasts are saved in
[paired_comparisons.csv](../results/mcts_ablation_analysis_100/paired_comparisons.csv);
the analysis manifest records 10,000 bootstrap samples and seed 42. No mode
has the highest observed win rate on every level. These are exploratory
development results, not evidence of universal superiority or human difficulty.

The result artifacts must accompany the source when sharing these numbers;
they are not regenerated by the test suite.

## Four-way experiment commands

From the repository root, validate reuse without playing any episodes:

```bash
python scripts/benchmark_mcts_baselines.py \
  --episodes 100 --simulations 30 \
  --env-seed-start 10001000 --agent-seed-start 20010000 \
  --agents MCTS_random_terminal MCTS_random_outcome MCTS_greedy_terminal MCTS_greedy_outcome \
  --reuse-from results/mcts_baselines_main_100 --validate-reuse-only
```

Validation-only checks the 1,200 original records and reports 600 reusable
RT/GO records without importing or playing episodes. The actual command below
imports those records and runs 600 new RO/GT episodes in a new directory:

```bash
python scripts/benchmark_mcts_baselines.py \
  --episodes 100 --simulations 30 \
  --env-seed-start 10001000 --agent-seed-start 20010000 \
  --agents MCTS_random_terminal MCTS_random_outcome MCTS_greedy_terminal MCTS_greedy_outcome \
  --reuse-from results/mcts_baselines_main_100 \
  --output-dir results/mcts_ablation_reproduction_100

python scripts/analyze_mcts_ablation.py \
  --input-dir results/mcts_ablation_reproduction_100 --output-dir results/mcts_ablation_reproduction_100_analysis \
  --bootstrap-samples 10000 --bootstrap-seed 42
```

The analyzer aligns whole boards and reports GT−RT, GO−RO, RO−RT, GO−GT,
and the interaction `(GO−GT)−(RO−RT)`. It reports paired sample sizes, excluded
pairs, discordant wins for binary contrasts, signed board counts, and pointwise
paired bootstrap intervals using a fixed RNG seed. Error/missing outcomes are
excluded explicitly rather than fabricated. No significance tests are performed;
intervals are exploratory and not multiplicity-adjusted. Winning-move summaries
are conditional on each agent's wins and are not treated as an unconditional
paired outcome measure.

Budgets 10, 20, and 30 already work through `--simulations`. Run each in a
separate new directory; do not reuse 30-simulation rows at another budget.
These runs are manual and are not launched by tests or analysis commands.

## Other inspection tools

```bash
python scripts/benchmark_mcts_baselines.py --help
python scripts/analyze_mcts_ablation.py --help
python scripts/inspect_mcts_search.py --level frozen-garden-v2 --seed 4 \
  --simulations 10 --value-mode terminal
```

The inspection script makes one decision for **each** of random, greedy, and
objective rollouts and prints search statistics; it does not run real episodes.
The objective rollout can make it slow. The older `benchmark_mcts_rollouts.py`
explores all three policies and both values; unlike the paired benchmark, it
writes a single CSV and can overwrite its output. Use the paired benchmark for
persisted, provenance-checked comparisons.

## Level evaluation and difficulty

Paired seeds separate initial-board variation from agent differences. Win rates,
objective completion, and moves used quantify how successfully a particular
agent plays a level under the recorded budget and seed distribution. A low win
rate indicates difficulty for that agent and protocol; it does not establish
human-perceived difficulty or a universal ranking of levels. Search budget and
rollout/value configuration are part of the measurement and must be reported.

The separate level-generation workflow's `evaluation/difficulty_evaluator.py`
can opt into `MCTS_greedy_outcome` alongside Random and Greedy; its default is
Random and Greedy only. It reuses the episode runner with lazy live-environment
binding. That PLG integration is separate from the core MCTS benchmark and
should be reviewed with the level-generation changes.

## Cost, validation, and limitations

The original baseline benchmark manifest records 54.7 minutes. The completed
four-way ablation records 44.7 minutes for importing RT/GO and running the 600
new RO/GT episodes. These are observations on the recorded machine, not runtime
guarantees. More simulations can change decisions and episode lengths, so cost
is not exactly linear and win rate need not increase. Cloning environments and
rolling out complete simulated games make MCTS substantially more expensive
per decision than a standalone Random or Greedy action.

Relevant tests cover constructor validation, UCT and ties, changing legal action
availability, expansion, exact backups, terminal handling, rollouts, values,
clone isolation, independent RNG streams, reference action/statistic equivalence,
paired benchmark metrics, and historical reuse. From the repository root:

```bash
python -m pytest -q tests/test_mcts_agent.py tests/test_mcts_simulator.py \
  tests/test_stochastic_mcts_agent.py tests/test_mcts_value.py \
  tests/test_objective_rollout.py tests/test_mcts_regression.py \
  tests/test_mcts_baseline_benchmark.py tests/test_mcts_ablation.py
```

Source compatibility is intentionally strict: edits to audited environment or
fingerprinting code can prevent benchmark/reuse execution until independently
reviewed. A matching version label alone is insufficient; do not replace hashes
merely to bypass that guard. Output directories must be new.

The agent has no parallel search, neural guidance, automatic tuning, or persistent
tree reuse. The shared agent interface and simulator allow future search agents
without speculative policy/value-network abstractions. The four-way comparison
uses one agent RNG stream per board, pools stochastic histories, and mixes timing
implementations as labeled above. No simulation-budget study or optimality claim
is established by these artifacts.
