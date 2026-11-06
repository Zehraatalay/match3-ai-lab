# Pygame UI

Install the optional UI dependency with `pip install -e ".[ui]"`.

Launch the default game with `python -m match3_ai.ui.pygame_view` or the
reproducible blocker demo with `python -m match3_ai.ui.pygame_view --demo-blockers`.

Launch level selection with `python -m match3_ai.ui.pygame_view --levels`, or
start a level directly with `python -m match3_ai.ui.pygame_view --level 3`.

Use the mouse to select and swap adjacent tiles. Press **R** to reset, **H** to
toggle move hints, **D** to show or hide blocker HP, and **Esc** to quit. In the
level menu, use **Up/Down** and **Enter**, number keys **1–6**, or click a level.
During play, **L** opens level selection. After winning, use **N** for the next
level; terminal screens also provide Restart/Retry, Next Level, and Level Select.

# Level evaluation

Run the paired Random/Greedy baseline benchmark with:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.evaluation.benchmark
```

The default configuration attempts 100 episodes per agent per level (1,200 total),
using environment seeds 52000–52099 and independent agent seeds 93000–93099.
Each episode has a 50-step bound. The experiment stops after two cascade-limit
errors or five minutes, checked between episodes. No failed episode is retried.
Use `--episodes`, `--env-seed`, `--agent-seed`, `--max-steps`, `--max-cascades`,
`--max-seconds`, and `--output-dir` to configure a run.

The engine, `LevelSession`, and `Match3LevelEnv` accept a positive integer
`max_cascades`. The default **200 is provisional** and should be reassessed using
benchmark distributions. The guard raises `CascadeLimitExceeded` with the limit,
processed cascades, and swap; the level environment adds level and episode context.
It never truncates a chain or retries with a larger budget.

Results are written to a new `experiments/results/<experiment_id>/` directory,
which is already Git-ignored. An existing output directory is never overwritten.
The files are `episodes.csv`, `moves.csv`, and `experiment_metadata.json`.
Nested CSV fields contain JSON; unavailable values are blank in CSV and null in
JSON. Experiment IDs include the configuration, level fingerprints, agent names
and their order, and a source fingerprint. Episode IDs also include the level,
agent, and episode index. Named agent factories must retain the same behavior
when reproducing a configuration.

`run_episode` shares its execution loop between the legacy and level environments.
Pass an `AnalyticsCollector` to opt into move records. `run_level_experiment`
accepts named seeded agent factories and explicit environment/agent seed tuples.
Legacy `run_experiment` and CSV APIs remain available.

Outcomes distinguish WON, LOST, ERROR, and TIME_LIMIT. Level move-budget exhaustion
is LOST; an external Gymnasium truncation or evaluator step bound is TIME_LIMIT.
Errors retain the attempted action, seed, exception, and available diagnostics.
On error, board/objective metrics describe the last completed transition; partial
resolution clears, damage, and activations are unavailable. Cascade totals include
processed phases reported by a cascade-limit exception. Moves used count consumed
level moves; move indices include invalid and failed attempts.

Objective completion is the mean of objective progress/target ratios, each capped
at one. Summary means use available values and include their sample counts. Win
rates use all attempted episodes as the denominator, including errors and time
limits, with Wilson 95% intervals. Inference time measures only `select_action`
and depends on hardware; gameplay traces are deterministic, timing is not.
These are descriptive agent comparisons, not estimates of human difficulty or
evidence of causality.

# Benchmark analysis

The optional analysis dependency is Matplotlib: install `.[analysis]` into the
analysis environment if needed. Pandas and SciPy are not required. Run:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.evaluation.analysis
```

Defaults read `experiments/results/6beb0e805eea3e9c25cc9de5/` and create a new
`experiments/results/sprint5_5_6beb0e805eea3e9c25cc9de5/` directory containing
`report.md`, `analysis.json`, and eight figures in both 300-dpi PNG and vector PDF.
Use `--input-dir` and `--output-dir` for another benchmark or a fresh output path.
Existing output paths are rejected; original benchmark inputs are hash-checked
and never rewritten.

The analysis validates typed CSV/JSON fields, missingness, IDs, move totals,
configuration identifiers and seed pairing. All attempts remain in descriptive
summaries; paired binary tests exclude only incomplete/error/time-limit pairs.
Comparisons use exact two-sided McNemar tests, Holm adjustments across six levels,
and paired percentile bootstrap intervals (20,000 draws; seed 55005 by default).
Use `--bootstrap-seed` and `--bootstrap-samples` to configure the bootstrap.
The report distinguishes all attempts, completed episodes, wins and losses and
records the nonmissing denominator for every numeric summary.

The initialization investigation replays First Match seed 52028 and samples
10,000 candidate boards without changing production generation rules. Tests and
quick analysis runs can use `--skip-initialization-investigation`; the full report
includes the diagnosis by default. No 1,200-episode benchmark is rerun by analysis.

# Initialization reliability and DQN design review

`Match3LevelEnv(max_initialization_attempts=1000)` retains the original
attempt budget. `LevelSession(max_initialization_attempts=1000)` and
`create_level_board(max_attempts=1000)` retain their prior defaults, as does the
legacy `Match3Env` initialization. The rejection algorithm and RNG streams are
unchanged. First Match now uses **five colors**, with all other level fields
preserved. A paired comparison on 100 seeds at budget 1000 found 99/100 successful
resets at three colors versus 100/100 at five; mean attempts fell from 150.15
(including the capped failure) to 6.26. The provisional budget increase to 5000
was therefore superseded. Exhaustion still raises `LevelGenerationError`.

The historical three-color seed 52028 regression first succeeds at attempt 1180
when explicitly given a higher budget; diagnostics freeze that old configuration.
**Previous First Match gameplay benchmarks are not directly comparable to the
new five-color configuration.** Sprint 5.6 initially kept target 300; Sprint 5.7
calibrates it to 30 while preserving five moves. Initialization checks alone do
not establish gameplay balance.

The level benchmark CLI also accepts `--max-initialization-attempts`; the value is
stored in experiment configuration/metadata. The historical Sprint 5.4 benchmark
used 1000 and is not rewritten. The initialization-only profile can be reproduced
into a new output directory with:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.evaluation.initialization_benchmark --output-dir <new-directory>
```

This profiles 100 deterministic seeds per level without running gameplay. It
reports success rate, actual candidate attempts and reset latency. A finite
budget and successful sample do not guarantee all future initializations succeed.

Reproduce the color comparison without modifying either configuration on disk:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.evaluation.initialization_benchmark --compare-first-match-colors --max-initialization-attempts 1000 --output-dir <new-directory>
```

The [Sprint 5.6 DQN architecture proposal](docs/sprint5_6_dqn_architecture.md)
documents dependencies, categorical encoding, MLP/CNN trade-offs, replay memory,
masked Bellman targets, seeds, training/evaluation stages and PPO compatibility.
It is retained as the historical design review. The approved implementation is
described in the [Sprint 5.7 DQN guide](docs/sprint5_7_dqn.md).

# Initial DQN

Install the optional `.[rl]` extra (tested with PyTorch 2.15.0). Legacy gameplay
and baseline agents do not require PyTorch. Run the bounded CPU smoke profile:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.dqn --level-id blocker-mayhem --steps 1000 --seed 5700 --output-dir <new-directory>
```

Use `--level-id first-match` for debugging, or configure hyperparameters through
`TrainingConfig`. The model encodes 1467 categorical/spatial/task features, outputs
144 Q-values, masks exploration and Bellman targets, and stores raw observations
in bounded uniform replay. Final full checkpoints support boundary resume;
periodic policy checkpoints support inference. Output and checkpoint paths are
never overwritten. Training metrics are separate from gameplay evaluation CSV.
The 1000-step smoke is a correctness check, not evidence of improved play; longer
training, PPO and reward shaping require separate approval.

# DQN pilot evaluation

Sprint 5.8 adds a bounded 5000-step Blocker Mayhem pilot with reserved training,
validation and final-test seed partitions. It retains the reviewed DQN defaults
and evaluates DQN/Random/Greedy on 100 paired validation seeds every 1000 steps.
The final-test partition is never used by the pilot.

See the [pilot protocol and configuration review](docs/sprint5_8_dqn_pilot.md)
for reproduction, metric definitions, analysis commands and interpretation limits.
The pilot and analysis commands both require fresh output directories. A longer
50000-step experiment requires separate approval.

# Multi-seed DQN learning evaluation

Sprint 5.9 runs the approved 50000-step profiles for master seeds 5901, 5902 and
5903. Each has step-zero validation, fixed held-out validation seeds and a fresh
model/replay/RNG setup. Milestone checkpoints and validation analytics are saved
independently; checkpoint publication is atomic and cannot overwrite prior files.

See the [multi-seed protocol](docs/sprint5_9_multi_seed_dqn.md) for exact settings,
commands, checkpoint continuation limits and correlated-validation caveats.
Final-test seeds, PPO and reward shaping remain outside this experiment.

# Frozen-agent behavior analysis

Sprint 5.10 compares real step-zero and trained DQN checkpoints, Random and
Greedy on paired validation episodes and identical shared observations. It adds
exact one-action comparisons using independent full environment/RNG clones,
source-stratified action agreement, paired statistics and measured reward sparsity.
It does not train models or modify gameplay. See the
[behavior-analysis protocol](docs/sprint5_10_agent_behavior.md) for commands,
dataset weighting, uncertainty methods and interpretation limits.

# Optional potential-based training rewards

Sparse rewards remain the default. Sprint 5.11 adds explicit
`TrainingConfig(reward_mode="pbrs", shaping_beta=0.5)` using normalized observable
objective progress, with true-terminal potential zero and truncation-aware
bootstrapping. Gameplay/evaluation rewards remain sparse. Separate logs retain
all reward components and sparse/shaped episode returns.

The [PBRS protocol and controlled pilot results](docs/sprint5_11_potential_reward_shaping.md)
document the telescoping assumptions, paired 5000-step experiment and eight
learning/reward figures. Longer training or beta tuning requires separate approval.

# Optional CNN DQN architecture

Sprint 5.12 adds `DQNAgent(architecture="cnn")` and
`TrainingConfig(architecture="cnn")`, with MLP remaining the default. The compact
spatial/global network receives exactly the original encoder features, shares
the existing masked DQN training pipeline, and has architecture-aware checkpoints
that retain legacy MLP loading.

See the [CNN design and controlled pilot](docs/sprint5_12_cnn_dqn.md) for input
equivalence, parameter counts, reproduction commands, paired uncertainty and
compute costs. The one-seed 5000-step sparse pilot does not establish architecture
superiority. Longer multi-seed training and PPO require separate approval.

# Masked PPO smoke training

Sprint 5.13 adds `PPOAgent` with a shared MLP actor-critic, masked categorical
actions, episode-aware GAE and clipped on-policy optimization. It uses the existing
encoder, level environment and evaluation runner; DQN remains unchanged.

See the [masked PPO design and smoke results](docs/sprint5_13_masked_ppo.md) for
hyperparameters, seed streams, checkpoint limits and the 5000-step experiment.
Run `python -m match3_ai.training.ppo --output-dir NEW_DIRECTORY --review-only`
to inspect the fixed smoke profile. Longer multi-seed training requires approval.

# PPO learning diagnostics

Sprint 5.14 adds raw rollout/critic diagnostics and a fixed 20000-step sparse PPO
pilot with exact 2500-step evaluation milestones. Intermediate evaluations preserve
the 256-step rollout schedule. Saved DQN checkpoints remain frozen references.

See the [audit, metric definitions and protocol](docs/sprint5_14_ppo_diagnostics.md)
for reproduction, paired comparisons, checkpoint scheduling and limitations.
No multi-seed training is started automatically.

# Multi-seed PPO replication

Sprint 5.15 provides the approved independent 50000-step PPO profiles for seeds
5915–5917 and aggregation that accounts for shared validation boards. It keeps
sparse PPO settings unchanged and uses frozen MLP-DQN references.

See the [configuration and statistical protocol](docs/sprint5_15_multi_seed_ppo.md)
for RNG derivation, checkpoint limits, commands and interpretation. Hyperparameter
tuning, PBRS, CNN-PPO and final-test evaluation require separate approval.

# Preregistered fresh-board evaluation

Sprint 5.16 evaluates eleven frozen policies on 500 previously unused recorded
validation seeds, with a hashed protocol saved before outcomes. Board-paired
statistics separate the primary PPO-versus-Greedy comparison from nine secondary
hypotheses and preserve shared-board dependence.

See the [fresh-board protocol and interpretation](docs/sprint5_16_fresh_paired_evaluation.md)
for provenance, checkpoint selection, critic diagnostic limits and artifacts.
This workflow performs no training or hyperparameter tuning.

# PPO training-budget study

Sprint 5.17 provides the approved 100000-transition profiles for seeds 5918–5920,
with a protocol frozen before training. Budget axes use collected transitions,
optimized counts remain explicit, and paired bootstrap draws share board indices
across all runs and checkpoints. PPO settings remain unchanged.

See the [protocol, accounting and statistical scope](docs/sprint5_17_ppo_budget.md)
for the eight-test Holm family, seed provenance, checkpoint limits and artifacts.
Sprint 5.18 and tuning require separate approval.
