# Sprint 5.9: multi-seed DQN experiment

The approved experiment comprises three fresh 50000-step vanilla DQN runs on
Blocker Mayhem. Master seeds are **5901, 5902 and 5903**. No algorithm or
hyperparameter changes are introduced: no Double/Dueling DQN, prioritized replay,
PPO or reward shaping. The six levels, engine, environment and baseline agents
are unchanged.

## Preflight and exact settings

- Encoder: 1467 float32 features, unchanged categorical/spatial/task encoding
- MLP: 1467 → 256 → 128 → 144, ReLU hidden layers
- Replay: 50000 raw transitions, uniform local-RNG sampling, batch 64
- Warm-up: 1000 environment steps; optimize every four steps
- Adam learning rate: 3e-4; Huber loss; gamma .99; L2 gradient clipping at 10
- Frozen target: hard synchronization every 1000 optimizer updates
- Exploration: epsilon 1 → .05 linearly over 50000 steps
- Last collected action epsilon: .050019; schedule value at step 50000: .05
- Expected updates: 12251 per run; expected target synchronizations: 12
- CPU: one intra-op Torch thread per process, deterministic algorithms enabled
- Resolution guard: 200 cascades; initialization guard: 1000 attempts
- Episode safety bound: 50 actions; natural move-budget exhaustion terminates
- Sparse rewards: +1 win, -1 loss, zero otherwise

Preflight reconfirmed legal-only exploration/exploitation and next-action target
maximization, reward-only terminal targets, final-observation bootstrapping on
truncation, finite-value checks and independent replay/model construction.
Normalization is unchanged and uses no validation-fitted statistics. Greedy
validation does not consume exploration RNG or update parameters. Scoped eval
mode is restored even when evaluation raises.

## Seeds and independence

The existing inclusive environment partitions remain:

| Purpose | Reserved range | Used by this experiment |
|---|---|---|
| Training | 200000–299999 | Sequential episode seeds beginning at 200000, 220000 and 240000 |
| Validation | 300000–300999 | Fixed 300000–300099 for every run and checkpoint |
| Final test | 400000–400999 | Never accessed |

Each master seed independently spawns agent and replay seeds. The agent seed
independently spawns model-initialization and exploration streams. Each episode
seed initializes the existing board-generation/gameplay streams through the
environment. Distinct training starts provide different reproducible board
sequences across the runs. Every run owns a new replay buffer. No weights,
optimizer states, replay samples or exploration RNG states are transferred
between runs.

RandomAgent uses independent seeds 600000–600099 for the paired validation
episodes. GreedyAgent is deterministic. Both baselines are evaluated afresh at
step zero in each run, checked for agreement, and referenced by later milestone
comparisons. Historical data are not assumed compatible or imported. Repeated
baseline results are not additional independent observations.

## Milestones and interruption behavior

Validation occurs at 0, 5000, 10000, ..., 50000 with 100 episodes per DQN policy,
exploration disabled. Step zero has no training transitions, optimizer updates
or replay insertion. Its validation duration is measured separately from active
training, although the total run wall timer includes it.

Every milestone writes a new `policy_step_<step>.pt` containing model/target/
optimizer, encoder/training configuration, master/derived seed metadata,
exploration RNG and validation metrics. It also writes independent gameplay
CSV/JSON under `validation/step_<step>/` and an episode-history JSON. Metrics CSV
is flushed at milestones.

Checkpoint serialization first writes an owned temporary file and flushes it to
disk. A hard link publishes the completed destination atomically and exclusively;
an existing checkpoint is never overwritten. An interruption cannot publish a
partially serialized `.pt` file or remove a completed checkpoint. Owned temporary
files are cleaned up during ordinary exception handling. Forced process death
can leave a `.tmp`, which is not treated as a completed checkpoint. Tests inject
both interrupted serialization and an interrupted later training step.

Periodic checkpoints support inference and model/optimizer recovery but omit
full replay and in-progress engine state. They are explicitly not exact
mid-episode continuation checkpoints. The final full checkpoint preserves the
existing replay/RNG boundary-resume mechanism. No new engine-state serialization
architecture is introduced.

## Commands and artifacts

Each command requires a fresh output directory. The three actual runs execute
in separate processes; no Torch training runs share process-global settings.

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.multi_seed \
  --master-seed 5901 --output-dir experiments/results/sprint5_9_seed_5901
```

The corresponding commands use seeds/directories 5902 and 5903. Add
`--review-only` to print a configuration without training. This CLI accepts only
the three approved master seeds and retains the approved fixed budget.

Analyze completed runs without executing an environment or selecting a model:

```sh
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run -n match3-ai \
  python -m match3_ai.training.multi_seed_analysis \
  --input-dirs experiments/results/sprint5_9_seed_5901 \
    experiments/results/sprint5_9_seed_5902 experiments/results/sprint5_9_seed_5903 \
  --output-dir experiments/results/<new-analysis-directory>
```

The report and numeric JSON retain input hashes and source directories. Ten
figures are saved in PNG/PDF: per-seed validation, mean validation, episode reward,
training win rate, TD loss, epsilon, Q statistics, gradient norm, objective
completion and remaining blocker HP. Existing output directories are rejected.

## Interpretation and limitations

Report final and best validation checkpoints separately. Best includes step zero
and uses the earliest checkpoint when tied; it is selected on validation and is
not an unbiased final-test estimate. Wilson 95% intervals are calculated for each
fixed policy on the 100 validation episode seeds. Across-training-seed means and
sample SD are descriptive; mean-curve shading is ±1 sample SD, not a confidence
interval.

The same validation boards are reused across all models/checkpoints. These
observations are correlated and must not be pooled into independent binomial
samples or analyzed with independent-sample tests across checkpoints. Three
training seeds cannot establish statistical superiority or convergence.

Training win rates use naturally terminated episodes under the exploratory
training policy. Budget truncations are recorded separately. Loss/Q/gradient
curves use trailing 100-update means; reward uses trailing 100 terminal episodes.
Raw records and all unsmoothed validation metrics remain available. Small
Q-value overshoot beyond true sparse-return bounds is not itself proof of a
gameplay or target-calculation defect; numerical drift and outcome trends must
be assessed together.

Concurrent run timers include validation/checkpoint overhead and reflect shared
machine load; their sum is not total experiment elapsed time. No result implies
permission for further training, algorithm changes, PPO, shaping or final-test
evaluation.

## Completed results

The three approved runs completed all **150000 training steps**. Output directories
are `experiments/results/sprint5_9_seed_5901/`, `sprint5_9_seed_5902/` and
`sprint5_9_seed_5903/`. Their report, numeric summary, ten PNG/PDF figures and
checkpoint audit are under `experiments/results/sprint5_9_multi_seed_analysis/`.
That directory's `findings.md` provides the final interpretation and recommendation.

| Seed | Step zero | Final | Best validation / step | Gain from zero |
|---:|---:|---:|---|---:|
| 5901 | 38% | 45% | 57% / 5000 | +7 pp |
| 5902 | 43% | 45% | 57% / 25000 | +2 pp |
| 5903 | 36% | 53% | 53% / 10000 | +17 pp |

Final mean ± sample SD is **47.67% ± 4.62 pp**, versus untrained 39.00% ± 3.61 pp.
Random remains 45%, Greedy 57%. Two final DQNs tie Random; all three remain below
Greedy. Mean improvement over step zero is 8.67 pp, but the fluctuating curves
show no sustained late gain. Best-checkpoint results are validation-selected.

Each run performed 12251 updates, 12 target synchronizations and filled replay
to 50000, with final collection epsilon .050019. No invalid actions, training/
validation errors or validation time limits occurred. Losses and gradients were
finite, while Q extrema -2.900 to 2.289 show value miscalibration beyond true
sparse-return bounds. No implementation correction was identified from that
overshoot alone. Further budget increases are not justified automatically; review
saved policy failures and value calibration first.

Concurrent per-run wall times were 1054.74–1057.79 seconds, including
219.68–221.46 seconds of validation, approximately 17.7 minutes for the batch.
All 33 milestone and three full final checkpoints loaded successfully. Actual
training seed sets were disjoint, and all validation board/gameplay seeds matched.
The final-test partition was not accessed.

Five new regression tests and the existing suite pass: 68 focused tests,
714 full-suite tests, clean Ruff and whitespace checks. Prior tracked and
untracked work was preserved; no staging/history operations were performed.
