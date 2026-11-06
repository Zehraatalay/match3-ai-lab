# Sprint 5.14: PPO learning diagnostics and extended pilot

This sprint audits the existing PPO implementation, adds diagnostics and runs one
approved 20,000-transition pilot. The architecture, sparse rewards and optimization
hyperparameters remain unchanged. No gameplay, DQN, PBRS or CNN-PPO changes occur.

## Implementation audit

| Area | Verified behavior |
|---|---|
| Old policy | Rollout collection runs without gradients and stores the sampled action's masked log probability as a detached scalar |
| Importance ratio | New and old log probabilities use the identical stored legal mask; ratio is exp(new−old) |
| Policy loss | Minimized negative clipped surrogate handles both advantage signs |
| Critic loss | Positive half squared-error loss, multiplied by value coefficient .5 |
| Entropy | Masked categorical entropy is subtracted from the minimized loss |
| GAE | Delta uses the stored final-observation next value; recurrence stops at termination or truncation |
| Termination | True terminals do not bootstrap, including when both terminal flags are set |
| Truncation | Final observations bootstrap before explicit reset; reset-state values never substitute |
| Rollout boundaries | Policy is frozen during collection; nonterminal boundaries bootstrap, while advantage recursion ends at the horizon |
| Normalization | Whole-rollout population-standard-deviation normalization; constant/singleton advantages remain finite |
| Evaluation | Separate environments, greedy masked policy, weights and policy/shuffle RNG checked unchanged |
| RNG streams | Spawned environment, model initialization, policy sampling and minibatch shuffle streams are separate |
| Seed partitions | Local environment seeds remain in 200000–299999; validation uses 300000–300099; final-test seeds are untouched |
| Checkpoints | Atomic exclusive publication; model, optimizer, encoder, architecture, RNGs and progress metadata saved |

No concrete algorithmic defect was found. Hand-computed tests cover masked ratios,
clipping, KL, entropy and GAE boundaries. The older tests already exercise update
gradients, target immutability, checkpoint/optimizer/RNG round trips and actual
environment actions on all six levels. Finite losses alone do not establish useful
learning or guarantee an implementation free of all possible defects.

The additions are observational. Default PPOTrainingConfig still selects the
5,000-step smoke profile with step-zero/final evaluation. The extended profile
changes only environment_steps=20000 and evaluation_every=2500.

## Metrics and interpretation

| Diagnostic | Definition and interpretation |
|---|---|
| Policy loss | Negative clipped surrogate on normalized advantages; not an episode-return estimate |
| Value loss | Half mean squared error against fixed GAE returns; scale depends on reward/return distribution |
| Entropy | Masked categorical entropy in nats; depends on both policy concentration and legal-action count |
| Approximate KL | mean((ratio−1)−log(ratio)); nonnegative estimator on sampled old-policy actions |
| Clip fraction | Fraction with abs(ratio−1)>.2; measures the prevalence of large likelihood changes, not whether performance improved |
| Ratio min/mean/max | Direct importance-ratio checks using stored rollout masks and old log probabilities |
| Explained variance | 1−Var(return−old_value)/Var(return), computed once per rollout before optimization |
| Raw advantage mean/std | Computed before normalization; expose return/critic scale that normalized inputs would hide |
| Gradient norm | Global norm before clipping at .5; values above .5 are expected in this log |
| Learning rate | Actual optimizer learning rate, fixed at .0003 |
| Training return/win rate | Cumulative natural-terminal episode statistics; artificial budget truncations excluded |

Explained variance near one indicates small residual variance; negative values
mean the predictor has larger residual variance than the targets. A constant
target variance at or below 1e-12 makes the statistic undefined: it is stored as
blank/None and plotted as a gap, never silently treated as zero. It is a pre-update
statistic against bootstrapped GAE returns, not an independent critic test score.

Policy/value losses, entropy, KL, ratios, clipping, gradients and learning rate are
logged per minibatch in optimization.csv. Explained variance, raw advantages,
learning rate, completed episode count, cumulative return and win rate are in
rollouts.csv. Each episode's sparse reward remains in training_episodes.json.
Plot smoothing uses trailing 20 optimizer updates, not 20 independent episodes.
Other curves use their raw rollout/checkpoint values. Aggregate optimization means
are per-update means, not transition-weighted estimates.

## Unchanged configuration

MLP shared backbone 1467→256→128, actor 128→144, critic 128→1: 427,409 parameters.
Master seed 5913; rollout 256; minibatch 64; four PPO epochs; gamma .99; GAE lambda
.95; policy clipping .2; Adam .0003; value coefficient .5; entropy coefficient .01;
gradient clipping .5. Guards remain 50 episode steps, 200 cascades and 1,000
initialization attempts. PBRS is disabled and evaluation rewards remain sparse.

The CLI profile runs exactly one 20,000-step blocker-mayhem experiment. Validation
uses 100 fixed seeds at 0, 2500, 5000, 7500, 10000, 12500, 15000, 17500 and 20000.

## Milestones inside rollouts

The evaluation interval is not divisible by 256. Splitting every rollout at 2500
would change optimization batches and hence learning dynamics. Instead, intermediate
validation observes the current frozen collection policy at the exact step and
does not interrupt or optimize the partially collected rollout. Progress metadata
records last_optimized_step and pending_rollout_steps. Final validation occurs
after optimizing the final partial rollout.

Consequently the intermediate 5,000-step checkpoint in this run differs from
Sprint 5.13's terminal 5,000-step checkpoint: that smoke run optimized an extra
partial rollout, whereas this extended run continues collecting its full rollout.
This is a scheduling distinction, not a different learning-rate or rollout setting.
An explicit test verifies identical final weights/statistics with and without
intermediate evaluations under the same total budget.

A historical-prefix audit also compared all first 5,000 recorded transitions
against Sprint 5.13. Actions, rewards, log probabilities, values, bootstrap values,
episode seeds and terminal flags match exactly. The only difference is the old
budget truncation at step 5,000, which is absent in this longer run. The old PPO
checkpoint still loads with identical weights and an unchanged file hash. Results
are saved in the pilot directory's sprint5_13_prefix_audit.json.

Every evaluation milestone publishes policy_step_N.pt. The final full policy
also has checkpoint_final.pt. Architecture metadata is explicit and versioned;
old PPO files without the additive field remain loadable. Publication never
replaces an existing checkpoint. Checkpoints retain optimizer, policy/shuffle RNG,
environment seed-generator state and configuration identity. Exact training
continuation is not supported: active engine state and unfinished rollout samples
are not serialized. RNG metadata alone does not provide exact resume.

## Baselines and uncertainty

Fresh deterministic evaluations compare final PPO with RandomAgent, GreedyAgent,
and the existing Sprint 5.9 seed-5903 MLP-DQN checkpoints at 20,000 and 50,000 steps.
Random uses independent local seeds 600000–600099. All policies use identical
validation board/engine seeds. DQN checkpoint files and historical metadata are
hashed and read only; no DQN training is executed.

The 20,000-step DQN reference matches transition count, but not optimizer updates,
collection seed policy, initialization, algorithm, or computational budget. The
50,000-step reference has 2.5× PPO's transition count. Historical timing metadata
and actual Adam update counters are reported where available. Historical timings
are machine/load dependent and cannot establish a controlled speed comparison.

The report gives Wilson win intervals, objective completion, remaining HP, moves,
rewards, paired outcome tables and PPO-minus-reference bootstrap differences.
Continuous paired intervals are retained in analysis.json. Bootstrap uses 5,000
paired seed resamples with seed 51414. Intervals are exploratory and pointwise,
without multiplicity correction; the reused checkpoint measurements are correlated.
Nonsignificance does not establish equivalence. One training seed does not measure
training-run variability. Best validation estimates are selection-biased, and
these familiar validation boards are not a held-out final test.

## Commands and artifacts

Use fresh output directories; these commands refuse existing destinations:

```sh
conda run -n match3-ai python -m match3_ai.training.ppo_pilot --output-dir experiments/results/NEW_ppo_pilot --review-only
conda run -n match3-ai python -m match3_ai.training.ppo_pilot --output-dir experiments/results/NEW_ppo_pilot
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run -n match3-ai python -m match3_ai.training.ppo_analysis --run-dir experiments/results/NEW_ppo_pilot --output-dir experiments/results/NEW_ppo_analysis --dqn-checkpoint experiments/results/sprint5_9_seed_5903/policy_step_20000.pt --dqn-checkpoint experiments/results/sprint5_9_seed_5903/policy_step_50000.pt
```

This sprint's fresh output directories are:

- `experiments/results/sprint5_14_ppo_pilot/`: all training/evaluation logs and checkpoints
- `experiments/results/sprint5_14_ppo_analysis/`: report.md, analysis.json,
  baseline gameplay CSVs, input/checkpoint hashes and ten PNG/PDF figure pairs

The final visual review saved report_final.md and figures_final/ with distinct
reference-line colors. The earlier exports are retained; no experiment artifacts
were overwritten. Both report versions contain identical numeric results.

The ten figures cover validation wins, objectives, remaining HP, policy/value loss,
entropy, approximate KL, clipping, pre-update explained variance and gradient norm.
Frozen reference lines are labeled by saved DQN training step and do not imply
that those models were trained on the PPO x-axis budget.

The analysis loads every milestone and verifies its first action on all validation
initial states against recorded gameplay, probability normalization, zero invalid
probabilities, finite values and unchanged sampling RNG. It checks source hashes
again before completion. CPU thread settings are restored after analysis.

## Validation and recommendation

The extended run completed 20,000 transitions, 79 rollouts and 1,252 optimizer
updates. Validation wins at the nine milestones were **42, 41, 38, 47, 47, 53,
46, 51, 51** out of 100. The final Wilson interval is [41.3%, 60.6%]; final
objective completion is 95.17%, remaining HP .65, moves used 12.51 and sparse
episode reward .02. No training or validation invalid actions occurred. The peak
53% at 12,500 is a validation-selected observation, not an unbiased final estimate.

Mean policy loss was −.02121, value loss .23533, entropy 2.79757 nats, approximate
KL .00531, clip fraction .07554 and pre-clipping gradient norm 1.40336. All were
finite. Explained variance averaged .02885 (range −.02696 to .39695), suggesting
weak critic fit on average. Its final .39695 belongs to a 32-sample partial rollout
and should not be interpreted as a reliable late improvement. Raw advantage means
ranged from −.52463 to .41603; standard deviations from .21472 to .75097. Final
cumulative natural-terminal training win rate was 44.99%, mean return −.10013.
These metrics do not identify a causal explanation for the fluctuating policy
results and do not justify automatically changing optimization settings.

The PPO wall timer was 390.46 seconds, including 140.72 seconds of validation;
active collection, optimization and checkpoint work took 249.74 seconds. Focused
and full tests overlapped part of this run, so timings include system contention.
See the generated report for the fresh frozen-reference results, paired intervals,
historical DQN budgets and all ten figures.

Fresh frozen-reference win rates were Random 45%, Greedy 57%, DQN-20k 35% and
DQN-50k 53%. Final PPO-minus-reference differences were respectively +6 pp
[-7, +19], −6 pp [-20, +8], +16 pp [+3, +29] and −2 pp [-16, +11]. Final versus
initial PPO was +9 pp [-5, +23]. These paired 95% intervals are exploratory and
do not establish algorithm superiority across training seeds. The 20k DQN model
has 4,751 optimizer updates versus PPO's 1,252; the 50k model has 12,251. Their
historical wall timers were 451.88 and 1,057.66 seconds, with different evaluation
schedules and system load. The baseline/audit pass took another 64.43 seconds.

All nine saved milestone policies passed inference/probability/RNG checks on
900 initial validation observations. Their recorded first actions matched, and
all input/checkpoint hashes remained unchanged. Ten PNG/PDF figure pairs and the
full paired tables are saved with the generated report.

Six new regression cases cover exact masked ratios, clipping/KL/entropy,
explained variance (including undefined targets), GAE boundaries, deterministic
milestones and pending-rollout metadata, seed-bound configuration, checkpoint
round trips, ten figure exports and read-only baseline artifacts. Focused PPO,
DQN and evaluation tests: **69 passed**. Full suite: **780 passed**. Ruff and
whitespace checks are clean. Existing tests were not weakened or rewritten.

Read the generated results before approving further work. Weak or fluctuating
curves do not justify automatic hyperparameter changes. Any next study should
preregister its seeds, budgets and endpoints and distinguish learning from
initialization performance. No multi-seed training was begun in this sprint.
