# Sprint 5.15: independent masked PPO replication

This study runs the approved three seeds, 5915, 5916 and 5917, for exactly
50,000 environment transitions each on blocker-mayhem (150,000 total). No
hyperparameters, gameplay rules, rewards, observations or DQN training change.
PBRS, CNN-PPO, tuning and final-test evaluation remain outside this sprint.

## Preflight findings and implementation

The PPO implementation from Sprints 5.13–5.14 remains correct under the focused
audits/tests: sampled old log probabilities are stored with their masks, ratios
reuse those masks, clipped-policy/value/entropy loss signs agree with analytic
tests, and GAE distinguishes bootstrap eligibility from episode-boundary carry.
True termination zeroes bootstrap; truncations use the actual final observation
before resetting. Ordinary rollout boundaries do not reset the environment.
Evaluation checks weights and policy/shuffle RNGs unchanged. Initialization runs
inside an isolated Torch RNG scope. No concrete algorithmic defect was identified.

The new `ppo_multi_seed` module supplies only approved profiles. Each invocation
owns a fresh environment, model, optimizer, rollout lists and RNG streams, in a
separate process and directory. Completed results stay intact if another process
fails. Existing exclusive checkpoint and output-directory publication rejects
collisions. There is no automatic retry into an existing directory or unsupported
resume from an incomplete checkpoint.

Checkpoint payloads now also retain validation metrics. Progress includes wall
and validation timing. These additive fields do not affect actions or updates.
Existing PPO checkpoints remain loadable. Frozen DQN reference labels now include
training seed as well as checkpoint step, preventing collisions between multiple
50k references. No DQN checkpoint files are changed.

## Exact shared configuration

| Setting | All three runs |
|---|---|
| Level | blocker-mayhem |
| Master seeds | 5915 / 5916 / 5917 |
| Environment sample budget | 50,000 per seed |
| Actor-critic | 1467→256→128 shared ReLU; 144-logit actor and scalar critic |
| Parameters | 427,409 |
| Rollout / minibatch / epochs | 256 / 64 / 4 |
| Gamma / GAE lambda | .99 / .95 |
| Policy clip ratio | .2 |
| Optimizer / learning rate | Adam / .0003 |
| Value / entropy coefficients | .5 / .01 |
| Gradient clipping | .5 global norm |
| Reward | Sparse terminal −1/0/+1, PBRS disabled |
| Guards | 50 episode steps, 200 cascades, 1,000 initialization attempts |
| Validation | Deterministic; 100 seeds 300000–300099 |
| Milestones | 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000 |
| Compute | Concurrent independent CPU processes, one Torch thread each |

Raw advantages use whole-rollout population-standard-deviation normalization.
The critic regresses fixed GAE returns with half squared error. Entropy is masked
categorical entropy in nats. There is no value clipping, learning-rate schedule,
KL early stopping or extra reward. The existing single-seed defaults remain intact.

## Seed derivation and partitions

`SeedSequence(master).spawn(2)` yields environment and agent streams. The agent
stream generates one uint32 agent seed; `SeedSequence(agent_seed).spawn(3)` then
yields uint32 model, policy and minibatch-shuffle seeds. Model initialization
uses local Torch manual_seed inside fork_rng; policy sampling uses a dedicated
CPU Torch generator; shuffling uses a local NumPy generator. The environment
generator is initialized from its child SeedSequence directly, then draws seeds
with replacement within 200000–299999. Do not replace that SeedSequence by the
displayed uint32 signature: it would initialize a different generator.

| Master | Agent seed | Environment stream signature | Model | Policy | Shuffle |
|---:|---:|---:|---:|---:|---:|
| 5915 | 200701294 | 3211586567 | 552181205 | 3799128560 | 194271621 |
| 5916 | 2885630954 | 29110978 | 2487139742 | 2754827312 | 2229925419 |
| 5917 | 242107743 | 3828225726 | 4145955981 | 2885351539 | 403747534 |

All twelve environment/model/policy/shuffle stream identifiers differ. Tests and
the aggregate audit reconstruct derivations, compare metadata and check distinct
initial weights. Some drawn environment seeds may overlap across runs because
sampling is with replacement; this is consistent with independent RNG streams.
Every recorded training seed is audited to remain in the training partition.

Validation is in 300000–300999, using only 300000–300099 here. It uses separate
fresh environments and never advances collection/policy/shuffle generators.
Reserved final-test seeds 400000–400999 are not accessed. Repeated episodes and
initial boards are reproducible under the installed CPU software; bitwise
reproducibility across Torch, BLAS, OS or CPU changes is not promised.

## Milestone checkpoints and termination

Milestones that lie inside a 256-step rollout evaluate the frozen collection
policy without shortening or optimizing that rollout. `last_optimized_step` and
`pending_rollout_steps` identify the difference between collected and optimized
transitions. Final evaluation follows optimization of the final partial rollout.
Training budget is never silently rounded. Evaluation episodes are excluded from
the 50,000 transition count. Step-zero evaluation precedes all training.

Atomic milestone checkpoints retain architecture, encoder, model, optimizer,
master/derived seeds, RNG states where supported, progress and evaluation metrics.
Each final checkpoint is loaded and inference checked. Individual audits also
compare every milestone's first actions on all validation boards with recorded
gameplay, verifying probabilities, values and unchanged policy sampling RNG.

Exact training continuation remains unsupported: active engine state and
unfinished rollout samples are not serialized. Saving generator state does not
make a partial checkpoint an exact mid-episode resume point. Failures remain
explicit and preserve other completed directories.

## Baselines and statistical interpretation

Frozen references are RandomAgent (local seeds 600000–600099), GreedyAgent and
all three existing Sprint 5.9 MLP-DQN final checkpoints (training master seeds
5901, 5902, 5903; checkpoint step 50,000). Their configuration identities, saved
training seeds, optimizer counters, historical timing and file hashes are retained.
They are not retrained. Equal environment-transition budgets do not equalize
optimizer updates, collection data, initialization or computation. Historical wall
times include different evaluation schedules and system conditions.

Individual audit reports give per-policy Wilson intervals, outcome tables,
paired bootstrap differences, continuous objective/HP/move/reward differences and
exact McNemar p values. References are re-evaluated within each individual audit;
the aggregate verifies their results agree and displays one set, without treating
repeated identical reference evaluations as additional independent boards.

The aggregate reports mean final rate and **sample** SD across the three training
seeds (ddof=1). This describes observed training-run variation; it is not a
confidence interval for algorithm performance. Best checkpoint ties choose the
earliest step. Validation-selected maxima are not unbiased final-test estimates.

All three policies face the same 100 validation boards. The aggregate conditional
interval first averages the three policy win indicators within each board, then
uses paired resampling of those 100 boards against the reference. It never treats
the 300 correlated policy/board episodes as independent. This interval measures
board uncertainty conditional on these three trained models, not population
training-seed uncertainty. Three seeds cannot establish algorithm superiority.

Bootstrap uses 5,000 resamples: seed 51414 for per-policy paired comparisons and
51515 for aggregate shared-board comparisons. Exact McNemar tests are exploratory
final-policy comparisons. Holm adjustment spans all 18 displayed tests (three
PPO policies × five frozen references plus their own initial policies). Pointwise
confidence intervals are not multiplicity-adjusted. Familiar validation boards
and correlated checkpoints limit interpretation; nonsignificance is not equivalence.

## Outputs and reproduction

Fresh run directories are `experiments/results/sprint5_15_ppo_seed_5915`,
`...5916` and `...5917`. Each retains transitions, rollouts, optimization metrics,
episode returns, evaluation gameplay, summary and all eleven milestone checkpoints.
Independent audit directories are `experiments/results/sprint5_15_ppo_audit_5915`,
`...5916`, `...5917`; aggregate outputs are in
`experiments/results/sprint5_15_ppo_aggregate/`.

Use report_final.md in each individual audit directory. A reused report template
initially contained the old pilot's seed/budget wording; the source now derives
both from actual metadata. Revised reports were saved under new names, preserving
the first exports and all numeric artifacts. The aggregate report was unaffected.

Run commands require fresh output directories. For each approved seed:

```sh
conda run -n match3-ai python -m match3_ai.training.ppo_multi_seed --master-seed 5915 --output-dir experiments/results/NEW_5915 --review-only
conda run -n match3-ai python -m match3_ai.training.ppo_multi_seed --master-seed 5915 --output-dir experiments/results/NEW_5915
```

Use the existing `match3_ai.training.ppo_analysis` command per run, supplying
`--dqn-checkpoint` for each existing seed-5901/5902/5903 checkpoint_final.pt.
Then pass three aligned `--run-dir` and `--analysis-dir` arguments to
`match3_ai.training.ppo_multi_seed_analysis`, with a new `--output-dir`.
Each completed analysis remains available independently if another fails.

The aggregate saves report.md, analysis.json, input hashes and twelve PNG/PDF
figure pairs: individual wins, mean wins ± sample SD, objectives, remaining HP,
policy/value losses, entropy, KL, clipping, explained variance, gradients and
final PPO versus references. Diagnostic plots use trailing 50-update means;
explained variance uses raw pre-update rollout values with undefined targets
left as gaps. Individual reports retain complete diagnostic ranges, training
returns/win rates, raw advantage scales and constant learning rates.

Existing work and experiment artifacts are preserved. Nothing is staged or
committed. Further tuning, PBRS, CNN-PPO or final-test work requires approval.

## Observed training results

All three approved runs completed 50,000 transitions: exactly 150,000 total,
excluding validation. Each produced 196 rollouts and 3,128 optimizer updates.

| Master seed | Initial wins / 100 | Final wins / 100 | Best wins / checkpoint | Final minus initial |
|---:|---:|---:|---|---:|
| 5915 | 45 | 59 | 59 / 50000 | +14 pp |
| 5916 | 35 | 45 | 56 / 45000 | +10 pp |
| 5917 | 38 | 55 | 59 / 45000 | +17 pp |

Mean final win rate is **53.0%**, sample SD **7.21 percentage points**. The positive
final-minus-initial changes are an observed replication signal, not proof of
general learning or algorithm superiority. Curves fluctuate and the three
validation-selected maxima should not replace final outcomes.

| PPO steps | 5915 | 5916 | 5917 |
|---:|---:|---:|---:|
| 0 | 45% | 35% | 38% |
| 5000 | 54% | 42% | 49% |
| 10000 | 46% | 40% | 53% |
| 15000 | 45% | 40% | 38% |
| 20000 | 44% | 54% | 44% |
| 25000 | 49% | 43% | 53% |
| 30000 | 50% | 54% | 54% |
| 35000 | 55% | 51% | 52% |
| 40000 | 45% | 49% | 49% |
| 45000 | 48% | 56% | 59% |
| 50000 | 59% | 45% | 55% |

Mean pre-update explained variance was .0835, .1045 and .0781. The last values
were −.2126, .2080 and .1608 on short final 80-sample rollouts. Critic fit is weak
on average; these values do not establish a causal explanation for policy results.
Final cumulative training win rates were 52.87%, 52.91% and 50.26%; mean sparse
terminal returns were .0573, .0582 and .0052. All diagnostics remained finite.

| Seed | Process wall seconds | Validation seconds | Active seconds |
|---:|---:|---:|---:|
| 5915 | 840.73 | 179.18 | 661.55 |
| 5916 | 841.24 | 181.40 | 659.85 |
| 5917 | 833.63 | 177.33 | 656.31 |

The processes ran concurrently, so their wall times must not be summed as elapsed
batch time. Timings include system contention and exclude final exports/plots.
The generated aggregate report provides full frozen-baseline results, uncertainty
tables, checkpoint audits and diagnostic figures.

## Validation

Five new focused cases cover approved configuration invariance, independent
streams, injected failure preservation, checkpoint evaluation/timing metadata,
shared-board uncertainty and complete three-run aggregation with figure exports.
Focused PPO/DQN/evaluation suite: **79 passed**. Full suite: **785 passed**. Ruff,
formatting and Git whitespace checks are clean. Existing tests were not weakened.

## Aggregate interpretation

Fresh frozen-reference results were Random 45%, Greedy 57%, and DQN seeds
5901/5902/5903 at 45%, 45% and 53%, each on the same 100 validation seeds. All DQN
references were already trained to 50,000 steps; each had 12,251 optimizer updates
versus PPO's 3,128. Their historical process wall times were 1,057.48, 1,054.58 and
1,057.66 seconds, including about 220 seconds validation. The current PPO timings
are not an isolated speed comparison. Random and Greedy have no learned training
sample budget; only inference/evaluation is required.

The mean fixed PPO policies beat Random by 8 pp, with conditional shared-board
95% CI [−2.67, +18.67], and trail Greedy by 4 pp [−15.67, +7.67]. Against the
three saved DQN models, differences are +8 pp [−3.33, +19.33], +8 pp [−3, +19]
and 0 pp [−10.67, +10.33]. These intervals condition on the three trained models.
No displayed exact McNemar comparison remains below .05 after Holm adjustment
across all 18 final-policy comparisons. This does not establish equivalence.

All 33 milestone policies were checked against their recorded first actions on
3,300 validation initial observations. Final checkpoints loaded successfully;
probabilities, policy RNG isolation and hashes passed audits. No training or
validation invalid actions/errors occurred, and all 150,000 transitions, 588
rollouts and 9,384 optimizer updates were retained. Twelve aggregate PNG/PDF
figure pairs and all per-policy paired tables are saved in the aggregate directory.

The next recommended study is a preregistered replication with more independent
training seeds and a fresh subset of the validation partition, retaining these
settings before tuning. This would separate training-seed variability from reuse
of the familiar boards. It requires separate approval; no additional training or
new validation/final-test evaluation was started.
