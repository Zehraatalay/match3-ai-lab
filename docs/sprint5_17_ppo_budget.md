# Sprint 5.17: preregistered PPO training budget study

This study follows explicit approval of Phase A, the computational budget and
two methodological requirements: use collected environment transitions on every
budget axis, and jointly resample the same development board indices across all
policies, training seeds and checkpoints. The eight-hypothesis Holm family is
unchanged. No PPO algorithm, reward, architecture, game or environment changes
are introduced.

## Frozen protocol

Before production training, the protocol was written to
`experiments/results/sprint5_17_preregistration/protocol.json`, with SHA-256:

```
d386db9e6b1de0d3aeb188dc1b9a6b29c1d5853af1f191706c2a7c7c153564c6
```

Its history_inventory.json hashes all 579 historical artifact files available at
registration (JSON, JSONL, CSV, NPZ, checkpoints). Development-validation seeds
**300600–300799** have no recorded overlap with earlier experiments, training or
Sprint 5.16's 300100–300599. The inventory is conservative and covers recorded
local artifacts, not unrecorded external experiments. No reserved final-test
episode seeds 400000–400999 are accessed. Provenance conflicts cause a stop,
not automatic selection of different seeds.

The protocol freezes configurations, master/derived streams, expected initial
model fingerprints, level configuration, source hashes, hypotheses and inferential
rules. Training and analysis reject changed source/protocol files. A required
protocol change following an unexpected correctness issue needs user approval;
the protocol is never silently revised after training starts.

## Exact configuration

| Setting | All three runs |
|---|---|
| Level | blocker-mayhem, unchanged configuration |
| Master seeds | 5918, 5919, 5920 |
| Budget | 100000 collected environment transitions each; 300000 total |
| Shared network | 1467→256→128, hidden ReLU |
| Actor / critic | 144 action logits / scalar value |
| Parameters | 427409 |
| Rollout / minibatch / optimization epochs | 256 / 64 / 4 |
| Gamma / GAE lambda | .99 / .95 |
| Policy clipping | .2 |
| Adam learning rate | .0003 |
| Value / entropy coefficients | .5 / .01 |
| Global gradient clipping | .5 |
| Reward | Sparse terminal −1/0/+1, no PBRS |
| Guards | 50 episode steps, 200 cascades, 1000 generation attempts |
| Development boards | Same 200 seeds 300600–300799 for all checkpoints/runs |
| Milestones | 0, 25000, 50000, 75000, 100000 collected transitions |
| Evaluation | Deterministic masked argmax; never counted as training |
| Compute | Independent concurrent CPU processes, one Torch thread each |

Each master SeedSequence spawns independent environment and agent streams.
The agent seed spawns model initialization, policy sampling and minibatch shuffle
streams. Environment episode seeds are locally sampled with replacement inside
200000–299999. All twelve derived environment/model/policy/shuffle signatures and
three expected initial model hashes are distinct and frozen. Some sampled training
episode seeds may overlap between runs, consistent with independent sampling.

The previous PPO implementation remains unchanged: masked categorical likelihoods,
on-policy rollout collection, correct terminal/truncation GAE, rollout-wide advantage
normalization, clipped policy loss, half squared critic loss, entropy bonus and
gradient clipping. No CNN-PPO, new dependency, learning-rate schedule or tuning is
added. Initial models are new; previous 50k models are not warm starts.

## Collected and optimized budget accounting

One transition is one completed valid single-environment step. Cascades and tile
clears inside that step do not add transitions. Resets and evaluations do not add
training transitions. A rollout is a collection chunk, not an episode.

| Collected axis | Optimized through | Pending samples | Completed optimization rollouts | Adam steps |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 25000 | 24832 | 168 | 97 | 1552 |
| 50000 | 49920 | 80 | 195 | 3120 |
| 75000 | 74752 | 248 | 292 | 4672 |
| 100000 | 100000 | 0 | 391 | 6252 |

Intermediate evaluations occur at the exact collected-step milestone while the
policy is frozen during collection; they do not split or optimize a partial
rollout. Final evaluation follows optimization of the last partial rollout.
Thus the within-run 50k checkpoint is not identical in optimized-sample meaning
to a separately terminated 50k historical run. Historical cohorts and different
validation boards are context, not the paired controls in this experiment.

Per run: 390×256 + 160 = 100000 transitions. Full rollouts produce 16 Adam steps
each; the 160-sample tail has minibatches 64, 64 and 32 in four epochs, producing
12 more. Total 390×16 + 12 = 6252. The four epochs present 400000 training samples
to optimization; these are reused samples, not new environment transitions.
Three completed runs yield 1173 rollouts and 18756 Adam steps.

## Eight fixed inferential hypotheses

1. Mean 75k win rate minus mean 50k win rate
2. Mean 100k win rate minus mean 50k win rate
3. Seed 5918: 75k minus 50k
4. Seed 5919: 75k minus 50k
5. Seed 5920: 75k minus 50k
6. Seed 5918: 100k minus 50k
7. Seed 5919: 100k minus 50k
8. Seed 5920: 100k minus 50k

One Holm correction applies to all eight p-values. Individual binary comparisons
use exact two-sided McNemar tests and retain discordant outcome tables. Cohort
comparisons use a two-sided null-centered paired bootstrap mean test with plus-one
resampling correction; this is approximate inference, not an exact label-swap
randomization test. Pointwise 95% percentile bootstrap CIs are not simultaneous
Holm-adjusted intervals. Other checkpoints and objective/HP changes are descriptive,
not additional uncorrected hypothesis tests. Nonsignificance is not equivalence.

## Joint resampling and units of uncertainty

The outcome tensor has axes `(training seed, checkpoint, board)`, shape
`(3, 5, 200)`. Each bootstrap replicate draws one 200-element board-index vector.
The same vector indexes the entire tensor, including all policies and checkpoints;
objective and HP metric tensors use those same indices too. Within-run checkpoint
dependence and shared-board dependence are preserved. Implementation uses linear
contrasts of the complete tensor, equivalent to resampling it jointly, without
allocating every full repeated tensor in memory.

Bootstrap uses 20000 replicates with seed 51717. The exact common index-stream
SHA-256 is retained in analysis.json. Tests compare the implementation with explicit
joint tensor indexing and verify that duplicating identical policies does not
artificially multiply the effective board sample size.

These intervals describe board uncertainty conditional on the three observed
training runs. Training-seed sample SD is reported separately at each budget;
three runs do not provide a large independent training sample. Repeated checkpoint
scores are not independent replications, nor are the 3000 evaluation episodes
3000 independent board realizations. There are 200 shared development boards.

## Checkpoints, diagnostics and failure handling

Five milestone files plus checkpoint_final.pt preserve the existing atomic,
exclusive publication scheme. Model, optimizer, encoder, architecture, master/RNG
metadata, validation metrics and collected/optimized/pending counts are retained.
Final checkpoint inference and all milestone first actions on the 200 development
initial boards are checked. Invalid probabilities, nonfinite values, RNG mutation,
changed initial conditions or source/hash discrepancies cause explicit failures.

The final 100k checkpoint is the fixed final endpoint. Best development-validation
checkpoint is separately described, with earliest ties and step zero eligible;
it is selected on these boards and is not an unbiased performance estimate.

Exact checkpoint continuation is unsupported: active engine state and unfinished
rollouts are not serialized. No automatic restart, failed-run replacement, artifact
overwrite or silent increase of the approved 300000-transition budget is allowed.
Other completed runs remain independent if one fails.

Logs retain episode return/win rate, losses, entropy, KL, clipping, gradient norm,
pre-update GAE explained variance, raw advantage statistics, learning rate and
training/evaluation time. Explained variance uses different on-policy trajectory
targets across runs, not a common held-out critic benchmark. Constant-target EV
is undefined; final-rollout diagnostics use only 160 samples. Finite losses are
not themselves evidence of useful learning or a causal explanation of outcomes.

## Artifacts, validation and cost

New modules ppo_budget.py and ppo_budget_analysis.py reuse the trainer and existing
seed, checkpoint, plotting and statistical infrastructure. No existing Python
implementation is modified. Unit tests use synthetic outcome cubes, historical
initial observations and temporary protocol files; they do not train the study
models or inspect production development boards before registration.

Directories:

- `experiments/results/sprint5_17_preregistration/`
- `experiments/results/sprint5_17_ppo_seed_5918/`, `...5919/`, `...5920/`
- `experiments/results/sprint5_17_ppo_analysis/`

The report retains full per-seed curves, paired outcomes, eight comparisons,
conditional CIs, training-seed SDs, checkpoint/provenance audits and figures.
All plot budget axes use collected transitions. Diagnostic smoothing uses trailing
50 optimizer steps; raw data remain in CSVs. Timing includes system contention,
validation and checkpoint work, excludes final exports/plots, and concurrent process
times cannot be added as elapsed batch time.

Phase A estimated 25–35 minutes for three concurrent runs and 30–45 minutes with
tests/analysis, based on previous measured CPU costs. Approximately 27 GiB was
available at preflight; expected artifacts were .2–.5 GiB with 1 GiB reserved.
These are estimates, not measured Sprint 5.17 outcomes. Actual costs are reported
after completion.

Before training, 73 focused PPO/DQN/evaluation tests passed and Ruff/formatting
checks were clean. No production source changes are made after protocol freeze.
The full suite is run during the study. Historical artifact hashes are verified
again after completion. No staging, commits, pushes, resets, rebases, stashes or
discarding existing work are performed. Sprint 5.18 remains outside this task.

## Completed results

All three runs completed successfully without restarts, training failures or
invalid training/evaluation actions. There were 300000 collected transitions,
1173 rollouts, 18756 optimizer steps and 3000 completed development evaluations
on 200 shared boards. No additional training or evaluations were run to finalize
the interrupted session.

| Master seed | 0 | 25000 | 50000 | 75000 | 100000 |
|---:|---:|---:|---:|---:|---:|
| 5918 | 29.5% | 45.5% | 51.0% | 63.0% | 49.0% |
| 5919 | 49.5% | 52.5% | 62.0% | 54.5% | 57.0% |
| 5920 | 40.5% | 51.0% | 51.0% | 56.5% | 63.0% |
| Mean | 39.83% | 49.67% | 54.67% | 58.00% | 56.33% |
| Training-seed sample SD, percentage points | 10.02 | 3.69 | 6.35 | 4.44 | 7.02 |

Each percentage uses 200 completed episodes. Columns use collected transitions;
the optimized/pending counts remain those in the accounting table above.

| Preregistered contrast | Difference, percentage points | Pointwise conditional 95% CI | Holm-adjusted p |
|---|---:|---|---:|
| Mean 75k minus 50k | +3.33 | [-2.17, +8.67] | 1.0000 |
| Mean 100k minus 50k | +1.67 | [-4.00, +7.17] | 1.0000 |
| 5918: 75k minus 50k | +12.00 | [+1.50, +22.00] | .2051 |
| 5919: 75k minus 50k | -7.50 | [-16.50, +1.50] | .8242 |
| 5920: 75k minus 50k | +5.50 | [-5.00, +15.50] | 1.0000 |
| 5918: 100k minus 50k | -2.00 | [-11.50, +7.50] | 1.0000 |
| 5919: 100k minus 50k | -5.00 | [-15.00, +5.50] | 1.0000 |
| 5920: 100k minus 50k | +12.00 | [+2.50, +21.50] | .1575 |

None of the eight comparisons survives Holm correction. Pointwise intervals
are not multiplicity-adjusted intervals. Joint board resampling preserves all
within-run checkpoint dependence; these intervals remain conditional on the
three trained models, not population uncertainty over training seeds.

Mean objective completion at 50k/75k/100k was 95.68%/96.03%/95.71%; mean remaining
blocker HP was .570/.542/.575. These descriptive metrics also fail to show a
consistent monotonic benefit. Per-seed best development checkpoints were
5918 at 75k (63%), 5919 at 50k (62%), and 5920 at 100k (63%). They remain
selection-biased development results, separate from the fixed final endpoint.

| Seed | Active training/checkpoint seconds | Validation seconds | Total process seconds | Mean / last explained variance |
|---:|---:|---:|---:|---|
| 5918 | 1951.05 | 248.10 | 2199.15 | .1378 / .3861 |
| 5919 | 1935.80 | 252.70 | 2188.50 | .0994 / .1871 |
| 5920 | 1951.24 | 246.73 | 2197.96 | .1009 / .1753 |

The concurrent training stage took approximately 36.7 minutes, including
validation. Summed process durations are 109.8 process-minutes, not elapsed
batch time or measured CPU time; validation totals 12.46 process-minutes.
This exceeded the 25–35 minute training estimate slightly and stayed within
the 30–45 minute overall estimate. Study artifacts occupy approximately
137.2 MiB; approximately 26.65 GiB remained free after completion.

All optimization diagnostics were finite. Mean value losses were .2306/.2369/.2342,
mean entropy 1.4413/1.6220/1.9090 nats, mean approximate KL .00541/.00551/.00611,
and mean clip fractions .0734/.0737/.0839 for seeds 5918/5919/5920. This supports
numerical stability, but not monotonic policy improvement. Critic explained
variance is modest, trajectory-dependent and not a controlled cross-run critic
comparison. No causal explanation for the policy regressions is established.

All 15 milestone checkpoints passed loading and deterministic first-action
reproduction on all 200 corresponding initial observations (3000 decisions).
Maximum probability-normalization error was below 2.4e-7; masked probabilities
were zero and inference preserved policy RNG and weights. Final checkpoints
matched 100k milestone weights and Adam step 6252. These are loading/initial-action
checks, not a second complete episode replay. Exact training continuation remains
unsupported. All saved run hashes, 579 historical artifact hashes, frozen
production source hashes and the protocol hash were verified unchanged.

Validation evidence: 73 focused tests passed in 10.29 seconds; the full suite
passed 796 tests in 73.68 seconds. Final Ruff checks, formatting checks for the
three new Python files and Git diff whitespace checks passed. The saved successful
tests were inspected rather than repeated after interruption; only documentation
was finalized afterward. Existing dirty/untracked work was preserved.

Full raw comparisons, Wilson intervals, checkpoint checks, diagnostic ranges and
13 PNG/PDF figure pairs are in
`experiments/results/sprint5_17_ppo_analysis/report.md` and `analysis.json`.
The frozen protocol remains unmodified; this section reports observed results.

## Recommendation for Sprint 5.18

Increasing this PPO configuration beyond 50k has not demonstrated a consistent
benefit: 5918 regressed after 75k, 5919 peaked at 50k, and 5920 improved through
100k. Nonsignificance is not evidence of equivalence. The 70% target was not
reached by any measured checkpoint or by the cohort mean.

Recommend a separately approved, preregistered replication with more independent
training seeds and the same within-run 50k/75k/100k endpoints before committing
to a larger training budget. Specify a practically meaningful improvement and
plan precision using both board and training-seed variability. Freeze the next
design before training, preserve development/final-test separation, and do not
promote best-validation checkpoints as unbiased evidence. These 200 boards are
now development data; independent confirmation needs a newly approved evaluation
partition. No Sprint 5.18 implementation or training has begun.
