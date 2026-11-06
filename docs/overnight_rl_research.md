# Overnight reinforcement-learning research

This report tracks the authorized overnight program on Blocker Mayhem. The
environment, level configuration, objectives, reward definitions and reserved
final-test partition are unchanged. Human Gameplay and Data Science Analytics
implementation are outside this mission.

## Frozen design and provenance

The experiment manifest is
`experiments/results/overnight_rl_research/manifest.json`, SHA-256
`0bdf0b4466c015595909abc6b5a5a13ad468b5abeef4b4509cc6172fec78f026`.
It was frozen before any production training or fresh-board evaluation. Source
fingerprints are rechecked by every worker before and after execution and by the
analysis. The mission permits execution without intervening approval; it
supersedes Sprint 5.18's unexecuted twelve-pair proposal with three pairs.

The clock starts 2026-11-05 20:18:34 UTC (23:18:34 Istanbul). The eight-hour hard
limit is 2026-11-06 04:18:34 UTC (07:18:34 Istanbul); training cutoff is one hour
earlier. The supervisor estimates each stage before launching it. At most three
training workers run concurrently, each using one Torch/BLAS thread. The machine
has eight CPU cores, 16 GiB physical RAM and approximately 27 GiB free disk at
preflight. Historical process durations were about 2195 seconds for 100k PPO and
1057 seconds for 50k DQN, including validation and CPU contention. Preferred
budgets therefore fit without reduction: about 74 minutes for Stage A, 25 minutes
each for B and C, plus smoke/baseline evaluation, implementation and finalization.
These are estimates, not measured costs of the new runs.

The common development set is **300800–300998 inclusive, 199 boards**. These do
not overlap training episode seeds 200000–299999, Sprint 5.16's 300100–300599,
Sprint 5.17's 300600–300799, earlier evaluation boards or reserved final-test seeds
400000–400999. The boundary 300999 remains conservatively excluded. No validation
partition expansion is made. Freshness is relative to recorded local artifacts;
external or unrecorded use cannot be ruled out.

Preflight's conservative inventory initially rejected the range because Sprint
5.18's summary lists it under `unexcluded_validation_seeds` and the unexecuted
experiment proposal names its endpoints. A field-level audit found the only
matching historical JSON integers in those explicitly prospective fields, no
matching evaluation/training CSV records, and no matching NPZ or checkpoint
metadata values. The raw inventory and all 201 prospective occurrences are
retained in `history_inventory_conservative.json` and `provenance_occurrences.json`.
`seed_provenance.json` documents this classification; `history_inventory.json`
protects all 633 pre-existing historical data/checkpoint artifacts and excludes
prospective references from the used-board ledger. This was resolved before
registration, without generating a board or inspecting an outcome. A registration
attempt without the completed inventory failed before a manifest or training run
was created; it is recorded in `preflight_events.json`.

## Algorithms and budgets

| Stage | Variant | Master seeds | Collected transitions/run | Checkpoints |
|---|---|---|---:|---|
| A | Existing masked PPO, constant LR .0003 | 5930–5932 | 100000 | 0, 50k, 100k |
| A | Masked PPO, late LR decay | Same paired seeds | 100000 | 0, 50k, 100k |
| B smoke | Standard Double DQN | 5960 | 512 | 0, 512 |
| B | Standard Double DQN | 5950–5952 | 50000 | 0, 25k, 50k |
| C smoke | Dueling Double DQN | 5961 | 512 | 0, 512 |
| C pilot | Dueling Double DQN | Same B seed policy | 50000 | 0, 25k, 50k |
| Baselines | Random and Greedy | Per-board Random seeds 619000000+index | No training | Fixed policy |
| Historical baselines | Frozen Vanilla DQN 5901–5903, PPO 5918–5920 | Historically trained | No new training | Fixed historical final |

There are 22 declared jobs including two smoke tests and eight frozen-policy
baseline evaluations. Full completion collects 901024 transitions: 600000 PPO,
300000 DQN-family pilots, and 1024 smoke transitions. Short unit-test training is
separate correctness work on legacy first-match boards, not additional production
replications. Production smoke results are not pooled into pilot estimates.

PPO settings remain rollout 256, minibatch 64, four epochs, gamma .99, lambda
.95, policy clip .2, value coefficient .5, entropy coefficient .01 and global
gradient threshold .5. Sparse rewards are used, without PBRS. Control/treatment
initialization, environment, policy-sampling and shuffle streams match within
each pair, and differ across master seeds. Process order balances arms across
the initial and subsequent waves; no outcomes influence that order.

Treatment rate at collected rollout completion c is
`.0003 * (1 - .9 * clip((c-50000)/50000,0,1))`, applied before that rollout's first
optimizer step and held fixed across its epochs/minibatches. Policies remain
frozen during collection. The first changed update is at 50176; the 50k policy
has optimized through 49920 with 80 samples pending. Final evaluation follows
optimization of the 160-sample tail; final LR is .00003. Each 100k PPO run has
391 rollouts and 6252 Adam steps. Equal model hashes at 0 and 50k and equality of
the first 50000 transition records are checked after training. Common RNG streams
do not guarantee identical trajectories after the intervention changes actions.

DQN-family pilots retain the existing feature encoder, MLP hidden dimensions
256/128, replay capacity 50000, batch 64, warmup 1000, update every four collected
transitions, hard target synchronization every 1000 optimizer steps, Adam .0003,
gamma .99 and gradient threshold 10. Epsilon decays from 1 to .05 over 50000
transitions. Training episode sequences start at 200000, 220000 and 240000 for
seeds 5950, 5951 and 5952, respectively. DDQN and dueling share this seed policy,
but different architectures do not have identical complete initial weights.
Each 50k pilot should complete 12251 updates and 12 target synchronizations.
Smoke runs explicitly use a 64-step warmup to exercise optimization within 512
transitions; all deviations are frozen in their configs.

Vanilla DQN remains the default. Double DQN selects the legal next action with
the online network and evaluates that action with the target network. True
termination uses reward-only targets; truncation bootstraps the returned final
observation. An empty nonterminal legal mask is an explicit error; a terminal
empty mask is accepted without an argmax. This follows the separation of action
selection and evaluation in [van Hasselt et al.](https://arxiv.org/abs/1509.06461).
A better win rate alone would not establish reduced overestimation bias; no
common-policy ground-truth action values are available in this pilot.

The optional dueling architecture uses the same MLP trunk with scalar V and 144
advantages: `Q=V+A-mean(A)` over the fixed 144-action dimension. This remains
well-defined for terminal observations and does not divide by legal-action count.
Legal masks are applied to action selection and Bellman targets, so illegal
advantages cannot cause an illegal action to be selected. They can affect the
shared baseline through the mean, as in standard fixed-action aggregation. There
are 129 extra parameters; the comparison isolates the architecture package, not
an exactly parameter-matched mechanism. See [Wang et al.](https://arxiv.org/abs/1511.06581).
No CNN changes are introduced.

## Evaluation and statistics

Every model/checkpoint uses the same frozen boards, unchanged guards and
deterministic greedy masked inference. Random is the explicitly stochastic
baseline with fixed independent per-board RNG seeds. Rows retain episode seeds,
derived board/engine seeds, win/loss, objective completion, remaining HP, episode
length, invalid actions and runtime. Initial board/config identities are checked
across algorithms. DQN milestone zero also evaluates Random and Greedy using the
same fixed seeds; those repeated copies are not independent board observations.

The **single primary comparison** is mean paired PPO LR-decay minus constant
win rate at the fixed 100k endpoint. A null-centered shared-board bootstrap test
with plus-one correction gives an approximate two-sided p-value; Holm correction
on the singleton leaves it unchanged. Percentile 95% CIs resample one common
board-index vector across all compared policies; 20000 replicates use seed 51919.
This is uncertainty over boards **conditional on the three observed training
pairs**, not a population claim about PPO training seeds. Per-pair effects and
sample SD are reported separately. A sign-flip sensitivity calculation over the
three seed-pair means has minimum attainable two-sided p=.25 and requires sign
symmetry/exchangeability under the null.

The separate **seven-test exploratory family** contains the three per-seed PPO
endpoint differences, mean dueling-minus-standard DDQN endpoint difference, and
its three per-seed differences. Individual binary contrasts use exact two-sided
McNemar tests with discordant outcome tables; the mean uses a null-centered
paired bootstrap. One Holm correction covers all seven tests. If a cohort is
incomplete, its mean is not estimated through seed substitution, and the complete
family is not silently reduced. CIs remain pointwise, not multiplicity-adjusted.

Objective/HP/episode-length/diagnostic comparisons are descriptive. Historically
trained Vanilla DQN and PPO are newly evaluated on these shared boards for
comparability, but their training seeds, budgets and histories are unmatched;
their comparisons remain descriptive. Equal transition counts also do not
equalize optimizer steps or wall-clock cost across algorithms. A development
score of 70% would be preserved and reported, not treated as held-out validation.
Nonsignificance is not equivalence; a best development checkpoint is selected on
these same boards and cannot be advertised as an unbiased final estimate.

## Execution, recoverability and safety

Each unique run has `records/<id>/config.json`, protocol/source/seed provenance,
an append-only fsynced `events.jsonl`, and an atomically updated `status.json`.
The record status is started/completed/failed/interrupted; supervisor events
retain process IDs, exit codes, queued jobs and skipped-stage reasons. Logs are
under `logs/`; training/evaluation/checkpoints are under `runs/<id>/`.

PPO journals optimizer/rollout/episode information incrementally, flushes its
transition file every rollout and exports cumulative milestone episode/move
records through the observer. DQN retains its incremental metrics file, flushes
every 256 transitions and at milestones, and saves independent milestone exports
plus episode histories. Model publication remains exclusive and atomic using
the existing temporary-file/hard-link scheme. Critical summaries/status use
atomic replacement only within the newly owned run directory. Historical outputs
are never overwritten. Interrupted runs are not exactly resumable because active
engine state and unfinished PPO rollouts are not serialized.

No automatic retries or seed replacements are configured. Failures and partial
artifacts remain in the inventory. Stage B requires a successful B smoke; Stage C
requires successful B pilots and C smoke. Time gates use only completion/resource
estimates, never win-rate selection. A major correctness issue requiring changed
methodology must stop execution for approval. The optional entropy stage has no
execution manifest: it will be skipped unless A–C finish, diagnostic evidence
supports it and enough time remains for a separately frozen exploratory addendum.
LR decay and entropy scheduling are never combined.

## Validation and results

Preflight: 61 focused tests passed in 11.64 seconds. Tests cover online selection
versus target evaluation, masking, terminal/truncated and no-action cases,
dueling aggregation/gradients/output dimensions, checkpoint round-trips,
deterministic repeated training, LR boundaries and equal PPO prefixes. Six actual
historical checkpoints loaded and produced repeatable legal actions on the legacy
board 300000. Repository Ruff and Git whitespace checks passed before freeze.

## Actual execution and failures

The program **partially completed**. The supervisor finished all queued work at
2026-11-05 21:43:01 UTC, approximately 72 minutes after the first workers started.
All workers exited within the execution window. However, the interactive session
waited for process-inspection approval until after the eight-hour deadline;
analysis and final reporting therefore occurred late. The overall reporting
window was not met. No long-running processes remain, and no training or new
evaluation was restarted after the deadline.

| Work | Original job outcome | Usable evidence |
|---|---|---|
| Six Stage A PPO attempts | Failed at collected step 0 | Six finite initial checkpoints, aggregate initial scores and paired equal initial weights; no training updates, 50k policies or final policies |
| B smoke and three DDQN pilots | Completed | Full logs, summaries, episodes/moves and checkpoints |
| C smoke and three dueling pilots | Completed | Full logs, summaries, episodes/moves and checkpoints |
| Eight baseline evaluation jobs | Export failed after all 199 episodes/job | Complete episode tables recovered from original journals; original failed statuses retained; baseline move traces were not persisted |
| D entropy scheduling | Skipped | No pilot was preregistered/executed; required A–C completion condition failed |

All 22 original jobs remain in the machine-readable inventory: eight completed
(six pilots, two smokes) and fourteen failed (six initial PPO exports, eight
baseline exports). There were **301024 production training transitions**, not
the planned 901024. Correctness-test training is separate from production runs.

I omitted required `experiment_id` metadata in the new PPO journal exporter and
baseline final exporter. The shared exporter correctly rejected the inconsistent
metadata. PPO encountered this after its initial checkpoint was saved, before
opening the transition-collection loop. Baselines encountered it after recording
all episode results into append-only journals. The focused tests initially used
an observer stub and did not exercise the actual journal/exporter integration;
new regression tests now cover that path and the baseline metadata path.

`operational_amendment_001.json` records the logging-only repair and an explicit
expired-window worker guard. The original manifest and its hash remain unchanged.
All 81 originally frozen source files were archived under `frozen_source/` before
the repair. The amendment identifies the old/new driver hashes; integrity checks
accept only this explicitly scoped driver replacement and verify its archived
original. No algorithms, hyperparameters, seeds, budgets, inference conditions or
statistical hypotheses were changed. There were **no training/evaluation retries**.
No failed files or directories were removed or relabeled as successful.

Baseline recovery is a postprocessing operation: it verifies exactly 199 unique
completed, valid episode rows per journal, writes a separate episode CSV, and
retains source journal hashes and failure statuses. It neither executes policies
nor reconstructs missing moves. All 14 final model groups (six new models and
eight baselines) have identical board/engine/config identities on all 199 boards.
The failed PPO step-zero scores are not treated as trained PPO LR results.

## Results on the common development boards

| Variant | Training budget/model | Final win rates by seed | Mean | Training-seed SD |
|---|---|---|---:|---:|
| Vanilla DQN, historically trained | 50k | 46.23%, 44.72%, 47.74% | 46.23% | 1.51 pp |
| Double DQN | 50k | 61.31%, 58.29%, 51.26% | 56.95% | 5.16 pp |
| Dueling Double DQN | 50k | 65.83%, 61.31%, 63.82% | **63.65%** | 2.27 pp |
| PPO, historically trained | 100k | 54.77%, 54.27%, 48.74% | 52.60% | 3.35 pp |
| Random | No training | 93/199 wins | 46.73% | Not applicable |
| Greedy | No training | 104/199 wins | 52.26% | Not applicable |
| New PPO constant / LR decay | Planned 100k | Failed before training | Unavailable | Unavailable |

Historical models were evaluated anew on the same boards; their *training* cohorts
remain unmatched. For example, the +10.72 pp DDQN-minus-historical-Vanilla mean
is descriptive, not a controlled algorithm-effect estimate. Reduced overestimation
bias was not empirically established because common-policy ground-truth Q-values
were not measured.

The highest observed scheduled checkpoint is **Dueling DDQN seed 5950 at 50k:
131/199 wins, 65.83%**. No observed checkpoint reaches 70%. Among the newly
trained variants, dueling has lower observed training-seed SD. Historical Vanilla
DQN has the smallest SD across the displayed three-model cohorts, but also much
lower mean performance and different training seeds; this is not evidence of a
more reliable algorithm in general. Single Random/Greedy policies have no
across-training-seed SD.

Mean objective completion is 95.77% for DDQN versus 96.51% for dueling; remaining
HP is .585 versus .472. Corresponding historical Vanilla/PPO values are
94.39%/.767 and 95.38%/.637. Random is 94.30%/.794 and Greedy 95.31%/.643.
These are descriptive secondary metrics. All completed/recovered evaluations
have zero invalid actions and completed WON/LOST outcomes.

Learning is not uniformly monotonic. DDQN seed 5952 falls from 57.29% at 25k to
51.26% at 50k; dueling seed 5951 plateaus at 61.31%. All recorded losses and
gradient diagnostics are finite. The six pilots each complete 12251 optimizer
updates and 12 target synchronizations; both smoke tests complete 113 updates.
There are no PPO training-time KL/entropy/critic/LR curves because no production
PPO optimization occurred. Sprint 5.18's historical diagnostics remain a separate
dataset and are not substituted for this missing experiment.

## Paired statistical results

Dueling minus standard DDQN has mean endpoint difference **+6.70 pp**, conditional
paired board-bootstrap 95% CI **[+1.01,+12.56] pp**, approximate raw p=.02340.
Seed-pair effects are +4.52, +3.02 and +12.56 pp, with sample SD 5.13 pp.
The three-pair sign-flip sensitivity p=.25; this is not population-level evidence
that dueling generally improves training.

Three planned PPO secondary tests are unestimable. To retain the declared
seven-test exploratory multiplicity burden rather than silently shrinking the
family, the final analysis conservatively uses non-rejection slots p=1 for those
three *only in adjustment*. They are not observed PPO p-values. The dueling mean
Holm-adjusted p is **.14039**. Per-seed exact McNemar raw p-values are
.39657/.59429/.01761, adjusted to 1/1/.12326. No tested contrast survives the
declared seven-slot correction. The positive pointwise interval and nonsignificant
adjusted p are compatible because the interval is not multiplicity-adjusted.

The primary PPO LR-decay comparison is **not estimable**. Neither improvement,
failure of LR scheduling, equivalence nor stability benefit can be inferred.
Exploratory dueling results are promising development evidence, not confirmation
of superiority or held-out performance. Three training seeds, shared boards,
architecture-dependent initial policies and 129 extra dueling parameters limit
mechanistic interpretation.

## Measured cost and retained models

DDQN total process durations were 1736.66/1744.57/1743.22 seconds; dueling durations
were 1760.51/1749.29/1754.72 seconds. Means are **29.02 versus 29.25 minutes/run**,
including validation. The six pilots total 174.82 process-minutes, including
28.40 process-minutes of validation. The two smokes total 7.32 process-minutes.
These sums are not elapsed batch time or measured CPU time. All baseline journal
process spans are about 57–63 seconds each; these upper-bound elapsed measures
include loading/export failure overhead. Their unavailable final wall timers are
not fabricated. Per-episode inference times remain recorded in recovered rows.

The training stages were slower than the historical 17.6-minute DQN processes;
the new runs evaluated 199 rather than 100 boards and used fsynced journals, with
potential system-contention differences. The available records cannot causally
attribute the runtime increase. The total autonomous worker interval was about
72 minutes. Interactive finalization occurred after the planned time window, as
explained above; no experiment was launched to consume the unused training time
after the deadline.

Retain **all six DQN-family final checkpoints**, their milestone policies and all
failed/partial artifacts. The strongest development candidate is
`runs/c_dueling_5950/checkpoint_final.pt`; selecting it on these boards introduces
selection bias. Its inference checkpoint is useful for later independently
approved evaluation, not a verified generalization claim.

Before moving to Data Science Analytics, use these artifacts as a traceable
research handoff. If further RL work is approved, first complete the unanswered
PPO LR study with the repaired exporter in a new authorized execution window,
and consider more independent dueling/DDQN pairs plus genuinely matched Vanilla
controls. Freeze any new design before outcomes; these 199 boards are now
development data. Do not run the reserved final test to choose between candidates.
No Human Gameplay or Data Science Analytics implementation was started.

## Final artifacts and validation

Authoritative combined outputs are in `final_analysis/summary.json`,
`final_analysis/comparison.csv`, recovered baseline episode tables, and three
PNG/PDF dashboard pairs: full algorithm/objective/HP comparison; learning curves,
per-model variability and cost; DQN-family optimization diagnostics. Raw values
remain in original CSVs. Diagnostic curves use trailing 50 recorded values except
LR, and never smooth across runs. The initial `analysis/` output is retained as
an audit of completed training jobs before baseline recovery; its empty PPO
panels do not represent measured PPO stability and are not substantive results.

`execution_complete.json` retains all original statuses. Additional audit files
include `checkpoint_audit.json`, `integrity_audit.json`,
`paired_board_identity_audit.json`, `stage_A_failure_audit.json`, and the explicit
operational amendment. The original manifest, logs, complete/failed run
directories and original source archive remain available.

Final validation: **812 full-suite tests passed in 58.49 seconds**, including the
new recovery/export regressions. Ten focused tests also passed in 4.59 seconds.
Repository-wide Ruff,
formatting of all nine changed/added Python files, and Git whitespace checks
passed. All 36 new checkpoints loaded with finite weights and matching configs:
30 belong to completed training/smoke runs, six to failed PPO initial attempts.
Completed run summaries reconcile budgets, optimizer steps, metric rows,
checkpoint round-trip and final/milestone weights. All **633 historical
artifacts remain hash-identical**. Only the intended pre-existing DQN agent,
DQN trainer and PPO trainer were extended; unrelated work was preserved. No
stage, commit, push, reset, rebase, stash or discard operation was performed.
