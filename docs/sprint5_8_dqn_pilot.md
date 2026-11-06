# Sprint 5.8: staged DQN pilot

This sprint retains vanilla DQN and the Sprint 5.7 encoder, network, replay,
Bellman targets and sparse rewards. It adds a reserved seed protocol, explicit
evaluation mode/isolation, recorded training statistics, learning curves and a
single approved 5000-step experiment. Gameplay, levels and baseline agents are
unchanged. No final-test evaluation or 50000-step run is authorized here.

## Configuration review

| Component | Review and decision |
|---|---|
| Observation | 1467 float32 features; categorical one-hot tiles/specials/blockers; COLORLESS has a separate category; masked holes and padding remain distinct |
| Scaling | HP/3, moves/15, normalized dimensions, log1p(target)/log1p(65535), capped progress/target; no running statistics or validation-fitted normalization |
| Masks | Separate from model features; uniform legal exploration, legal argmax exploitation, and legal-only target maximization |
| Targets | Frozen target MLP; terminal reward only; truncation bootstraps from the returned final observation; empty nonterminal masks fail explicitly |
| Replay | 50000 raw transitions, copied arrays, local uniform sampling without replacement within each 64-entry minibatch |
| Optimizer | Adam 3e-4, Huber loss, gamma .99, global norm clip 10; finite losses/targets/gradients checked |
| Collection | One update per four steps after 1000-step warm-up |
| Target sync | Hard copy every 1000 optimizer updates; only one copy is expected in this 5000-step pilot, at step 4996 |
| Exploration | Linear 1 → .05 over 50000 steps, unchanged; epsilon at step 5000 is .905, so this pilot remains highly exploratory |
| Evaluation | Every 1000 steps, 100 episodes per agent, epsilon zero, scoped eval mode, separate environments, parameter/RNG mutation checks |
| Checkpoints | Exclusive creation, CPU loading, weights/target/optimizer/local RNG restoration, full final replay state; boundary resume is preserved |

The review found no reproducible algorithm defect requiring a change. In
particular, this MLP has no dropout or BatchNorm, so adding explicit eval mode
does not change its numerical policy. It clarifies and tests the boundary for
future layers. Existing Sprint 5.7 full checkpoints remain compatible because
the new optional seed-policy field defaults to None when absent.

## Seed protocol

Environment partitions are inclusive:

- Training: **200000–299999**
- Validation: **300000–300999**
- Final test: **400000–400999**, reserved and unused

The pilot starts sequential episode seeds at 200000 and uses exactly
300000–300099 for every validation checkpoint. Its master seed is **5800**;
network/exploration/replay streams are independently spawned. Random baseline
agent seeds are 600000–600099; Greedy is deterministic and does not consume them.
The agent seeds are metadata namespaces separate from environment partitions.

`TrainingConfig(seed_policy="sprint5_8")` rejects training intervals outside the
training partition and validation intervals outside the validation partition.
The conservative training interval allows at most one new episode per step.
Other existing callers can keep the legacy unrestricted policy. The pilot CLI
has no final-test selection option. Analysis checks recorded training and
validation seeds before writing a report.

## Reproduction

Use the existing `match3-ai` Conda environment with the optional `[rl]` and
`[analysis]` dependencies already installed. No new dependency is added.

Review without training:

```sh
conda run -n match3-ai python -m match3_ai.training.pilot \
  --review-only --output-dir <new-pilot-directory>
```

Run only the approved pilot:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.pilot \
  --output-dir experiments/results/<new-pilot-directory>
```

Generate seven learning curves in PNG and PDF, descriptive summaries and a
Markdown report from that existing run:

```sh
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run -n match3-ai \
  python -m match3_ai.training.analysis \
  --input-dir experiments/results/<pilot-directory> \
  --output-dir experiments/results/<new-analysis-directory>
```

Existing output directories, figure directories and checkpoint files are
rejected. Analysis hash-checks its source artifacts and never evaluates a policy
or generates another training transition.

## Metric definitions and interpretation

Training win rate and mean reward use naturally terminated training episodes;
the final budget truncation is recorded separately and excluded from that win
denominator. Episode reward curves include that explicitly recorded truncation.
Evaluation win rates include every attempted episode in the denominator and
use the existing Wilson 95% intervals. Objective completion is the existing
mean of capped per-objective completion fractions. Remaining HP is total HP
across surviving blockers. Missing/error values are retained by existing analytics.

Loss, Q-value and gradient statistics are logged on optimization steps only.
Gradient norm is measured before clipping. The replay occupancy and target-sync
counters are cumulative; completed-episode statistics describe the current
training invocation when resuming. Evaluation time is measured separately from
collection/optimization plus checkpoint and orchestration overhead. The total
training timer ends before gameplay CSV export and analysis/plot generation.

The longer-run estimate scales measured active time by ten and validation time
by two: 50000 steps with ten 300-episode validation rounds, versus the pilot's
five rounds. Three independent runs triple that estimate. It is a linear planning
estimate with substantial hardware, episode-length and workload uncertainty.

Reused validation seeds support paired comparisons across checkpoints but
cannot serve as an unbiased final test after selection. Baseline repeats on
the same seeds are consistency checks, not additional independent samples.
One short training seed, high remaining epsilon, sparse rewards and one late
target copy limit learning conclusions. Fast tests check repeatability and
checkpoint evaluation; the full pilot is not repeated during pytest.

PPO, reward shaping, advanced DQN variants and longer training remain subject
to separate approval.

## Completed pilot

The single approved run used experiment ID `773e04273a8b05e31f9eea46` and wrote
artifacts to `experiments/results/sprint5_8_dqn_pilot/`. The final report, numeric
analysis and seven curves in PNG/PDF are in
`experiments/results/sprint5_8_dqn_pilot_analysis_final/`. Earlier analysis outputs
were preserved. The final checkpoint is `checkpoint_final.pt` in the run directory.

| Steps | DQN validation wins / 100 | Random wins / 100 | Greedy wins / 100 |
|---:|---:|---:|---:|
| 1000 | 34 | 45 | 57 |
| 2000 | 52 | 45 | 57 |
| 3000 | 53 | 45 | 57 |
| 4000 | 46 | 45 | 57 |
| 5000 | 53 | 45 | 57 |

Final Wilson 95% intervals: DQN 43.29–62.49%, Random 35.61–54.76%,
Greedy 47.22–66.27%. Final objective completion was 95.25%, 93.92% and 94.92%,
respectively; mean remaining blocker HP was .64, .87 and .73. The DQN point
estimate exceeds Random and remains below Greedy; this does not establish
superiority. No step-zero untrained policy was evaluated.

The run took 303.87 seconds, including 231.87 seconds of validation and 72.00
seconds of collection/optimization and overhead. Training seeds actually used
were 200000–200393: 184 wins, 209 losses and one budget truncation. Natural
training win rate was 46.82%, mean reward -.06361. Replay reached 5000 / 50000;
1001 optimizer updates occurred and the target synchronized at step 4996.

All losses and gradients were finite. Mean loss was .02266; the first and last
100 updates averaged .04046 and .01828. Maximum pre-clip norm was .3362. Observed
valid-action Q extrema were -1.0925 and 1.1724: modest function-approximation
overshoot beyond the true sparse-return bounds, with no observed divergence.
Collection epsilon decreased from 1 to .905019 on the last collected action.
There were zero invalid actions, errors or validation time limits; checkpoint
inference passed. All five baseline rounds reproduced identical gameplay
outcomes, and paired board/engine seeds matched across all policies/checkpoints.

The measured projection for a separately approved 50000-step run with validation
every 5000 steps is **19.73 minutes per run**, **59.19 minutes for three**. Allow
headroom, approximately 20–30 minutes per run. Candidate master seeds are 5801,
5802 and 5803, with the same partitions, hyperparameters and validation seeds.
No longer run or final-test evaluation was executed. Longer training is practical,
but one seed, high exploration and one late target copy limit conclusions here.

Validation added 11 fast test cases covering partitions, deterministic/checkpoint
evaluation, mode restoration, unchanged policy/RNG/optimizer state, metrics,
figures, seed-leakage rejection and overwrite protection. The focused DQN and
evaluation suites passed 63 tests. The full suite passed 709 tests; Ruff and
Git whitespace checks passed. Existing tracked and untracked work was preserved.
