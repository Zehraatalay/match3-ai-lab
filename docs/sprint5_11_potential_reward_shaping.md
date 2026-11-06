# Sprint 5.11: potential-based reward shaping

PBRS is an optional training reward adapter. Sparse reward remains the default
in `TrainingConfig`, `Match3LevelEnv`, `LevelSession` and all gameplay analytics.
No engine, objective, observation, model architecture, optimizer, replay schema,
exploration schedule or DQN algorithm was changed. No new dependency is added.

## Potential and reward definition

For every active observable objective slot, progress is normalized by its target
and clamped to [0,1]. The potential is the **equal-weight mean** of those ratios.
Inactive padding is excluded. Active invalid/nonpositive/nonfinite targets and
nonfinite progress contribute zero while retaining their slot weight, avoiding
NaN/division-by-zero or silent weight redistribution. With no active objectives,
potential is zero. Unsupported active objective types fail explicitly.

The current codes support `CLEAR_ALL_BLOCKERS` (1) and `COLLECT_TILES` (2).
Blocker progress means completed removals from the existing objective counters;
partial HP damage is not separately rewarded. Tile progress comes from the
existing objective accounting, including color-specific collection. No unrelated
tile clears enter a blocker-only potential. No future RNG or hidden outcomes
are used.

```text
Phi(s) = mean_active(clip(objective_progress / objective_target, 0, 1))
Phi(true_terminal) = 0
r_training = r_sparse + beta * (gamma * Phi(next) - Phi(current))
```

Enable with `TrainingConfig(reward_mode="pbrs", shaping_beta=0.5)`. Default
`reward_mode="sparse"` ignores beta for training rewards. A PBRS beta of zero is
exactly sparse-compatible. The adapter always uses `config.gamma`, the same
discount passed to the unchanged DQN update; the pilot uses **.99**.

True termination includes both wins and move-budget losses. Artificial safety
or collection-budget truncations retain next-state potential and bootstrap from
the actual returned next observation. If both flags are true, termination wins.
There is no reward clipping or adaptive beta. At stalled positive potential,
the discount creates a small negative shaping term; a denser signal does not
mean every transition provides new objective-progress information.

## Mathematical behavior and assumptions

For any T-step prefix, direct cancellation gives:

```text
sum(t=0..T-1) gamma^t * (r_shaped[t] - r_sparse[t])
    = beta * (-Phi(s0) + gamma^T * Phi(sT))
```

For true-terminal episodes, Phi(sT)=0. At fixed initial state, the offset is
independent of actions, even for different episode lengths. On this level,
initial progress and Phi(s0) are zero, so complete discounted sparse/shaped
returns agree. Undiscounted reward sums need not agree.

The relevant policy-invariance result is potential reward transformation with
the same discount; see [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
Here the claim requires a fixed bounded potential and beta, matching discount,
unchanged transitions/actions, policy-independent initial-state distribution,
and correctly treated terminal/continuation boundaries. A truncated prefix has
a remaining boundary term; bootstrapping must use the shaped continuation value.
Using the wrong discount, retaining a nonzero true-terminal potential or falsely
zeroing an artificial truncation can violate those conditions. Tests demonstrate
both the valid identity and invalid-boundary/discount counterexamples.

This preserves the original **discounted-return objective** under the stated
conditions; it does not equate that objective to undiscounted win rate. It does
not guarantee finite-budget function-approximation convergence, identical learned
actions or empirical improvement. Float32 replay adds normal numerical rounding.
No stronger invariance claim is made when these assumptions are violated.

## Integration and logging

`training/reward_shaping.py` contains pure observable-potential/reward functions.
`training/dqn.py` computes the reward after deciding the actual termination and
artificial truncation flags, then passes the selected training reward into the
existing replay buffer. Evaluation uses the unwrapped sparse environment and
existing evaluator; shaping is never applied there.

Per-transition logs keep `sparse_reward`, `shaping_bonus`, `shaped_reward`,
`potential_current` and `potential_next`. Episode logs retain the original sparse
`reward` field plus sparse/shaped/bonus sums and discounted sparse/shaped returns.
The existing `episode_reward` and training sparse-return statistics retain their
meaning. Natural-terminal means exclude the separately recorded budget truncation.
Full checkpoints retain reward settings, so incompatible reward-mode resumes
are rejected; older sparse checkpoints receive backward-compatible defaults.

The audit recomputes every shaping component from replay observations, checks
float32 replay rewards against configured training rewards, verifies episode
sums and discounted boundary identities, and verifies sparse validation rewards.
Zero initial models, seed streams and paired validation board/gameplay seeds are
also checked.

## Controlled pilot

Both fresh arms use master seed **5903**, Blocker Mayhem, 5000 environment steps,
gamma .99, beta .5, 1467→256→128→144 MLP, replay 50000, batch 64, warm-up 1000,
Adam 3e-4, Huber loss, clipping 10, update every four steps and hard target copy
every 1000 optimizer updates. Epsilon retains the existing 1→.05 schedule over
50000 steps; it reaches .905 at this pilot's endpoint.

Training episode seeds start at **240000**. Validation uses **300000–300099** at
steps 0,1000,...,5000, with exploration disabled; baseline agent seeds start at
600000. Reserved final-test seeds 400000–400999 are never used. Model initialization,
optimizer/replay settings and local RNG stream seeds match between arms. Actions
and subsequent stochastic trajectories may diverge after learning differs.

The two configurations differ only in `reward_mode`. Processes run concurrently,
each using one Torch CPU thread. The sparse control reproduces the exact online
weights in the historical Sprint 5.9 seed-5903 step-5000 checkpoint, despite the
additional logging and more frequent isolated evaluation.

| Step | Sparse validation wins / 100 | PBRS validation wins / 100 |
|---:|---:|---:|
| 0 | 36 | 36 |
| 1000 | 42 | 43 |
| 2000 | 51 | 51 |
| 3000 | 50 | 47 |
| 4000 | 53 | 55 |
| 5000 | 49 | 47 |

Final PBRS-minus-sparse win difference is **-2 pp**, paired percentile-bootstrap
95% CI **[-15, +10.025] pp**, exact exploratory McNemar p **.8776**. Paired counts:
27 both win, 20 PBRS-only wins, 22 sparse-only wins, 31 both lose. Final objective
completion is 94.50% versus 94.83%; remaining HP .75 versus .72. Their paired
differences are -.00333 [-.01833,.01167] and +.03 [-.20,.26]. There is no clear
pilot advantage and no evidence of equivalence; one short seed and correlated
validation checkpoints limit interpretation. Best validation scores are selected
on validation and are not unbiased final-test estimates.

Both arms complete 1001 updates, one target synchronization and replay occupancy
5000, with no invalid actions or evaluation errors. PBRS changes zero training
reward fraction from sparse's 92.26% to 0%; 31.26% of bonuses are positive and
68.74% negative. The maximum logged discounted-identity error is 2.22e-16.
Mean terminal-episode sparse/shaped sums are -.10078/-.10078 for sparse and
-.15707/-.20381 for PBRS. These undiscounted shaping sums are not the invariant
discounted returns or a substitute for sparse evaluation.

Mean Huber loss is .02235 versus .03449; maximum pre-clip gradient norm .2772
versus .4593. All losses/Q-values/gradients remain finite. Q extrema are
[-1.1717,1.1694] versus [-1.5014,.8415]. Reward scales/value functions differ, so
these differences do not establish improved value calibration or policy quality.

Wall times are 439.96 versus 440.70 seconds, including validation 242.19 versus
242.58 seconds. Active-time difference is about +.18% for PBRS. Shared system
load, trajectory differences, checkpoint work and potential logging affect these
timings; they are not an isolated causal estimate of formula overhead.

## Reproduction and artifacts

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.pbrs_pilot \
  --mode sparse --output-dir experiments/results/<new-sparse-directory>

conda run --no-capture-output -n match3-ai python -m match3_ai.training.pbrs_pilot \
  --mode pbrs --output-dir experiments/results/<new-pbrs-directory>
```

Use `--review-only` to inspect either configuration without training. The CLI
retains the approved fixed 5000-step profiles; it does not launch long training.
Actual outputs are `experiments/results/sprint5_11_sparse/` and
`experiments/results/sprint5_11_pbrs/`.

```sh
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run -n match3-ai \
  python -m match3_ai.training.pbrs_analysis \
  --sparse-dir experiments/results/sprint5_11_sparse \
  --pbrs-dir experiments/results/sprint5_11_pbrs \
  --output-dir experiments/results/<new-analysis-directory> \
  --historical-checkpoint experiments/results/sprint5_9_seed_5903/policy_step_5000.pt
```

The completed report, JSON audit and eight PNG/PDF figures are under
`experiments/results/sprint5_11_pbrs_analysis/`. Existing experiment/checkpoint
paths are rejected rather than overwritten. Input hashes are verified unchanged.

## Recommendation and validation

Keep beta .5 fixed; do not automatically increase it or extend the budget.
If this research direction continues, the next approved experiment should be
a small paired replication on two additional master seeds at the same 5000-step
budget/settings before deciding whether a longer experiment is justified. The
current pilot verifies reward-design correctness and density, not superiority.

Tests cover normalization/invalid targets, multi-objective weights, termination,
truncation, matching/mismatched discounts, telescoping examples, beta-zero exact
training equivalence, observation/RNG preservation, replay integration, sparse
evaluation isolation, deterministic repeatability, checkpoint compatibility and
analysis/plot export. Seventeen new test cases pass; the focused DQN/PBRS/
evaluation selection passes 85 tests and the full suite passes 743 tests.
Ruff, tracked Git whitespace and additional new-file whitespace checks pass.
No long training, PPO, architecture changes, level changes or Git staging/history
operations were performed.
