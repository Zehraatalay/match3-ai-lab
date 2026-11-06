# Sprint 5.13: masked PPO implementation and smoke training

PPO is an additional agent; MLP remains the default DQN architecture. No engine,
level, environment, reward, DQN, Pygame or existing evaluation behavior changed.
The implementation uses existing Torch support and adds no dependencies.

## Integration and architecture

New files:

- `src/match3_ai/agents/ppo_agent.py`: actor-critic, masked distribution, local
  sampling, deterministic agent interface and exclusive checkpoint publication.
- `src/match3_ai/training/ppo_update.py`: GAE, advantage normalization, clipped
  policy/value/entropy losses and bounded minibatch optimization.
- `src/match3_ai/training/ppo.py`: validated configuration, single-environment
  on-policy collection, sparse smoke profile, evaluation and logs.
- `tests/test_ppo.py`: 18 deterministic focused cases.
- This report; README links to the new entry point.

The shared ReLU backbone is **1467 → 256 → 128**, with separate actor
**128 → 144** and critic **128 → 1** heads: **427,409 parameters**. Hidden widths
are configurable. There is no CNN-PPO, dropout, batch normalization or separate
copy of game rules. Standard Torch Linear initialization runs inside an isolated
CPU RNG scope. Actor and critic share backbone gradients.

`ObservationEncoder` is reused unchanged. The input contains the existing
20×9×8 spatial planes and 27 board/objective values, including masks, bounds,
blocker HP, objective targets, progress and moves. The legal-action mask remains
separate. Encoding restrictions and normalizations are identical to DQN version 1.

`PPOAgent.select_action(observation)` is deterministic and satisfies the existing
Agent protocol. `act(observation)` samples for training and returns action, masked
log probability and state value. `value(observation)` can evaluate terminal/final
observations without requiring a nonempty action mask.

## Legal-action masking

The shared `masked_distribution` helper validates finite logits and a nonempty
boolean mask for every state, replaces invalid logits with negative infinity,
then constructs a categorical distribution. Invalid probabilities are exactly
zero; valid probabilities sum to one within float32 precision. Deterministic
evaluation selects the valid maximum probability, with ordinary first-index
argmax tie breaking. Training samples using `torch.multinomial` with a dedicated
CPU generator, avoiding Categorical.sample's global RNG.

Collection stores masks and old masked log probabilities. Optimization reuses
those masks to compute new log probabilities and masked entropy; stored actions
are checked against them. Entropy is in nats and is not normalized by the number
of legal actions. All-invalid action selection raises ValueError. A true terminal
observation may have an empty mask because no distribution is constructed for its
bootstrap. Unexpected recovery/termination failures are surfaced explicitly,
with training_error.json, rather than silently resetting or inventing outcomes.

## GAE and PPO objective

For each transition, stored next_value refers to the actual final observation:

```
delta[t] = reward[t] + gamma * (1 - terminated[t]) * next_value[t] - value[t]
adv[t] = delta[t] + gamma * lambda * (1 - (terminated[t] or truncated[t])) * adv[t+1]
return[t] = adv[t] + value[t]
```

True termination has zero bootstrap, including when both flags are true.
External/safety/budget truncation retains the final observation's value but stops
the recurrence before the next reset episode. Ordinary rollout boundaries also
bootstrap their final next observation; they do not reset the environment.
The reverse scan starts with zero trailing advantage. These choices distinguish
value bootstrapping from propagation of sampled advantages across episodes.
Tests verify hand-calculated deltas, boundary precedence and reset isolation.

Collected values, log probabilities, returns, masks and advantages are detached
fixed targets. Advantage normalization uses the complete rollout's population
standard deviation, clamped to at least 1e-8. Constant or singleton advantages
therefore normalize to zero without NaNs. They still permit value/entropy updates.

```
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv))
value_loss = 0.5 * mean((value - return)^2)
loss = policy_loss + value_coef * value_loss - entropy_coef * mean(entropy)
```

This is clipped policy PPO with unclipped squared value regression. No value
clipping, replay, target network, reward shaping, learning-rate decay or KL early
stopping is added. Diagnostic approximate KL is mean((ratio−1)−log(ratio)); clip
fraction counts abs(ratio−1)>eps. Logs describe each minibatch before its update;
gradient norm is measured before clipping. Aggregate means are per optimizer
update, not transition-weighted averages. Rollouts are discarded after their
fixed optimization epochs and never reused as off-policy replay.

## Configuration and reproducibility

| Setting | Smoke value |
|---|---|
| Level / master seed | blocker-mayhem / 5913 |
| Budget | Exactly 5,000 environment transitions |
| Rollout / minibatch / epochs | 256 / 64 / 4 |
| Gamma / GAE lambda | .99 / .95 |
| Clip ratio | .2 |
| Value / entropy coefficients | .5 / .01 |
| Optimizer / learning rate | Adam / .0003 |
| Global gradient clipping | .5 |
| Reward | Existing sparse −1/0/+1; PBRS disabled |
| Episode / cascade / initialization guards | 50 / 200 / 1,000 |
| Validation | 100 fixed seeds 300000–300099, step zero and final |
| Compute | CPU, one Torch thread, deterministic Torch algorithms |

The master SeedSequence spawns independent environment and agent streams. The
agent stream spawns model initialization, policy sampling and minibatch shuffling
streams. Actual derived seeds are in training_summary.json. Environment resets
draw seeds locally with replacement from **200000–299999**; each episode's seed
is recorded in transitions.csv and training_episodes.json. Evaluation uses new
environments and deterministic actions without advancing policy/shuffle RNGs.
The reserved final-test seeds **400000–400999 were not accessed**. No new Random
or Greedy baseline runs were necessary for this smoke experiment.

Same-config tiny runs reproduce transition logs, optimization statistics and
final weights exactly in the installed CPU environment. Model initialization and
sampling do not consume global Torch RNG. CPU thread count/determinism settings
are restored after training. Bitwise reproducibility across different Torch,
BLAS, CPU or OS versions is not promised. Installed versions: Python 3.11.17,
Torch 2.15.0, NumPy 2.4.6 in the match3-ai Conda environment.

Reproduce only into a fresh directory:

```sh
conda run -n match3-ai python -m match3_ai.training.ppo --output-dir experiments/results/NEW_ppo_smoke --review-only
conda run -n match3-ai python -m match3_ai.training.ppo --output-dir experiments/results/NEW_ppo_smoke
```

The CLI runs this fixed smoke profile. Python callers can use PPOTrainingConfig
and train_ppo for explicitly approved future profiles. No long or multi-seed
training was launched here. Architecture/optimization/boundary tests and existing
DQN/evaluation regressions passed before the smoke run started.

## Smoke results

| Metric | Step zero | Final (5,000) |
|---|---:|---:|
| Wins / losses / errors | 42 / 58 / 0 | 34 / 66 / 0 |
| Win rate, Wilson 95% CI | 42% [32.8%, 51.8%] | 34% [25.5%, 43.7%] |
| Mean objective completion | 94.08% | 91.92% |
| Mean remaining blocker HP | .82 | 1.14 |
| Mean episode reward | −.16 | −.32 |
| Mean moves used | 13.20 | 13.12 |
| Mean special activations | 20.16 | 17.04 |
| Invalid actions / time limits | 0 / 0 | 0 / 0 |

The run collected 20 rollouts, including a final partial rollout of 136 steps,
and performed **316 optimizer updates**. Training completed 379 natural-terminal
episodes (151 wins) plus one budget-truncated episode. Mean terminal training
reward was −.20317. The policy weights changed. Training used zero invalid
actions and sparse rewards throughout.

| Optimization diagnostic | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| Policy loss | −.52562 | .34505 | −.02510 |
| Value loss | .10733 | .32485 | .21904 |
| Entropy (nats) | 2.80595 | 3.25226 | 3.03571 |
| Approximate KL | .00000 | .01972 | .00477 |
| Clip fraction | .00000 | .39063 | .06542 |
| Gradient norm, pre-clip | .40263 | 4.68106 | 1.09904 |

Every recorded loss, entropy, KL estimate and gradient norm was finite. Norms
above .5 are expected in pre-clipping diagnostics. Total timed training was
**93.30 s**, including **32.10 s** validation; active collection/optimization and
checkpoint work took **61.20 s**. Full-suite tests ran concurrently, so timings
include system contention and exclude final JSON/CSV exports. These numbers are
not an isolated PPO-versus-DQN speed comparison.

An additional read-only audit loaded both checkpoints on all 100 validation
initial states. Their first actions matched recorded evaluation actions exactly
(200/200), repeated inference was deterministic and policy RNG remained unchanged.
Invalid probabilities were exactly zero; maximum probability-sum error was
1.79e-7. Checkpoint file hashes were unchanged by the audit.

Final-minus-initial win difference is **−8 percentage points**, with exploratory
paired bootstrap 95% CI **[−22, +5]**; exact McNemar p=.31233. The paired table is
14 both-win, 20 final-only, 28 initial-only, 38 both-lose. Completion fell by
2.17 pp (paired CI [−4.08, −.33]); remaining HP rose by .32 (CI [.05, .61]).
These are pointwise descriptive comparisons on 100 reused validation seeds,
with 5,000 paired resamples and fixed analysis seed 51313. No multiple-endpoint
confirmatory claim is made. They condition on one training run and do not
measure training-seed uncertainty. Nonsignificance is not equivalence.

This smoke run verifies the implementation path, not useful learning or PPO
superiority. Several observed metrics deteriorated. There is no basis to call
PPO better than DQN: initialization, training seed policy, update budget and
algorithm differ. Do not infer an implementation bug solely from a weak short
learning result; separate numerical, boundary, masking and checkpoint checks pass.

## Checkpoints, artifacts and limitations

Checkpoints use algorithm-specific/versioned metadata and contain actor-critic
weights, Adam state, encoder configuration, model dimensions, stream seeds,
policy generator state, shuffle RNG state, training configuration and progress.
Trainer progress includes environment RNG, step/update/rollout counts and level
configuration identity. Loading restores the agent for inference, including RNG
continuation. DQN checkpoint code and existing checkpoint files are untouched.
Atomic publication uses an exclusive hard link, preventing silent replacement.

Exact training resume is **not implemented**: checkpoints do not serialize a
mid-episode engine state or unfinished rollout. Saving RNG and optimizer states
does not imply continuation equivalence across missing environment state. No
vectorized/autoresetting environment integration is provided; the single-env
collector consumes final observations directly before explicit resets.

Fresh artifacts are in `experiments/results/sprint5_13_ppo_smoke/`:

- `policy_step_0.pt`, `checkpoint_final.pt`
- `transitions.csv`: actions, sparse rewards, old log probabilities, values,
  final-observation bootstraps, episode seeds and both termination flags
- `optimization.csv`: all per-minibatch diagnostics
- `training_summary.json`, `training_episodes.json`
- `evaluation/episodes.csv`, `evaluation/moves.csv`, evaluation metadata
- `smoke_audit.json`: checkpoint hashes, probability/action checks, paired results
- A copy of this report and the isolated Sprint 5.13 patch

Focused validation: **111 tests passed**, including **18 new PPO cases**. They
cover output shapes, finite outputs, illegal-probability/gradient zeros, entropy,
log probabilities, all six levels, deterministic evaluation, local RNGs, GAE,
clipping signs, normalization, final-observation bootstrapping, immutable rollout
targets, finite updates, checkpoint/optimizer/RNG round trips, seed bounds and
repeatable tiny training. Full suite: **774 passed**. Ruff, formatting and Git
whitespace checks passed. Existing work was preserved; nothing was staged or
committed.

The next experiment should require approval of a bounded, preregistered
multi-seed PPO learning study with sparse outcome evaluation, explicit budgets
and equal evaluation boards. Retain these smoke settings as a documented starting
point; do not automatically extend training, enable PBRS or tune several factors
in response to this single result. No long multi-seed run was started.
