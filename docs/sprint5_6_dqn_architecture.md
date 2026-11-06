# Sprint 5.6: initialization reliability and proposed DQN architecture

Historical design review: the approved implementation and current First Match
target-30 calibration are documented in [Sprint 5.7](sprint5_7_dqn.md).

Status: initialization budget plumbing is implemented; the DQN design below is a
proposal for approval. No neural networks, training loop, replay buffer or PyTorch
dependency were implemented or installed in this sprint.

## Initialization API

`create_level_board(config, max_attempts=1000, seed=...)` retains its algorithm and
default. `LevelSession(..., max_initialization_attempts=1000)` forwards the budget
on construction and reset. `Match3LevelEnv(..., max_initialization_attempts=1000)`
also retains the original budget following the paired color-count comparison. The legacy
`Match3Env`, direct generator calls, JSON-configured seeds and Pygame's ordinary
`LevelSession` initialization retain their previous behavior. The engine cascade
budget remains a separate 200-phase guard.

Budgets are positive Python integers; booleans, fractions, strings and None are
rejected. Exhaustion still raises `LevelGenerationError` naming the level and
configured attempt count. There is no alternate-seed retry or constructive fill.
Increasing a budget leaves every candidate and the first accepted board unchanged
for runs that previously succeeded. Engine randomness remains an independent
seed stream. Reset info exposes the initialization budget; level experiment
configuration/metadata and the benchmark CLI record and forward it explicitly.

First Match now uses five colors; all other level fields are unchanged. The
provisional 5000-attempt default was superseded rather than adopted. With the
historical three-color configuration, environment seed 52028 still first accepts
at attempt 1180. The Sprint 5.5 diagnostic explicitly freezes that old three-color
configuration and requests 1000, preserving the regression independently of the
live level. Old datasets and reports are not modified.

### Paired color-count decision

At a fixed budget of 1000 on identical environment seeds 52000–52099:

| First Match colors | Successes | Mean attempts, all / successful | Maximum observed attempts | Mean initialization ms | Valid accepted boards |
|---|---:|---:|---:|---:|---:|
| 3 | 99/100 | 150.15 / 141.57 | 1000 (failed run capped) | 26.36 | 99/99 |
| 5 | 100/100 | 6.26 / 6.26 | 29 | 2.83 | 100/100 |

All accepted boards had no initial matches and at least one legal move. Board and
engine seed streams were paired exactly; configurations differed only in num_colors.
Raw comparison data are in `experiments/results/sprint5_6_first_match_colors/`.
The increase to 5000 is unnecessary on this sample; a finite budget still cannot
guarantee every future seed succeeds. The configuration fingerprint changes:
**previous First Match gameplay benchmark results cannot be compared directly to
the new five-color level**. Collection target 300 and move limit 5 were preserved;
gameplay balance needs fresh evaluation and is not established by reset validity.

Initialization profiling is separate from gameplay benchmarking:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.evaluation.initialization_benchmark
```

It measures all six levels on seeds 52000–52099 (600 resets), counts actual
candidate generator calls, and writes raw CSV and JSON under the already ignored
`experiments/results/sprint5_6_initialization/`. Existing paths are rejected.
Constructor/resource-loading time is excluded. Timings include reset, generation,
observation/action-mask construction and a lightweight call-count wrapper. The
counter does not draw RNG values. These are serial, hardware-dependent timings.
Per-level results are recorded in the benchmark JSON and the sprint report.
A finite successful sample cannot guarantee all future seeds/configurations fit
1000 attempts. Rejection sampling can remain expensive or fail on impossible
configurations; training must preserve ERROR diagnostics instead of inventing losses.

### Measured initialization results

The earlier isolated serial run, using the provisional 5000 budget and the old
three-color First Match, succeeded on **600/600 resets**. These are historical
measurements, not results for the current five-color First Match.
No gameplay episodes were rerun. The earlier profile ran alongside validation;
use the isolated results below for latency comparisons. Timing remains specific
to this machine and includes the lightweight instrumentation described above.

| Level | Successes | Attempts mean / p95 / max | Reset mean ms | Reset p95 ms |
|---|---:|---:|---:|---:|
| First Match | 100/100 | 151.95 / 464.30 / 1180 | 28.50 | 80.94 |
| Wooden Trouble | 100/100 | 21.25 / 60.00 / 106 | 7.19 | 14.01 |
| Double Trouble | 100/100 | 5.42 / 13.00 / 30 | 4.96 | 6.70 |
| Frozen Garden | 100/100 | 16.95 / 49.05 / 126 | 9.33 | 15.23 |
| Stone Breaker | 100/100 | 11.97 / 32.20 / 92 | 8.13 | 11.93 |
| Blocker Mayhem | 100/100 | 25.27 / 79.70 / 137 | 13.54 | 24.81 |

Raw per-reset attempts, board fingerprints, seeds, times and failures are in
`experiments/results/sprint5_6_initialization_final/initialization.csv` and
`initialization_benchmark.json`. First Match seed 52028 is the maximum at 1180.
The finite sample includes known development seeds and is not an unseen guarantee
of reliability. All boards that accepted before 1000 retain their prior candidate
and gameplay RNG streams; seed 52028 now proceeds farther in its same candidate stream.

## Existing components to reuse

- `Match3LevelEnv` supplies the fixed observation, 144 actions, legal-action mask,
  sparse reward, seed handling and level outcome contract.
- `LevelSession` retains move/objective accounting; `Match3Engine` retains rules.
- `Agent.select_action(observation) -> int` remains the evaluation protocol.
- `run_episode`, `run_level_experiment`, `AnalyticsCollector`, CSV export and the
  paired statistical analysis remain the evaluation path.
- Random and Greedy remain fixed reference policies. No training assumptions
  should be added to the game engine, session, Pygame UI or observation schema.

## Dependency and device proposal

Inspected environment: macOS arm64, Python 3.11.17, NumPy 2.4.6, Gymnasium 1.4.0,
Matplotlib 3.11.2; PyTorch is not installed. Propose one optional dependency,
`torch`, under a future `[rl]` extra, with the tested stable macOS arm64 version
pinned after approval. No torchvision, torchaudio, TorchRL, Stable Baselines,
TensorBoard or data-science packages are required for this initial DQN.
Check the selected wheel against Python and NumPy before committing to a version.
The official installation documentation supports macOS and currently lists
Python 3.11+ for the latest stable build. Nothing is installed in this sprint.
[PyTorch installation requirements](https://pytorch.org/get-started/locally/).

Start on CPU for functional/reproducibility tests; consider MPS only after local
availability and performance are measured. No CUDA assumption fits this machine.
The small model may be cheaper on CPU once transfer and environment latency are
included. Record device, thread settings, library versions and numerical precision.

## Observation encoder: proposed fixed contract

Use an independent, pure NumPy encoder accepting only the observation. It returns
float32 features and a separate bool action mask, without mutating input arrays,
consulting LevelSession or reading info/engine RNG state.

All current JSON levels use at most five colors. Freeze the initial vocabulary
explicitly rather than treating raw integer IDs as ordered quantities. Reject
unsupported IDs or changed objective capacity; never silently map unknown values
onto existing colors. A later vocabulary change creates a new encoder/checkpoint
version. The COLORLESS value 255 is a distinct tile category, not a large scalar.

| Spatial feature | Channels | Encoding |
|---|---:|---|
| Tiles | 7 | EMPTY, colors 1–5, COLORLESS; one-hot |
| Specials | 6 | NONE plus five existing special IDs; one-hot |
| Blockers | 4 | NONE, CRATE, ICE, STONE; one-hot |
| Blocker HP | 1 | HP / 3 for current variants; retain raw HP in replay |
| cell_mask | 1 | True only for playable board coordinates |
| in_bounds | 1 | Derived from board_shape; includes masked holes, excludes padding |

Total: 20 × 9 × 8 = **1440 spatial features**. Multiply categorical planes and HP
by cell_mask so padded/masked cells contribute zero. A playable fixed blocker
still has cell_mask=1, an EMPTY tile category, its blocker category, and HP.
ICE preserves its underlying tile and special categories. A masked hole has
in_bounds=1/cell_mask=0; padding has both zero. This keeps the two concepts distinct.
Current HP values outside 0–3 should fail the initial encoder contract; future
variants require an explicit scale/version update, not silent clipping.

Append rows/9, cols/8 and moves_remaining/15: **3 scalars**. These scales are
global constants for the six-level experiment, not calculated separately per
episode. Increased move limits require an explicit contract decision.

For each of two objective slots append:

- Three type indicators: padding, CLEAR_ALL_BLOCKERS, COLLECT_TILES; gate by
  objective_mask so inactive slots are zero.
- Six color indicators: wildcard/not-color-specific zero, colors 1–5; also gated.
- `log1p(target)/log1p(65535)` to preserve distinctions between target sizes.
- Completion fraction `min(progress/target, 1)` for an active positive target.
- objective_mask itself.

That is 12 × 2 = **24 objective features**, giving **1467 float32 features** total.
Progress beyond target is intentionally saturated only in the encoder because
the current objective semantics do not need additional progress after completion;
raw counts/targets remain available in replay. Inactive targets are zero and must
never be divided by zero. Fixed slot order follows the observation; no reordering.

Do not feed level ID or effective seeds to the network. Objectives, shape and mask
describe the task without identifying a particular PRNG sequence. Keep action_mask
outside the feature vector initially and apply it to exploration/action choice and
Bellman targets. Embeddings could reduce categorical dimensionality later; one-hot
planes are easier to inspect and test for the initial six-level vocabulary.

## Initial Q-network

Propose `1467 -> Linear(256) -> ReLU -> Linear(128) -> ReLU -> Linear(144)`.
This is **427,280 parameters** with an unconstrained scalar Q-value per action;
no softmax, dropout or BatchNorm. Masking is applied outside the model so training
loss gathers the raw Q-value at the selected legal action.

An MLP is a transparent baseline but does not share spatial features across
coordinates. A later controlled CNN comparison can use the same 20 planes,
two small padded convolution layers, a flattened position-preserving head and
the 27 nonspatial features. Mask planes are still needed for irregular boards.
Avoid global pooling that discards the coordinate information required by the
action head. Keep reward, seeds, training budget and evaluation protocol matched
when comparing encoders. No CNN or agent is implemented here.

## Replay buffer proposal

Use a preallocated NumPy ring buffer with detached copies. Encode batches only
when sampling; storing float32 one-hot observations wastes memory. Each record
contains current/next raw observation fields, action, reward, terminated,
truncated, current mask and next mask. Sample uniformly using a local Generator.

| Per-state data, excluding action mask | Dtype | Bytes |
|---|---|---:|
| tiles, specials, blockers, blocker_hp | 4 × 72 uint8 | 288 |
| cell_mask | 72 bool | 72 |
| board_shape | 2 uint8 | 2 |
| moves_remaining | uint16 | 2 |
| objective_type, objective_color | 2 × 2 uint8 | 4 |
| objective_progress, objective_target | 2 × 2 uint16 | 8 |
| objective_mask | 2 bool | 2 |
| **Raw observation total** | | **378** |

Two raw observations (756 bytes), two bool action masks (288), an int16 action
(2), float32 reward (4) and two bool flags (2) total **1052 bytes/transition**.
50,000 transitions require about **52.6 MB / 50.2 MiB** of payload; 100,000 need
105.2 MB / 100.3 MiB, before container overhead, sample batches and checkpoints.
Storing both encoded states instead would exceed 586 MB at 50,000 transitions
even before masks and metadata. Use 50,000 initially. Avoid Python-object lists
for each numeric field and avoid retaining animation snapshots or full info dicts.

Keep both masks exactly once per state. Preserve terminal next observations and
both Gymnasium flags. Copy arrays to prevent environment/replay aliasing. Check
all dtype bounds before storing; do not wrap moves or targets on overflow.
Initialization failures have no transition; engine exceptions have no valid
returned next state and must not become fabricated negative-reward transitions.

## Masked DQN update

Propose the conventional online/target-network DQN baseline with uniform replay,
epsilon-greedy behavior and a Huber/SmoothL1 TD loss. A separate target model
supplies detached next-state values; target parameters are never optimized by
the online optimizer. These are standard components described in the
[official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
The concrete sizes and hyperparameters below are proposals for this project.

For a transition with a returned next observation:

`target = reward` when terminated, otherwise
`target = reward + gamma * max_{a: next_mask[a]} Q_target(next_observation, a)`.

- Select epsilon exploration uniformly among current mask-valid IDs using the
  exploration RNG. Exploitation takes the lowest-ID masked argmax on a tie.
- Assert a nonempty current mask before acting. Never recompute legal swaps in
  the agent. Assert info.valid_action after the environment returns.
- Build zero-valued next-state values for the batch; compute maxima only on
  nonterminated rows with a nonempty next mask. Use `-inf` for invalid actions in
  those rows. Never multiply a terminal zero by an all-`-inf` maximum: that can
  produce NaN. Masked Q-values are not passed directly into the chosen-action loss.
- Terminal rows may have all-zero next masks and receive no bootstrap. A
  nonterminated row with an all-zero next mask is a contract error: retain its
  trace as ERROR and stop/quarantine it; do not silently bootstrap zero or call it
  a level loss. Match3LevelEnv already marks unrecoverable dead boards terminal.
- Truncation ends collection but normally still bootstraps from the actual final
  observation. Level move-limit loss is a natural termination, so it does not
  bootstrap. If both flags are true, terminated takes precedence.
- Initially use explicit reset between episodes and no autoreset wrappers.
  Later vector/autoreset support must extract the final observation and mask,
  never the reset observation, before constructing a truncation target.

This distinction follows the [Gymnasium termination/truncation guidance](https://gymnasium.farama.org/introduction/migration_guide/).
An exception or an evaluator safety interruption without a returned final state
is not automatically a usable truncated training transition.

## Training configuration and reproducibility proposal

| Setting | Initial proposal, subject to smoke validation |
|---|---|
| Optimizer | Adam, learning rate 3e-4 |
| Discount | gamma 0.99 per engine step, not per cascade |
| Batch size | 64 |
| Replay capacity / warmup | 50,000 / 1,000 valid transitions |
| Optimization frequency | One update every four valid environment steps |
| Target update | Hard copy every 1,000 optimizer updates |
| Gradient clipping | Global L2 norm capped at 10; log norm before clipping |
| Epsilon | Linear 1.0 to 0.05 over 50,000 valid steps |
| Initial level | Blocker Mayhem, sparse rewards |
| Single-level pilot | 100,000 valid steps; assess learning rather than assume it |
| Environment guards | 1000 initialization attempts, 200 cascades, explicit 50-step safety bound |

Spawn separate local RNG streams for training-seed selection, exploration, replay
sampling and network initialization from one logged master SeedSequence. Pass
explicit episode seeds to the existing environment. Use a dedicated Torch
Generator where supported. Construct layers in a scoped `torch.random.fork_rng`
context with an explicitly seeded Torch RNG, so default Linear initialization is
seeded and does not permanently alter unrelated process RNG state. No unseeded
global NumPy/Python randomness is needed. Preserve RNG states in resume artifacts.

Use deterministic CPU operations for the smoke run and request deterministic
algorithms; record failures rather than silently relaxing reproducibility.
Bitwise identity across devices/releases is not promised by PyTorch. Record
the environment, seed streams and backend settings as advised in its
[reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
No random initialization or training is performed in this sprint.

On initialization/step errors, record the seed and trace through existing ERROR
analytics, insert no fabricated transition, and apply a configured error-stop
threshold. A higher fixed budget is not a reason to retry with a larger one.
Invalid/no-op actions should be treated as trainer bugs, preventing a policy from
avoiding natural loss indefinitely through unconsumed moves.

## Staged training and evaluation proposal

1. After approval, add and pin only PyTorch; implement/test the pure encoder and
   Q-network. Test fixed shape, categorical channels, masks, padding and all six
   levels before any optimization.
2. Add raw replay and masked-target unit tests. Test terminal/all-zero masks,
   truncation bootstrapping, no NaNs, mask-invalid large Q-values, array isolation,
   gradient flow and deterministic sampling using small synthetic transitions.
3. Run a deterministic CPU smoke of about 1,000 valid steps on First Match with a
   small fixed seed set. Its short horizon/collection objective is useful for
   debugging. Historical three-color ceiling performance does not apply to the
   revised five-color configuration. Confirm finite losses, gradients and legal actions;
   do not infer learning or superiority from the smoke.
4. Train the first substantive single-level pilot on **Blocker Mayhem**. Its
   Random/Greedy rates (45%/49% on the historical sample) leave more evaluation
   headroom than near-ceiling levels. Its blocker/objective complexity also makes
   failure to learn under sparse reward plausible; report that result honestly.
5. Evaluate fixed checkpoints every 5,000 valid steps with epsilon=0, an independent
   environment and the Agent protocol. Freeze training updates/replay insertion
   during evaluation. Pair all policies on the same seed lists and retain
   WON/LOST/ERROR/TIME_LIMIT separately. Existing paired statistics and confidence
   intervals apply; compare inference costs as well as win rate/objective progress.
6. Pre-specify a small learning-rate/epsilon/target-update experiment, multiple
   independent training seeds (at least three initially), and equal environment-step
   budgets. Use validation results for selection; do not select checkpoints on the
   final held-out test. Consider Double DQN/CNN only as later controlled extensions.
7. Add deterministic, balanced level sampling outside the environment for
   multi-level training. Replay must retain task observations, not hidden level/RNG
   state. Reassess vocabulary/scales if levels are expanded.
8. Evaluate held-out seeds and held-out levels/configurations separately. A
   six-level seen-level evaluation is not proof of generalization to new levels.

Reserve disjoint explicit seed namespaces: training 1,000,000–9,999,999,
validation 100,000–100,099, sealed final evaluation 200,000–200,999. The exact lists
and training master seeds are open decisions; validate disjointness before runs.
The historical 52000–52099 sample has already informed development, so use it as
a regression reference, not an unseen final test. Re-evaluate Random/Greedy on
new held-out seeds for paired comparisons. Initialization budget and level hashes
must be identical across policies. No new benchmark or training was run here.

## Tracking and checkpoint boundaries

Reuse episode/move analytics for gameplay only. Put training diagnostics in a
separate optional `training_metrics.csv` keyed by run ID, environment step and
optimizer update: reward, evaluation win rate, TD loss, epsilon, chosen/masked Q
statistics, pre-clip gradient norm, environment steps, update count and wall time.
Do not add gradients or optimizer state to StepResult/environment info.

Checkpoints should include online/target weights, optimizer state, replay contents
and write position when exact training resume is required, all local RNG states,
epsilon/step/update counters, next episode index, architecture/encoder version,
level fingerprints, budgets, seed lists, software/device versions and configuration.
Save initially only at episode boundaries: mid-episode exact resume would additionally
require full engine state and RNG, which are intentionally absent from observations.
Inference checkpoints can omit replay/optimizer but must retain encoder metadata.
Checkpoints are already Git-ignored; result outputs must not overwrite prior runs.

## PPO, custom rewards and objective-aware agents

The encoder can later be shared with a PPO actor/value model, using masked logits
and on-policy rollouts instead of a replay/target-network algorithm. PPO advantage
calculation must distinguish termination bootstrap from truncation and stop GAE
recursion across an episode boundary. Do not add DQN-specific fields to the
Gymnasium observation or game engine.

Reward shaping/custom rewards should be separate, versioned wrappers or reward
functions after approval, retaining sparse terminal reward as the comparable
evaluation baseline. Keep raw objective progress/targets in replay and the
existing info contract for future objective-aware heuristic agents. Do not
conflate a reward experiment with a gameplay change.

## Open decisions requiring approval before implementation

- The PyTorch version/install, CPU-first device policy and optional `[rl]` extra.
- The proposed 1467-feature encoder contract (five colors, HP<=3, move scale 15),
  objective completion saturation and MLP sizes versus starting with a CNN.
- Replay capacity, initial hyperparameters, training step budget and master seeds.
- Validation/final seed lists, checkpoint-selection criterion and number of
  independent training seeds. Define practical improvement thresholds in advance.
- Keep sparse reward for the first pilot; treat shaped reward as a later experiment.
- If initialization failures occur at 1000, investigate their traces and
  profile a policy separately; never silently increase limits or change seeds.
