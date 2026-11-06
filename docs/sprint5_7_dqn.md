# Sprint 5.7: calibration and initial DQN

This implements the reviewed MLP DQN baseline with sparse rewards and CPU execution.
No Double/Dueling DQN, prioritized replay, PPO, CNN or reward shaping is included.

## First Match calibration

Only the objective target changed, from 300 to **30**. Five colors, five moves,
6×6 dimensions, configured seed 41001 and all other JSON fields were preserved.
The other five level JSON files were not changed.

Held-out environment seeds **100000–100099** were paired across agents, with
separate agent seeds **193000–193099**. These environment seeds were not used for
the five-color target-300 benchmark. Initialization/cascade guards stayed 1000/200.

| Agent | Wins | Losses | Errors | Win rate / Wilson 95% CI | Mean moves | Mean objective completion | Initialization failures |
|---|---:|---:|---:|---|---:|---:|---:|
| Random | 65 | 35 | 0 | 65%, 55.25–73.64% | 4.20 | 92.03% | 0 |
| Greedy | 71 | 29 | 0 | 71%, 61.46–78.99% | 3.81 | 93.77% | 0 |

All rates use 100 attempted episodes. There were no invalid actions, errors or
time-limit interruptions. Greedy's point estimate is within the expected 70–80%
range; Random's 65% is below it. The result is retained without additional tuning.
No serious gameplay defect was observed. This is a calibration sample, not proof
of human difficulty or an optimal policy's performance.

Raw episode/move CSV and frozen configuration/seed metadata are saved under
`experiments/results/sprint5_7_first_match_calibration/`. Historical three-color
or five-color target-300 results describe different tasks and must stay labeled.

## Dependencies and module boundaries

The optional `[rl]` extra adds only **torch==2.15.0**, the version installed and
tested on macOS arm64 with Python 3.11 and NumPy 2.4.6. No separate RL framework,
torchvision or torchaudio dependency is added. PyTorch's normal transitive runtime
dependencies are installed by pip. Existing environment/Random/Greedy imports do
not import PyTorch; the encoder/replay modules need only NumPy. Neural-network tests
skip when the optional dependency is absent; the tested full environment includes it.

```sh
conda run -n match3-ai python -m pip install -e '.[rl]'
```

Modules:

- `agents/observation_encoder.py`: pure observation validation, encoding and raw
  packing; no engine, LevelSession or environment references.
- `agents/replay_buffer.py`: bounded NumPy ring buffer with private sampling RNG.
- `agents/dqn_agent.py`: MLP, epsilon-greedy selection, targets, TD update,
  target synchronization and checkpoint serialization.
- `training/dqn.py`: single-level orchestration, local seed namespaces, evaluation,
  training metrics, error records and resume. Game progression stays in the
  existing Match3LevelEnv/LevelSession/Match3Engine stack.

## Encoder and network

Encoder version 1 implements exactly **1467 float32 features**:

- 20 × 9 × 8 = 1440 spatial features: tile one-hot channels (EMPTY, colors 1–5,
  COLORLESS=255), six special categories, four blocker categories, HP/3,
  cell_mask and an in-bounds plane derived from board_shape.
- Rows/9, cols/8 and moves_remaining/15: three global features.
- Two objective slots with 12 features each: three type channels, six color
  channels, log1p(target)/log1p(65535), capped completion fraction and active mask.

Categorical planes and HP are zeroed outside playable cells. The separate in-bounds
plane distinguishes masked holes from padding. COLORLESS maps to its own channel,
not a large continuous value. Raw target/progress counts remain in replay.
Unsupported categories, shapes, HP>3, move counts>15 or encoder versions fail
explicitly. Inactive objectives are zero; active zero targets are rejected.
The **144-entry bool action mask is returned separately** and is not a model feature.

The default Q-network is **1467 → 256 → 128 → 144**, using ReLU hidden activations,
427,280 parameters and unrestricted scalar Q-values. Hidden dimensions are
configurable. It has no dropout, BatchNorm, softmax or environment-specific access.

## Replay, actions and targets

Replay preallocates raw observations and next observations, int16 action IDs,
float32 rewards, bool terminated/truncated flags and two 144-entry bool masks.
It copies inputs and returns sampled copies. Sampling is uniform without replacement
within a minibatch, using a local NumPy Generator. Capacity is fixed; old entries
are replaced in ring order. Its payload is 1052 bytes/transition, approximately
52.6 MB for 50,000 entries; the smoke configuration uses only 2048 entries.

Exploration samples uniformly from mask-valid IDs using the agent's own Generator.
Exploitation chooses the largest legal Q-value, with lowest-ID tie-breaking.
Empty current masks fail explicitly. Evaluation calls the existing
`select_action(observation) -> int` interface with the default epsilon=0 and does
not advance exploration RNG or modify parameters.

Vanilla DQN uses the frozen target network's maximum over **legal next actions**.
Terminated transitions use reward alone. Truncated transitions bootstrap from
the actual final observation, following Gymnasium's semantics; if both flags are
true, termination wins. Terminal all-zero masks are safe: those rows never enter
the maximum reduction, avoiding 0×infinity NaNs. Nonterminal empty masks fail as
contract errors. Finite targets/losses/gradients are checked. No Double DQN online
next-action selection is used. Loss is SmoothL1 and gradients are clipped by
global L2 norm before the Adam step.

## Training and evaluation

All hyperparameters are configurable through the frozen `TrainingConfig` Python
API. The CLI supplies a bounded **smoke profile**, not an unapproved long run:

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.dqn \
  --level-id blocker-mayhem --steps 1000 --seed 5700 \
  --output-dir experiments/results/<new-run-directory>
```

The smoke profile uses replay 2048, minibatches 32, warm-up 64 valid transitions,
one optimizer update every four environment steps, Adam learning rate 3e-4,
gamma .99, clipping norm 10 and hard target copies every 50 optimizer updates.
Epsilon decays from 1.0 to 0.1 over 1000 steps. The Python API retains the reviewed
baseline defaults (50,000 replay, batch 64, warm-up 1000, target period 1000);
budgets must leave room after warm-up for useful optimization.

Training episode seeds start at **1000000**. Validation seeds **110000–110009** are
disjoint from training and the First Match calibration sample. Initialization of
network weights is explicitly seeded inside torch.random.fork_rng; its state is
restored afterwards. Exploration and replay use independently spawned local
NumPy streams. CPU thread count is one and deterministic Torch algorithms are
enabled during training; prior process settings are restored when it ends.

At steps 500 and 1000, the existing evaluation runner evaluates DQN with epsilon=0,
Random and Greedy on identical held-out seeds. No optimization, replay insertion
or exploration-RNG consumption occurs during evaluation. Baseline RNG seeds start
at 210000. The same validation set is reused between checkpoints, so these are
development evaluations, not independent final tests. No superiority/convergence
claim follows from ten episodes per policy.

A returned transition at an evaluator/training budget boundary is stored as a
truncation if it is not already terminal, and bootstraps normally. A natural move
budget loss remains terminated. Invalid actions and unrecognized level-terminal
states are trainer errors; exceptions produce `training_error.json` and abort,
without fabricated replay transitions, retries or increased guards.

## Checkpoints and records

Every existing output directory and checkpoint path is rejected rather than
overwritten. Periodic `policy_step_<step>.pt` checkpoints support inference;
their absent replay/training state deliberately prevents treating them as exact
training-resume checkpoints. `checkpoint_final.pt` additionally contains replay
contents/write position/RNG, target state, optimizer state, exploration RNG,
absolute step/update/synchronization counters, next episode index and level hash.
Both kinds retain seed metadata, architecture, encoder configuration and training
configuration. Loading is CPU-only and uses `weights_only=True`.

The final checkpoint ends collection at a boundary. Resume starts a new episode
with the next explicit seed; it does **not** restore a mid-episode engine RNG/state.
Cutting a run mid-episode introduces a documented truncation, so split runs need
not match one uninterrupted longer trajectory. Hyperparameters and level hashes
must match for resume; only the absolute environment-step budget may increase.

```sh
conda run --no-capture-output -n match3-ai python -m match3_ai.training.dqn \
  --steps 1100 --seed 5700 --resume-from <full-checkpoint.pt> \
  --output-dir experiments/results/<new-resume-directory>
```

`training_metrics.csv` records step/action validity, epsilon, replay size, update
count, TD loss, valid-Q mean/min/max, pre-clip gradient norm, episode reward/outcome,
validation win rate and elapsed time. `training_episodes.json` records training
seed/reward/flags per episode. `training_summary.json` records settings, model
fingerprints, update/sync counts, checkpoint checks and evaluation summaries.
Full evaluation gameplay episode/move CSV and metadata are kept separately in
`evaluation/`, using the existing AnalyticsCollector/export path.

## Smoke results and limits

The bounded 1000-step Blocker Mayhem smoke filled replay with 1000 transitions,
performed 235 optimizer updates and four target synchronizations. Parameters
changed, losses were finite, and checkpoint loading reproduced policy decisions
and model weights. There was one intentional training-budget truncation; this
was not a level loss. No invalid actions or evaluation errors were observed.

At step 1000, evaluation on ten seeds produced DQN 6 wins / 4 losses, Random
5 / 5 and Greedy 3 / 7. This tiny, reused validation sample establishes execution
compatibility only, not agent improvement. The implementation still needs longer
approved experiments, multiple independent training seeds, hyperparameter studies
and larger held-out comparisons. It is CPU-only, uses sparse reward, supports the
fixed reviewed encoder contract and does not promise cross-platform bitwise replay.

All experiment datasets are generated under Git-ignored `experiments/results/`.
No history/staging operations were performed. PPO, reward shaping and advanced
DQN variants remain outside this sprint.

The final smoke artifacts are in
`experiments/results/sprint5_7_dqn_smoke_final/`, including training metrics,
evaluation CSV, policy checkpoints and the full final checkpoint. Repeating the
1000-step run with the same seeds produced identical final model weights and mean
TD loss. Wall-clock times are not expected to match.

## Validation

The focused DQN/environment/evaluation/configuration suites passed 167 tests.
The final full suite passed **698 tests** in the `match3-ai` Conda environment
with PyTorch installed. Ruff and `git diff --check` passed; new untracked source
files were also checked for trailing whitespace and final newlines. Seventeen
new test cases cover encoding, replay isolation/reproducibility, masked actions,
Bellman targets, target synchronization, checkpoints, bounded training and resume.
Existing First Match configuration and HUD expectations were updated to target
30; no gameplay implementation changes were necessary.
