# Sprint 5.12: CNN architecture and controlled pilot

The hypothesis is that spatial structure may help DQN learn board relationships.
It is not an established advantage. This sprint adds a compact CNN, retains MLP
as the default, and runs the approved 5,000-step paired sparse-reward pilot.
Gameplay, environment observations, reward semantics and DQN targets are unchanged.

## Architecture and exact input equivalence

The original encoder already constructs channel-first board planes before
flattening them. The CNN splits that same float32 vector into views; it never
reinterprets raw observations with different normalization or categories.

| Channels | Original features, retained exactly |
|---|---|
| 0–6 | Tile categories: EMPTY, colors 1–5, COLORLESS=255 |
| 7–12 | Six special categories, including NONE |
| 13–16 | Four blocker categories, including NONE |
| 17 | Blocker HP / 3 |
| 18 | Playable-cell mask |
| 19 | In-bounds rectangle derived from board_shape |

Each plane is 9×8: 20×9×8 = 1,440 spatial values. Tile/special/blocker one-hot
planes and HP are zero on non-playable cells. The bounds plane distinguishes
masked holes inside the board from padded cells outside its rectangle. Zero
convolution padding does not make cells playable; the input geometry planes
remain observable and the independent output mask enforces legal actions.

The 27 global values, in original order, are:

- Rows / 9, columns / 8, moves remaining / 15.
- Two objective slots, each with 12 values: type one-hot (3), color one-hot (6),
  log1p(target) / log1p(65535), capped progress/target, and active flag.

Inactive objective slots are zero. Normalizations, supported bounds and clipping
remain exactly those of encoder version 1. Exact equivalence is relative to the
existing encoded representation; it does not claim to invert its existing
progress clipping or compress nothing inside the learned network.

The 144-entry legal-action mask remains separate from network features for both
architectures. It is used for greedy inference, epsilon exploration and target
maxima. The canonical RIGHT/DOWN action encoding is unchanged. No future RNG
values are exposed and Match3LevelEnv is untouched.

`ObservationEncoder.encode_cnn()` returns `(spatial, global_features, mask)`.
Concatenating `spatial.ravel()` and `global_features` reconstructs every original
1467 float32 value bit-for-bit. Parameterized tests check all six board shapes,
masked holes, padding, objective targets and observation isolation. Runtime
replay encoding stays shared: CNN.forward accepts the same flat minibatch and
splits it into views, avoiding a second replay or training implementation. The
explicit spatial/global forward interface is also tested against the flat one.

| Network | Structure | Parameters |
|---|---|---:|
| MLP | 1467 → 256 → 128 → 144 | 427,280 |
| CNN spatial | Conv3×3 20→8→16, padding 1/stride 1, ReLU; flatten to 1152 | 2,616 |
| CNN global | 27 → 32, ReLU | 896 |
| CNN fusion | 1184 → 256 → 128 → 144, hidden ReLU | 354,832 |
| CNN total | Spatial + global + fusion | 358,344 |

No pooling removes board resolution. There is no dropout, batch normalization,
new dependency, or duplicate rule implementation. The existing configurable
hidden dimensions apply to the fusion head. The default CNN uses **16.13% fewer
parameters** (68,936 fewer). Float32 online weights occupy approximately 1.37 MiB
instead of 1.63 MiB; target and Adam allocations add further memory. Convolutions
add computation, so fewer parameters do not imply lower latency. This comparison
is not parameter-matched and does not isolate inductive bias from capacity.

## Integration and checkpoint compatibility

`DQNAgent(architecture="cnn")` and `TrainingConfig(architecture="cnn")` select
the new network. Both default to `"mlp"`; unknown names fail explicitly.
Replay, Adam, Huber loss, gradient clipping, epsilon schedule, masked Bellman
targets, update cadence and hard target copies use their existing shared code.

Checkpoint payloads add architecture and versioned network configuration. Old
payloads without these fields load as MLP version 1. CNN checkpoints validate
the convolution configuration before loading weights. Encoder configuration,
hidden dimensions, online/target weights, optimizer and exploration RNG still
round-trip. Existing checkpoint files are never rewritten. Existing exclusive,
atomic publication prevents accidental replacement. Resume checks architecture
compatibility and defaults absent historical training-config architecture to MLP.

Periodic policy checkpoints retain the prior limited-resume behavior: they do
not contain the complete replay/in-flight episode. Full final checkpoints retain
the existing replay and boundary-resume support; this sprint does not expand it.

Initialization remains isolated from global Torch/NumPy RNG. Seed 5903 selects
the same named model, replay and exploration streams in both arms. Different
layer shapes/fan-in and initialization draw counts mean different initial
weights and distributions. Once actions differ, trajectories and subsequent
environment RNG consumption can diverge despite identical episode seed policy.

## Approved experimental protocol

| Setting | Both arms |
|---|---|
| Level / master seed | blocker-mayhem / 5903 |
| Training transitions | 5,000 |
| Training episode seed start | 240000 |
| Validation episode seeds | 300000–300099, paired across architectures |
| Baseline agent seed start | 600000 |
| Validation steps | 0, 1000, 2000, 3000, 4000, 5000 |
| Reward | Sparse; PBRS disabled |
| Replay / minibatch | 50,000 / 64 |
| Warmup / update cadence | 1,000 steps / every 4 steps |
| Optimizer / learning rate | Adam / 0.0003 |
| Gamma / gradient clipping | 0.99 / 10 |
| Target synchronization | Every 1,000 optimizer updates |
| Exploration | 1.0 → 0.05 over 50,000 steps |
| Guards | 200 cascades, 1,000 initialization attempts, 50 episode steps |
| Compute | CPU, one Torch thread per pilot |

The two configurations differ only in architecture. Each arm evaluates Random
and Greedy once at step zero, and DQN at all six checkpoints (800 validation
episodes per arm). Evaluation is isolated from training RNG and always sparse.
Reserved final-test seeds 400000–400999 were not used. These validation seeds
are familiar from earlier experiments; they are not a new unbiased test set.

Architecture/masking/checkpoint/update tests passed **before** pilot launch.
The pilots ran concurrently in separate processes. Every artifact uses a fresh
output directory. Reproduction commands require new directory names:

```sh
conda run -n match3-ai python -m match3_ai.training.architecture_pilot --architecture mlp --output-dir experiments/results/NEW_mlp
conda run -n match3-ai python -m match3_ai.training.architecture_pilot --architecture cnn --output-dir experiments/results/NEW_cnn
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run -n match3-ai python -m match3_ai.training.architecture_analysis --mlp-dir experiments/results/NEW_mlp --cnn-dir experiments/results/NEW_cnn --output-dir experiments/results/NEW_architecture_analysis --historical-checkpoint experiments/results/sprint5_9_seed_5903/policy_step_5000.pt
```

Add `--review-only` to either pilot command to print its configuration without
training. The dedicated profile deliberately exposes no longer-training flag.

## Pilot findings

| Step | MLP win rate | CNN win rate |
|---:|---:|---:|
| 0 | 36% | 44% |
| 1000 | 42% | 55% |
| 2000 | 51% | 47% |
| 3000 | 50% | 45% |
| 4000 | 53% | 51% |
| 5000 | 49% | 42% |

At the predeclared final checkpoint, MLP won 49/100, Wilson 95% CI
[39.4%, 58.7%]; CNN won 42/100, [32.8%, 51.8%]. No validation errors or invalid
actions occurred. Final CNN-minus-MLP is **−7 percentage points**, with paired
bootstrap 95% CI **[−21, +6]** and exploratory exact McNemar p=0.39160. The paired
table is 21 both-win, 21 CNN-only, 28 MLP-only and 30 both-lose. Bootstrap uses
5,000 paired seed resamples and fixed analysis RNG seed 51212.

Final completion is 94.83% for MLP versus 94.08% for CNN (difference −0.75 pp,
paired CI [−2.33, +0.75] pp). Remaining HP is .72 versus .85 (difference +.13,
CI [−.09, +.36]). Best validation results are MLP 53% at step 4000 and CNN 55%
at step 1000. These are validation-selected estimates, not final-test results.
The CNN best checkpoint follows just one optimizer update; its early advantage
cannot be confidently attributed to learned spatial reasoning.

Both arms completed 1001 optimizer updates and one target synchronization.
Final logged epsilon is .90502. Mean terminal training returns are −.10078
(MLP) and −.17436 (CNN). TD losses, valid-action Q statistics and pre-clipping
gradient norms are finite. Loss means are .02235 and .02335; mean gradient norms
are .15130 and .11947. Different Q scales are not policy-quality measurements.

The fresh MLP final weights match the historical Sprint 5.9 seed-5903 step-5000
weights exactly. Historical checkpoint hashes are recorded and checked again
after analysis. This supplies a concrete sparse-training backward-compatibility
check in addition to synthetic legacy-checkpoint tests.

| Timing | MLP | CNN |
|---|---:|---:|
| Active collection/optimization/checkpoint time | 78.17 s | 81.10 s |
| Validation time | 132.04 s | 131.67 s |
| Training wall timer | 210.22 s | 212.77 s |
| Shared-state batch-one median decision latency | 118.75 µs | 158.08 µs |

The CNN uses 16.13% fewer parameters but measured median decisions are about
33% slower, and active time about 3.8% longer. Timings are descriptive, affected
by hardware/load and differing trajectories. Both pilots ran concurrently;
process times must not be summed as elapsed batch time. Wall timers exclude
final gameplay CSV/plot export. The separate latency benchmark uses identical
100 initial observations, 20 warmups per policy, three measured passes with
alternating order, one CPU thread, and includes encoding, inference and masked
argmax. It does not step environments or alter policy weights/RNG. Mean, median,
p95 and actual rollout decision latency are saved in analysis.json.

## Artifacts, tests and interpretation

Outputs are in fresh, ignored experiment directories:

- `experiments/results/sprint5_12_mlp/`
- `experiments/results/sprint5_12_cnn/`
- `experiments/results/sprint5_12_architecture_analysis/`: detailed report,
  machine-readable statistics/input hashes and eight PNG/PDF figure pairs.

Figures cover validation wins (pointwise Wilson bands), objective completion,
remaining HP, TD loss, valid-action Q min/mean/max, pre-clipping gradient norms,
runtime and parameter count. Training losses/Q/gradients use trailing-50-update
means; raw values remain in CSV. Training episode returns and epsilon remain
in logs and the report. Source artifacts are read-only during analysis.

Thirteen new test cases cover all six levels' encoding equivalence, COLORLESS,
objective targets, shape checks, parameter counts, RNG isolation, forward
equivalence, greedy/exploratory masking, finite updates/gradients, frozen target
and hard copy, checkpoint/Adam round trips, legacy MLP loading, deterministic
tiny training, resume compatibility, paired analysis, figure export and
checkpoint immutability. Existing DQN tests cover the unchanged shared Bellman
and replay logic. Full validation: **756 passed**, Ruff clean, whitespace checks
clean. No production gameplay or existing test behavior was changed.

The curves do not establish CNN superiority or inferiority. Paired confidence
intervals describe evaluation-seed uncertainty conditional on these trained
models, not training-seed variability; repeated checkpoints are correlated.
Nonsignificance is not equivalence. Feature parity, deterministic updates,
finite gradients and the historical control reveal no implementation failure
that explains the short final CNN result. Initialization and normal short-run
learning variability remain plausible; causation is unresolved.

Keep MLP as default. If approved, the next experiment should preregister a
multi-seed architecture comparison with an equal budget and isolated sparse
evaluation, before architecture resizing or reward changes. No longer training,
automatic model enlargement, PBRS activation or PPO work was performed.
