# Sprint 5.10: agent behavior and learning analysis

This is a frozen-policy analysis of Sprint 5.9, not another training experiment.
It asks whether DQN changes useful behavior relative to its initialization and
uniform random actions, how its decisions relate to the immediate-match Greedy
heuristic, and what evidence can guide a later controlled experiment. Agreement
with Greedy is not an objective or an optimality certificate.

## Data and protocol

- Level: unchanged `blocker-mayhem`, sparse rewards, 200-cascade guard,
  1000 initialization attempts, 50-step episode safety bound
- Primary validation seeds: **300000–300099**, within 300000–300999
- Reserved final test 400000–400999: never used
- Random trajectory agent seeds: **600000–600099**, separate from environment RNG
- Checkpoints: all 33 step-zero/intermediate/final policy checkpoints plus the
  three full final checkpoints for master seeds 5901, 5902 and 5903
- Inference: CPU, one Torch thread, deterministic algorithms, epsilon zero
- No weight updates, checkpoint rewrites, retraining or algorithm changes

Checkpoint configuration comparison normalizes JSON lists and checkpoint tuples
without dropping fields. Real step-zero models are compared with newly initialized
models using their exact saved seeds and hidden dimensions. Encoder version,
level fingerprint, evaluation settings, checkpoint step and final weight hash
must match. Input CSV/JSON/checkpoint SHA-256 hashes are rechecked after analysis.

The recorded 33 checkpoint evaluations supply learning-progression comparisons.
The analysis additionally runs 500 complete episodes: Random, Greedy and each of
three final DQNs on the paired 100 initial seeds. Fresh action sequences, outcomes,
clear/cascade/special totals, moves and remaining HP must reproduce historical
results. Same initial seeds do not imply identical later boards after policies
choose different actions and consume refill RNG differently.

## Shared observations and weighting

`observations.npz` stores full typed observations without object arrays. Identical
observations are deduplicated by a stable SHA-256 fingerprint over all fields,
including objective targets, raw progress, remaining moves and action masks.
`state_occurrences.jsonl` retains every sampled decision's source policy, episode,
seed, move index, board shape, objective progress, HP, legal count, fingerprint,
actions, engine RNG state and full-state digest. Thus inference storage is
deduplicated while trajectory provenance and dependence remain observable.

All zero/final DQNs and Greedy are evaluated on each identical observation.
Uniform Random's probability of agreeing with any legal deterministic action is
computed analytically as 1/K, where K is legal-action count. Forced K=1 states
are distinguished. Greedy's lowest-ID tie-breaking is verified against its exact
existing score; equal-score alternative actions are reported separately.

Source populations are reported separately for Random-generated, Greedy-generated
and DQN-generated states. Within a source policy, decisions are occurrence-weighted,
so longer episodes contribute more states. DQN-generated results average the
three generating policies equally. The pooled estimate averages the three source
families equally: Random 1/3, Greedy 1/3, each DQN generating policy 1/9. This
chosen mixture does not represent all possible board states.

## Exact immediate counterfactuals

Inspection found that the complete live environment consists of ordinary Python
state and NumPy arrays/local RNGs, with no rendering handles in the engine. An
analysis-only `deepcopy` preserves BoardState, raw/capped objective counters,
moves/status, settings and engine/Gym RNG state. Independence and reproducibility
are tested on advanced states. Neither observation-only reconstruction nor an
approximate RNG is used.

At each sampled decision, each distinct candidate action is simulated on an
independent full environment clone. The original full-state/RNG digest must stay
unchanged. The source action's simulated next observation must match the actual
trajectory's next observation. Identical observations with different RNG state
remain distinct counterfactual contexts. Exact action-prefix restoration is also
available and validates the complete state/RNG digest. Engine, session and
environment implementations are not modified.

Immediate outputs include tile clears, capped objective gain, blocker hits and
destruction, remaining HP, special activations, cascades, reward and terminal
status. Both all-state and disagreement-only comparisons are reported by source.
Simulation errors are explicit and failed pairs are excluded rather than assigned
zero effects. These comparisons measure one-action consequences under this exact
RNG realization, not integrated future randomness or long-term policy value.

## Statistics

The primary learning comparisons are three final-vs-zero paired win differences.
Exact two-sided McNemar tests use discordant pairs, with Holm adjustment across
the three tests. Six final-vs-baseline comparisons form a separate exploratory
Holm family. Continuous metrics use initial-seed paired percentile bootstrap
intervals. Relative effects are withheld for nonpositive reference means and
signed sparse episode rewards.

Shared-state intervals resample initial environment-seed clusters, retaining all
within-episode moves, identical states and correlated cross-policy trajectories.
Ratios and source-family weights are recomputed in each draw. Pooled/stratified
agreement is not treated as independent move-level binomial data. The primary
bootstrap uses 5000 draws, with deterministic seed-derived streams and seed 51010
for shared-state summaries. Rare strata report their cluster/draw counts.

Wilson intervals describe each fixed policy's win rate. Reused validation seeds,
checkpoint selection and only three training seeds constrain generalization.
Nonsignificance is not equivalence. Parameter changes and deterministic action
preferences do not alone establish useful learning. Greedy is neither an
objective-aware policy nor an optimal baseline.

## Reward and case analysis

Fresh move records measure zero/positive/negative reward fractions, episode
lengths, objective completion immediately before the terminal move, and blocker
hits accumulated before terminal reward. The three full final replay buffers are
available, so their actual reward/termination distributions are also measured.
They are not inferred from evaluation episodes or invented historical buffers.

For each DQN, representative discordant cases use the earliest seed in each
direction: DQN-only win and Greedy-only win. Reports retain full action, objective,
HP, special and cascade sequences with board fingerprints. Differences after
the first disagreement may occur on different boards and are descriptive,
not evidence that a single action caused the eventual episode outcome.

## Reproduction and artifacts

Use the existing `match3-ai` environment with optional Torch and Matplotlib
already installed. No new dependencies are added.

```sh
MPLCONFIGDIR=/tmp/match3-ai-matplotlib conda run --no-capture-output -n match3-ai \
  python -m match3_ai.evaluation.behavior_analysis \
  --input-root experiments/results \
  --output-dir experiments/results/<new-behavior-directory>
```

The default is 100 episodes per source policy and 5000 bootstrap draws.
`--episodes 2 --bootstrap-samples 100` is a bounded integration check, not primary
scientific evidence. Existing directories and observation archives are rejected.

Outputs include `analysis.json`, `report.md`, `observations.npz`, occurrence and
counterfactual JSONL, `representative_cases.json`, fresh gameplay episode/move CSV,
and ten Matplotlib figures in PNG/PDF. The two-seed check is kept separately in
`experiments/results/sprint5_10_behavior_smoke/`; the primary run uses
`experiments/results/sprint5_10_behavior_primary/`.

Implementation modules are `evaluation/behavior_data.py`, `behavior_statistics.py`,
`behavior_analysis.py` and `behavior_report.py`; all gameplay and agent modules
remain unchanged. Focused tests in `tests/test_behavior_analysis.py` do not train
models. Further training, reward shaping, new agents and final-test evaluation
require separate approval.

## Completed findings and validation

The primary run collected **500 complete episodes**, **6451 decision occurrences**
and **5969 unique observations**. It verified 25159 distinct occurrence/action
counterfactuals, represented by 32255 labeled policy records, with zero errors.
All source next-observation comparisons passed. The post-export audit verifies
every observation fingerprint, action legality and all 48 historical/checkpoint
input hashes; no source artifact or weight was modified.

Primary outputs are in `experiments/results/sprint5_10_behavior_primary/`:
`findings.md` synthesizes the research conclusions; `report.md` contains full
tables/cases/figures; `analysis.json` retains estimates and source hashes;
`dataset_audit.json` records export checks. There are ten requested PNG/PDF figures.

Final-minus-zero win-rate changes are +7 pp (95% paired CI -8 to +21), +2 pp
(-12 to +16) and +17 pp (+4 to +30). Exact McNemar p-values are .41875, .88543
and .01862; three-test Holm values are .83751, .88543 and .05587. Seed 5903 is a
positive but uncertain signal after multiplicity correction. Neither those
uncertainties nor failure to exceed Greedy establish absence of learning.

Pooled final–Greedy action agreement is 4.08%, 4.35% and 2.75%, against exact
masked-random expectation 4.40%. Final–zero agreement is only 2.19–3.43%.
Disagreement-only immediate objective-gain differences are -.01041, -.00818 and
-.00666, while special-activation differences are +.1491, +.0835 and +.1356.
The full report gives clustered confidence intervals and source-specific effects;
these mixed local effects do not determine long-term policy quality.

Actual replay zero-reward fractions are 92.104%, 92.122% and 92.194%. Delayed
credit is a plausible hypothesis, not an established cause of weak performance.
The proposed next experiment compares frozen zero/final actions under exact
counterfactuals on these same saved states before choosing a single controlled
reward or architecture intervention. No new training or shaping was performed.

Supplemental `action_preferences.json` measures source-balanced empirical action
ID distributions against exact uniform-mask probabilities on the stored data.
Final action entropies are 5.36, 5.09 and 4.77 bits, versus 3.58, 3.36 and 3.07
at initialization. Marginal total variation from masked uniform falls from about
.80 to .49–.61. This documents changed behavior, not beneficial learning by itself.
The supplement adds no new episodes or inference and is reproducible with:

```sh
conda run -n match3-ai python -m match3_ai.evaluation.behavior_statistics \
  --input-dir experiments/results/sprint5_10_behavior_primary \
  --output-file <new-preferences-file.json>
```

Fresh episode CSV `inference_time_seconds` includes the instrumented observer's
other-policy inference and counterfactual simulations. It is excluded from
comparisons and must not be treated as agent-only latency. The primary timer
(about 598 seconds) excludes plot/report generation. Statistical intervals are
pointwise unless the report explicitly gives a Holm-adjusted p-value.

Twelve fast new regression tests cover checkpoint compatibility and true zero,
fingerprints/deduplication/export, exact cloning/restoration/RNG isolation,
agreement/provenance, analytic random probabilities, paired outcomes/statistics,
cluster sampling, actual sparse reward boundaries and reserved-seed protection.
The focused selection passes 62 tests and the full suite passes 726; Ruff and
Git whitespace checks are clean. Existing tracked/untracked work is preserved,
with no staging, commits or history changes.
