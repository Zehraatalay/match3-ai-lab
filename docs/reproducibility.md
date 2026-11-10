# Architecture and reproducibility

## Contracts and separation

`Match3LevelEnv` resolves the complete move, including stochastic specials,
gravity/refill, cascades, and dead-board recovery. It exposes padded 9×8 arrays,
two objectives, and 144 action IDs: `((row * 8) + col) * 2 + direction`, where
RIGHT=0 and DOWN=1. Ordinary swaps require a match involving an endpoint;
special swaps can be legal without an ordinary match. Agents must use the mask.

Classic MCTS stores action histories and rechecks actions in each sampled state.
Neural MCTS stores separate observable successor states per action, including
raw objective progress, and excludes hidden RNG state from state identity.
Both clone the live environment and replace only the simulation engine RNG.
Actual chance outcomes are sampled; search does not choose favorable outcomes.

The separate policy-value encoder returns spatial `(20,9,8)`, global `(30,)`,
and legal-mask `(144,)` arrays. Static context supplies original move limit and
color count. The 157,459-parameter CNN uses row/column/direction policy ordering
and a tanh value head. Empty terminal masks are permitted for value inference;
policy selection without a legal action raises an explicit error.

## Search-guided training

Collection uses a frozen network within each round, independent reset/search/
exploration RNG streams, and normalized untempered root visits as policy targets.
Only naturally completed wins/losses enter replay with +1/-1 labels. External
truncations and execution errors are recorded and excluded. CPU Adam minimizes
masked soft-target cross-entropy plus weighted terminal-outcome MSE. Illegal
log-probabilities are zeroed before multiplication to avoid `0 * -inf`.

Checkpoints include weights, optimizer, bounded replay, counters, metrics,
configuration identities, architecture/encoder versions, and RNG states.
Resume is supported at completed collection/update round boundaries; it is not
mid-episode recovery. Existing output/checkpoint paths are protected against
accidental overwrite.

The following is a **manual smoke configuration**, not reproduction of the
published overnight model. It is not launched during installation:

```python
from match3_ai.levels.loader import load_level
from match3_ai.training.neural_mcts import NeuralMCTSTrainer, NeuralMCTSTrainingConfig

level = load_level("experiments/fixtures/az4_profiles/medium/level.json")
config = NeuralMCTSTrainingConfig(checkpoint_dir="results/manual_neural_smoke")
trainer = NeuralMCTSTrainer(config, level)
trainer.run()
```

Defaults are two rounds, four episodes/round, eight simulations/decision,
256 replay positions, batch size 16, and ten updates/round when replay is large
enough. This validates execution only. No depth cap is introduced: AZ-2 does not
expose one. Do not use the training task's win rate as a model-selection metric
without separate development configurations and seeds.

## AZ-4 evaluation

Portable configurations are in `experiments/fixtures/az4_profiles/`. Their bytes
match the original local PLG artifacts. The five-color compatible Hard candidate
is reconstructed from candidate 0 of the saved Medium generation specification;
candidate 1 must exactly reproduce the accepted Medium configuration before use.
The saved 100-episode confirmation of candidate 0 had 22% wins and Wilson 95%
interval 15.0–31.1%, at ten classic-MCTS simulations. The accepted six-color Hard
configuration is retained as provenance and explicitly excluded from neural use.

With the original six round checkpoints supplied under a local directory:

```bash
python scripts/benchmark_neural_mcts.py \
  --checkpoint-dir results/neural_mcts/overnight_01/checkpoints \
  --episodes 20 --easy-episodes 5 --control-episodes 5 --simulations 8 \
  --env-seed-start 31000000 --agent-seed-start 41000000 \
  --output-dir results/neural_mcts/evaluation/manual_az4
```

The loader reads inference weights directly, does not construct a trainer or
optimizer, disables gradients and training exploration, and checks model/file
hashes. Round 1, 100, 200, 400, 600, and 669 are evaluated alongside initialization,
Random, Greedy, and classic greedy-rollout/outcome-value MCTS. Round 1 has zero
optimizer updates and equals the initialization weights.

The published protocol has 20 boards each for Medium/compatible Hard and five
each for Easy/training control: 500 episodes on 50 profile/seed boards.
Environment seeds are 31000000–31000019, with the first five used for smaller
profiles; smoke used the separate seed 31001000. They are outside training
200000–299999, PLG screening/confirmation seeds, and reserved final-test seeds
400000–400999. Paired neural checkpoints share agent streams to control the
comparison; other baselines have separate deterministic stream slots.

Episode execution reuses `run_episode()`. Initial observation/configuration and
board/engine seed identities are verified within each profile. Results are
persisted incrementally; an interrupted run is not automatically resumable.

Win rates/Wilson intervals use attempted records, with errors/time limits listed
separately. Moves use available episode values, including losses. Winning moves
are conditional on wins. Decision time is total selection time divided by calls,
excluding loading/binding/integrity checks. There is no native score field;
reported tiles cleared is a score proxy, separate from sparse return. Simulated
transition diagnostics are measured only for neural agents; older publication
records contain placeholder zeros for other agents. New runs leave those cells
empty. Equal simulations are not equal wall-clock or transition budgets.

These are development comparisons across single profile configurations, not
unbiased final-test results or statistical significance claims. Inspecting six
checkpoints on these boards does not justify selecting one and treating the same
boards as final validation. The final critic's +1 probes and positive-only replay
explain the zero value loss; they do not establish useful outcome discrimination.

## Public evidence and historical reuse

Small evidence snapshots are under `docs/benchmarks/`. Local `results/`, model
weights, overnight metrics, and smoke outputs are ignored and not deleted.
Two historical JSON manifests have home-directory prefixes redacted in the
publication copies. Original and published hashes are listed separately; do not
feed redacted copies into the strict historical-reuse validator.

The classic-MCTS source guard intentionally rejects unreviewed critical-source
changes. The release audit checks whitespace AST equivalence, byte-equivalent
packing on packaged levels, generated-level registry compatibility tests, and
fixed action/search-statistic reference traces. It updates pins only after that
audit; this does not authorize arbitrary future hash refreshes. See
[release validation](release_validation.md).

Original RL and Neural MCTS weights are not distributed. Fresh installations can
run new experiments, inspect publication records, and execute tests, but cannot
reproduce a specific trained policy without its original weights. CPU resume
identity was tested under the same software configuration; cross-platform
bitwise training equivalence is not promised.
