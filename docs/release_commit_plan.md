# Release commit plan — approval required

No staging, unstaging, commits, pushes, or index removals have been performed.
All paths below exist and exhaust the current pending public source/evidence
files. Classic MCTS is already committed as `17d6692`; preserve that commit and
all earlier history. No duplicate implementation commit is proposed.

The commit messages below are lowercase, imperative, and have no prefixes.
Each commit should be validated in an isolated exported index/tree so later
working-tree changes cannot mask a missing dependency. Do not discard or alter
the original working files while preparing those snapshots.

## Shared file and partial staging

`src/match3_ai/evaluation/mcts_provenance.py` is the only shared source file.
Commit 1 includes **only** its `mcts_simulator.py` SHA256 replacement. Leave the
`behavior_data.py` and `environment.py` SHA256 replacements for commit 2, alongside
the corresponding PLG support. This preserves a coherent source boundary at
each commit. Do not stage the whole provenance file in commit 1.

## 1. `repair mcts provenance checks`

Depends on existing classic-MCTS commit; repairs its whitespace-related source pin.

```text
src/match3_ai/evaluation/mcts_provenance.py
tests/test_mcts_simulator.py
```

## 2. `add procedural level generation and difficulty calibration`

Depends on commit 1; includes PLG compatibility support and its matching source pins.

```text
scripts/analyze_level_difficulty.py
scripts/benchmark_targeted_generation.py
src/match3_ai/evaluation/behavior_data.py
src/match3_ai/evaluation/difficulty_benchmark.py
src/match3_ai/evaluation/difficulty_comparison.py
src/match3_ai/evaluation/difficulty_evaluator.py
src/match3_ai/evaluation/mcts_provenance.py
src/match3_ai/games/match3/environment.py
src/match3_ai/levels/batch_generation.py
src/match3_ai/levels/difficulty.py
src/match3_ai/levels/export.py
src/match3_ai/levels/generation.py
src/match3_ai/levels/screening.py
src/match3_ai/levels/targeted_generation.py
src/match3_ai/levels/validation.py
tests/test_batch_generation.py
tests/test_difficulty_benchmark.py
tests/test_difficulty_comparison.py
tests/test_difficulty_evaluator.py
tests/test_difficulty_mcts.py
tests/test_generated_level_environment.py
tests/test_generated_level_fingerprint.py
tests/test_level_difficulty.py
tests/test_level_export.py
tests/test_level_generation.py
tests/test_level_screening.py
tests/test_targeted_generation.py
```

## 3. `implement neural mcts and training pipeline`

Depends on commit 2 for custom generated-level construction.

```text
src/match3_ai/agents/policy_value_encoder.py
src/match3_ai/agents/policy_value_network.py
src/match3_ai/agents/neural_mcts_agent.py
src/match3_ai/training/search_replay.py
src/match3_ai/training/neural_mcts.py
tests/test_policy_value_network.py
tests/test_neural_mcts_agent.py
tests/test_neural_mcts_training.py
```

## 4. `add neural mcts evaluation and benchmarks`

Depends on commits 2–3; includes public fixtures and small AZ-4 evidence.

```text
scripts/benchmark_neural_mcts.py
src/match3_ai/evaluation/neural_mcts_benchmark.py
tests/test_neural_mcts_benchmark.py
experiments/fixtures/az4_profiles/easy/generation_report.json
experiments/fixtures/az4_profiles/easy/level.json
experiments/fixtures/az4_profiles/hard/level.json
experiments/fixtures/az4_profiles/medium/generation_report.json
experiments/fixtures/az4_profiles/medium/level.json
docs/benchmarks/az4/checkpoint_progression.csv
docs/benchmarks/az4/episodes.csv
docs/benchmarks/az4/evaluation_findings.json
docs/benchmarks/az4/evaluation_findings.md
docs/benchmarks/az4/manifest.json
docs/benchmarks/az4/summary.csv
```

## 5. `finalize documentation and project setup`

Depends on the feature commits; finalizes setup, publication, and documentation.

```text
.gitignore
README.md
pyproject.toml
docs/stochastic_mcts.md
docs/reproducibility.md
docs/release_validation.md
docs/release_commit_plan.md
docs/benchmarks/README.md
docs/benchmarks/publication_sources.json
docs/benchmarks/mcts_ablation_100/episodes.csv
docs/benchmarks/mcts_ablation_100/manifest.json
docs/benchmarks/mcts_ablation_100/summary.csv
docs/benchmarks/mcts_ablation_analysis_100/analysis_manifest.json
docs/benchmarks/mcts_ablation_analysis_100/paired_comparisons.csv
docs/benchmarks/mcts_baselines_main_100/episodes.csv
docs/benchmarks/mcts_baselines_main_100/manifest.json
docs/benchmarks/mcts_baselines_main_100/summary.csv
```

### Index-only removals in commit 5 — additional explicit approval required

These eight original result files are already tracked. The proposed action is
`git rm --cached` on these exact paths, followed by committing their public copies
above. It removes them from the release index **without deleting local files**.
No index removal has been performed. Do not use ordinary `git rm`, `clean`, or
filesystem deletion. The working originals remain under ignored `results/`.

```text
results/mcts_ablation_100/episodes.csv
results/mcts_ablation_100/manifest.json
results/mcts_ablation_100/summary.csv
results/mcts_ablation_analysis_100/analysis_manifest.json
results/mcts_ablation_analysis_100/paired_comparisons.csv
results/mcts_baselines_main_100/episodes.csv
results/mcts_baselines_main_100/manifest.json
results/mcts_baselines_main_100/summary.csv
```

The two original manifests with home-directory prefixes remain in earlier Git
history. Current public copies redact those paths and explicitly distinguish
original versus published digests. Erasing prior history is not part of this
plan and would require separate authorization.

## Excluded local work

- All `results/` runs, overnight metrics, raw smoke outputs, and model checkpoints
- `*.pt`, `*.pth`, local environments, caches, logs, temporary and build outputs
- Local `AGENTS.md`, ignored sprint notes, archives, and intermediate analysis
  outputs; retain the existing committed Stage 2 dataset and selected figures

Exclusion means retaining files on disk, not deleting or discarding experiments.
The copied profile fixtures and publication snapshots listed above are the only
new experiment artifacts proposed for inclusion.

## Approval boundary

Approve the five named commits, their exact file/hunk groups, and the eight
index-only result removals before any Git mutation. After approval, create each
commit separately, report validation and final status, and do not push unless
separately instructed. No license choice is made by this plan.
