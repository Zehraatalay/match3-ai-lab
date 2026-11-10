# Release validation and compatibility audit

## Scope

This pass adds no new agent, training algorithm, or experiment. Existing commits,
working files, checkpoints, training metrics, and local experiment outputs are
preserved. No index changes or commits were performed. See the exact
[approval-gated commit plan](release_commit_plan.md).

## Findings and repairs

- Initial full suite: 1,307 passed, 3 failed, 14 setup errors, zero skips. All
  failures/errors came from a stale simulator source hash after an already
  committed whitespace cleanup
- All 27 formatter changes retain identical Python ASTs; the import-order repair
  also changes only blank-line formatting
- Removed a duplicate plain `self.level_id` assignment in generated-level support
- Raw observation packing remains byte-equivalent to the previous neural packer
  on 27 initial boards: all nine packaged levels at seeds 300000, 300001, 31000000
- Generated-level tests preserve packaged-level order and support larger move
  limits; the fixed 12 pre-optimization action/search-statistic traces pass
- Updated the three critical-source pins below after this audit; no game rule,
  reward, rollout, search selection, or backup behavior was changed
- Declared pandas >=2.2, SciPy, and scikit-learn in the analysis extra. pandas
  2.2 is required by `include_groups=False` in the analysis workflow
- AZ-4 now reads portable profile fixtures when available and retains the local
  result-path fallback. Unmeasured classic-MCTS transition counts are left empty
  in new runs; archived evidence preserves the original placeholder-zero records

| Source | Previous pin | Audited release pin |
|---|---|---|
| `src/match3_ai/agents/mcts_simulator.py` | `15140b62988ce458f5fa3c78f6eec95332c6f21f372a590b42eec98b5981a6d9` | `3be1602c57ceca8040c28ba70653405517dc25ab615903de328ffd1e98262734` |
| `src/match3_ai/evaluation/behavior_data.py` | `b7f92c748b9327800c5d95009b495153a43589117b42395ae8b78b52ce631d82` | `c510fa42a47cd4fc89099a4a09bbc0a42315d9e9f5f2d3fa730db1544ed4462c` |
| `src/match3_ai/games/match3/environment.py` | `7706cf3c3847b977c142cb35d72e791995bd7cd5af43e85c5c4dd20b65710cfb` | `d4e8e45a21d34e2f10c9d75f05111b1740b37f7d2a3d234b6dae19e713681b49` |

This pins valid current sources; it does not assert that arbitrary future edits
are compatible. Modern historical records with different critical-source hashes
still require an explicit compatibility audit. Redacted publication manifests
are not accepted replacements for original historical import artifacts.

## Validation

Final full suite: **1,325 passed, 0 failed, 0 errors, 0 skipped**, in 115.34
seconds. This includes the public-fixture test and classic-MCTS reference
regressions. An earlier repaired run passed 1,324 tests before that case was added.

The complete Ruff lint and formatting checks pass. Eight CLI help checks pass:
Pygame UI, DQN, PPO, baseline evaluation, classic MCTS benchmarking, Neural MCTS
benchmarking, targeted generation, and difficulty analysis. An offline wheel
build/install into a temporary target successfully imports the installed package,
loads packaged level JSON, resets an environment, and imports the neural network.
The existing environment's `pip check` reports no broken requirements.

A bounded classic-MCTS CLI smoke uses one episode per configuration/level and
one simulation: 12 attempted episodes, zero errors and time limits, with outputs
confined to a temporary directory. No training,
large benchmark, or original experiment output is overwritten by release checks.

## Publication and Git safety

`results/`, checkpoints, training outputs, logs, caches, and build artifacts are
ignored and retained locally. Small public evidence is copied into
`docs/benchmarks/`; original/published SHA256 digests are listed separately.
Two already committed JSON manifests contain home-directory prefixes. Their
publication copies redact those prefixes. Removing original results from the
release index is proposed only after approval, while retaining local files.
Earlier commits will still contain those historical paths: this task does not
rewrite history. No credential-pattern matches were found in the proposed public
text files. README/supporting-document links are checked before completion.

There is no LICENSE file. The project can be published as a portfolio source
repository, but granting a reuse license is an owner decision left unchanged.
Trained weights are excluded, so exact checkpoint-specific results cannot be
regenerated from a fresh clone without separately supplied original weights.

## Tested software

Python 3.11.17; numpy 2.4.6, torch 2.15.0, gymnasium 1.4.0, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1, pytest 9.1.1, ruff 0.17.0.
All declared core and optional dependency bounds accept these installed versions.
The wheel was built offline with build isolation/dependency downloads disabled;
installation used a separate temporary target, leaving the active environment
unchanged. The pandas upper bound includes the tested 3.x release.
