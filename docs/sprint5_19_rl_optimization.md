# Sprint 5.19 — Controlled PPO optimization and dueling DDQN budget

Frozen manifest SHA-256: `5f307d41cb8bee83dcc1fc2c8263017a9511f811d2c4f80a0a0801eb328e9d03`. Actual start 2026-11-06T07:43:20+00:00; deadline 2026-11-06T15:43:20+00:00. Finalized 2026-11-06T09:25:44.123191+00:00; within deadline: True.
Istanbul execution window: 2026-11-06T10:43:20+03:00 to 2026-11-06T18:43:20+03:00.

The study uses 200 fresh development boards, 310000–310199, disjoint from all previously recorded seeds. This explicit study block leaves the legacy training/validation/final-test contracts unchanged. No reserved final-test board was evaluated.

## Findings

| Variant | Completed seeds | Mean 100k win rate | Training-seed SD (pp) | Mean process minutes | Objective completion | Remaining HP |
|---|---:|---:|---:|---:|---:|---:|
| PPO constant | 3 | 56.83% | 4.31 | 24.07 | 96.06% | 0.515 |
| PPO LR decay | 3 | 61.17% | 2.08 | 24.08 | 96.36% | 0.477 |
| Dueling DDQN | 3 | 77.17% | 1.76 | 30.37 | 97.90% | 0.278 |

| Primary hypothesis | Difference(pp) | Pointwise conditional95% CI(pp) | Raw p | Holm(two) p |
|---|---:|---|---:|---:|
| A | +4.33 | [-1.17, +9.83] | 0.128894 | 0.128894 |
| B | +17.00 | [+12.00, +22.17] | 0.000050 | 0.000100 |

A compares PPO LR decay minus constant at fixed100k. B compares the same new dueling runs at100k minus50k. Both were preregistered; one Holm correction covers both. Pointwise CIs are not multiplicity-adjusted. Bootstrap uses20000 replicates and seed519519 with common board draws across all policies/checkpoints; its index digest is saved. Raw p-values are approximate null-centered paired board-bootstrap values, not exact permutation tests.

These intervals condition on the three observed training pairs. Per-seed variation and sign-flip sensitivity do not establish population-level superiority; with three pairs the minimum two-sided sign-flip p is.25. Nonsignificance is not equivalence. A development score above70% is not held-out generalization.

## Per-seed outcomes

| Run | Checkpoint wins/200 in collected-transition order | Total process seconds | Validation seconds |
|---|---|---:|---:|
| a_constant_5970 | 0: 71 (35.5%); 50000: 125 (62.5%); 75000: 115 (57.5%); 100000: 106 (53.0%) | 1433.90 | 123.33 |
| a_late_linear_5971 | 0: 96 (48.0%); 50000: 110 (55.0%); 75000: 122 (61.0%); 100000: 121 (60.5%) | 1432.29 | 121.12 |
| a_constant_5972 | 0: 99 (49.5%); 50000: 100 (50.0%); 75000: 102 (51.0%); 100000: 112 (56.0%) | 1441.87 | 124.87 |
| a_late_linear_5970 | 0: 71 (35.5%); 50000: 125 (62.5%); 75000: 120 (60.0%); 100000: 127 (63.5%) | 1448.35 | 124.25 |
| a_constant_5971 | 0: 96 (48.0%); 50000: 110 (55.0%); 75000: 129 (64.5%); 100000: 123 (61.5%) | 1456.68 | 124.04 |
| a_late_linear_5972 | 0: 99 (49.5%); 50000: 100 (50.0%); 75000: 116 (58.0%); 100000: 119 (59.5%) | 1453.66 | 126.33 |
| b_dueling_5980 | 0: 86 (43.0%); 25000: 124 (62.0%); 50000: 125 (62.5%); 75000: 146 (73.0%); 100000: 154 (77.0%) | 1832.99 | 213.07 |
| b_dueling_5981 | 0: 85 (42.5%); 25000: 110 (55.0%); 50000: 117 (58.5%); 75000: 134 (67.0%); 100000: 158 (79.0%) | 1814.15 | 213.60 |
| b_dueling_5982 | 0: 93 (46.5%); 25000: 115 (57.5%); 50000: 119 (59.5%); 75000: 125 (62.5%); 100000: 151 (75.5%) | 1820.33 | 213.58 |

## Interpretation and late stability

PrimaryA: no positive effect established after correction. Individual effects(pp): [10.5, -1.0, 3.5]; pair-effect SD 5.80pp; training-pair sign-flip sensitivity p=0.500.
PrimaryB: positive conditional evidence after correction. Individual effects(pp): [14.5, 20.5, 16.0]; pair-effect SD 3.12pp; training-pair sign-flip sensitivity p=0.250.
PPO constant: mean75k→100k signed change -0.83pp; negative-part regression 2.50pp; absolute change 4.17pp. These are exploratory descriptive summaries.
PPO LR decay: mean75k→100k signed change +1.50pp; negative-part regression 0.17pp; absolute change 1.83pp. These are exploratory descriptive summaries.
Decay-minus-control mean negative-part regression is -2.33 pp: lower observed regression. This descriptive result from three pairs is not a confirmatory stability test.

| PPO run / interval | KL | Clip fraction | Entropy | Value loss | Explained variance | Gradient clipping frequency |
|---|---:|---:|---:|---:|---:|---:|
| a_constant_5970: 50000–75000 | 0.00449 | 0.0542 | 0.7109 | 0.2301 | 0.1518 | 100.00% |
| a_constant_5970: 75000–100000 | 0.00399 | 0.0478 | 0.4819 | 0.2291 | 0.1771 | 100.00% |
| a_late_linear_5971: 50000–75000 | 0.00369 | 0.0481 | 0.5536 | 0.2321 | 0.1393 | 100.00% |
| a_late_linear_5971: 75000–100000 | 0.00093 | 0.0114 | 0.3249 | 0.2260 | 0.1800 | 100.00% |
| a_constant_5972: 50000–75000 | 0.00358 | 0.0459 | 0.6357 | 0.2321 | 0.1387 | 100.00% |
| a_constant_5972: 75000–100000 | 0.00430 | 0.0553 | 1.0352 | 0.2247 | 0.1780 | 100.00% |
| a_late_linear_5970: 50000–75000 | 0.00394 | 0.0508 | 1.0880 | 0.2291 | 0.1571 | 100.00% |
| a_late_linear_5970: 75000–100000 | 0.00096 | 0.0089 | 0.6659 | 0.2298 | 0.1818 | 100.00% |
| a_constant_5971: 50000–75000 | 0.00478 | 0.0630 | 0.9266 | 0.2309 | 0.1402 | 100.00% |
| a_constant_5971: 75000–100000 | 0.00477 | 0.0561 | 1.1523 | 0.2211 | 0.1848 | 100.00% |
| a_late_linear_5972: 50000–75000 | 0.00278 | 0.0378 | 0.4548 | 0.2314 | 0.1482 | 100.00% |
| a_late_linear_5972: 75000–100000 | 0.00063 | 0.0068 | 0.3684 | 0.2328 | 0.1626 | 100.00% |

Lower KL or smaller gradients alone do not establish better performance or stability. KL, clip fraction, entropy, value loss, pre-update trajectory-dependent EV and raw advantage summaries are retained by interval. Gradient norms are the total L2 norms returned before clipping; clipping frequency is count(norm>.5)/optimizer steps. The entropy coefficient remains.01. Learning-rate trajectories verify the sole PPO intervention. Associations across three pairs and sparse checkpoints cannot identify a causal diagnostic mechanism.

Dueling TD loss/Q statistics, epsilon, replay size, updates, target synchronizations and gradients remain in raw CSVs and summary distributions. Epsilon stays.05 after50k, replay capacity remains50000, and target synchronization occurs every1000 optimizer updates. Different on-policy or replay states make loss magnitudes imperfect quality comparisons.

## Strongest checkpoint and compute trade-off

Highest fixed100k endpoint: b_dueling_5981, 79.00%, `experiments/results/sprint5_19_rl_optimization/runs/b_dueling_5981/checkpoint_final.pt`. Keep all final and intermediate checkpoints. This candidate is development-selected.
Highest scheduled checkpoint, descriptive only: b_dueling_5981 at100000, 79.00%. Aspirational70% reached by a measured checkpoint: True.
Largest observed endpoint win-rate/process-minute ratio among100k cohorts: Dueling DDQN. This simple descriptive ratio is not a statistically controlled compute-efficiency claim; budgets, optimizer counts and initial models differ. Within-run dueling50k cumulative time is also plotted. Process time includes evaluation/checkpoint work and contention; sums are not elapsed batch time or measured CPU time.

## Protocol and reproducibility

Three new PPO master-seed pairs5970–5972 use equivalent initial parameters and controlled environment/model/policy/shuffle streams. Identical0/50k weights and the first50000 transition records are audited. LR is.0003 through50k, then linearly declines by collected rollout completion to.00003 at100k; each rollout shares its rate across all epochs. PPO intermediate policies optimize through49920/74752 at50k/75k, retaining80/248 pending samples. Final100k includes the160-sample tail:391 rollouts and6252 Adam steps/run. Scheduled PPO checkpoints are exactly0,50k,75k,100k.

Dueling masters5980–5982 initialize new networks/replay. Old50k checkpoints include online/target weights, optimizer, replay/sampling RNG, exploration RNG and counters, but omit the active environment and in-progress episode. Their budget-truncated final sample can also differ from a continuous100k run. Exact continuation is unsupported; no model was resumed. The new within-run50k checkpoint is the matched control. Each100k run has24751 updates and24 target synchronizations, with checkpoints0/25k/50k/75k/100k.

Dueling runs use private model/exploration/replay streams under a common deterministic episode-seed curriculum starting200000. Reported training variability is conditional on that curriculum; shared environment seeds do not make the three initial models identical. Training and validation seeds are disjoint. Sparse rewards, legal masks, architecture, replay/target settings, game rules and guards are unchanged.

The first smoke attempt failed configuration validation before any transitions because default baseline agent seeds overlapped the training partition. It is preserved. A separate successful attempt used explicit independent baseline seeds. Both PPO and dueling smoke runs collected512 transitions, optimized32/113 steps, exported actual evaluation metadata and statuses, saved/reloaded checkpoints and wrote summaries. Smoke outcomes are excluded from research estimates and used legacy validation boards, not the fresh production set.

## Historical context, disclosure and limitations

The overnight199-board means were dueling63.65%, standardDDQN56.95%, historicalPPO52.60%, Greedy52.26%, Random46.73%, and historicalVanillaDQN46.23%. Those checkpoints/boards/training cohorts are different. These values are context only, not paired controls for this sprint. No historical result was overwritten or substituted.

The six individual endpoint tests have a separate exploratory six-slot Holm family; raw/adjusted McNemar p-values and discordant tables are in statistics.json. Missing planned comparisons retain non-rejection correction slots and are never silently replaced. Other checkpoints and diagnostic correlations are descriptive. Three training seeds, reused development boards and candidate selection limit generalization claims. No reserved final-test evaluation was performed.

Publish a concise protocol/results/limitations report, source changes and reproducibility commands after a separate user-approved Git review. Include negative results, per-seed tables, confidence intervals and failures; do not publish a70%/superiority claim from a selected development score. Do not commit large checkpoint/replay artifacts blindly; preserve them locally and describe their provenance. No Git publication operation is performed here.

The experiment and artifact pipeline can support a Human Gameplay/Data Science Analytics handoff, but mature product infrastructure is distinct from verified RL superiority. Broader seed replication and genuinely independent evaluation would be needed for strong performance claims. Do not automatically expand RL experiments to chase70%, and do not begin the next phase in this sprint.

## Inventory, validation and artifacts

| Job | Status |
|---|---|
| a_constant_5970 | completed |
| a_late_linear_5971 | completed |
| a_constant_5972 | completed |
| a_late_linear_5970 | completed |
| a_constant_5971 | completed |
| a_late_linear_5972 | completed |
| b_dueling_5980 | completed |
| b_dueling_5981 | completed |
| b_dueling_5982 | completed |

Historical artifacts verified unchanged: 1425. Only intended existing files changed: ['src/match3_ai/training/seed_policy.py', 'src/match3_ai/training/dqn.py', 'src/match3_ai/training/ppo.py']. All completed policy checkpoints load, match frozen config, retain finite parameters, reproduce200 recorded initial decisions, preserve inference RNG/weights and match final Adam counters. These are reset-only/initial-decision checks, not additional full evaluation episodes.

Validation statuses: {"analysis": {"command": ["/Users/zehraatalay/miniforge3/envs/match3-ai/bin/python", "-m", "match3_ai.training.sprint519_analysis", "analyze", "--root", "experiments/results/sprint5_19_rl_optimization"], "returncode": 0, "status": "passed"}, "diff_check": {"command": ["git", "diff", "--check"], "returncode": 0, "status": "passed"}, "focused_pytest": {"command": ["/Users/zehraatalay/miniforge3/envs/match3-ai/bin/python", "-m", "pytest", "-q", "tests/test_sprint519.py"], "returncode": 0, "status": "passed"}, "format": {"command": ["/Users/zehraatalay/miniforge3/envs/match3-ai/bin/python", "-m", "ruff", "format", "--check", "src/match3_ai/training/sprint519.py", "src/match3_ai/training/sprint519_analysis.py", "src/match3_ai/training/ppo.py", "src/match3_ai/training/dqn.py", "src/match3_ai/training/seed_policy.py", "tests/test_sprint519.py"], "returncode": 0, "status": "passed"}, "full_pytest": {"command": ["/Users/zehraatalay/miniforge3/envs/match3-ai/bin/python", "-m", "pytest", "-q"], "returncode": 0, "status": "passed"}, "ruff": {"command": ["/Users/zehraatalay/miniforge3/envs/match3-ai/bin/python", "-m", "ruff", "check", "."], "returncode": 0, "status": "passed"}}

Machine-readable data: `experiments/results/sprint5_19_rl_optimization/final_summary.json`, `analysis/per_seed_results.json`, `analysis/statistics.json`, `analysis/checkpoint_integrity.json`, frozen manifest/inventory, actual smoke outputs, per-run append-only events/status/config/provenance, raw training/evaluation CSVs and source archive. All PNG/PDF figures are reproducible from `python -m match3_ai.training.sprint519_analysis analyze --root <fresh-root-with-manifest-and-runs>`; existing output directories are rejected. Original raw optimizer data remain available; plots use trailing50 recorded values except unsmoothed LR, with no smoothing across runs.

No staging, committing, pushing, resetting, rebasing, stashing or discarding work occurred. Failed and partial attempts remain preserved. No Human Gameplay or Data Science Analytics implementation was started.
