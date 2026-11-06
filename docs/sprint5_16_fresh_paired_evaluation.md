# Sprint 5.16: preregistered fresh-board paired evaluation

This is evaluation only. No weights, architectures, hyperparameters, rewards,
gameplay, levels or training routines are changed. Existing checkpoints are
loaded for inference; no optimizer step or new PPO/DQN training is invoked.

## Frozen protocol and provenance

Before production board generation, the protocol was saved in
`experiments/results/sprint5_16_preregistration/protocol.json`, with SHA-256:

```
1f465cfc1797365a292d01e2d3b21e7e2af880dcd944d1c234c19eb45bb1160d
```

The corresponding history_inventory.json records SHA-256 hashes for 513 historical
artifact files: local experiment JSON, JSONL, CSV, NPZ and Torch checkpoints.
The scan conservatively excludes validation-range numeric values, including
declared seed ranges and unrelated matching numeric values. It found 101 excluded
values (300000–300099 and the metadata boundary 300999). The selected 500 seeds are
**300100–300599**, frozen in the protocol before evaluation.

These do not overlap recorded training/validation seeds or reserved final-test
episode seeds 400000–400999. Freshness is relative to the available local recorded
artifacts; it cannot certify unrecorded or external experiments. Hashes of all nine
checkpoints, the level configuration and source implementation are preregistered.
Changing a frozen source/checkpoint or protocol causes execution to fail.

The protocol is not edited after outcomes. Tests use historical seeds 300000 and
300001 under synthetic temporary protocols; they do not inspect the production
500 boards before preregistration. A JSON tuple/list canonicalization issue found
in testing was corrected before production registration.

## Eleven fixed policies

- Random, uniformly sampling legal actions with independent local RNG
- Existing GreedyAgent, maximizing immediate match count with deterministic ties
- Existing MLP-DQN final checkpoints: training seeds 5901, 5902 and 5903, step 50000
- PPO initial and final checkpoints for each training seed 5915, 5916 and 5917

No best-validation checkpoints are selected. PPO initial files are policy_step_0.pt;
final files are checkpoint_final.pt from Sprint 5.15. DQN files are checkpoint_final.pt
from Sprint 5.9. Exact paths/hashes are in protocol.json. Random's 500 local seeds
are frozen as 615160000–615160499 and are independent of environment RNG.

All 11×500 = **5500** episodes use blocker-mayhem, its unchanged 15-move limit and
objectives, sparse rewards, cascade budget 200, initialization budget 1000 and a
50-step external safety bound. Each episode has a new environment with the same
selected level configuration. Reset derives independent board-generation and
gameplay RNG child seeds from the environment seed. Initial observation fingerprints,
board/engine seeds and level-config identities must match across all eleven policies.
Equal initial RNG seeds do not imply identical future tile sequences after different
action trajectories and RNG consumption.

Policy model, optimizer and sampling/shuffle state are fingerprinted before and
after inference. File hashes are rechecked. Deterministic PPO/DQN inference does
not advance training RNGs. Random intentionally advances its own fresh per-board
generator. The engine, environment, existing agents and episode runner are reused
unchanged. No Pygame comparison UI is added.

## Primary estimand and inference

For each board b:

```
d[b] = mean(win[PPO5915_final,b], win[PPO5916_final,b], win[PPO5917_final,b])
       - win[Greedy,b]
```

The primary result is mean(d), with a two-sided paired board-bootstrap percentile
95% confidence interval. All models are resampled together by board. The interval
is conditional on the three observed PPO training seeds; it does not describe
uncertainty over the population of possible training runs. Training-seed sample
SD (ddof=1) is reported separately. There is no 1500-independent-episode binomial
interval for mean PPO performance.

The frozen protocol additionally specifies a two-sided null-centered board-bootstrap
test for the primary mean. Bootstrap uses 20000 samples and seed 51616. Centering
the empirical per-board differences at their observed mean approximates a null
sampling distribution. This is not an exact permutation or label-exchangeability
test. Its p-value uses the plus-one finite-resampling correction. Inference assumes
the selected seeded generator realizations reasonably represent the target board
distribution; it does not establish performance across levels.

## Nine secondary hypotheses

1. Mean final PPO minus Random
2. Mean final PPO minus mean frozen DQN
3. Mean final PPO minus mean initial PPO
4. PPO5915 final minus corresponding initial
5. PPO5916 final minus corresponding initial
6. PPO5917 final minus corresponding initial
7. PPO5915 final minus Greedy
8. PPO5916 final minus Greedy
9. PPO5917 final minus Greedy

All are board-paired. The three aggregate comparisons use the same null-centered
bootstrap mean test. The six individual binary comparisons use exact two-sided
McNemar tests on discordant pairs. One Holm correction covers these nine secondary
p-values; the single primary comparison is separate. Reported secondary CIs are
pointwise, not simultaneous Holm-adjusted intervals. Raw and adjusted p-values are
kept. Nonsignificance is not equivalence, and conditional evidence about these
checkpoints is not algorithm superiority across training seeds.

Individual policy rates use Wilson 95% intervals with 500 boards. Reports preserve
wins, losses, reward, episode length/moves, objective completion, invalid-action
counts and every per-board outcome. Equal training-transition budgets for PPO and
DQN do not equalize optimizer updates, collection trajectories or wall-clock cost.

## Critic diagnostic scope

For each frozen PPO model and each initial board, save the critic prediction and
the observed discounted terminal return from that model's deterministic greedy
actor. With sparse terminal rewards, target = gamma^(steps−1) × terminal_reward,
where gamma=.99. Report mean V, target variance, mean(V−target), RMSE and
1−Var(target−V)/Var(target). Constant target variance yields undefined EV.

These are held-out **greedy-behavior return-agreement** diagnostics. PPO's critic
was trained under stochastic masked actions and on-policy GAE targets, so this is
not an unbiased test of its training-policy value function. One trajectory per
board also includes gameplay-return variability. Different models induce different
return targets; do not rank their EV as a common fixed-target accuracy benchmark.
Training-time GAE metrics remain separate and unchanged. EV does not detect
constant prediction bias, which is why bias and RMSE are retained.

No extra stochastic critic rollouts are generated; the evaluation remains exactly
5500 episodes. More precise on-policy critic testing would require a separately
approved protocol with repeated policy-matched trajectories.

## Behavioral examples fixed before outcomes

PPO means majority outcome across the three final PPO models. For each category,
select the three lowest seed IDs, preserving empty categories without substitution:

- PPO majority wins, Greedy loses
- Greedy wins, PPO majority loses
- All eleven policies win
- All eleven policies lose

Initial observations, all eleven action sequences, moves, outcomes and final
objective progress are retained for selected boards. Initial snapshots for all
500 boards are also preserved. These examples are descriptive, not independent
statistical tests or explanations of causal strategy differences.

## Execution and outputs

The new modules separate protocol provenance, execution and analysis:

- `evaluation/fresh_protocol.py`: historical inventory, exclusive protocol creation,
  checkpoint/source hashes and protocol validation
- `evaluation/fresh_evaluation.py`: frozen-policy loading, existing runner/analytics,
  initial observations, state checks and isolated policy outputs
- `evaluation/fresh_analysis.py`: alignment, preregistered statistics, critic agreement,
  behavioral traces, CSV exports and figures
- `tests/test_fresh_evaluation.py`: seed provenance, frozen protocol, policy loading,
  deterministic inference, shared-board alignment, statistics and complete export

Failures preserve partial/completed outputs. No boards are replaced or excluded.
Confirmatory analysis stops unless all planned episodes complete with valid
WON/LOST outcomes and zero invalid actions. Output directories and files are
created exclusively; historical artifacts are not overwritten.

Production outputs:

- `experiments/results/sprint5_16_preregistration/`: immutable protocol, its digest
  and historical inventory
- `experiments/results/sprint5_16_frozen_policies/POLICY/`: episodes/moves CSVs,
  complete initial observations, summaries, elapsed time and state/hash checks
- `experiments/results/sprint5_16_analysis/`: report.md, analysis.json, episodes.csv,
  policy_summary.csv, paired_outcomes.csv, comparisons.csv, critic_predictions.csv,
  representative_boards.json and diagnostic PNG/PDF figures

Run the protocol command only into a fresh directory. Workers accept --protocol,
repeated --policy names and --output-root; analysis accepts --protocol,
--results-root and a fresh --output-dir. Workers and analysis verify the frozen
protocol digest, source/checkpoint hashes and result identity. Reproducing an
existing protocol uses its fixed seeds; creating a later study requires a new
inventory and must not call those already observed boards fresh again.

No source files are changed after the production protocol is frozen. Documentation
and descriptive result interpretation do not amend the protocol. Any future longer
training needs separate approval and preregistration; exact mid-episode checkpoint
resume remains unsupported. No Sprint 5.17 work is started.

## Observed results and supported conclusions

All 5500 planned episodes completed, with zero invalid actions, errors or
replacements. All 500 initial fingerprints are distinct and match across policies.
Frozen source, checkpoint, optimizer/RNG and protocol checks passed. The numerical
tables, comparisons and every policy/board pair passed an independent consistency
audit in reproducibility_audit.json. Initial/final weights are unchanged.

| Policy | Wins / 500 | Win rate [Wilson 95% CI] |
|---|---:|---|
| Random | 225 | 45.0% [40.69%, 49.38%] |
| Greedy | 257 | 51.4% [47.03%, 55.75%] |
| DQN 5901 | 243 | 48.6% [44.25%, 52.97%] |
| DQN 5902 | 240 | 48.0% [43.65%, 52.38%] |
| DQN 5903 | 239 | 47.8% [43.46%, 52.18%] |
| PPO 5915 initial | 207 | 41.4% [37.16%, 45.77%] |
| PPO 5915 final | 250 | 50.0% [45.63%, 54.37%] |
| PPO 5916 initial | 202 | 40.4% [36.19%, 44.76%] |
| PPO 5916 final | 262 | 52.4% [48.02%, 56.74%] |
| PPO 5917 initial | 224 | 44.8% [40.50%, 49.18%] |
| PPO 5917 final | 283 | 56.6% [52.22%, 60.88%] |

Mean final PPO is **53.0%**, training-seed sample SD **3.34 pp**. Mean frozen DQN
is **48.13%**, sample SD **.42 pp**. Mean initial PPO is **42.2%**, sample SD
**2.31 pp**. These SDs describe only the three observed training runs.

The single primary comparison is **+1.60 pp** for mean final PPO minus Greedy,
paired 95% CI **[−3.20, +6.47]**, approximate bootstrap p=.52747. It does not
establish an advantage over Greedy or establish equivalence.

Preregistered secondary mean differences:

| Reference | Final PPO difference pp [pointwise 95% CI] | Holm p |
|---|---|---:|
| Random | +8.00 [+2.93, +13.07] | .01020 |
| Mean frozen DQN | +4.87 [+1.40, +8.33] | .03200 |
| Mean initial PPO | +10.80 [+7.13, +14.47] | .00045 |

Individual final-minus-initial gains are +8.6, +12.0 and +11.8 pp; exact McNemar
Holm-adjusted p values are .04239, .00064 and .00197. This supports improvement
for these frozen models on the fresh seeded board distribution. No individual
PPO-final-versus-Greedy comparison passes secondary Holm correction. The primary
result is not replaced by favorable secondary results. Conditional fixed-model
evidence is not proof of algorithm superiority across training runs or efficiency.

The six critic behavior-return tables are in report.md. Final-model behavior
agreement EV values are −.018, −.015 and −.033, with RMSE .913, .910 and 1.009.
Their V-minus-return biases are +.151, −.144 and −.454. These indicate weak agreement
with observed greedy-play returns, not demonstrated on-policy critic failure;
the stochastic-training/deterministic-evaluation distinction remains essential.

Predefined behavioral categories contain 120 PPO-majority-win/Greedy-loss boards,
106 converse boards, one all-eleven-win board (300208) and one all-eleven-lose
board (300443). Only the existing single universal examples are used. Behavioral
details are in behavior_notes.md and representative_boards.json. On seed 300102,
final PPO models clear all six blockers in 6–7 moves while Greedy ends at five
blockers after 15. On seed 300114, Greedy clears all six in nine moves while every
final PPO model ends at five after 15, despite collecting many tiles. These are
descriptive counterexamples, not causal explanations of agent strategy.

Per-policy runtime metadata sums to 1864.94 process-seconds. Three workers ran
concurrently; the longest worker's accumulated policy time is about 670 seconds
(11.2 minutes), excluding small launch/export overhead. These timings include
system contention and concurrent regression tests and are not isolated inference
latency or elapsed-time sums.

## Validation and next-study recommendation

Five new tests cover historical usage, protocol/checkpoint consistency, deterministic
loading, shared-board identity (including rejection of a mismatched state), paired
statistics, critic metric interpretation, predefined representative rules and full
temporary exports. Focused PPO/DQN/evaluation suite: **72 passed**. Full suite:
**790 passed**. Ruff, formatting and Git whitespace checks are clean. No existing
tests were weakened, and no production source was changed after preregistration.

The fresh-board study supports a learning signal versus initialization while
leaving the primary Greedy advantage unresolved. If longer training is approved,
preregister matched initialization seeds, unchanged hyperparameters, fixed final
budgets/endpoints and additional training-seed replication. Budgets aligned with
256-step rollout boundaries, such as 51200 and 102400, avoid ambiguity from partial
rollouts. Exact continuation from existing 50k checkpoints is unsupported; a
longer-budget run must not be presented as an exact resumed trajectory.

Longer training is a hypothesis, not a guaranteed improvement. Keep critic
calibration a separate on-policy evaluation question if extra diagnostic rollouts
are approved. The current conservative local inventory leaves 399 unused validation
seed candidates after this study; another 500-board fresh study cannot fit that
remaining pool without an approved seed-policy change. Do not silently consume
reserved final-test seeds. No longer training, tuning or Sprint 5.17 was started.
