# AZ-4 evaluation findings

Final checkpoint improved observed Medium wins over initialization (6 vs 2), but had no Hard wins, fewer Easy wins, and fewer Medium/Hard wins than classic MCTS. No broad gameplay superiority or monotonic checkpoint improvement established.

## Controlled comparison

All search agents used 8 simulations. Each Medium/Hard cell is wins/20; Easy and training-control cells are wins/5.

| Agent | Easy | Medium | Compatible Hard | Training control |
|---|---:|---:|---:|---:|
| Random | 4/5 | 0/20 | 0/20 | 5/5 |
| Greedy | 1/5 | 6/20 | 0/20 | 5/5 |
| Classic_MCTS_GO | 5/5 | 14/20 | 6/20 | 5/5 |
| Neural_untrained | 4/5 | 2/20 | 0/20 | 5/5 |
| Neural_round_0669 | 3/5 | 6/20 | 0/20 | 5/5 |

## Checkpoint progression

| Round | Medium wins/20 | Compatible Hard wins/20 |
|---|---:|---:|
| 1 | 2 | 0 |
| 100 | 6 | 2 |
| 200 | 4 | 1 |
| 400 | 6 | 1 |
| 600 | 5 | 0 |
| 669 | 6 | 0 |

## Intervals and paired differences

- Final, medium: 30%; Wilson 95% CI 14.5%–51.9%
- Paired with initialization: n=20, final-only wins=5, initialization-only wins=1
- Final, hard_compatible: 0%; Wilson 95% CI 0.0%–16.1%
- Paired with initialization: n=20, final-only wins=0, initialization-only wins=0

## Why training looked perfect

The checkpoint records an 8×8, five-color board without blockers, collecting 30 tiles of any color in 10 moves. All 5,352 recorded episodes won. Final replay contains 4,096 positive outcomes and no negative outcomes. A constant +1 critic fits these targets; final initial-state value predictions were exactly +1 on every evaluated profile. No target-assignment defect was identified in the inspected code/checkpoint. These findings support an easy training task and uninformative value-target balance, rather than evidence of broad learned competence.

Round 1 has zero optimizer updates and exactly matches the initialization weights. Final replay policy-target mean entropy is 0.01321 nats; 97.8% of targets are one-hot. Lower policy loss therefore also reflects concentrated search targets and is not a gameplay-quality metric.

## Scope and limitations

Accepted Hard artifact has 6 colors, unsupported by AZ-1. hard_compatible is candidate 0 from the saved Medium generation search; original artifacts unchanged.

The compatible candidate was historically confirmed at 22% wins, 95% interval 15.0%–31.1%, using 100 classic-MCTS episodes at 10 simulations; no new level was tuned using Neural MCTS outcomes. The original six-color Hard profile was not evaluated or modified.

This is a small development comparison across one configuration per profile, not final-test evidence. Checkpoints were inspected on the same development boards; no best-checkpoint selection claim is made. Equal simulations do not equalize compute: classic MCTS performs full rollouts; neural MCTS uses leaf values. Simulated-transition counts are instrumented only for neural agents.

There were 500 verified paired episode records on 50 profile/seed boards, with zero errors, time limits, or pairing failures. Models remained frozen and checkpoint hashes were checked after evaluation. Environment seeds were 31000000–31000019 (Medium/Hard) and 31000000–31000004 (Easy/control); smoke used the separate seed 31001000. Training, PLG confirmation, and reserved RL final-test seeds were not reused.

See summary.csv for Wilson intervals, mean moves (all available episodes), winning moves, objective completion, total tiles cleared, sparse returns, and call-weighted decision times. The environment has no native score field: tiles cleared is explicitly a score proxy.

No statistical significance tests or causal claims were made.
