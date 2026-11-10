# AI Agents

Match3-AI-Lab implements reinforcement learning and search-based agents using a shared game environment. All agents select actions from the legal-action mask.

## Agents

| Agent | Method |
|---|---|
| Random | Uniformly samples a legal action |
| Greedy | Selects the swap producing the largest immediate match |
| DQN / DDQN | Learns action values with experience replay and masked actions |
| PPO | Learns an actor-critic policy using clipped policy updates |
| Classic MCTS | Uses open-loop UCT search and simulated game rollouts |
| Neural MCTS | Combines PUCT search with a learned policy-value network |

## Reinforcement Learning

**DQN / DDQN** approximate action values with neural networks. Double DQN separates action selection from target-value estimation to reduce overestimation. The project explores MLP, CNN, and dueling architectures.

**PPO** uses a masked actor-critic network. Training optimizes a clipped policy objective, while evaluation selects actions deterministically.

Both approaches learn from interactions with the Match-3 environment and are evaluated separately from training.

## Classic MCTS

Classic MCTS uses an open-loop search tree containing action histories rather than fixed successor boards.

Each simulation:
1. Clones the current environment.
2. Samples stochastic outcomes using an independent RNG.
3. Selects actions using UCT.
4. Expands the search tree and performs a rollout.
5. Backpropagates the resulting value.

The implementation supports random and greedy rollout policies with terminal or outcome-based evaluation.

A new search tree is created for each real decision.

## Neural MCTS

Neural MCTS is AlphaZero-inspired and adapted to a single-player stochastic environment.

It combines:
- A CNN policy-value network
- PUCT action selection
- Simulated successor states
- Search-visit policy targets
- Terminal win/loss value targets

The model uses the real game simulator rather than a learned dynamics model. Values are not sign-flipped during backup because there is no opposing player.

Training uses completed episodes, bounded replay, and checkpoint-based recovery.

## Evaluation

Agents are compared using seeded gameplay episodes, win rates, confidence intervals, and paired board configurations.

Search agents require additional computation at decision time, so equal simulation counts do not imply equal computational cost.

See [Results](RESULTS.md) for measured performance and limitations.
