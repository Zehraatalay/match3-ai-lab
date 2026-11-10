# Match3-AI-Lab

**AI agents, automated playtesting, and difficulty analysis for Match-3 games.**

Match3-AI-Lab is a Python project combining **reinforcement learning**, **Monte Carlo Tree Search**, **procedural level generation**, and **gameplay analytics** in a playable Match-3 environment.

## Gameplay

<table>
<tr>
<td align="center" width="50%"><img src="docs/images/level4.png" alt="Frozen Garden" width="100%"><br><strong>Frozen Garden</strong></td>
<td align="center" width="50%"><img src="docs/images/level6.png" alt="Blocker Mayhem" width="100%"><br><strong>Blocker Mayhem</strong></td>
</tr>
</table>

The game supports special tiles, blockers, cascades, multiple objectives, and human play through a Pygame interface.

## Features

| Component | Capabilities |
| :--- | :--- |
| **Game Engine** | Match-3 mechanics, special tiles, blockers, and objectives |
| **AI & Search** | Random, Greedy, DQN/DDQN, PPO, Classic MCTS, Neural MCTS |
| **Level Generation** | Seeded generation, validation, and difficulty calibration |
| **Evaluation** | Paired benchmarks, win rates, and confidence intervals |
| **Analytics** | Statistical testing, win prediction, and spatial features |
| **Interface** | Playable Pygame UI |

## AI Agents

| Agent | Approach |
| :--- | :--- |
| **Random / Greedy** | Baseline legal-action policies |
| **DQN / DDQN** | Value-based reinforcement learning |
| **PPO** | Masked actor-critic reinforcement learning |
| **Classic MCTS** | Stochastic open-loop UCT search |
| **Neural MCTS** | PUCT search with a CNN policy-value network |

Implementation details: [AI Algorithms](docs/algorithms.md).

## Benchmark Results

### Reinforcement Learning

**Protocol:** 100 evaluation episodes per agent per level, using paired boards.

| Level | Random | Greedy | DDQN | PPO |
| :--- | ---: | ---: | ---: | ---: |
| Double Trouble | 62% | 64% | **77%** | 58% |
| Wooden Trouble | 43% | 32% | **63%** | 51% |
| Frozen Garden | 48% | 54% | **76%** | 66% |
| **Macro average** | **51.0%** | **50.0%** | **72.0%** | **58.3%** |

**Finding:** DDQN had the highest observed win rate on all three evaluated levels.

### Classic Monte Carlo Tree Search

**Protocol:** 100 paired boards per level, 30 simulations per decision.

| Level | Random | Greedy | MCTS (RT) | MCTS (GO) |
| :--- | ---: | ---: | ---: | ---: |
| Double Trouble V2 | 52% | 55% | 91% | **93%** |
| Wooden Trouble V2 | 41% | 37% | **95%** | 93% |
| Frozen Garden V2 | 52% | 64% | 94% | **95%** |

*RT = random rollout + terminal value; GO = greedy rollout + outcome value.*

**Finding:** Classic MCTS outperformed both baselines in these development benchmarks.

[Baseline results](docs/benchmarks/mcts_baselines_main_100/summary.csv) · [MCTS ablation](docs/benchmarks/mcts_ablation_100/summary.csv)

### Neural Monte Carlo Tree Search

**Protocol:** 20 paired boards per configuration, 8 simulations per decision.

| Agent | Medium | Hard |
| :--- | ---: | ---: |
| Random | 0% | 0% |
| Greedy | 30% | 0% |
| **Classic MCTS** | **70%** | **30%** |
| Neural MCTS (untrained) | 10% | 0% |
| Neural MCTS (trained) | 30% | 0% |

**Finding:** Training improved the observed Medium result relative to initialization, but Neural MCTS did not surpass Classic MCTS. Training on overly easy levels limited useful outcome learning.

[Neural MCTS findings](docs/benchmarks/az4/evaluation_findings.md)

> **Evaluation note:** These studies use different levels and evaluation protocols; their win rates are not a single cross-study ranking.

## Procedural Level Generation

The pipeline generates level candidates, validates their constraints, and calibrates difficulty through seeded AI playtesting.

| Difficulty | Target Classic MCTS win rate |
| :--- | ---: |
| Easy | 75–100% |
| Medium | 40–75% |
| Hard | 10–40% |
| Very Hard | 0–10% |

Difficulty reflects **agent performance**, not measured human difficulty. See [Level Generation](docs/level_generation.md).

## Gameplay Analytics

**2,400 gameplay episodes** support paired statistical comparisons, win prediction, and spatial feature analysis.

| Model | Accuracy | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: |
| Historical baseline | **0.608** | 0.625 | 0.582 |
| Logistic Regression | 0.604 | **0.647** | **0.623** |
| Random Forest | 0.558 | 0.606 | 0.605 |

**Finding:** Logistic Regression achieved the best ROC-AUC and PR-AUC in this comparison. Spatial features modestly improved ranking metrics but not probability calibration.

<p align="center"><img src="analysis/figures/win_rate_comparison.png" alt="Win rates by level and agent" width="80%"></p>

<p align="center"><img src="analysis/figures/stage2_board_difficulty_distribution.png" alt="Board difficulty distribution" width="80%"></p>

[Full Results & Analysis](docs/RESULTS.md)

## Getting Started

**Requirements:** Python 3.11 and pip.

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl,ui,analysis,dev]"
```

### Run

```bash
# Play the game
python -m match3_ai.ui.pygame_view --levels

# Validate the project
python -m pytest -q
ruff check .
ruff format --check .
```

### Explore Experiments

```bash
python -m match3_ai.training.dqn --help
python -m match3_ai.training.ppo --help
python scripts/benchmark_mcts_baselines.py --help
python scripts/benchmark_neural_mcts.py --help
python scripts/benchmark_targeted_generation.py --help
```

## Project Structure

```text
match3-ai-lab/
├── src/match3_ai/
│   ├── agents/         # AI agents
│   ├── games/match3/   # Game engine
│   ├── training/       # Training pipelines
│   ├── levels/         # Level generation
│   ├── evaluation/     # Benchmarks
│   └── ui/             # Pygame interface
├── analysis/           # Gameplay analytics
├── experiments/        # Experiment fixtures
├── scripts/            # Evaluation tools
├── tests/              # Automated tests
└── docs/               # Documentation and evidence
```

## Documentation

| Document | Description |
| :--- | :--- |
| [AI Algorithms](docs/algorithms.md) | Agent and search methods |
| [Level Generation](docs/level_generation.md) | Generation and calibration |
| [Results & Analysis](docs/RESULTS.md) | Methods, results, limitations |
| [Reproducibility](docs/reproducibility.md) | Setup and evaluation commands |
| [Benchmark Evidence](docs/benchmarks/README.md) | Published CSV/JSON records |

## Tech Stack

**Python · PyTorch · Gymnasium · NumPy · pandas · scikit-learn · SciPy · Pygame · pytest**
