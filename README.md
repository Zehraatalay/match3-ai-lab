# Match-3 AI Lab

I built a playable Match-3 game in Python to experiment with reinforcement learning and gameplay data analysis.

The idea started with a simple question: **Can an AI agent learn to play Match-3 better than a rule-based strategy?**

I developed the game engine, introduced different level mechanics, trained AI agents, and analyzed gameplay data to understand what makes certain levels more challenging.

## The Game

The game includes six playable levels with different board layouts, objectives, and obstacles.

Features include:
- Match detection, cascades, and special tiles
- Wooden, frozen, and stone blockers
- Different board shapes and level objectives
- Move limits and win/loss conditions
- A playable Pygame interface

| Frozen Garden | Blocker Mayhem |
|---|---|
| ![Frozen Garden](docs/images/level4.png) | ![Blocker Mayhem](docs/images/level6.png) |

## AI Agents

I implemented and evaluated four agents:

- **Random:** Selects a random valid move.
- **Greedy:** Uses a rule-based strategy to select moves.
- **DDQN:** Learns action values through reinforcement learning.
- **PPO:** Learns a policy using an actor-critic approach.

### Evaluation Results

Each agent was evaluated on 100 paired episodes per level.

| Level | Random | Greedy | DDQN | PPO |
|---|---:|---:|---:|---:|
| Double Trouble | 62% | 64% | **77%** | 58% |
| Wooden Trouble | 43% | 32% | **63%** | 51% |
| Frozen Garden | 48% | 54% | **76%** | 66% |
| **Macro average** | **51.0%** | **50.0%** | **72.0%** | **58.3%** |

DDQN achieved the highest observed win rate across all three evaluated levels.

These results describe the selected trained policies on a limited evaluation set. They do not establish statistical superiority across other levels or training runs.

## Gameplay Data Analysis

I also explored gameplay data to understand how board characteristics and agent behavior relate to winning.

The analysis includes:
- Exploratory data analysis
- Paired statistical comparisons
- Feature engineering from board states
- Win prediction using Logistic Regression and Random Forest
- Spatial feature experiments and grouped cross-validation

Adding spatial features improved mean cross-validated ROC-AUC from **0.577 to 0.591**, although probability calibration metrics did not improve.

For the methodology, detailed results, and limitations, see the [Results & Analysis Report](docs/RESULTS.md).

## Getting Started

Requires Python 3.11 or later.

```bash
git clone https://github.com/Zehraatalay/match3-ai-lab.git
cd match3-ai-lab
python -m pip install -e ".[ui,analysis,dev]"
```

Launch the game:

```bash
python -m match3_ai.ui.pygame_view --levels
```

Run the tests:

```bash
python -m pytest -q
```

The optional RL dependencies are not included in the installation command above.

## Tech Stack

Python · NumPy · PyTorch · Gymnasium · Pygame · Matplotlib · pytest

The game engine, AI agents, evaluation tools, and UI are kept separate so that gameplay mechanics can be tested independently of rendering and training.
