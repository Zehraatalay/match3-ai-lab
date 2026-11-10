Match3-AI-Lab

Building intelligent agents for Match-3 games

Match3-AI-Lab is a Python-based environment for building, training, and evaluating AI agents in Match-3 games.

The project combines reinforcement learning, Monte Carlo Tree Search, procedural level generation, and gameplay analytics to explore how AI can support automated playtesting, difficulty evaluation, and game balancing.

Beyond agent development, it includes a fully playable Match-3 game, configurable levels, reproducible evaluation pipelines, and data-driven gameplay analysis.

⸻

🎮 Gameplay Showcase

The game features different level configurations, objectives, blockers, and special tile mechanics.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/level4.png" alt="Frozen Garden gameplay" width="100%"/>
      <br/>
      <b>Frozen Garden</b>
    </td>
    <td align="center" width="50%">
      <img src="docs/images/level6.png" alt="Blocker Mayhem gameplay" width="100%"/>
      <br/>
      <b>Blocker Mayhem</b>
    </td>
  </tr>
</table>

Players can interact with the game through a Pygame interface, while AI agents use the same underlying game environment for automated gameplay and evaluation.

⸻

✨ Key Features

Component	Capabilities
Match-3 Engine	Match detection, cascades, gravity, refill, special tiles, blockers, and objectives
AI Agents	Random, Greedy, DQN/DDQN, PPO, Classic MCTS, and Neural MCTS
Reinforcement Learning	Gymnasium-compatible environment, legal-action masking, training and evaluation
Procedural Generation	Configurable level generation, validation, and difficulty calibration
Automated Playtesting	Paired evaluations, win-rate comparisons, and confidence intervals
Gameplay Analytics	Statistical testing, win prediction, and spatial feature engineering
Interactive UI	Human-playable game with Pygame

The game engine, agents, training pipelines, and visualization are implemented as separate components to support testing and experimentation.

⸻

🤖 AI Agents

The project implements multiple approaches to solving Match-3 levels, ranging from simple heuristics to neural-guided search.

Agent	Method	Description
Random	Baseline	Selects a random legal action
Greedy	Heuristic	Prioritizes immediate match-based rewards
DQN / DDQN	Value-based RL	Learns action values through deep Q-learning
PPO	Policy-gradient RL	Learns a masked stochastic policy using actor-critic optimization
Classic MCTS	Tree Search	Uses stochastic open-loop UCT search and simulated rollouts
Neural MCTS	Neural-Guided Search	Combines PUCT search with a CNN policy-value network

All agents interact with the same game rules and legal-action interface.

Technical details: AI Algorithms

⸻

📊 Benchmark Results

The project evaluates agents using controlled game configurations, explicit random seeds, and gameplay metrics.

1. Reinforcement Learning

Random, Greedy, DDQN, and PPO were evaluated across three levels, using 100 episodes per agent per level.

Level	Random	Greedy	DDQN	PPO
Double Trouble	62%	64%	77%	58%
Wooden Trouble	43%	32%	63%	51%
Frozen Garden	48%	54%	76%	66%
Macro Average	51.0%	50.0%	72.0%	58.3%

Key finding: DDQN achieved the highest observed win rate on all three evaluated levels.

These results reflect the specific training and evaluation configurations rather than general performance guarantees.

2. Classic Monte Carlo Tree Search

Classic MCTS uses stochastic simulations to evaluate possible moves.

The main comparison used 100 paired boards per level and 30 simulations per decision.

Level	Random	Greedy	MCTS (RT)	MCTS (GO)
Double Trouble V2	52%	55%	91%	93%
Wooden Trouble V2	41%	37%	95%	93%
Frozen Garden V2	52%	64%	94%	95%

RT: Random rollout + terminal value. GO: Greedy rollout + outcome value.

Key finding: Classic MCTS substantially outperformed Random and Greedy in these development benchmarks.

The evaluation also includes a four-configuration ablation study examining rollout policies and value functions.

Classic MCTS benchmark data · Ablation results

3. Neural Monte Carlo Tree Search

An AlphaZero-inspired agent combines PUCT search with a CNN policy-value network.

The evaluation used 20 paired boards per configuration and 8 simulations per decision.

Agent	Medium	Hard
Random	0%	0%
Greedy	30%	0%
Classic MCTS	70%	30%
Neural MCTS (Untrained)	10%	0%
Neural MCTS (Trained)	30%	0%

Key finding: The trained Neural MCTS improved over its untrained initialization on Medium, but did not outperform Classic MCTS.

The experiments revealed limitations in training diversity and outcome prediction, highlighting the importance of representative training environments.

Neural MCTS evaluation findings

Evaluation note: Reinforcement Learning, Classic MCTS, and Neural MCTS results come from separate experimental protocols. Their win rates should not be interpreted as a single cross-study ranking.

For detailed methodology, results, and limitations, see Results & Analysis.

⸻

🧩 Procedural Level Generation

Match3-AI-Lab includes a procedural generation pipeline for creating and evaluating Match-3 levels.

The pipeline supports:

1. Level Generation: Generate candidate boards and configurations using controlled random seeds.
2. Validation: Check structural constraints and gameplay compatibility.
3. Difficulty Screening: Evaluate candidate levels using AI-based playtesting.
4. Difficulty Calibration: Estimate win rates and confidence intervals.
5. Level Selection: Retain configurations that satisfy the defined difficulty criteria.

Difficulty Categories

Difficulty is estimated using Classic MCTS performance under a fixed evaluation protocol.

Difficulty	Target Win Rate
Easy	75–100%
Medium	40–75%
Hard	10–40%
Very Hard	0–10%

These categories represent agent-estimated difficulty, not direct measurements of human player experience.

Learn more: Procedural Level Generation

⸻

📈 Gameplay Analytics

The project also explores how gameplay data can support difficulty analysis and predictive modeling.

Gameplay Dataset

A dataset of 2,400 gameplay episodes was generated using Random and Greedy agents.

The analysis includes:

* Gameplay performance and win-rate comparisons
* Paired statistical testing using McNemar’s test
* Multiple-comparison correction
* Win prediction with Logistic Regression and Random Forest
* Spatial board feature engineering
* Grouped cross-validation

Win Prediction

Model	Accuracy	ROC-AUC	PR-AUC
Historical Baseline	0.608	0.625	0.582
Logistic Regression	0.604	0.647	0.623
Random Forest	0.558	0.606	0.605

Logistic Regression achieved the highest ROC-AUC and PR-AUC in this experiment.

Gameplay Visualizations

<p align="center">
  <img src="analysis/figures/win_rate_comparison.png" alt="Agent win-rate comparison" width="80%"/>
</p>
<p align="center">
  <img src="analysis/figures/stage2_board_difficulty_distribution.png" alt="Board difficulty distribution" width="80%"/>
</p>

Spatial features provided modest improvements in win-prediction ranking metrics, although probability calibration did not improve.

Full analysis: Results & Analysis

⸻

🚀 Getting Started

Requirements

* Python 3.11
* pip
* Git

Installation

git clone <https://github.com/Zehraatalay/match3-ai-lab>
cd match3-ai-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl,ui,analysis,dev]"

Play the Game

Launch the Pygame interface:

python -m match3_ai.ui.pygame_view --levels

Run Tests

python -m pytest -q
ruff check .
ruff format --check .

Explore Training and Benchmarks

# Reinforcement learning
python -m match3_ai.training.dqn --help
python -m match3_ai.training.ppo --help
# Search-based agents
python scripts/benchmark_mcts_baselines.py --help
python scripts/benchmark_neural_mcts.py --help
# Procedural generation
python scripts/benchmark_targeted_generation.py --help

See Reproducibility for evaluation protocols and checkpoint requirements.

⸻

📁 Project Structure

match3-ai-lab/
├── src/match3_ai/
│   ├── agents/          # AI agents and search algorithms
│   ├── games/match3/    # Game engine and environments
│   ├── training/        # RL and neural training pipelines
│   ├── levels/          # Level configuration and generation
│   ├── evaluation/      # Agent evaluation and benchmarks
│   └── ui/              # Pygame interface
│
├── analysis/            # Gameplay data science
├── experiments/         # Evaluation configurations and fixtures
├── scripts/             # Benchmark and experiment utilities
├── tests/               # Automated tests
└── docs/                # Documentation and published evidence

⸻

📚 Documentation

Document	Contents
AI Algorithms	Agent implementations and algorithmic approaches
Level Generation	Procedural generation and difficulty calibration
Results & Analysis	Evaluation methods, benchmark results, and limitations
Reproducibility	Installation, experiments, and checkpoint requirements
Benchmark Evidence	Published CSV/JSON evaluation records

⸻

Tech Stack

Python · PyTorch · Gymnasium · NumPy · pandas · scikit-learn · SciPy · Pygame · pytest

⸻

Match3-AI-Lab explores how reinforcement learning, search algorithms, and gameplay analytics can contribute to automated game testing and difficulty evaluation.