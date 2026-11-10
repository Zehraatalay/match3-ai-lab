Match3-AI-Lab

A Python-based Match-3 game environment for developing, training, and evaluating AI agents. The project combines reinforcement learning, Monte Carlo Tree Search, procedural level generation, and gameplay analytics to explore automated playtesting and game difficulty.

It includes a playable game, multiple AI agents, reproducible experiments, and data-driven evaluation tools.

Features

* Game Engine: Match detection, cascades, gravity, special tiles, blockers, objectives, and configurable levels.
* AI Agents: Random, Greedy, DQN/DDQN, PPO, Classic MCTS, and Neural MCTS.
* Reinforcement Learning: Gymnasium-compatible environments with legal-action masking.
* Procedural Generation: Seeded level generation, validation, and difficulty calibration.
* Automated Playtesting: Agent comparisons using win rates, confidence intervals, and gameplay metrics.
* Gameplay Analytics: Statistical analysis, win prediction, and spatial feature engineering.
* Interactive Gameplay: Playable Pygame interface.

Gameplay

Frozen Garden	Blocker Mayhem


AI Agents

Agent	Approach
Random	Random legal-action selection
Greedy	Immediate match-based action selection
DQN / DDQN	Deep Q-learning with action masking
PPO	Masked actor-critic reinforcement learning
Classic MCTS	Stochastic open-loop UCT search
Neural MCTS	AlphaZero-inspired PUCT search with a policy-value CNN

The agents share a common game environment, enabling controlled evaluation across different strategies.

For implementation details, see AI Algorithms.

Procedural Level Generation

The project generates configurable Match-3 levels and estimates their difficulty through AI-based playtesting.

Generated candidates are validated, screened, and evaluated using independent random seeds. Difficulty categories are based on Classic MCTS win rates and Wilson confidence intervals.

These estimates describe agent performance under specific evaluation settings, rather than human-perceived difficulty.

See Level Generation.

Experimental Results

Reinforcement Learning

DDQN achieved the highest observed macro-average win rate across three evaluated levels, with 100 episodes per agent per level.

Agent	Macro Win Rate
Random	51.0%
Greedy	50.0%
DDQN	72.0%
PPO	58.3%

Monte Carlo Tree Search

Classic MCTS achieved high win rates on three calibrated levels using 30 simulations per decision. The greedy-rollout/outcome-value configuration achieved 93–95% win rates across these levels.

A separate Neural MCTS evaluation compared search-based agents on Medium and Hard configurations using eight simulations per decision.

Agent	Medium	Hard
Random	0%	0%
Greedy	30%	0%
Classic MCTS	70%	30%
Trained Neural MCTS	30%	0%

Neural MCTS did not outperform Classic MCTS in this evaluation. Its training results highlighted the importance of task difficulty and diverse training environments.

These experiments use different evaluation protocols and should not be interpreted as a single head-to-head comparison.

For methods, additional metrics, and limitations, see Results & Analysis.

Installation

Python 3.11 is recommended.

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl,ui,analysis,dev]"

Usage

Launch the game:

python -m match3_ai.ui.pygame_view --levels

Run tests:

python -m pytest -q
ruff check .
ruff format --check .

Explore training and evaluation tools:

python -m match3_ai.training.dqn --help
python -m match3_ai.training.ppo --help
python scripts/benchmark_mcts_baselines.py --help
python scripts/benchmark_neural_mcts.py --help
python scripts/benchmark_targeted_generation.py --help

For experiment configurations and checkpoint requirements, see Reproducibility.

Project Structure

src/match3_ai/
├── games/match3/    # Game engine and environments
├── agents/          # AI agents and search algorithms
├── training/        # Reinforcement learning and neural training
├── levels/          # Level configurations and generation
├── evaluation/      # Agent evaluation and benchmarking
└── ui/              # Pygame interface
scripts/             # Experiment utilities
analysis/            # Gameplay data analysis
experiments/         # Evaluation fixtures
docs/                # Technical documentation and results
tests/               # Automated tests

Documentation

* AI Algorithms — Agent architectures and search methods.
* Level Generation — Procedural generation and difficulty calibration.
* Results & Analysis — Experimental methods, results, and limitations.
* Reproducibility — Setup and experiment execution.

Tech Stack: Python · NumPy · PyTorch · Gymnasium · pandas · SciPy · scikit-learn · Pygame · pytest