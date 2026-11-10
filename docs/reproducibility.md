# Reproducibility

Match3-AI-Lab supports seeded experiments, paired agent evaluations, and reproducible benchmark configurations.

## Installation

Python 3.11 is recommended.

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[rl,ui,analysis,dev]"

On Windows, activate the environment using `.venv\Scripts\activate`.

## Validation

    python -m pytest -q
    ruff check .
    ruff format --check .

## Running Experiments

Available entry points:

- DQN: `python -m match3_ai.training.dqn --help`
- PPO: `python -m match3_ai.training.ppo --help`
- Classic MCTS: `python scripts/benchmark_mcts_baselines.py --help`
- Neural MCTS: `python scripts/benchmark_neural_mcts.py --help`
- Level generation: `python scripts/benchmark_targeted_generation.py --help`

Example Classic MCTS evaluation:

    python scripts/benchmark_mcts_baselines.py --episodes 1 --simulations 2 --output-dir results/manual_mcts_smoke

Use a new output directory for each experiment.

## Evaluation Protocol

Agent comparisons use explicit environment seeds and independent agent random streams. Paired evaluations share board configurations to reduce variation between agents.

Reported metrics include win rates, Wilson confidence intervals, and gameplay statistics.

Published benchmark records are available in [benchmarks](benchmarks/README.md).

## Checkpoints and Limitations

Large experiment outputs and trained model checkpoints are excluded from GitHub.

New experiments can be run from a fresh installation, but reproducing historical trained-policy results requires the original model weights.

Neural MCTS evaluations additionally require compatible level configurations and checkpoint files.

Results from different evaluation protocols should not be treated as directly comparable.
