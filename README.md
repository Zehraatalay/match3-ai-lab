# Match3-AI-Lab

A Python Match-3 environment for gameplay AI, reinforcement learning, automated
playtesting, procedural level generation, and reproducible agent comparisons.
The project includes a playable game, learned policies, simulation-based search,
and gameplay data analysis. Difficulty estimates describe agent performance;
they are not measurements of human-perceived difficulty.

## Features

- Match detection, cascades, gravity/refill, special tiles, and blocker damage
- Crate, ice, and stone blockers; masked boards and finite-move objectives
- Six original levels and three calibrated V2 variants
- Gymnasium observations with legal-action masks
- Random, Greedy, DQN/DDQN, PPO, classic MCTS, and Neural MCTS
- Seeded level generation, validation, screening, and difficulty confirmation
- Paired evaluation, Wilson confidence intervals, episode records, and provenance
- Pygame UI and a numbered Data Science analysis workflow

| Frozen Garden | Blocker Mayhem |
|---|---|
| ![Frozen Garden](docs/images/level4.png) | ![Blocker Mayhem](docs/images/level6.png) |

## Architecture and agents

Game rules stay independent of agents, training, evaluation, and rendering.
Agents share `select_action(observation)`; search agents additionally bind to an
independent simulator of the current environment.

| Agent | Implemented approach |
|---|---|
| Random | Uniform legal-action sampling |
| Greedy | Largest immediate line-match count; ignores blocker/objective effects |
| DQN / DDQN | Masked action values; MLP, CNN, and dueling architectures |
| PPO | Masked MLP actor-critic with deterministic evaluation |
| Classic MCTS | Fresh open-loop UCT tree; sampled stochastic futures and full rollouts |
| Neural MCTS | Policy-value CNN, sampled successor states, PUCT, neural leaf values |

Neural MCTS is **AlphaZero-inspired**, adapted to a single-player stochastic
environment. It uses the exact simulator, independent simulation RNG streams,
and backups without alternating signs. It has no learned dynamics model,
parallel search, or optimality guarantee. Its current encoder supports boards up
to 9×8, 3–5 colors, two objectives, and move limits up to 32. Existing DQN/PPO
encoder version 1 retains its stricter 15-move limit.

See [classic MCTS](docs/stochastic_mcts.md) and
[architecture and reproducibility](docs/reproducibility.md).

## Procedural level generation

The generator samples reproducible level configurations with configurable board
size, color count, move limits, and crate counts. Generated candidates are
validated, screened, and confirmed on separate seed sets. The current generator
uses single-layer crates; it does not generate every mechanic supported by the
engine.

Difficulty bands use classic MCTS win rates: Easy 75–100%, Medium 40–75%, Hard
10–40%, and Very Hard 0–10%. A confirmed match requires its Wilson interval to
fit inside the requested band; borderline cases remain uncertain. Labels depend
on the reference agent, simulation budget, and seed distribution. They do not
establish solvability or human difficulty. Screening and confirmation can be
computationally expensive.

## Experimental results

### Neural MCTS evaluation (AZ-4)

Twenty paired development boards per Medium/Hard configuration, with **eight
simulations per decision** for all search agents:

| Agent | Medium | Compatible Hard |
|---|---:|---:|
| Random | 0% | 0% |
| Greedy | 30% | 0% |
| Classic MCTS (greedy rollout / outcome value) | **70%** | **30%** |
| Untrained Neural MCTS | 10% | 0% |
| Trained Neural MCTS (round 669) | 30% | 0% |

The final checkpoint's Wilson 95% intervals are 14.5–51.9% on Medium and
0–16.1% on compatible Hard. Neural MCTS did not outperform classic MCTS in this
sample. Training improved the observed Medium count over initialization, but
checkpoint progression was not monotonic and Hard performance did not improve.

The accepted Hard artifact uses six colors and is incompatible with the neural
encoder. Evaluation explicitly used a previously confirmed **five-color Hard
candidate**, preserving both original artifacts. Equal simulation counts do not
equalize computation: classic MCTS rolls out to termination, while neural MCTS
evaluates new leaves.

The training environment was too easy: a blocker-free task collecting 30 tiles
of any color in 10 moves produced 5,352 wins and only positive outcome targets.
The final critic predicted exactly +1 on all 50 inspected evaluation boards.
Random also won all five fresh training-control boards. Decreasing policy loss
and near-zero value loss therefore did not establish general gameplay quality.

[Evaluation report](docs/benchmarks/az4/evaluation_findings.md) ·
[full metrics and intervals](docs/benchmarks/az4/summary.csv) ·
[checkpoint progression](docs/benchmarks/az4/checkpoint_progression.csv)

### Earlier RL and Data Science studies

A separate 100-episode-per-level comparison reported DDQN macro win rate 72.0%,
versus Random 51.0%, Greedy 50.0%, and PPO 58.3%. Those checkpoint-based results
use a different protocol and must not be combined with AZ-4 as one head-to-head
study. Spatial features increased mean grouped-CV ROC-AUC from 0.577 to 0.591,
without improving the probability-calibration metrics.

See [Results & Analysis](docs/RESULTS.md) for methods and limitations. The trained
weights and some intermediate analysis data are not distributed in this source
repository. Archived results are evidence, not promises of exact reproduction
from a fresh clone.

## Installation and usage

Python **3.11** is recommended. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[rl,ui,analysis,dev]"
python -m pip check
```

On Windows, activate with `.venv\Scripts\activate`. The `rl` extra includes
PyTorch; accelerator-specific installation depends on the platform. The core
engine can instead be installed with `python -m pip install -e .`.

```bash
# Run tests and static checks
python -m pytest -q
ruff check .
ruff format --check .

# Launch the game or inspect command-line options
python -m match3_ai.ui.pygame_view --levels
python -m match3_ai.training.dqn --help
python -m match3_ai.training.ppo --help
python scripts/benchmark_targeted_generation.py --help
python scripts/benchmark_neural_mcts.py --help
```

A small new classic-MCTS development comparison:

```bash
python scripts/benchmark_mcts_baselines.py \
  --episodes 1 --simulations 2 \
  --env-seed-start 10002000 --agent-seed-start 20020000 \
  --output-dir results/manual_mcts_smoke
```

This runs Random, Greedy, and the default two classic-MCTS configurations on
three V2 levels. Use a new output directory. Even small search experiments take
longer than direct policy inference.

For Neural MCTS training, checkpoint-based evaluation, profile fixtures, and
exact AZ-4 commands, see [reproducibility](docs/reproducibility.md). The six
selected checkpoint files must be supplied locally to repeat the trained
comparison; they are intentionally excluded from Git. No experiment runs merely
by importing a training module.

## Project structure

```text
src/match3_ai/
  games/match3/   engine, board mechanics, Gymnasium environments
  agents/         direct policies, neural networks, search and simulator
  training/       DQN/PPO and search-guided policy-value training
  levels/         configurations, generation, screening, calibration, export
  evaluation/     episode runner, analytics, comparisons and benchmarks
  ui/             Pygame interface
scripts/          experiment command-line tools
analysis/         gameplay data analysis and spatial-feature studies
experiments/fixtures/  portable evaluation configurations
results/          ignored local runs and checkpoints, retained on disk
docs/             methodology, figures, and small published benchmark snapshots
tests/            unit, integration, reproducibility, and regression tests
```

## Findings and limits

- Task difficulty and target diversity matter more than training win rate alone
- The neural critic learned an uninformative constant on a single easy task
- Search-based difficulty is conditional on the agent and compute budget
- Small development samples and one configuration per profile limit generalization
- Neural training currently uses one fixed level, CPU Adam, bounded replay, and
  completed-round checkpoint/resume; it has no configurable search-depth cap
- Original checkpoints and full local training outputs stay outside the public
  source tree; small benchmark evidence includes explicit provenance caveats

Python · NumPy · PyTorch · Gymnasium · pandas · SciPy · scikit-learn · Pygame · pytest
