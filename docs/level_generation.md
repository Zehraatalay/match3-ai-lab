# Level Generation and Difficulty

Match3-AI-Lab supports predefined levels and procedural level generation.

## Level Configurations

The game includes six original levels and three calibrated V2 variants.

Levels define board dimensions, tile colors, move limits, objectives, blockers, and other gameplay constraints.

Supported mechanics include crates, ice, stone blockers, special tiles, cascades, and masked boards.

The current procedural generator focuses on configurable boards with single-layer crates; it does not generate every mechanic supported by the engine.

## Procedural Generation

The generation pipeline:

1. Samples level configurations using deterministic seeds.
2. Validates configurations and gameplay constraints.
3. Screens candidates using a reference AI agent.
4. Confirms difficulty on separate evaluation seeds.

This separates candidate selection from difficulty confirmation.

## Difficulty Calibration

Difficulty is estimated using Classic MCTS win rates.

| Difficulty | Reference-agent win rate |
|---|---:|
| Easy | 75–100% |
| Medium | 40–75% |
| Hard | 10–40% |
| Very Hard | 0–10% |

Wilson confidence intervals are used to assess whether a candidate reliably falls within the requested difficulty band.

Candidates whose intervals overlap difficulty boundaries remain uncertain.

These categories describe performance under a specific agent, search budget, and seed distribution. They are not direct measurements of human difficulty.

## Calibrated Levels

| Level | Random | Greedy | Status |
|---|---:|---:|---|
| Wooden Trouble V2 | 38.5% | 31.0% | Did not meet the Greedy target |
| Double Trouble V2 | 56.5% | 56.0% | Passed |
| Frozen Garden V2 | 50.5% | 59.5% | Passed |

Each agent was evaluated on 200 development boards per configuration.

The original levels remain available; V2 configurations are registered separately.

See [Results](RESULTS.md) for agent comparisons.
