"""Frozen-policy Stone Breaker diagnostic protocol; execution is explicit."""

import argparse
import json
from pathlib import Path

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.cross_level_benchmark import SELECTION, validate_seeds
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.fresh_evaluation import policy_state
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.gameplay_events import GameplayCollector, InstrumentedLevelEnv
from match3_ai.evaluation.runner import run_episode

SEEDS = tuple(range(730000, 730050))


def run_diagnostics(results_root, output):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(results_root)
    validate_seeds(SEEDS, root, expected_count=50)
    policies, specs = {}, []
    for seed, step, directory in SELECTION:
        path = (
            root / "sprint5_20_rl_optimization" / directory / f"policy_step_{step}.pt"
        )
        policy, payload = DQNAgent.load_checkpoint(path)
        config = payload["training_config"]
        if (
            config["seed"] != seed
            or payload["training_step"] != step
            or config["level_id"] != "blocker-mayhem"
            or policy.architecture != "dueling"
            or policy.algorithm != "double"
        ):
            raise ValueError(f"Checkpoint selection mismatch: {path}")
        policy.online.eval()
        policy.online.requires_grad_(False)
        name = f"DDQN_{seed}_{step}"
        policies[name] = policy
        specs.append(
            {
                "name": name,
                "checkpoint": str(path.resolve()),
                "sha256": sha256(path),
                "training_config": config,
            }
        )
    level_path = Path(__file__).resolve().parents[1] / "levels/data/stone-breaker.json"
    manifest = {
        "schema_version": 1,
        "source_sha256": sha256(__file__),
        "telemetry_source_sha256": sha256(
            Path(__file__).with_name("gameplay_events.py")
        ),
        "agents": [*policies, "Greedy", "Random"],
        "level_id": "stone-breaker",
        "environment_seeds": SEEDS,
        "random_agent_seeds": list(range(740000, 740050)),
        "policies": specs,
        "level_sha256": sha256(level_path),
        "planned_episodes": 250,
        "max_cascades": 200,
        "max_initialization_attempts": 1000,
        "development_only": True,
        "final_test_used": False,
        "cascade_round": "0: direct special phase; 1+: MATCH resolution including initial match",
        "interrupted_resolution": "move_start without move_end is incomplete; engine events are emitted after a successful step",
        "damage_attribution": "Engine source categories; combinations may have multiple sources, not unique special instance IDs",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    experiment_id = sha256(output / "manifest.json")[:24]
    before = {name: policy_state(policy) for name, policy in policies.items()}
    records, identities = [], {}
    status = "interrupted"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (output / "events.jsonl").open("x") as handle:
            for index, seed in enumerate(SEEDS):
                agents = {
                    **policies,
                    "Greedy": GreedyAgent(),
                    "Random": RandomAgent(740000 + index),
                }
                for name, policy in agents.items():
                    collector = GameplayCollector(
                        handle, experiment_id, f"stone-breaker:{name}:{seed}"
                    )
                    env = InstrumentedLevelEnv(
                        "stone-breaker", collector, initial_identities=identities
                    )
                    try:
                        record = run_episode(
                            env,
                            policy,
                            seed=seed,
                            max_steps=15,
                            experiment_id=experiment_id,
                            episode_id=collector.episode_id,
                            agent_name=name,
                            collector=collector,
                            agent_seed=740000 + index if name == "Random" else None,
                        )
                    finally:
                        env.close()
                    records.append(record)
                    if env.pairing_error:
                        raise RuntimeError("Initial boards differ across agents")
                    collector.validate()
                    if name in policies and policy_state(policy) != before[name]:
                        raise RuntimeError("Frozen policy state changed")
            for spec in specs:
                if sha256(spec["checkpoint"]) != spec["sha256"]:
                    raise RuntimeError("Checkpoint changed")
        status = "complete"
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_records": len(records),
                },
                indent=2,
            )
        )
        raise
    finally:
        try:
            summary = summarize_level_episodes(records)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            lines = [
                "# Stone Breaker development diagnostics",
                "",
                f"Status: {status}",
                "",
                "Frozen zero-shot policies; 50 boards per agent; failures are not replaced",
                "",
                "| Agent | n/50 | Win rate |",
                "|---|---:|---:|",
            ]
            lines.extend(
                f"| {row['agent_name']} | {row['attempted']} | {row['win_rate']:.1%} |"
                for row in summary
            )
            (output / "report.md").write_text("\n".join(lines) + "\n")
        finally:
            torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostics(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
