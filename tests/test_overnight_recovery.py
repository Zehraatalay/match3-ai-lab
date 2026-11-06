import json
from dataclasses import asdict

import pytest

from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.training.overnight import baseline
from match3_ai.training.overnight_recovery import recover_rows


def episode(seed=1, experiment_id="synthetic", episode_id="synthetic:0"):
    return EpisodeResult(
        seed=seed,
        total_reward=1.0,
        steps=0,
        valid_actions=0,
        invalid_actions=0,
        total_cleared=0,
        total_cascades=0,
        reshuffles=0,
        completed=True,
        experiment_id=experiment_id,
        episode_id=episode_id,
        agent_name="synthetic",
        outcome="WON",
        moves_used=0,
        objective_completion_ratio=1.0,
        blocker_hp_remaining=0,
    )


def test_episode_recovery_requires_complete_unique_records_and_keeps_fields(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [
        {
            "event": "evaluation_episode",
            "utc": "2026-01-01",
            **asdict(episode(seed=i, episode_id=str(i))),
        }
        for i in (1, 2)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    recovered = recover_rows(path, [1, 2])
    assert len(recovered) == 2 and recovered[0]["outcome"] == "WON"
    assert "event" not in recovered[0] and "utc" not in recovered[0]
    with pytest.raises(ValueError, match="coverage"):
        recover_rows(path, [1, 3])
    rows[0]["invalid_actions"] = 1
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="invalid"):
        recover_rows(path, [1, 2])


def test_baseline_export_uses_matching_metadata_without_executing_environment(
    tmp_path, monkeypatch
):
    from match3_ai.training import overnight

    class FakeEnvironment:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def fake_episode(
        env, policy, *, seed, collector, experiment_id, episode_id, **kwargs
    ):
        result = episode(seed, experiment_id, episode_id)
        collector.record_episode(result)
        return result

    monkeypatch.setattr(overnight, "Match3LevelEnv", FakeEnvironment)
    monkeypatch.setattr(overnight, "run_episode", fake_episode)
    result = baseline(
        {"id": "job-namespace", "kind": "random"},
        tmp_path / "baseline",
        {"validation_seeds": [1]},
        lambda *_: None,
    )
    assert result["experiment_id"] == "job-namespace"
    assert (tmp_path / "baseline/evaluation/episodes.csv").exists()
