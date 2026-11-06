import json

import numpy as np
import pytest

pytest.importorskip("torch")

from match3_ai.training.ppo_stability import (
    PERFORMANCE,
    analyze,
    distribution,
    figures,
    paired_intervals,
    reconstruct_targets,
    trailing_mean,
    update_summary,
    verified_semantics,
    verify_hashes,
)
from match3_ai.training.ppo_update import rollout_diagnostics


def test_clipping_uses_preclip_norm_and_entropy_sign():
    import torch

    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    recorded = torch.nn.utils.clip_grad_norm_([parameter], 0.5)
    assert float(recorded) == pytest.approx(5.0)
    assert float(parameter.grad.norm()) == pytest.approx(0.5)
    rows = [
        dict.fromkeys(
            (
                "approximate_kl",
                "clip_fraction",
                "entropy",
                "policy_loss",
                "value_loss",
                "gradient_norm",
                "learning_rate",
                "ratio_min",
                "ratio_max",
            ),
            "0",
        )
        for _ in range(3)
    ]
    for row, norm in zip(rows, (0.4, 0.5, 5.0)):
        row.update(gradient_norm=str(norm), entropy="2", minibatch_size="64")
    result = update_summary(rows, threshold=0.5, coefficient=0.01)
    assert result["gradient_clipped_count"] == 1
    assert result["gradient_clipping_frequency"] == pytest.approx(1 / 3)
    assert result["entropy_bonus"]["mean"] == pytest.approx(0.02)
    assert result["sampled_presentations"] == 192
    semantics = verified_semantics(
        {"configurations": {"x": {"gradient_clip": 0.5, "entropy_coefficient": 0.01}}}
    )
    assert semantics["constant_entropy_coefficient"]
    with pytest.raises(ValueError, match="coefficients"):
        verified_semantics(
            {
                "configurations": {
                    "x": {"gradient_clip": 1.0, "entropy_coefficient": 0.01}
                }
            }
        )


def test_reconstructed_targets_preserve_truncation_bootstrap_and_terminal_reset():
    rows = [
        {
            "reward": "0",
            "value": ".2",
            "next_value": ".4",
            "terminated": "False",
            "truncated": "True",
        },
        {
            "reward": "1",
            "value": ".1",
            "next_value": "0",
            "terminated": "True",
            "truncated": "False",
        },
        {
            "reward": "-1",
            "value": "-.1",
            "next_value": "0",
            "terminated": "True",
            "truncated": "False",
        },
    ]
    expected_a = np.array([0.16, 0.9], dtype=np.float32)
    expected_t = (expected_a + np.array([0.2, 0.1])).astype(np.float32)
    d1 = rollout_diagnostics(np.array([0.2, 0.1]), expected_t, expected_a)
    a2 = np.array([-0.9], dtype=np.float32)
    t2 = (a2 + np.array([-0.1])).astype(np.float32)
    d2 = rollout_diagnostics(np.array([-0.1]), t2, a2)
    recorded = [
        {k: "" if v is None else str(v) for k, v in d.items()} for d in (d1, d2)
    ]
    a, t = reconstruct_targets(
        rows, recorded, {"rollout_steps": 2, "gamma": 0.9, "gae_lambda": 0.95}
    )
    np.testing.assert_allclose(a, [0.16, 0.9, -0.9], atol=1e-6)
    np.testing.assert_allclose(t, [0.36, 1.0, -1.0], atol=1e-6)
    recorded[0]["advantage_mean"] = "100"
    with pytest.raises(ValueError, match="differs"):
        reconstruct_targets(
            rows, recorded, {"rollout_steps": 2, "gamma": 0.9, "gae_lambda": 0.95}
        )


def test_exploratory_pairing_common_boards_and_holm_family():
    boards = [11, 12, 13, 14]
    binary = [[0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]]
    checkpoints = [
        {
            b: {
                "outcome": "WON" if wins[i] else "LOST",
                **{k: float(wins[i]) for k in PERFORMANCE},
            }
            for i, b in enumerate(boards)
        }
        for wins in binary
    ]
    results = paired_intervals(
        dict.fromkeys((5918, 5919, 5920), checkpoints), boards, samples=200, seed=1
    )
    assert len(results) == 12
    assert all(
        r["exploratory"] and r["holm_p_exploratory_12"] >= r["mcnemar_p"]
        for r in results
    )
    first = results[0]
    assert first["counts"] == {
        "both_win": 1,
        "later_only_win": 1,
        "earlier_only_win": 1,
        "both_lose": 1,
    }
    assert first["win_difference"] == 0
    assert first["win_difference_ci95"] == results[4]["win_difference_ci95"]
    rng = np.random.default_rng(1)
    indices = rng.integers(4, size=(200, 4))
    explicit = np.array([1.0, 0.0, 0.0, -1.0])[indices].mean(-1)
    np.testing.assert_allclose(
        first["win_difference_ci95"], np.quantile(explicit, [0.025, 0.975])
    )
    checkpoints[0].pop(11)
    with pytest.raises(ValueError, match="coverage"):
        paired_intervals({5918: checkpoints}, boards, samples=10)


def test_preservation_and_exclusive_publication(tmp_path):
    from match3_ai.evaluation.fresh_protocol import sha256

    path = tmp_path / "protected.json"
    path.write_text(json.dumps({"original": True}))
    hashes = {str(path): sha256(path)}
    verify_hashes(hashes)
    path.write_text("changed")
    with pytest.raises(ValueError, match="protected artifacts"):
        verify_hashes(hashes)
    with pytest.raises(FileExistsError):
        analyze(tmp_path / "missing.json", tmp_path / "missing-history.json", tmp_path)


def test_missing_nonfinite_and_smoothing_are_explicit():
    assert distribution([None, ""]) == {"count": 0, "undefined": 2}
    assert distribution([1.0, "", 3.0])["mean"] == 2.0
    with pytest.raises(ValueError, match="finite"):
        distribution([float("nan")])
    np.testing.assert_allclose(
        trailing_mean([1.0, 2.0, 6.0, 4.0], window=2), [1.0, 1.5, 4.0, 5.0]
    )
    with pytest.raises(ValueError):
        trailing_mean([1.0, float("inf")])


def test_diagnostic_figures_export_from_synthetic_records_only(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    original_plot = Axes.plot

    def numeric_plot(self, x, y, *args, **kwargs):
        assert not any(isinstance(value, str) for value in x)
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", numeric_plot)
    run = {
        "checkpoints": [
            {
                "environment_step": str(step),
                "win_rate": 0.5,
                **dict.fromkeys(PERFORMANCE, 1.0),
            }
            for step in (0, 100000)
        ],
        "updates": [
            {
                "environment_step": str(step),
                **dict.fromkeys(
                    (
                        "approximate_kl",
                        "clip_fraction",
                        "entropy",
                        "entropy_bonus",
                        "policy_loss",
                        "gradient_norm",
                        "learning_rate",
                    ),
                    0.01,
                ),
            }
            for step in (256, 512)
        ],
        "rollouts": [
            {
                "environment_step": "256",
                "value_loss": 0.2,
                "explained_variance": None,
                "advantage_std": 0.6,
                "target_mean": 0.0,
            }
        ],
        "intervals": [
            {
                "optimization": {
                    k: {"mean": 0.1}
                    for k in ("entropy", "approximate_kl", "value_loss")
                },
                "paired": {"win_difference": 0.05},
            }
        ],
    }
    paths = figures(dict.fromkeys((5918, 5919, 5920), run), tmp_path / "figures")
    assert len(paths) == 8
    for name in paths:
        content = (tmp_path / "figures" / name).read_bytes()
        assert content.startswith(b"\x89PNG" if name.endswith(".png") else b"%PDF")
