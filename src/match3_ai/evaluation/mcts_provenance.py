"""Audited compatibility boundary for the Sprint 3 and Sprint 4 MCTS searches.

The legacy package hash was verified before edits. The current critical hashes
pin the behavior-preserving implementation checked against recorded reference
traces. Any further critical-source edit requires an explicit compatibility audit.
"""

import hashlib
from pathlib import Path

ALGORITHM_SEMANTICS = "open-loop-uct-v1"
IMPLEMENTATION_VERSION = "step-observation-reuse-v1"
LEGACY_IMPLEMENTATION = "observations-recomputed-v1"
BENCHMARK_SCHEMA_VERSION = 2
LEGACY_PACKAGE_SHA256 = (
    "d8b6d7db7ab7a273cb586a0a46d688f84c951c2965a3049d229df42b2888ff93"
)
LEGACY_CLI_SHA256 = "7fe16a3bdb4f1eba60d518bcae6d0a8537cfa3c1bf5c0f58d7813574a594636e"
VERIFIED_SOURCE_SHA256 = {
    "src/match3_ai/agents/base.py": "02d726a09e10fb99f9bb5f485a689dd1ae7314f42792602e4d1a4e98e0b3c247",
    "src/match3_ai/agents/greedy_agent.py": "457cc61849d9fb970e40e4943ad934fca96d59969c930f71cfc08428cb16a4c0",
    "src/match3_ai/agents/mcts_simulator.py": "3be1602c57ceca8040c28ba70653405517dc25ab615903de328ffd1e98262734",
    "src/match3_ai/agents/mcts_value.py": "11ca252d26f4d285eb7610fcc875e553401d6a40deacc2420ba311432f91ceda",
    "src/match3_ai/agents/objective_rollout.py": "25d9edbfb43a4fd848d89324d8ddb5e7a777fa2ad186e9a9ef9b2ec76df4292e",
    "src/match3_ai/agents/observation_encoder.py": "adb520ea6844a5578f09a3264c8e1ba50cdd58c74019b068079e71e51e6f94be",
    "src/match3_ai/agents/random_agent.py": "1e29a53eb0221e1efb0fc3614f4e061fcb77c8fcefddf4d96dab2804aa86fef3",
    "src/match3_ai/agents/stochastic_mcts_agent.py": "4e06747da39a1f272cf701a9e6cfb044547bf06ad5868cbfdfeec09abb027310",
    "src/match3_ai/core/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/match3_ai/core/protocols.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/match3_ai/core/types.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/match3_ai/evaluation/analytics.py": "4aa9480decb4a47ba4dfd88cf1fe500963d20b1422df03ebffa77e22657622cd",
    "src/match3_ai/evaluation/behavior_data.py": "c510fa42a47cd4fc89099a4a09bbc0a42315d9e9f5f2d3fa730db1544ed4462c",
    "src/match3_ai/evaluation/experiments.py": "f1d4cd81a80559bc48c8ce85fc7ddcbf71904c285ec274f27ce13539f3a49b65",
    "src/match3_ai/evaluation/runner.py": "2d09e78776af520dc2f36c1f5832a5603282361d04dfd94b07243946706fc9d5",
    "src/match3_ai/games/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/match3_ai/games/match3/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/match3_ai/games/match3/animation_events.py": "a7b67c72d24f473615f8fd497f72f97e24979e86f0997606dad97833f4f80ac1",
    "src/match3_ai/games/match3/blockers.py": "879f782f15dff8ca04dacf5c4748fffc2e0865a605111fa01638d814b4e328e1",
    "src/match3_ai/games/match3/board.py": "f5e4d2f73bb3404a30ddb47af22db4c4110f59ace10dc885e0750340176bac55",
    "src/match3_ai/games/match3/board_mechanics.py": "9034fef97b27fd9c63af9a3c2ef13436433e297f4b3e69425efa54d4b4884843",
    "src/match3_ai/games/match3/constants.py": "9d8d891f385697b59a5d4c38c4ae0fe88576d55bbc511261e1b20ddf8e714c2a",
    "src/match3_ai/games/match3/engine.py": "266dc6f4a44b2d468ba2eb5d206fcdedb4bd34890641af1d71eeeee5c0ea9815",
    "src/match3_ai/games/match3/environment.py": "d4e8e45a21d34e2f10c9d75f05111b1740b37f7d2a3d234b6dae19e713681b49",
    "src/match3_ai/games/match3/find_collision_board.py": "ac1ad9aa33f04696225fcf87b6c2b008d8078c391c204cc94d70ce6ee78f9193",
    "src/match3_ai/games/match3/generator.py": "1b2b3441adb69a6b7b217369a4a36b433b2be3518eb6f1ff01978d25c5a55df3",
    "src/match3_ai/games/match3/mechanics.py": "9bf4f2af04bda8a60af92d1ee309235cb06b1861b78a7ef0cd2403b5a75a44e0",
    "src/match3_ai/games/match3/moves.py": "d1a95d4e14c4ed427cd9df667a64832da905e7114fcc11235e585ae6ffdf1e52",
    "src/match3_ai/games/match3/pattern_detection.py": "8ca9677da89d3871ed7baf102d405996f3f0b8210b80f237bbf334d11c41309f",
    "src/match3_ai/games/match3/recovery.py": "f090dc754c97daa2debd38aef00df68451d5bfed82683dce1963250c3a256a4e",
    "src/match3_ai/games/match3/rules.py": "2c894369f378555b8c0f6e419fa3c4d9056f3ddf31f2dd53e385475b032b0478",
    "src/match3_ai/games/match3/special_activation.py": "338e37cca7e5d55c51d67fef5a4301ad8c0841e5bc4f89fcf7be428f3c70a444",
    "src/match3_ai/games/match3/special_resolver.py": "79e423716f26275ea00e141b6b7a3613c2901074dc601c49be1d40566b7bddb1",
    "src/match3_ai/games/match3/special_selection.py": "fea306246d1bfac67d8a1515fef6448d681f992488f9df966858706386915511",
    "src/match3_ai/games/match3/specials.py": "c027d16a05b15472e7fe310a2f10f158c0f4c797070c42261f39b8efd113416f",
    "src/match3_ai/levels/__init__.py": "e5dd6665b3a00e84bd7db2101798dab155727f6d6b1673253ed3b9a2595df934",
    "src/match3_ai/levels/config.py": "1054c527b3593439db3a96f73b09d12ba2c8274ed1c2c325adad0e378937a5db",
    "src/match3_ai/levels/initialization.py": "7284b7a2c1907936a7b152ef6494e909ed89efcd927b5d0d977ba898c8c02907",
    "src/match3_ai/levels/loader.py": "3e378ac980becb38bd3129ae8f2f64d1fc78d705779ea453ba6a6e8b0c5dfa98",
    "src/match3_ai/levels/session.py": "28fc156f3977c27a3a03ccbdbf01c984afe399eb2a931733eecb977c6d0a040b",
    "src/match3_ai/training/seed_policy.py": "ba3dd6b370fb63acee0775aba22b3630018ae453eac047e0d36ec5616caa523e",
}


def semantic_source_hashes() -> dict[str, str]:
    """Refuse to claim audited semantics after an unreviewed source edit."""
    root = Path(__file__).resolve().parents[3]
    actual = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in VERIFIED_SOURCE_SHA256
    }
    if actual != VERIFIED_SOURCE_SHA256:
        changed = [
            name for name in actual if actual[name] != VERIFIED_SOURCE_SHA256[name]
        ]
        raise ValueError(
            f"MCTS compatibility audit required for changed sources: {changed}"
        )
    return actual
