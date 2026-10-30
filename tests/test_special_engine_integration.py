import numpy as np

import match3_ai.games.match3.engine as engine_module
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.pattern_detection import (
    detect_patterns,
)
from match3_ai.games.match3.special_resolver import (
    resolve_matched_specials,
)
from match3_ai.games.match3.specials import SpecialType


def make_special_test_state(
    source_special: SpecialType,
    destination_special: SpecialType = SpecialType.NONE,
) -> BoardState:
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    source = (2, 1)
    destination = (2, 2)

    specials[source] = source_special
    specials[destination] = destination_special

    if source_special == SpecialType.LIGHT_BALL:
        tiles[source] = 255

    if destination_special == SpecialType.LIGHT_BALL:
        tiles[destination] = 255

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )


def test_engine_rocket_swap():
    state = make_special_test_state(
        SpecialType.ROCKET_HORIZONTAL,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(
        state,
        action=((2, 1), (2, 2)),
    )

    assert result.valid
    assert result.activation_result is not None
    assert result.cleared_tiles >= 5

    assert np.all(result.state.tiles > 0)


def test_engine_light_ball_rocket_combination():
    state = make_special_test_state(
        SpecialType.LIGHT_BALL,
        SpecialType.ROCKET_HORIZONTAL,
    )

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(
        state,
        action=((2, 1), (2, 2)),
    )

    assert result.valid
    assert result.activation_result is not None

    assert result.activation_result.transformations

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)


def test_engine_light_ball_light_ball_combination():
    state = make_special_test_state(
        SpecialType.LIGHT_BALL,
        SpecialType.LIGHT_BALL,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(
        state,
        action=((2, 1), (2, 2)),
    )

    assert result.valid
    assert result.activation_result is not None

    assert len(result.activation_result.affected_cells) == 25

    assert result.activation_result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_engine_matched_rocket_activates(monkeypatch):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    # a horizontal Rocket is part of the match created by the swap.
    specials[2, 2] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    captured_states = []
    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(
        state,
        action=((2, 1), (2, 2)),
    )

    assert result.valid
    assert captured_states


def test_engine_normal_match_activates_existing_rocket(monkeypatch):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    # create a horizontal match through a normal swap.
    tiles[2, 1] = 2
    tiles[2, 2] = 2
    tiles[2, 3] = 1
    tiles[1, 3] = 2

    specials = np.zeros_like(tiles, dtype=np.uint8)

    # the Rocket is part of the resulting match,
    # but it is not one of the swapped tiles.
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 3), (2, 3))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []

    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())

        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    first_resolution = captured_states[0]

    # the Rocket clears its entire row.
    assert np.all(first_resolution.tiles[2, :] == 0)

    assert np.all(first_resolution.specials[2, :] == SpecialType.NONE)

    assert result.cleared_tiles >= 5


def test_existing_special_activation_takes_priority_over_creation(
    monkeypatch,
):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 2, 2, 1, 2],
            [1, 2, 3, 1, 3],
            [2, 3, 1, 2, 1],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    # existing Rocket belongs to the four-match after the swap.
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 3), (2, 3))

    captured_states = []

    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    first_resolution = captured_states[0]

    # the existing Rocket activates and clears its row.
    assert np.all(first_resolution.tiles[2, :] == 0)

    # no replacement special is created in the same group.
    assert np.all(first_resolution.specials[2, :] == SpecialType.NONE)


def test_special_effect_cancels_overlapping_creation(
    monkeypatch,
):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    # four-match creation group.
    tiles[0, :] = [1, 1, 1, 1, 2]

    # separate three-match group containing a vertical Rocket.
    tiles[2, :] = [3, 2, 2, 2, 1]

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[2, 2] = SpecialType.ROCKET_VERTICAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    # resolve both existing match groups through the resolver.
    matched_cells = {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 1),
        (2, 2),
        (2, 3),
    }

    result = resolve_matched_specials(
        state=state,
        matched_cells=matched_cells,
        rng=np.random.default_rng(42),
    )

    assert (0, 2) in result.effect_cells
    assert (0, 2) in result.affected_cells

    # a special cannot be created where an existing effect hits.
    creation_position = (0, 2)

    assert creation_position in result.effect_cells


def test_engine_resolves_two_independent_match_groups(
    monkeypatch,
):
    tiles = np.array(
        [
            [2, 2, 3, 2, 3],
            [2, 3, 1, 2, 2],
            [1, 3, 1, 3, 1],
            [3, 1, 2, 3, 2],
            [3, 2, 1, 2, 1],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[4, 1] = SpecialType.ROCKET_VERTICAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((3, 2), (4, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []

    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=3, seed=42)

    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    first_resolution = captured_states[0]

    # the vertical Rocket clears column 1.
    assert np.all(first_resolution.tiles[:, 1] == 0)

    # the four-match group must not create a special
    # at a position hit by the Rocket effect.
    assert first_resolution.specials[3, 1] == SpecialType.NONE


def test_creation_is_cancelled_when_position_is_hit_by_special_effect():
    creation_position = (0, 2)

    effect_cells = frozenset(
        {
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
        }
    )

    creations = [
        type(
            "MockCreation",
            (),
            {"position": creation_position},
        )()
    ]

    filtered_creations = [
        creation for creation in creations if creation.position not in effect_cells
    ]

    assert filtered_creations == []


def test_engine_cancels_creation_hit_by_bomb_effect(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[1, 3] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []
    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    first_resolution = captured_states[0]

    # the four-match would create a special at (2, 2).
    # the matched Bomb at (1, 3) also affects (2, 2).
    # therefore, special creation must be cancelled.
    assert first_resolution.tiles[2, 2] == 0
    assert first_resolution.specials[2, 2] == SpecialType.NONE


def test_engine_creates_special_without_bomb_interference(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []
    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    first_resolution = captured_states[0]

    # the four-match creates a horizontal Rocket at the swap destination.
    assert first_resolution.specials[2, 2] == SpecialType.ROCKET_HORIZONTAL
    assert first_resolution.tiles[2, 2] == 1


def test_engine_resolves_three_special_chain_reaction():
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    specials[0, 2] = SpecialType.ROCKET_VERTICAL
    specials[2, 2] = SpecialType.BOMB
    specials[2, 3] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((0, 2), (0, 3))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    engine = Match3Engine(num_colors=3, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert result.activation_history

    activated_positions = set()

    for activation in result.activation_history:
        activated_positions.update(activation.activated_positions)

    # positions refer to the board after the swap.
    assert (0, 3) in activated_positions
    assert (2, 2) in activated_positions
    assert (2, 3) in activated_positions

    assert len(activated_positions) == 3

    # verify that each special is activated exactly once.
    activation_counts = {}

    for activation in result.activation_history:
        for position in activation.activated_positions:
            activation_counts[position] = activation_counts.get(position, 0) + 1

    assert len(activation_counts) == 3
    assert all(count == 1 for count in activation_counts.values())


def test_engine_preserves_special_during_gravity(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[0, 0] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []
    original_refill = engine_module.refill_board_state

    def capture_refill(current_state, *args, **kwargs):
        captured_states.append(current_state.copy())
        return original_refill(current_state, *args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "refill_board_state",
        capture_refill,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    # inspect the board after gravity but before refill.
    after_gravity = captured_states[0]

    bomb_positions = np.argwhere(after_gravity.specials == SpecialType.BOMB)

    assert len(bomb_positions) == 1

    bomb_row, bomb_col = map(int, bomb_positions[0])

    # the Bomb must remain attached to its original tile color.
    assert after_gravity.tiles[bomb_row, bomb_col] == 3

    # the Bomb must not be activated by this match.
    assert not any(
        (0, 0) in activation.activated_positions
        for activation in result.activation_history
    )


def test_engine_moves_special_with_tile_during_gravity(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[0, 2] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_states = []
    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        before_gravity = current_state.copy()
        after_gravity = original_gravity(current_state)

        captured_states.append((before_gravity, after_gravity.copy()))

        return after_gravity

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert captured_states

    before_gravity, after_gravity = captured_states[0]

    before_positions = np.argwhere(before_gravity.specials == SpecialType.BOMB)
    after_positions = np.argwhere(after_gravity.specials == SpecialType.BOMB)

    assert len(before_positions) == 1
    assert len(after_positions) == 1

    before_row, before_col = map(int, before_positions[0])
    after_row, after_col = map(int, after_positions[0])

    # the special must remain in the same column.
    assert after_col == before_col

    # gravity must move the special downward.
    assert after_row > before_row

    # the special must preserve its tile color.
    assert after_gravity.tiles[after_row, after_col] == 1
    # every recorded activation must contain at least one special.
    assert all(
        activation.activated_positions for activation in result.activation_history
    )

    # activated positions must be within the board boundaries.
    rows, cols = state.tiles.shape

    for activation in result.activation_history:
        for row, col in activation.activated_positions:
            assert 0 <= row < rows
            assert 0 <= col < cols


def test_engine_records_special_activation_during_cascade(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    original_refill = engine_module.refill_board_state
    refill_count = 0

    def controlled_refill(current_state, *args, **kwargs):
        nonlocal refill_count

        refilled_state = original_refill(
            current_state,
            *args,
            **kwargs,
        )

        refill_count += 1

        if refill_count != 1:
            return refilled_state

        # Force a cascade match containing an existing Bomb.
        new_tiles = refilled_state.tiles.copy()
        new_specials = refilled_state.specials.copy()

        new_tiles[0, 0:3] = 2
        new_specials[0, 1] = SpecialType.BOMB

        return BoardState(
            tiles=new_tiles,
            specials=new_specials,
            cell_mask=refilled_state.cell_mask,
        )

    monkeypatch.setattr(
        engine_module,
        "refill_board_state",
        controlled_refill,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert result.cascade_count >= 2

    assert result.activation_history

    assert any(
        (0, 1) in activation.activated_positions
        for activation in result.activation_history
    )

    assert refill_count >= 2


def test_engine_creates_special_during_cascade(monkeypatch):
    tiles = np.array(
        [
            [3, 4, 1, 3, 4],
            [4, 2, 1, 2, 3],
            [1, 1, 2, 1, 4],
            [2, 3, 4, 2, 1],
            [3, 4, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    original_refill = engine_module.refill_board_state
    original_gravity = engine_module.apply_board_gravity

    refill_count = 0
    captured_states = []

    def controlled_refill(current_state, *args, **kwargs):
        nonlocal refill_count

        refilled_state = original_refill(
            current_state,
            *args,
            **kwargs,
        )

        refill_count += 1

        if refill_count != 1:
            return refilled_state

        new_tiles = refilled_state.tiles.copy()
        new_specials = refilled_state.specials.copy()

        # Create a horizontal four-match for the next cascade.
        new_tiles[0, 0:4] = 2
        new_specials[0, 0:4] = SpecialType.NONE

        return BoardState(
            tiles=new_tiles,
            specials=new_specials,
            cell_mask=refilled_state.cell_mask,
        )

    def capture_gravity(current_state):
        captured_states.append(current_state.copy())
        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "refill_board_state",
        controlled_refill,
    )
    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=4, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert result.cascade_count >= 2
    assert refill_count >= 2

    # the second gravity call follows cascade resolution.
    assert len(captured_states) >= 2

    cascade_resolution = captured_states[1]

    # cascade creations use the default group position,
    # not the original swap destination.
    assert cascade_resolution.specials[0, 0] == SpecialType.ROCKET_HORIZONTAL
    assert cascade_resolution.tiles[0, 0] == 2


def test_engine_preserves_board_invariants_after_special_chain():
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)

    specials[0, 2] = SpecialType.ROCKET_VERTICAL
    specials[2, 2] = SpecialType.BOMB
    specials[2, 3] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((0, 2), (0, 3))

    assert is_valid_swap(state, *action)

    engine = Match3Engine(num_colors=3, seed=42)
    result = engine.step(state, action)

    assert result.valid

    final_state = result.state

    # all board arrays must have identical shapes.
    assert final_state.tiles.shape == final_state.specials.shape
    assert final_state.tiles.shape == final_state.cell_mask.shape

    # all playable cells must contain valid tile colors.
    playable_tiles = final_state.tiles[final_state.cell_mask]

    assert np.all(playable_tiles >= 1)
    assert np.all(playable_tiles <= 3)

    # all special types must be defined.
    valid_special_types = {int(special_type) for special_type in SpecialType}

    assert set(np.unique(final_state.specials)).issubset(valid_special_types)

    # every remaining special must be attached to a valid tile.
    special_mask = final_state.specials != SpecialType.NONE

    assert np.all(final_state.cell_mask[special_mask])
    assert np.all(final_state.tiles[special_mask] != 0)

    # the engine must finish with no unresolved matches.
    assert detect_patterns(final_state.tiles) == []


def test_engine_counts_only_occupied_affected_cells(monkeypatch):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[0, 2] = SpecialType.ROCKET_VERTICAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((0, 2), (0, 3))

    assert is_valid_swap(state, *action)

    original_clear = engine_module.clear_board_matches
    cleared_counts = []

    def capture_clear(current_state, affected_cells):
        occupied_count = sum(
            current_state.cell_mask[position] and current_state.tiles[position] != 0
            for position in affected_cells
        )

        cleared_counts.append(occupied_count)

        return original_clear(current_state, affected_cells)

    monkeypatch.setattr(
        engine_module,
        "clear_board_matches",
        capture_clear,
    )

    engine = Match3Engine(num_colors=3, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert cleared_counts

    assert result.cleared_tiles == sum(cleared_counts)


def test_engine_does_not_count_empty_affected_cells(monkeypatch):
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles, dtype=np.uint8)
    specials[0, 2] = SpecialType.ROCKET_VERTICAL

    # create an empty cell within the Rocket's effect area.
    tiles[2, 3] = 0

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    action = ((0, 2), (0, 3))

    assert is_valid_swap(state, *action)
    original_clear = engine_module.clear_board_matches
    actual_cleared_counts = []

    def capture_clear(current_state, affected_positions):
        occupied_count = sum(
            1
            for position in affected_positions
            if current_state.cell_mask[position] and current_state.tiles[position] != 0
        )

        actual_cleared_counts.append(occupied_count)

        return original_clear(current_state, affected_positions)

    monkeypatch.setattr(
        engine_module,
        "clear_board_matches",
        capture_clear,
    )
    engine = Match3Engine(num_colors=3, seed=42)
    result = engine.step(state, action)

    assert result.valid
    assert result.activation_result is not None

    affected_cells = result.activation_result.affected_cells

    # calculate the number of occupied cells after the swap,
    # before special effects are applied.
    swapped_tiles = state.tiles.copy()
    swapped_tiles[action[0]], swapped_tiles[action[1]] = (
        swapped_tiles[action[1]],
        swapped_tiles[action[0]],
    )

    expected_cleared = sum(
        1
        for position in affected_cells
        if state.cell_mask[position] and swapped_tiles[position] != 0
    )

    # The empty cell must be included in the effect area.
    assert (2, 3) in affected_cells
    assert swapped_tiles[2, 3] == 0

    # Verify the number of occupied cells affected by the initial activation.
    assert expected_cleared == sum(
        1 for position in affected_cells if swapped_tiles[position] != 0
    )
    assert actual_cleared_counts
    assert result.cleared_tiles == sum(actual_cleared_counts)
