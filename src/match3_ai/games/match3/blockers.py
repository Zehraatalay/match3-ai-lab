from dataclasses import dataclass
from enum import IntEnum, auto


class BlockerType(IntEnum):
    NONE = 0
    CRATE = auto()
    ICE = auto()
    STONE = auto()


class DamageSource(IntEnum):
    ADJACENT_MATCH = auto()
    DIRECT_MATCH = auto()
    ROCKET = auto()
    BOMB = auto()
    PROPELLER = auto()
    LIGHT_BALL = auto()


@dataclass(frozen=True, slots=True)
class BlockerProperties:
    contains_tile: bool
    allows_swap: bool
    allows_gravity: bool
    default_hp: int
    damage_sources: frozenset[DamageSource]

    blocks_special_effects: bool = False
    blocks_refill: bool = False
    movable: bool = False


BLOCKER_PROPERTIES: dict[BlockerType, BlockerProperties] = {
    BlockerType.CRATE: BlockerProperties(
        contains_tile=False,
        allows_swap=False,
        allows_gravity=False,
        default_hp=1,
        damage_sources=frozenset(
            {
                DamageSource.ADJACENT_MATCH,
                DamageSource.ROCKET,
                DamageSource.BOMB,
                DamageSource.PROPELLER,
                DamageSource.LIGHT_BALL,
            }
        ),
    ),
    BlockerType.ICE: BlockerProperties(
        contains_tile=True,
        allows_swap=True,
        allows_gravity=True,
        default_hp=1,
        damage_sources=frozenset(
            {
                DamageSource.DIRECT_MATCH,
                DamageSource.ROCKET,
                DamageSource.BOMB,
                DamageSource.PROPELLER,
                DamageSource.LIGHT_BALL,
            }
        ),
    ),
    BlockerType.STONE: BlockerProperties(
        contains_tile=False,
        allows_swap=False,
        allows_gravity=False,
        default_hp=2,
        damage_sources=frozenset(
            {
                DamageSource.ROCKET,
                DamageSource.BOMB,
                DamageSource.PROPELLER,
                DamageSource.LIGHT_BALL,
            }
        ),
    ),
}