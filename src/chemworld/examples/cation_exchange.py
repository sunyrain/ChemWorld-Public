"""Caller-authored finite mixed cation exchange and rinse example."""

from collections.abc import Mapping, Sequence

from chemworld.physchem.cation_exchange import CationExchangeSpec
from chemworld.world.cation_exchange_world import cation_exchange_world_spec
from chemworld.world.composition import WorldCompositionSpec


def cation_exchange_example(
    spec: CationExchangeSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    initial_calcium_equivalent_fraction: float = 0.0,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    """Use set_flow_rate with process/regenerate/rinse, then wait; source and media are finite."""
    return cation_exchange_world_spec(
        spec,
        amounts,
        initial_calcium_equivalent_fraction=initial_calcium_equivalent_fraction,
        shared_clock=shared_clock,
    )
