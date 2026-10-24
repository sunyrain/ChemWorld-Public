"""Declare an osmotic experiment using caller-owned sourced aqueous properties."""

from collections.abc import Mapping

from chemworld.physchem.osmotic_membrane import OsmoticMembraneSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.osmotic_membrane_world import osmotic_membrane_world_spec


def osmotic_membrane_example(
    model: OsmoticMembraneSpec,
    left_amounts: Mapping[str, float],
    right_amounts: Mapping[str, float],
    *,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    """Use wait to evolve FO/PRO/reverse flow at declared fixed pressures.

    Salt stays in its original tank. The finite bath supplies Q=delta H and
    moving pressure boundaries exchange sum P delta V; no generator is implied.
    """
    return osmotic_membrane_world_spec(
        model, left_amounts, right_amounts, shared_clock=shared_clock
    )
