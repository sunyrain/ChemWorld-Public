"""Caller-authored finite mixed aqueous adsorption and rinse example."""

from collections.abc import Mapping, Sequence

from chemworld.physchem.aqueous_adsorption import AqueousAdsorptionSpec
from chemworld.world.aqueous_adsorption_world import aqueous_adsorption_world_spec
from chemworld.world.composition import WorldCompositionSpec


def aqueous_adsorption_example(
    spec: AqueousAdsorptionSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    initial_loading_mol_per_kg: Mapping[str, float] | None = None,
    shared_clock: bool = True,
) -> WorldCompositionSpec:
    """Use set_flow_rate with process/rinse, then wait; source and media are finite."""
    return aqueous_adsorption_world_spec(
        spec,
        amounts,
        initial_loading_mol_per_kg=initial_loading_mol_per_kg,
        shared_clock=shared_clock,
    )
