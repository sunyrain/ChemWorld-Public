"""Author an explicit stochastic network on the common native runtime.

This generic constructor is not a biochemical task, measurement or score preset.
"""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger, VesselRecord
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.stochastic_kinetics import StochasticSpec, counts_to_mol
from chemworld.runtime.stochastic import install_stochastic_state
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def stochastic_world(
    network: ReactionNetworkSpec, counts: Mapping[str, int], *, seed: int = 0, budget: int = 40,
) -> WorldCompositionSpec:
    if network.stochastic is None:
        raise ValueError("stochastic_world requires an explicitly discrete network")
    spec = StochasticSpec(**network.stochastic)
    if set(counts) - set(network.species_ids):
        raise ValueError("unknown initial stochastic species")
    amounts = counts_to_mol({key: counts.get(key, 0) for key in network.species_ids})
    state = WorldState(
        amounts,
        spec.volume_L,
        spec.temperature_K,
        101325,
        "liquid",
        "reactor",
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "closed",
                    spec.volume_L,
                    470.0,
                    550000.0,
                    ("liquid",),
                    spec.temperature_K,
                    101325.0,
                )
            }
        ),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"},
    )
    state = install_stochastic_state(state, seed=seed, algorithm=spec.algorithm)
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"stochastic:{network.network_id}",
            "vessels": [{"id": "reactor", "capacity_L": spec.volume_L}],
            "components": [{"kind": "stochastic", "parameters": {}}],
            "physics": CompositionPhysics(network, {"reactor": state}).to_dict(),
            "task": {
                "objective": "characterize an explicitly stochastic network",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
            },
        }
    )
