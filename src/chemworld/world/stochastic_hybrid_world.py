"""Author one finite, fixed-partition chemical PDMP on the original world."""

from dataclasses import replace
from typing import cast

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import ProcessLedger, VesselLedger, VesselRecord
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.stochastic_hybrid import StochasticHybridSpec, initial_hybrid_continuation
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def stochastic_hybrid_world(network: ReactionNetworkSpec, spec: StochasticHybridSpec,
                             amounts_mol: dict[str, float], *,
                             seed: int = 0, budget: int = 40) -> WorldCompositionSpec:
    continuation = initial_hybrid_continuation(network, spec, amounts_mol, seed=seed)
    state = WorldState(dict(amounts_mol), spec.volume_L, spec.temperature_K, 101325.,
        "liquid", "reactor", vessels=VesselLedger({"reactor": VesselRecord(
            "reactor", "closed", spec.volume_L, 470., 550000., ("liquid",),
            spec.temperature_K, 101325.)}),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"})
    state = state.replace(process=replace(cast(ProcessLedger, state.process),
                                          stochastic_continuation=continuation))
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": f"stochastic-hybrid:{network.network_id}",
        "vessels": [{"id": "reactor", "capacity_L": spec.volume_L}],
        "components": [{"kind": "stochastic", "parameters": {}}],
        "physics": CompositionPhysics(network, {"reactor": state},
            models={"stochastic_hybrid": {"reactor": spec}}).to_dict(),
        "task": {"objective": "characterize a fixed-partition chemical PDMP", "budget": budget,
                 "operations": ["wait", "terminate"], "instruments": []}})
