"""Install an adaptive variant in the original chemical world and stochastic strategy."""

from dataclasses import replace

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger, VesselRecord
from chemworld.physchem.stochastic_adaptive_advance import initial_adaptive_continuation
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def stochastic_adaptive_world(network, spec, amounts_mol, *, seed=0, budget=40):
    memory = initial_adaptive_continuation(network, spec, amounts_mol, seed=seed)
    state = WorldState(
        dict(amounts_mol),
        spec.volume_L,
        spec.temperature_K,
        101325.0,
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
    state = state.replace(process=replace(state.process, stochastic_continuation=memory))
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "stochastic-adaptive:" + network.network_id,
            "vessels": [{"id": "reactor", "capacity_L": spec.volume_L}],
            "components": [
                {"kind": "stochastic", "parameters": {}},
                {"kind": "thermal", "parameters": {"duration_range_s": [1e-6, 14400.0]}},
                {"kind": "material_input", "parameters": {}},
                {"kind": "observation", "parameters": {"instruments": ["sde_amount_meter"]}},
            ],
            "physics": CompositionPhysics(
                network, {"reactor": state}, models={"stochastic_hybrid": {"reactor": spec}}
            ).to_dict(),
            "task": {
                "objective": "characterize adaptive stochastic and deterministic reaction subnets",
                "budget": budget,
                "operations": ["wait", "add_component", "measure", "terminate"],
                "instruments": ["sde_amount_meter"],
            },
        }
    )
