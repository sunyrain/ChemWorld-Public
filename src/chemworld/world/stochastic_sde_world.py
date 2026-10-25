"""Declare continuous reaction SDE chemistry in the original physical world."""

from dataclasses import replace

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger, VesselRecord
from chemworld.physchem.stochastic_sde import initial_sde_continuation
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def stochastic_sde_world(network, spec, amounts_mol, *, seed=0, budget=40):
    from chemworld.world.stochastic_sde_instruments import SDE_ALIQUOT_METER, SDE_ALIQUOT_VOLUME_L

    instruments = ["sde_amount_meter"]
    if (
        spec.minimum_volume_L is not None
        and spec.volume_L - SDE_ALIQUOT_VOLUME_L >= spec.minimum_volume_L
    ):
        instruments.append(SDE_ALIQUOT_METER)
    continuation = initial_sde_continuation(network, spec, amounts_mol, seed=seed)
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
    state = state.replace(process=replace(state.process, stochastic_continuation=continuation))
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": f"stochastic-sde:{network.network_id}",
            "vessels": [{"id": "reactor", "capacity_L": spec.volume_L}],
            "components": [
                {"kind": "stochastic", "parameters": {}},
                {"kind": "thermal", "parameters": {"duration_range_s": [1e-6, 14400.0]}},
                {"kind": "material_input", "parameters": {}},
                {"kind": "observation", "parameters": {"instruments": instruments}},
            ],
            "physics": CompositionPhysics(
                network, {"reactor": state}, models={"stochastic_sde": {"reactor": spec}}
            ).to_dict(),
            "task": {
                "objective": "characterize continuous stochastic reaction chemistry",
                "budget": budget,
                "operations": ["wait", "add_component", "measure", "terminate"]
                + (["set_sde_parameters"] if spec.controls else []),
                "instruments": instruments,
            },
        }
    )
