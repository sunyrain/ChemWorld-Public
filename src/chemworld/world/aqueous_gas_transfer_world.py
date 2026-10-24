"""One original world containing finite gas and aqueous phases of a rigid vessel."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.aqueous_headspace_minerals import AqueousHeadspaceMineralSpec
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.aqueous_gas_transfer import (
    AQUEOUS_GAS_TRANSFER_TYPE,
    HeadspaceSpec,
    headspace_container_volume,
    headspace_pressure,
    install_headspace,
    prepare_aqueous_headspace_state,
)
from chemworld.runtime.aqueous_mineral_kinetics import database_mineral_geometry_slot
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_aqueous_headspace_world(
    spec: HeadspaceSpec,
    amounts_mol: Mapping[str, float],
    *,
    max_pressure_Pa: float | None = None,
    budget: int = 100,
    mineral_geometries: Mapping[str, MineralGeometry] | None = None,
) -> WorldCompositionSpec:
    amounts = prepare_aqueous_headspace_state(spec, dict(amounts_mol))
    pressure = headspace_pressure(spec, amounts)
    max_pressure = spec.pressure_bounds_Pa[1] if max_pressure_Pa is None else max_pressure_Pa
    volume = headspace_container_volume(spec) * 1000
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        if mineral_geometries is None or set(mineral_geometries) != set(spec.mineral_ids):
            raise ValueError("every headspace mineral requires its original material geometry")
    elif mineral_geometries is not None:
        raise ValueError("gas/liquid v1 does not accept mineral geometry")
    state = WorldState(
        amounts,
        volume,
        spec.temperature_K,
        pressure,
        "fluid",
        "reactor",
        equipment=EquipmentLedger(
            {
                "headspace-model": EquipmentRecord(
                    "headspace-model",
                    AQUEOUS_GAS_TRANSFER_TYPE,
                    "reactor",
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict()},
                )
            }
        ),
        vessels=VesselLedger(
            {
                "reactor": VesselRecord(
                    "reactor",
                    "rigid-aqueous-headspace",
                    volume,
                    spec.temperature_K,
                    max_pressure,
                    ("fluid", "liquid", "gas", "solid")
                    if mineral_geometries is not None
                    else ("fluid", "liquid", "gas"),
                    spec.temperature_K,
                    pressure,
                )
            }
        ),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"},
    )
    state = install_headspace(state, spec, amounts)
    if mineral_geometries is not None:
        state = state.replace(
            state_slots=StateSlots(
                tuple(
                    database_mineral_geometry_slot("reactor", "mineral:" + m, m, geometry)
                    for m, geometry in mineral_geometries.items()
                )
            )
        )
    network = ReactionNetworkSpec(
        "finite-aqueous-headspace",
        tuple(
            SpeciesSpec(
                s.species_id,
                hill_formula(dict(s.composition)),
                charge=s.charge,
                phase="gas" if s.phase == "gas" else "solid" if s.phase == "mineral" else "liquid",
            )
            for s in spec.equilibrium.database.species
        ),
        (),
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-headspace",
            "vessels": [{"id": "reactor", "capacity_L": volume}],
            "components": [
                {
                    "id": "headspace",
                    "kind": "aqueous_gas_transfer",
                    "vessel": "reactor",
                    "parameters": {},
                },
            ],
            "connections": [],
            "physics": CompositionPhysics(
                network,
                {"reactor": state},
                advance_bindings={"reactor": "finite-aqueous-gas-transfer"},
            ).to_dict(),
            "task": {
                "objective": "study finite headspace absorption and desorption",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
            },
        }
    )
