"""Installed generic fixed-SHE catalytic half-cell on the original thermal strategy."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, make_catalytic_slot
from chemworld.physchem.catalysis_reservoirs import CatalyticReservoirBoundary
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.chemostatted_catalysis import pack_chemostatted_state, validate_boundary
from chemworld.physchem.heat_transfer_units import ThermalSystemSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.chemostatted_catalysis import (
    CHEMOSTATTED_CATALYSIS_TYPE,
    install_chemostatted_material,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def chemostatted_catalysis_world(
    spec: CatalyticMechanismSpec,
    boundary: CatalyticReservoirBoundary,
    gas_amounts_mol: Mapping[str, float],
    *,
    capacities_mol: Mapping[str, float],
    initial_adsorbates_mol: Mapping[str, float],
    potential_V: float,
    support_heat_capacity_J_K: float = 0.0,
    budget: int = 100,
) -> WorldCompositionSpec:
    validate_boundary(spec, boundary, potential_V)
    name, owner = "reactor", "reactor:catalyst"
    slot = make_catalytic_slot(
        owner, name, spec, capacities_mol=capacities_mol, initial_amounts_mol=initial_adsorbates_mol
    )
    physical = pack_chemostatted_state(spec, slot, boundary, gas_amounts_mol)
    record = EquipmentRecord(
        "working-electrode",
        CHEMOSTATTED_CATALYSIS_TYPE,
        name,
        batch_reset="preserve",
        settings={
            "physical_model": {
                "mechanism": spec.to_dict(),
                "boundary": boundary.to_dict(),
                "support_heat_capacity_J_K": support_heat_capacity_J_K,
            },
            "electrical_control": {
                "limits_V": [boundary.minimum_potential_V, boundary.maximum_potential_V],
                "supported_units": ["V"],
                "constant_potential_only": True,
                "program": ScalarControlProgram(((0, potential_V),), "V").to_dict(),
            },
        },
    )
    state = WorldState(
        {k: physical.amounts_mol[k] for k in (*spec.gas_species, *spec.reservoir_species)},
        physical.volume_L,
        physical.temperature_K,
        1.0,
        "gas",
        name,
        equipment=EquipmentLedger(
            {
                owner: EquipmentRecord(owner, "fixed_catalyst", name, batch_reset="preserve"),
                record.equipment_id: record,
                "heater": EquipmentRecord(
                    "heater",
                    "thermal_device",
                    name,
                    batch_reset="preserve",
                    settings={
                        "thermal_device": boundary.bath.to_dict(),
                        "lab_role": "fixed_isothermal_bath",
                    },
                ),
            }
        ),
        vessels=VesselLedger(
            {
                name: VesselRecord(
                    name,
                    "chemostatted-catalytic-half-cell",
                    physical.volume_L,
                    spec.maximum_temperature_K,
                    boundary.maximum_pressure_Pa,
                    ("gas", "electrolyte"),
                    physical.temperature_K,
                    1.0,
                )
            }
        ),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"},
    )
    state = install_chemostatted_material(state, spec, boundary, slot, physical, record)
    caloric = CatalyticThermodynamics(
        spec.species_by_id,
        support_heat_capacity_J_K + boundary.carrier.volumetric_cv_J_L_K * boundary.liquid_volume_L,
        boundary.carrier.feed_temperature_K,
    )
    thermal = ThermalSystemSpec(
        caloric,
        minimum_temperature_K=spec.minimum_temperature_K,
        maximum_temperature_K=spec.maximum_temperature_K,
        stop_temperature_K=(physical.temperature_K + spec.maximum_temperature_K) / 2,
        environment_temperature_K=physical.temperature_K,
        loss_conductance_W_K=0,
        maximum_pressure_Pa=boundary.maximum_pressure_Pa,
    )
    physics = CompositionPhysics(
        ReactionNetworkSpec(spec.mechanism_id, tuple(s.species for s in spec.species), ()),
        {name: state},
        models={"surface": {spec.mechanism_id: spec}, "thermal": {name: thermal}},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "chemostatted-catalysis:" + spec.mechanism_id,
            "vessels": [{"id": name, "capacity_L": physical.volume_L}],
            "connections": [],
            "components": [
                {"id": "heater", "vessel": name, "kind": "thermal", "parameters": {}},
                {
                    "id": owner,
                    "vessel": name,
                    "kind": "surface_reaction",
                    "parameters": {"surface_id": spec.mechanism_id},
                },
                {
                    "id": "meter",
                    "vessel": name,
                    "kind": "observation",
                    "parameters": {"instruments": ["electrical_meter", "chemostatted_coverage"]},
                },
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study declared fixed-SHE/constant-activity catalysis",
                "budget": budget,
                "operations": ["configure_electrical_control", "wait", "measure", "terminate"],
                "instruments": ["electrical_meter", "chemostatted_coverage"],
            },
        }
    )
