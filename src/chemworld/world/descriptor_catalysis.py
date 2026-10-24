"""Declared catalytic apparatus using the original composition and thermal runtime."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, make_catalytic_slot
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def descriptor_catalysis_world(
    spec: CatalyticMechanismSpec,
    gas_amounts_mol: Mapping[str, float],
    *,
    capacities_mol: Mapping[str, float],
    initial_adsorbates_mol: Mapping[str, float],
    volume_L: float,
    temperature_K: float,
    support_heat_capacity_J_K: float,
    stop_temperature_K: float,
    maximum_pressure_Pa: float,
    isothermal: bool = True,
    budget: int = 100,
    gas_network: ReactionNetworkSpec | None = None,
) -> WorldCompositionSpec:
    """One sealed reactor; only declared finite material and original owned sites."""
    if set(gas_amounts_mol) - set(spec.gas_species):
        raise ValueError("catalytic world gas inventory refers to unknown species")
    if not spec.minimum_temperature_K < temperature_K < spec.maximum_temperature_K:
        raise ValueError("initial catalytic temperature must be inside its declared domain")
    name, owner, heater = "reactor", "reactor:catalyst", "reactor:heater"
    amounts = {key: gas_amounts_mol.get(key, 0.0) for key in spec.gas_species}
    pressure = sum(amounts.values()) * R * temperature_K / (volume_L / 1000)
    slot = make_catalytic_slot(
        owner, name, spec, capacities_mol=capacities_mol, initial_amounts_mol=initial_adsorbates_mol
    )
    device = ThermalDeviceSpec(
        setpoint_K=temperature_K,
        mode="isothermal" if isothermal else "finite_power",
        minimum_setpoint_K=spec.minimum_temperature_K,
        maximum_setpoint_K=spec.maximum_temperature_K,
        enabled=isothermal,
    )
    state = WorldState(
        amounts,
        volume_L,
        temperature_K,
        pressure,
        "gas",
        name,
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"},
        phases=PhaseLedger({"gas": PhaseRecord("gas", name, "gas", volume_L, amounts)}),
        vessels=VesselLedger(
            {
                name: VesselRecord(
                    name,
                    "sealed-catalytic-reactor",
                    volume_L,
                    spec.maximum_temperature_K,
                    maximum_pressure_Pa,
                    ("gas",),
                    temperature_K,
                    pressure,
                )
            }
        ),
        equipment=EquipmentLedger(
            {
                owner: EquipmentRecord(owner, "fixed_catalyst", name, batch_reset="preserve"),
                heater: EquipmentRecord(
                    heater,
                    "thermal_device",
                    name,
                    batch_reset="preserve",
                    settings={"thermal_device": device.to_dict()},
                ),
            }
        ),
        state_slots=StateSlots((slot,)),
    )
    caloric = CatalyticThermodynamics(spec.species_by_id, support_heat_capacity_J_K)
    thermal = ThermalSystemSpec(
        caloric,
        minimum_temperature_K=spec.minimum_temperature_K,
        maximum_temperature_K=spec.maximum_temperature_K,
        stop_temperature_K=stop_temperature_K,
        environment_temperature_K=temperature_K,
        loss_conductance_W_K=0,
        maximum_pressure_Pa=maximum_pressure_Pa,
    )
    if gas_network is not None:
        if set(gas_network.species_ids) != set(spec.gas_species):
            raise ValueError("bulk mechanism must use exactly the declared catalytic gas species")
        for key, model in gas_network.imported_gas_thermochemistry.items():
            if model.to_dict() != spec.species_by_id[key].gas_model.to_dict():
                raise ValueError("bulk and surface kinetics require identical gas thermochemistry")
        if gas_network.reactions and set(gas_network.imported_gas_thermochemistry) != set(
            spec.gas_species
        ):
            raise ValueError("imported bulk reactions require the full same-source thermal table")
    network = ReactionNetworkSpec(
        spec.mechanism_id,
        tuple(s.species for s in spec.species),
        () if gas_network is None else gas_network.reactions,
        metadata={} if gas_network is None else dict(gas_network.metadata),
    )
    physics = CompositionPhysics(
        network,
        {name: state},
        models={"surface": {spec.mechanism_id: spec}, "thermal": {name: thermal}},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-catalysis:" + spec.mechanism_id,
            "vessels": [{"id": name, "capacity_L": volume_L}],
            "components": [
                {
                    "id": heater,
                    "vessel": name,
                    "kind": "thermal",
                    "parameters": {
                        "duration_range_s": [1, 10000],
                    },
                },
                {
                    "id": owner,
                    "vessel": name,
                    "kind": "surface_reaction",
                    "parameters": {"surface_id": spec.mechanism_id},
                },
                *(
                    [{"id": "reactor:bulk", "vessel": name, "kind": "reaction", "parameters": {}}]
                    if network.reactions
                    else []
                ),
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study declared finite-site catalytic chemistry",
                "budget": budget,
                "operations": ["wait", "terminate"],
                "instruments": [],
            },
        }
    )
