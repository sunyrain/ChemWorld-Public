"""Authored surface apparatus; task policies remain independently replaceable."""

from dataclasses import replace

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
from chemworld.physchem.coupled_system import ConstantCvThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.physchem.surface_rates import gas_amounts_from_pressure
from chemworld.physchem.surface_specs import (
    DEFAULT_SURFACE_MECHANISM,
    GAS_SPECIES,
    bulk_co_oxidation_network,
    make_surface_slot,
    surface_species,
    surface_species_energies,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def surface_world_spec(
    *, include_bulk=False, mobile=False, include_surface=True, terminal_assay=False,
    surface_spec=DEFAULT_SURFACE_MECHANISM,
):
    """Two research vessels and two finite, fixed-temperature gas supplies.

    Both research vessels have the same device type. Mobile mode starts one
    carrier in the reactor and an empty share of that material in the receiver.
    Parameters are the synthetic benchmark in NATIVE_SURFACE_DESIGN.md.
    """
    network = replace(
        bulk_co_oxidation_network(),
        species=surface_species(surface_spec),
        reactions=bulk_co_oxidation_network().reactions if include_bulk else (),
    )
    states, components, models = {}, [], {}
    instruments = ["gas_pressure", "gas_composition", "gas_final_assay"]
    gas = gas_amounts_from_pressure(
        {"N2": 1.0}, pressure_Pa=100000, temperature_K=350, volume_L=0.01
    )
    for name in ("reactor", "receiver", "co_supply", "oxygen_supply"):
        equipment, slots = {}, ()
        amounts = dict(gas)
        if name in {"co_supply", "oxygen_supply"}:
            amounts = dict.fromkeys(GAS_SPECIES, 0.0)
            amounts["CO" if name == "co_supply" else "O2"] = 1e-4
            components.append(
                {
                    "id": f"{name}:material",
                    "vessel": name,
                    "kind": "material_input",
                    "parameters": {},
                }
            )
        else:
            heater = f"{name}:heater"
            equipment[heater] = EquipmentRecord(
                heater,
                "thermal_device",
                name,
                settings={"thermal_device": ThermalDeviceSpec().to_dict()},
                batch_reset="preserve",
            )
            models[name] = ThermalSystemSpec(
                ConstantCvThermodynamics(surface_species_energies(surface_spec), 20.0)
            )
            components += [
                {
                    "id": f"{name}:instruments", "vessel": name, "kind": "observation",
                    "parameters": {"instruments": instruments},
                },
                {
                    "id": f"{name}:thermal",
                    "vessel": name,
                    "kind": "thermal",
                    "parameters": {"temperature_range_K": [300, 430], "duration_range_s": [1, 600]},
                },
                {
                    "id": f"{name}:material",
                    "vessel": name,
                    "kind": "material_input",
                    "parameters": {},
                },
            ]
            if include_surface:
                owner = "mobile-catalyst" if mobile else f"{name}:catalyst"
                if not mobile:
                    equipment[owner] = EquipmentRecord(
                        owner, "fixed_catalyst", name, batch_reset="preserve"
                    )
                slot = make_surface_slot(
                    owner,
                    name,
                    fixed=not mobile,
                    phase_id="gas" if mobile else None,
                    heat_capacity_J_K=0.2 if mobile else 0,
                    spec=surface_spec,
                )
                if mobile and name == "receiver":
                    slot = slot.portion(0, name)
                slots = (slot,)
                components.append(
                    {
                        "id": f"{name}:surface",
                        "vessel": name,
                        "kind": "surface_reaction",
                        "parameters": {"surface_id": surface_spec.mechanism_id},
                    }
                )
            if include_bulk:
                components.append(
                    {"id": f"{name}:bulk", "vessel": name, "kind": "reaction", "parameters": {}}
                )
        pressure = sum(amounts.values()) * R_J_PER_MOL_K * 350 / 0.00001
        states[name] = WorldState(
            amounts,
            0.01,
            350.0,
            pressure,
            "gas",
            name,
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
            phases=PhaseLedger({"gas": PhaseRecord("gas", name, "gas", 0.01, amounts)}),
            vessels=VesselLedger(
                {name: VesselRecord(name, "sealed", 0.01, 450, 500000, ("gas",), 350, pressure)}
            ),
            equipment=EquipmentLedger(equipment),
            state_slots=StateSlots(slots),
        )
    physics = CompositionPhysics(
        network,
        states,
        thermal_models=models,
        feed_sources={"CO-feed": "co_supply", "O2-feed": "oxygen_supply"},
        surface_models={surface_spec.mechanism_id: surface_spec},
    )
    payload = {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "native-surface-apparatus",
        "components": components,
        "vessels": [{"id": name, "capacity_L": 0.01} for name in states],
        "connections": [
            {
                "id": "reactor-to-receiver",
                "unit": "mol",
                "source": {"component": "reactor:material", "port": "out"},
                "target": {"component": "receiver:material", "port": "in"},
            },
            {
                "id": "receiver-to-reactor",
                "unit": "mol",
                "source": {"component": "receiver:material", "port": "out"},
                "target": {"component": "reactor:material", "port": "in"},
            },
        ],
        "physics": physics.to_dict(),
        "task": {
            "objective": "study chemical response",
            "budget": 40,
            "operations": [
                "add_component",
                "heat",
                "wait",
                "pause_control",
                "resume_control",
                "route_material",
                "select_vessel",
                "measure",
                "terminate",
            ],
            "instruments": instruments if terminal_assay else instruments[:2],
            "termination": "final-assay-or-budget" if terminal_assay else "explicit-stop-or-budget",
        },
    }
    return WorldCompositionSpec.from_dict(payload)
