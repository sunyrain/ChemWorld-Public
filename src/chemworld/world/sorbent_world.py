"""Authored finite liquid contact apparatus, using the existing composition."""

from chemworld.foundation import WorldState
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.heat_transfer_units import (
    CondensedBackgroundSpec,
    ThermalDeviceSpec,
    ThermalSystemSpec,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.sorbent_kinetics import make_sorbent_slot
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def sorbent_world_spec(spec, thermodynamics, component_formulas, *, mobile=False, preloaded=False):
    """Finite contact, feed, waste and wash stocks with authored formulas."""
    if set(spec.component_ids) != {"A", "B"} or set(component_formulas) != {"A", "B"}:
        raise ValueError("this research apparatus requires authored A/B molecular identities")
    network = ReactionNetworkSpec(
        "finite-sorbent-contact",
        tuple(
            SpeciesSpec(
                f"{key}_ads" if bound else key,
                component_formulas[key],
                phase="surface" if bound else "liquid",
                metadata={"synthetic": True},
            )
            for key in spec.component_ids
            for bound in (False, True)
        ),
        (),
    )
    states, models, components = {}, {}, []
    for name, volume, capacity in (
        ("reactor", 0.02, 0.1),
        ("feed", 0.06, 0.1),
        ("waste", 0.0, 0.1),
        ("wash_supply", 0.03, 0.1),
    ):
        owner = "mobile-bed" if mobile else f"{name}:bed"
        slot = make_sorbent_slot(
            spec,
            owner_id=owner,
            location_id=name,
            mass_kg=0.01 if name == "reactor" else 0.0,
            fixed=not mobile,
            phase_id="liquid" if mobile else None,
            initial_loading_mol_per_kg=({"A": 0.3, "B": 0.2} if preloaded else None),
            bound_species_ids={key: f"{key}_ads" for key in spec.component_ids},
        )
        heater_id = f"{name}:heater"
        equipment = {
            heater_id: EquipmentRecord(
                heater_id,
                "thermal_device",
                name,
                batch_reset="preserve",
                settings={
                    "thermal_device": ThermalDeviceSpec(
                        maximum_power_W=5, setpoint_K=350, minimum_setpoint_K=280, enabled=False
                    ).to_dict()
                },
            )
        }
        if not mobile:
            equipment[owner] = EquipmentRecord(
                owner, "stationary_sorbent", name, batch_reset="preserve"
            )
        amounts = {"A": volume, "B": volume * 0.5}
        if name == "wash_supply":
            amounts = {"A": 0.0, "B": 0.0}
        states[name] = WorldState(
            amounts,
            volume,
            298.15,
            101325.0,
            "liquid",
            name,
            phases=PhaseLedger(
                {
                    "liquid": PhaseRecord(
                        "liquid",
                        name,
                        "liquid",
                        volume,
                        amounts,
                        selected=True,
                        settled=True,
                        solvents=SolventInventory.pure(0, volume),
                    )
                }
            ),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-liquid-contact",
                        capacity,
                        430,
                        500000,
                        ("liquid",),
                        298.15,
                        101325.0,
                    )
                }
            ),
            equipment=EquipmentLedger({} if name == "wash_supply" else equipment),
            state_slots=StateSlots(() if name == "wash_supply" else (slot,)),
            units={
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
                "cost": "currency",
                "risk": "risk",
            },
        )
        if name == "wash_supply":
            components.append({"id": f"{name}:material", "kind": "material_input",
                               "vessel": name, "parameters": {}})
            continue
        models[name] = ThermalSystemSpec(
            thermodynamics,
            pressure_law="fixed_pressure",
            environment_temperature_K=298.15,
            loss_conductance_W_K=0.1,
            minimum_temperature_K=280,
            maximum_temperature_K=430,
            stop_temperature_K=425,
            condensed_background=CondensedBackgroundSpec(600.0, 298.15, 0, "wash_supply"),
        )
        components += [
            {"id": f"{name}:material", "kind": "material_input", "vessel": name, "parameters": {}},
            {
                "id": f"{name}:thermal",
                "kind": "thermal",
                "vessel": name,
                "parameters": {"temperature_range_K": [280, 425], "duration_range_s": [1, 3600]},
            },
            {
                "id": f"{name}:sorbent",
                "kind": "sorbent_contact",
                "vessel": name,
                "parameters": {"sorbent_id": spec.sorbent_id},
            },
            {
                "id": f"{name}:probe",
                "kind": "observation",
                "vessel": name,
                "parameters": {"instruments": ["phase_state"]},
            },
        ]
    physics = CompositionPhysics(
        network, states, thermal_models=models, sorbent_models={spec.sorbent_id: spec}
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "finite-sorbent-research",
            "components": components,
            "vessels": [
                {"id": name, "capacity_L": state.vessels.vessels[name].max_volume_L}
                for name, state in states.items()
            ],
            "connections": [
                {
                    "id": f"{a}-to-{b}",
                    "unit": "mol",
                    "source": {"component": f"{a}:material", "port": "out"},
                    "target": {"component": f"{b}:material", "port": "in"},
                }
                for a, b in (("reactor", "waste"), ("feed", "reactor"), ("reactor", "feed"))
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study chemical response",
                "budget": 120,
                "operations": [
                    "dry",
                    "heat",
                    "wait",
                    "add_solvent",
                    "route_material",
                    "select_vessel",
                    "measure",
                    "pause_control",
                    "resume_control",
                    "terminate",
                ],
                "instruments": ["phase_state"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
