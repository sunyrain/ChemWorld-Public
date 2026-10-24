"""Two finite aqueous tanks with original finite-power heating and cooling."""

from collections.abc import Mapping

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.elements import hill_formula
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.membrane_distillation import (
    MembraneDistillationSpec,
    membrane_distillation_inventory,
    validate_membrane_thermal_devices,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.membrane_distillation import (
    MEMBRANE_DISTILLATION_TYPE,
    install_membrane_liquid,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def membrane_distillation_world_spec(
    spec: MembraneDistillationSpec,
    hot_amounts: Mapping[str, float],
    cold_amounts: Mapping[str, float],
    *,
    hot_temperature_K: float,
    cold_temperature_K: float,
    heater: ThermalDeviceSpec,
    cooler: ThermalDeviceSpec,
    budget: int = 100,
    shared_clock: bool = False,
) -> WorldCompositionSpec:
    """Local heat/pause controls, or shared passive wait with disabled utilities.

    Shared scheduling retains its original queue ownership: this builder does
    not expose queue operations or claim active shared thermal control.
    """
    validate_membrane_thermal_devices(heater, cooler)
    if shared_clock and (heater.enabled or cooler.enabled):
        raise ValueError("shared passive membrane world requires initially disabled utilities")
    initial = membrane_distillation_inventory(
        spec, hot_amounts, cold_amounts, hot_temperature_K, cold_temperature_K
    )
    names = ("reactor", spec.cold_vessel_id)
    if names[0] == names[1]:
        raise ValueError("thermal membrane requires distinct vessels")
    states = {}
    for i, (name, device, temperature) in enumerate(
        zip(names, (heater, cooler), (hot_temperature_K, cold_temperature_K), strict=True)
    ):
        equipment = {
            name + "-utility": EquipmentRecord(
                name + "-utility",
                "thermal_device",
                name,
                batch_reset="preserve",
                settings={"thermal_device": device.to_dict()},
            )
        }
        if i == 0:
            equipment["membrane"] = EquipmentRecord(
                "membrane",
                MEMBRANE_DISTILLATION_TYPE,
                name,
                batch_reset="preserve",
                settings={"physical_model": spec.to_dict()},
            )
        state = WorldState(
            initial.amounts_mol[i],
            0,
            temperature,
            spec.pressures_Pa[i],
            "liquid",
            name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger(
                {
                    name: VesselRecord(
                        name,
                        "finite-thermal-membrane-tank",
                        spec.capacities_L[i],
                        spec.temperature_bounds_K[1],
                        max(spec.pressures_Pa) * 1.1,
                        ("liquid",),
                        temperature,
                        spec.pressures_Pa[i],
                    )
                }
            ),
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
        states[name] = install_membrane_liquid(
            state, spec, i, initial.amounts_mol[i], initial.enthalpies_J[i]
        )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "declared-direct-contact-membrane",
            tuple(
                SpeciesSpec(
                    s.species_id, hill_formula(dict(s.composition)), charge=s.charge, phase="liquid"
                )
                for s in spec.properties.database.species
            ),
            (),
        ),
        states,
        lab_clock=LabClockSpec(
            heater_ids=tuple(n + "-utility" for n in names), analyzers=(), vessel_capacity=2
        )
        if shared_clock
        else None,
    )
    components = [
        {"id": "separator", "kind": "ro_membrane", "vessel": "reactor", "parameters": {}},
        {"id": "cold", "kind": "material_input", "vessel": names[1], "parameters": {}},
    ]
    if shared_clock:
        components.append(
            {
                "id": "clock",
                "kind": "lab_schedule",
                "vessel": "reactor",
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
        )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": "declared-direct-contact-membrane",
            "components": components,
            "vessels": [
                {"id": n, "capacity_L": c} for n, c in zip(names, spec.capacities_L, strict=True)
            ],
            "connections": [
                {
                    "id": "membrane-water",
                    "unit": "mol",
                    "source": {"component": "separator", "port": "out"},
                    "target": {"component": "cold", "port": "in"},
                }
            ],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite thermally driven water separation",
                "budget": budget,
                "operations": ["wait", "terminate"]
                if shared_clock
                else ["heat", "wait", "pause_control", "resume_control", "terminate"],
                "instruments": [],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
