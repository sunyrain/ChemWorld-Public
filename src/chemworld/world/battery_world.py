"""A synthetic battery fixture; physics is independent of task completion."""

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
from chemworld.physchem.battery_degradation import SEI_FORMULAS
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.battery import BATTERY_TYPE, initial_battery_slot
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def battery_world_spec(spec=None, *, budget=80, thermal_device=None):
    spec = spec or BatterySPMSpec()
    vessel = "reactor"
    record = EquipmentRecord(
        "battery",
        BATTERY_TYPE,
        vessel,
        batch_reset="preserve",
        settings={
            "physical_model": spec.to_dict(),
            "electrical_control": {
                "limits_A": [-spec.maximum_current_A, spec.maximum_current_A],
                "limits_V": [spec.minimum_voltage_V, spec.maximum_voltage_V],
                "staged_control": True,
                "program": ScalarControlProgram(((0, 0),), "A").to_dict(),
            },
        },
    )
    equipment = {record.equipment_id: record}
    operations = [
        "configure_electrical_control",
        "queue_electrical_stage",
        "wait",
        "measure",
        "terminate",
    ]
    if thermal_device is not None:
        equipment["heater"] = EquipmentRecord(
            "heater",
            "thermal_device",
            vessel,
            settings={"thermal_device": thermal_device.to_dict()},
        )
        operations.extend(thermal_device.supported_operations)
    state = WorldState(
        dict.fromkeys(spec.inventory_species.values(), 0.0),
        spec.volume_l,
        spec.temperature_K,
        101325,
        "solid",
        vessel,
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        },
        equipment=EquipmentLedger(equipment),
        vessels=VesselLedger(
            {
                vessel: VesselRecord(
                    vessel, "battery-fixture", 0.1, 400, 5e5, (), spec.temperature_K, 101325
                )
            }
        ),
        # This phase describes only occupied housing volume. All active lithium
        # is equipment-owned below; material reset cannot clone its initial SOC.
        phases=PhaseLedger({"housing": PhaseRecord("housing", vessel, "solid", spec.volume_l, {})}),
        state_slots=StateSlots((initial_battery_slot(record, spec),)),
    )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "battery-lithium",
            tuple(
                SpeciesSpec(name, formula)
                for name, formula in {
                    "Li": "Li",
                    **({} if spec.sei is None else SEI_FORMULAS),
                    **({} if spec.electrolyte is None else {ELECTROLYTE_SPECIES: "LiPF6"}),
                }.items()
            ),
            (),
        ),
        {vessel: state},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": ("native-battery-mpm"
                               if getattr(spec, "model_family", None) == "mpm-isothermal-v1"
                               else "native-battery-dfn"
                               if getattr(spec, "model_family", None) == "dfn-isothermal-v1"
                               else "native-battery-spm" if spec.electrolyte is None
                               else "native-battery-spme"),
            "components": [
                {"id": "cell", "kind": "battery_spm", "vessel": vessel, "parameters": {}},
                {
                    "id": "meter",
                    "kind": "observation",
                    "vessel": vessel,
                    "parameters": {"instruments": ["battery_meter"]},
                },
            ],
            "vessels": [{"id": vessel, "capacity_L": 0.1}],
            "connections": [],
            "physics": physics.to_dict(),
            "task": {
                "objective": "study finite battery response",
                "budget": budget,
                "operations": list(dict.fromkeys(operations)),
                "instruments": ["battery_meter"],
                "termination": "explicit-stop-or-budget",
            },
        }
    )
