"""A battery world consuming its explicit physical model and equipment state."""

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
from chemworld.runtime.battery import battery_equipment_type, initial_battery_slot
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def battery_world_spec(spec=None, *, budget=80, thermal_device=None):
    spec = spec or BatterySPMSpec()
    if getattr(spec, "parameter_binding", None) is not None and thermal_device is not None:
        raise ValueError("source SPM thermal-device coupling is unsupported")
    plating = getattr(spec, "plating", None) is not None
    if plating and thermal_device is not None:
        raise ValueError("plating thermal coupling is unsupported")
    split_ocvr = getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1"
    if split_ocvr and thermal_device is not None:
        raise ValueError("Split OCV-R thermal coupling is unsupported")
    lead_acid = getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1"
    if lead_acid and thermal_device is not None:
        raise ValueError("lead-acid LOQS thermal coupling is unsupported")
    newman_tobias = getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1"
    if newman_tobias and thermal_device is not None:
        raise ValueError("Newman-Tobias thermal coupling is unsupported")
    msmr = getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1"
    if msmr and thermal_device is not None:
        raise ValueError("MSMR thermal coupling is unsupported")
    half_cell_dfn = getattr(spec, "model_family", None) == "half-cell-dfn-isothermal-v1"
    if half_cell_dfn and thermal_device is not None:
        raise ValueError("DFN half-cell thermal coupling is unsupported")
    half_cell = getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1"
    if half_cell and thermal_device is not None:
        raise ValueError("finite half-cell thermal coupling is unsupported")
    composite_dfn = getattr(spec, "model_family", None) == "composite-dfn-isothermal-v1"
    if composite_dfn and thermal_device is not None:
        raise ValueError("composite DFN thermal coupling is unsupported")
    collector_spm = getattr(spec, "model_family", None) == "collector-spm-isothermal-v1"
    if collector_spm and thermal_device is not None:
        raise ValueError("collector SPM thermal coupling is unsupported")
    composite = getattr(spec, "model_family", None) == "composite-spm-isothermal-v1"
    if composite and thermal_device is not None:
        raise ValueError("composite thermal coupling is unsupported")
    spatial = getattr(spec, "model_family", None) == "spm-spatial-thermal-v1"
    if spatial and thermal_device is not None:
        raise ValueError("spatial thermal cell requires declared surface contacts; "
                         "external device unsupported")
    vessel = "reactor"
    record = EquipmentRecord(
        "battery",
        battery_equipment_type(spec),
        vessel,
        batch_reset="preserve",
        settings={
            "physical_model": spec.to_dict(),
            "electrical_control": {
                "limits_A": [-spec.maximum_current_A, spec.maximum_current_A],
                "limits_V": [spec.minimum_voltage_V, spec.maximum_voltage_V],
                "staged_control": True,
                **({"supported_units": list(spec.supported_control_units)} if split_ocvr else {}),
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
        # This phase describes only occupied housing volume. All cell inventory
        # is equipment-owned below; material reset cannot clone its initial SOC.
        phases=PhaseLedger({"housing": PhaseRecord("housing", vessel, "solid", spec.volume_l, {})}),
        state_slots=StateSlots((initial_battery_slot(record, spec),)),
    )
    physics = CompositionPhysics(
        ReactionNetworkSpec(
            "battery-lead-acid" if lead_acid else "battery-lithium",
            tuple(
                SpeciesSpec(name, formula)
                for name, formula in (spec.species_formulas if lead_acid else {
                    "Li": "Li",
                    **({} if spec.sei is None else SEI_FORMULAS),
                    **({} if spec.electrolyte is None else {ELECTROLYTE_SPECIES: "LiPF6"}),
                }).items()
            ),
            (),
        ),
        {vessel: state},
    )
    return WorldCompositionSpec.from_dict(
        {
            "schema_version": "chemworld-world-composition-0.1",
            "composition_id": ("native-battery-mpm-plating"
                               if plating and getattr(spec, "model_family", None)
                               == "mpm-isothermal-v1"
                               else "native-battery-spm-plating" if plating
                               else "native-battery-split-ocvr"
                               if split_ocvr
                               else "native-battery-lead-acid"
                               if lead_acid
                               else "native-battery-newman-tobias"
                               if newman_tobias
                               else "native-battery-msmr"
                               if msmr
                               else "native-battery-spatial-thermal"
                               if spatial
                               else "native-battery-composite-dfn"
                               if composite_dfn
                               else "native-battery-collector-spm"
                               if collector_spm
                               else "native-battery-composite"
                               if composite
                               else "native-battery-half-cell-dfn"
                               if half_cell_dfn
                               else "native-battery-half-cell"
                               if half_cell
                               else "native-battery-mpm"
                               if getattr(spec, "model_family", None) == "mpm-isothermal-v1"
                               else "native-battery-dfn"
                               if getattr(spec, "model_family", None) == "dfn-isothermal-v1"
                               else "native-battery-spm" if spec.electrolyte is None
                               else "native-battery-spme"),
            "components": [
                {"id": "cell", "kind": "battery_cell"
                 if lead_acid or split_ocvr else "battery_spm",
                 "vessel": vessel, "parameters": {}},
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
