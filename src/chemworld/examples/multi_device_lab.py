"""Four authored bottles, two finite heaters and two independent chromatographs."""

from copy import deepcopy
from dataclasses import replace

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.chromatography_methods import EmpiricalChromatographyAnalyteSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime.chromatography import initial_memory
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.lab_world import shared_lab_world


def multi_device_lab_world(*, noise_sd=1e-5):
    """Synthetic gains/calibration are authored, not fitted from hidden readouts."""
    base = shared_lab_world()
    payload = base.to_dict()
    physics = payload["physics"]
    source = base.physics.initial_states["reactor"]
    analyzer = source.equipment.equipment["instrument:programmed_hplc"]
    for name in tuple(physics["initial_states"]):
        local = WorldState.from_dict(physics["initial_states"][name])
        removed = {r.equipment_id for r in local.equipment.equipment.values()
                   if r.equipment_type in {"instrument", "thermal_device"}}
        physics["initial_states"][name] = local.replace(
            equipment=EquipmentLedger({
                key: record for key, record in local.equipment.equipment.items()
                if key not in removed
            }),
            state_slots=StateSlots(tuple(slot for slot in local.state_slots.slots
                                         if slot.spec.owner_id not in removed)),
        ).to_dict()
    for index, name in enumerate(("vial-c", "vial-d"), 1):
        physics["initial_states"][name] = WorldState(
            dict.fromkeys(source.species_amounts, (index + 1) * 1e-5),
            .01 * (index + 1), 298.15, 101325, "liquid", name, units=source.units,
            equipment=EquipmentLedger({}),
        ).to_dict()
        physics["models"]["thermal"]["bindings"][name] = deepcopy(
            physics["models"]["thermal"]["bindings"]["reactor"],
        )
        payload["vessels"].append({"id": name, "capacity_L": .1})
        payload["components"] += [
            {"id": name + ":" + kind, "kind": kind, "vessel": name,
             "parameters": {"instruments": ["programmed_hplc"]} if kind == "observation" else {}}
            for kind in ("thermal", "observation", "lab_schedule")
        ]
    for index, name in enumerate(("reactor", "sample")):
        local = WorldState.from_dict(physics["initial_states"][name])
        heater_id, analyzer_id = ("heater-a", "hplc-a") if index == 0 else ("heater-b", "hplc-b")
        gain = 1.0 if index == 0 else 2.0
        settings = deepcopy(analyzer.settings)
        settings["instrument_id"] = "programmed_hplc"
        settings["chromatography"].update(initial_gain=gain, noise_sd=noise_sd)
        settings["calibration"] = [
            {"detector_id": f"peak_{i}", "slope": gain * (1 - settings[
                "chromatography"]["carryover_fraction"]), "intercept": 0.0,
             "provenance_id": "authored-synthetic-device-calibration-not-a-fit"}
            for i in (1, 2, 3)
        ]
        analytes = tuple(EmpiricalChromatographyAnalyteSpec(**item)
                         for item in settings["chromatography"]["analytes"])
        local = local.replace(
            equipment=EquipmentLedger({
                **local.equipment.equipment,
                heater_id: EquipmentRecord(heater_id, "thermal_device", name, settings={
                    "thermal_device": ThermalDeviceSpec(
                        conductance_W_K=.05, maximum_power_W=3,
                        calibration_factor=.9 if index == 0 else 1.1, enabled=False,
                    ).to_dict(),
                }),
                analyzer_id: replace(analyzer, equipment_id=analyzer_id, attached_vessel_id=name,
                                     settings=settings),
            }),
            state_slots=StateSlots((*local.state_slots.slots, initial_memory(
                "programmed_hplc", name, analytes, device_id=analyzer_id,
            ))),
        )
        physics["initial_states"][name] = local.to_dict()
    for component in payload["components"]:
        if component["kind"] == "lab_schedule":
            component["parameters"]["duration_range_s"] = [1e-6, 14400]
    physics["lab_clock"] = LabClockSpec(
        heater_ids=("heater-a", "heater-b"),
        analyzers={"hplc-a": "programmed_hplc", "hplc-b": "programmed_hplc"},
        vessel_capacity=4,
    ).to_dict()
    payload["composition_id"] = "synthetic-two-heaters-two-analyzers"
    payload["task"]["budget"] = 160
    return WorldCompositionSpec.from_dict(payload)
