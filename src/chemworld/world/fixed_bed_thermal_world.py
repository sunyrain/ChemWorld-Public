"""Author an actual finite apparatus on the original composition and clocks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, VesselLedger
from chemworld.physchem.fixed_bed_thermal_specs import FixedBedThermalSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.thermal_fixed_bed_world import (
    THERMAL_BED_CONTROL,
    THERMAL_BED_MODEL,
    controlled_spec,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def thermal_fixed_bed_world(
    spec: FixedBedThermalSpec, initial_state: WorldState, formulas: Mapping[str, str], *,
    inlet_phase_id: str | None = None, outlet_phase_id: str | None = None,
    shared_clock: bool = True, budget: int = 100,
) -> WorldCompositionSpec:
    """Declare fresh original stock; existing experiments resume by session snapshots."""
    if (set(formulas) != set(spec.bed.component_ids) or not isinstance(shared_clock, bool)
            or initial_state.inactive_vessels or initial_state.lab_clock is not None
            or initial_state.ledger.time_s != 0
            or any(time != 0 for time in initial_state.vessel_elapsed_s.values())):
        raise ValueError("thermal-bed authoring needs complete formulas and fresh original stock")
    if "closed" in set(spec.gas_phase_ids)-set(spec.bed.cell_phase_ids):
        raise ValueError("closed is reserved for a disconnected finite boundary")
    native_units = {"amount": "mol", "volume": "L", "temperature": "K",
                    "pressure": "Pa", "time": "s"}
    if any(initial_state.units.get(key, unit) != unit for key, unit in native_units.items()):
        raise ValueError("thermal-bed authoring cannot relabel a conflicting physical unit")
    device = spec.jacket
    controls = {"inlet_phase_id": inlet_phase_id, "outlet_phase_id": outlet_phase_id,
                "jacket_enabled": False if device is None else device.enabled,
                "jacket_setpoint_K": None if device is None else device.setpoint_K}
    controlled_spec(spec, controls)
    equipment = cast(EquipmentLedger, initial_state.equipment)
    record = equipment.equipment[spec.bed.bed_id]
    model = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    record = replace(record, settings={**record.settings, THERMAL_BED_MODEL: model,
                                      THERMAL_BED_CONTROL: controls})
    state = initial_state.replace(
        units={**initial_state.units, **native_units},
        equipment=EquipmentLedger({**equipment.equipment, record.equipment_id: record}))
    vessel = state.vessel_id
    hardware = cast(VesselLedger, state.vessels).vessels[vessel]
    network = ReactionNetworkSpec(
        f"{spec.bed.bed_id}:finite-thermal-bed",
        tuple(SpeciesSpec(key, formulas[key], phase="gas") for key in spec.bed.component_ids), ())
    components = [{"id": spec.bed.bed_id, "vessel": vessel,
                   "kind": "thermal_fixed_bed", "parameters": {}}]
    if shared_clock:
        components.append({"id": f"{vessel}:clock", "vessel": vessel, "kind": "lab_schedule",
                           "parameters": {"duration_range_s": [1e-6, 14400]}})
    physics = CompositionPhysics(
        network, {vessel: state}, lab_clock=LabClockSpec(
            heater_ids=(), analyzers=(), vessel_capacity=1) if shared_clock else None,
        advance_bindings={vessel: "finite-thermal-fixed-bed"})
    clock_kind = "shared" if shared_clock else "local"
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": f"{spec.bed.bed_id}:finite-thermal-bed:{clock_kind}",
        "vessels": [{"id": vessel, "capacity_L": hardware.max_volume_L}],
        "components": components, "connections": [], "physics": physics.to_dict(),
        "task": {"objective": "study the declared finite gas-solid thermal apparatus",
                 "budget": budget, "operations": ["configure_thermal_bed", "wait", "terminate"],
                 "instruments": []},
    })
