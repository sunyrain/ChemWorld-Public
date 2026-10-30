"""Bind original finite-bed owners to the existing physical advance coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite
from typing import Any, cast

from chemworld.foundation.lifecycle import LifecycleTransition
from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.fixed_bed_thermal_specs import FixedBedThermalSpec
from chemworld.runtime.fixed_bed_thermal import advance_thermal_fixed_bed
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.world.composition import WorldCompositionSpec

THERMAL_BED_MODEL = "thermal_fixed_bed_model"
THERMAL_BED_CONTROL = "thermal_fixed_bed_control"


def bed_record(state: WorldState) -> EquipmentRecord | None:
    records = [record for record in cast(EquipmentLedger, state.equipment).equipment.values()
               if record.attached_vessel_id == state.vessel_id
               and THERMAL_BED_MODEL in record.settings]
    if len(records) > 1:
        raise ValueError("one thermal-bed strategy cannot claim multiple apparatus owners")
    return records[0] if records else None


def controlled_spec(spec: FixedBedThermalSpec, controls: Mapping[str, Any]) -> FixedBedThermalSpec:
    fields = {"inlet_phase_id", "outlet_phase_id", "jacket_enabled", "jacket_setpoint_K"}
    if set(controls) != fields:
        raise ValueError("bed controls must retain their complete original layout")
    boundaries = set(spec.gas_phase_ids)-set(spec.bed.cell_phase_ids)
    inlet, outlet = controls["inlet_phase_id"], controls["outlet_phase_id"]
    if (any(value is not None and (not isinstance(value, str) or value not in boundaries)
            for value in (inlet, outlet)) or (inlet is not None and inlet == outlet)):
        raise ValueError("bed controls require distinct actual finite boundary compartments")
    enabled, target = controls["jacket_enabled"], controls["jacket_setpoint_K"]
    if not isinstance(enabled, bool):
        raise ValueError("thermal equipment enable must be boolean")
    if spec.jacket is None:
        if enabled or target is not None:
            raise ValueError("controls cannot create an undeclared thermal device")
        return spec
    if type(target) not in {int, float} or not isfinite(target):
        raise ValueError("an installed thermal device needs a finite setpoint")
    return replace(spec, jacket=replace(spec.jacket, enabled=enabled, setpoint_K=target))


def bed_boundary_choices(state: WorldState, _validator: Any = None) -> tuple[str, ...]:
    record = bed_record(state)
    if record is None:
        return ()
    spec = FixedBedThermalSpec.from_dict(record.settings[THERMAL_BED_MODEL])
    boundaries = set(spec.gas_phase_ids)-set(spec.bed.cell_phase_ids)
    if "closed" in boundaries:
        raise ValueError("closed is reserved for a disconnected bed boundary")
    return ("closed", *sorted(boundaries))


class ThermalFixedBedRuntime:
    supported_operations = ("wait", "configure_thermal_bed")

    def __init__(self, composition: WorldCompositionSpec | None) -> None:
        self.composition = composition

    def _original(self, vessel_id: str) -> EquipmentRecord | None:
        physics = None if self.composition is None else self.composition.physics
        original = None if physics is None else physics.initial_states.get(vessel_id)
        return None if original is None else bed_record(original)

    def supports(self, state: WorldState) -> bool:
        return bed_record(state) is not None or self._original(state.vessel_id) is not None

    def _bound(self, state: WorldState) -> tuple[EquipmentRecord, FixedBedThermalSpec]:
        original, record = self._original(state.vessel_id), bed_record(state)
        if (original is None or record is None or record.equipment_id != original.equipment_id
                or record.equipment_type != "fixed_bed" or record.batch_reset != "preserve"):
            raise ValueError("thermal fixed bed must retain its original physical declaration")
        try:
            spec = FixedBedThermalSpec.from_dict(record.settings[THERMAL_BED_MODEL])
            declared = FixedBedThermalSpec.from_dict(original.settings[THERMAL_BED_MODEL])
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            raise ValueError("invalid original thermal-bed model declaration") from exc
        if spec.to_dict() != declared.to_dict():
            raise ValueError("thermal fixed bed must retain its original physical declaration")
        if record.equipment_id != spec.bed.bed_id:
            raise ValueError("bed declaration does not belong to its original equipment owner")
        controls = record.settings.get(THERMAL_BED_CONTROL)
        if not isinstance(controls, Mapping):
            raise ValueError("thermal bed must retain its declared controls")
        return record, controlled_spec(spec, controls)

    def validate_state(self, state: WorldState) -> None:
        record, spec = self._bound(state)
        controls = record.settings[THERMAL_BED_CONTROL]
        result = advance_thermal_fixed_bed(
            state.replace(inactive_vessels={}, terminated=False, quenched=False), spec, 0.,
            inlet_phase_id=controls["inlet_phase_id"], outlet_phase_id=controls["outlet_phase_id"])
        if not result.success:
            raise ValueError(f"invalid original thermal-bed state: {result.diagnostics}")

    def validate_world(self) -> None:
        physics = None if self.composition is None else self.composition.physics
        if self.composition is None or physics is None:
            raise ValueError("thermal-bed runtime needs original composition physics")
        installed = {v for v, state in physics.initial_states.items()
                     if bed_record(state) is not None}
        declared = {component.vessel for component in self.composition.components
                    if component.kind == "thermal_fixed_bed"}
        if installed != declared:
            raise ValueError("thermal-bed components must identify their actual installed owners")
        for vessel in installed:
            self.validate_state(physics.initial_states[vessel])

    @property
    def preserved_source_vessels(self) -> set[str]:
        physics = None if self.composition is None else self.composition.physics
        return set() if physics is None else {
            vessel for vessel, state in physics.initial_states.items()
            if bed_record(state) is not None}

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def reconcile_initialization_from(
        self, transition: LifecycleTransition, previous: WorldState | None,
    ) -> LifecycleTransition:
        protected = self.protected_lifecycle_vessels
        affected = set(transition.receipt.get("vessels", ()))
        identity = transition.receipt.get("equipment_id")
        if identity is not None:
            for vessel in protected:
                owner = self._original(vessel)
                if owner is not None and owner.equipment_id == identity:
                    affected.add(vessel)
        if protected & affected:
            raise ValueError("a finite thermal bed requires a complete new experiment")
        return transition

    def advance(self, state: WorldState, action: Mapping[str, Any], *, heat: bool = False,
                jacket_program: Any = None) -> ModelProviderResult:
        try:
            if heat or jacket_program is not None:
                raise ValueError("bed heater controls belong to its declared finite thermal device")
            record, spec = self._bound(state)
            controls = record.settings[THERMAL_BED_CONTROL]
            result = advance_thermal_fixed_bed(
                state.replace(inactive_vessels={}), spec, float(action["duration_s"]),
                inlet_phase_id=controls["inlet_phase_id"],
                outlet_phase_id=controls["outlet_phase_id"])
            if result.state is not None:
                result = replace(result, state=result.state.replace(
                    inactive_vessels=state.inactive_vessels))
            return result
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            return ModelProviderResult(
                {}, diagnostics={"error": str(exc)}, success=False, outcome="rejected",
                failure_reason="thermal_fixed_bed_binding_failed")


def configure_thermal_bed(state: WorldState, action: Mapping[str, Any]) -> ModelProviderResult:
    """Change actual connections/finite utility settings without replacing stock."""
    try:
        record = bed_record(state)
        if record is None or state.terminated or state.quenched:
            raise ValueError("bed control needs its active original apparatus")
        spec = FixedBedThermalSpec.from_dict(record.settings[THERMAL_BED_MODEL])
        controls = dict(record.settings[THERMAL_BED_CONTROL])
        for field, key in (("bed_inlet_phase", "inlet_phase_id"),
                           ("bed_outlet_phase", "outlet_phase_id")):
            value = action[field]
            if not isinstance(value, str):
                raise ValueError("bed boundary selection must name a declared compartment")
            controls[key] = None if value == "closed" else value
        if "bed_heater_enabled" in action:
            if action["bed_heater_enabled"] not in (0, 1):
                raise ValueError("bed heater enable must be zero or one")
            controls["jacket_enabled"] = bool(action["bed_heater_enabled"])
        if "bed_target_temperature_K" in action:
            controls["jacket_setpoint_K"] = action["bed_target_temperature_K"]
        current = controlled_spec(spec, controls)
        validation = advance_thermal_fixed_bed(
            state.replace(inactive_vessels={}), current, 0.,
            inlet_phase_id=controls["inlet_phase_id"], outlet_phase_id=controls["outlet_phase_id"])
        if not validation.success:
            raise ValueError(
                f"bed control leaves the actual operating domain: {validation.diagnostics}")
        equipment = cast(EquipmentLedger, state.equipment)
        updated = replace(record, settings={**record.settings, THERMAL_BED_CONTROL: controls})
        candidate = state.replace(equipment=EquipmentLedger({
            **equipment.equipment, record.equipment_id: updated}))
        return ModelProviderResult(
            {"bed_controls": controls}, diagnostics={"bed_control_only": True},
            state=candidate, actual_duration_s=0.)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return ModelProviderResult(
            {}, diagnostics={"error": str(exc)}, success=False, outcome="rejected",
            failure_reason="thermal_fixed_bed_control_failed")
