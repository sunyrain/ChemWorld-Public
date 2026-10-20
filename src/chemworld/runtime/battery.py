"""Equipment-owned SPM inventory and one declared shared advance strategy."""

from dataclasses import replace

from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.battery_models import (
    BATTERY_COORDINATES,
    BatterySPMSpec,
    advance_battery,
    battery_control_current,
    battery_readout,
)
from chemworld.physchem.coupled_system import CoupledEvent, CoupledState
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

BATTERY_TYPE = "single_particle_battery"
BATTERY_STATE_TYPE = "battery-lithium-and-charge-v1"


def battery_record(state):
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == BATTERY_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("this battery system requires exactly one installed cell")
    return records[0] if records else None


def initial_battery_slot(record, spec):
    initial = spec.initial_state()
    declaration = StateSlotSpec(
        f"{record.equipment_id}:inventory",
        "equipment",
        record.equipment_id,
        BATTERY_STATE_TYPE,
        (
            *(
                StateFieldSpec(key, "mol", species_id=species)
                for key, species in spec.inventory_species.items()
            ),
            StateFieldSpec("discharged_charge_C", "C", minimum=None),
            StateFieldSpec("delivered_work_J", "J", minimum=None),
        ),
        mechanism_id="synthetic-SPM-v1",
        batch_reset="preserve",
    )
    return StateSlot(
        declaration, record.attached_vessel_id, {**initial.amounts_mol, **initial.coordinates}
    )


def unpack_battery(state):
    record = battery_record(state)
    if record is None:
        raise ValueError("no SPM battery is installed")
    spec = BatterySPMSpec(**record.settings["physical_model"])
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.owner_kind == "equipment"
        and s.spec.owner_id == record.equipment_id
        and s.spec.state_type == BATTERY_STATE_TYPE
    ]
    if len(slots) != 1 or slots[0].spec != initial_battery_slot(record, spec).spec:
        raise ValueError("battery must retain its declared inventory layout and ownership")
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("battery requires its connected electrical source")
    slot = slots[0]
    physical = CoupledState(
        {key: slot.values[key] for key in spec.inventory_keys},
        state.temperature_K,
        state.volume_L,
        state.ledger.time_s,
        coordinates={key: slot.values[key] for key in BATTERY_COORDINATES},
    )
    programme = ScalarControlProgram(**record.settings["electrical_control"]["program"])
    return record, slot, spec, physical, programme


class BatteryRuntime:
    supported_operations = ("configure_electrical_control", "queue_electrical_stage", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return battery_record(state) is not None

    def validate_world(self):
        declared = {c.vessel for c in self.composition.components if c.kind == "battery_spm"}
        installed = {
            v for v, s in self.composition.physics.initial_states.items() if self.supports(s)
        }
        if declared != installed:
            raise ValueError("battery components must match their installed equipment")
        for vessel in installed:
            self.validate(self.composition.physics.initial_states[vessel])

    def validate(self, state):
        _, _, spec, physical, programme = unpack_battery(state)
        _, device = connected_device(state, required=False)
        advance_battery(spec, physical, programme, 0, thermal_device=device)
        if self.composition is not None:
            physics = self.composition.physics
            if (
                any(
                    state.vessel_id in model
                    for model in (
                        physics.thermal_models,
                        physics.flash_models,
                        physics.liquid_models,
                        physics.column_models,
                        physics.concentration_models,
                    )
                )
                or physics.network.reactions
            ):
                raise ValueError("SPM cannot bind another kinetic or thermal strategy")

    def advance_controlled(self, state, action, control_stage, **kwargs):
        stage, index = control_stage
        _, _, spec, _, programme = unpack_battery(state)

        def distance(candidate):
            current = battery_control_current(spec, candidate, programme)
            reading = battery_readout(spec, candidate, current)
            if "stop_voltage_below_V" in stage:
                return reading.voltage_V - stage["stop_voltage_below_V"]
            if "stop_voltage_above_V" in stage:
                return stage["stop_voltage_above_V"] - reading.voltage_V
            return abs(current) - stage["stop_abs_current_A"]

        events = (
            (CoupledEvent(f"control:stage:{index}", distance),)
            if any(key.startswith("stop_") for key in stage)
            else ()
        )
        return self.advance(state, action, control_events=events, **kwargs)

    def advance(self, state, action, *, heat=False, jacket_program=None, control_events=()):
        if jacket_program is not None:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="battery_requires_shared_thermal_device",
            )
        thermal_record, device = connected_device(state, required=False)
        try:
            if heat:
                if device is None:
                    raise ValueError("heat requires a connected thermal device")
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
                records = dict(state.equipment.equipment)
                records[thermal_record.equipment_id] = replace(
                    thermal_record,
                    settings={**thermal_record.settings, "thermal_device": device.to_dict()},
                )
                state = state.replace(equipment=EquipmentLedger(records))
            _, _, spec, physical, programme = unpack_battery(state)
            if device is not None and (
                spec.thermal is None
                or not spec.thermal.minimum_temperature_K
                <= device.setpoint_K
                <= spec.thermal.maximum_temperature_K
                or (
                    device.enabled
                    and device.mode == "isothermal"
                    and abs(device.setpoint_K - state.temperature_K) > 1e-8
                )
            ):
                raise ValueError("thermal device exceeds the declared battery boundary")
        except ValueError as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_thermal_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            self.validate(state)
            _, slot, spec, physical, programme = unpack_battery(state)
            result = advance_battery(
                spec,
                physical,
                programme,
                float(action["duration_s"]),
                thermal_device=device,
                control_events=control_events,
            )
            final = result.final_state
            updated = replace(slot, values={**final.amounts_mol, **final.coordinates})
            candidate = state.replace(
                temperature_K=final.temperature_K,
                state_slots=StateSlots(
                    tuple(
                        updated if s.spec.slot_id == slot.spec.slot_id else s
                        for s in state.state_slots.slots
                    )
                ),
                process=replace(state.process, last_observation_record=None),
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    energy_deltas={
                        "flow_electrical_work_J": physical.coordinates["delivered_work_J"]
                        - final.coordinates["delivered_work_J"],
                        "energy_jacket_J": final.boundary_heat_J,
                        "heat_loss_J": final.heat_loss_J,
                        "heat_reaction_J": -final.chemical_energy_change_J,
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="battery_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            outcome=result.outcome,
            actual_duration_s=result.actual_duration_s,
            diagnostics={
                "solver": dict(result.solver_diagnostic),
                "stopping_event": result.stopping_event,
                "battery_conservation": dict(result.conservation_residuals),
                "energy_scope": "synthetic OCV chemical energy and lumped heat capacity"
                if spec.thermal
                else "isothermal SPM; full cell heat balance not modelled",
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "electrical_control": {
                    "charge_C": final.coordinates["discharged_charge_C"],
                    "work_J": final.coordinates["delivered_work_J"],
                    "unit": programme.unit,
                    "target": programme.value_at(final.time_s),
                    "current_A": battery_control_current(spec, final, programme),
                    "voltage_V": battery_readout(
                        spec, final, battery_control_current(spec, final, programme)
                    ).voltage_V,
                    "current_limited": programme.unit == "V"
                    and abs(battery_control_current(spec, final, programme))
                    == spec.maximum_current_A,
                },
            },
        )
