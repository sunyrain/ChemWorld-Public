"""Equipment-owned electrochemical inventory and one shared advance strategy."""

from dataclasses import replace

from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.battery_models import (
    advance_battery,
    battery_control_current,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.coupled_system import CoupledEvent, CoupledState
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

BATTERY_TYPE = "single_particle_battery"
BATTERY_STATE_TYPE = "battery-lithium-and-charge-v1"
LEAD_ACID_BATTERY_TYPE = "electrochemical_battery"
LEAD_ACID_STATE_TYPE = "battery-lead-acid-inventory-and-charge-v1"


def battery_equipment_type(spec):
    return (LEAD_ACID_BATTERY_TYPE
            if getattr(spec, "model_family", None) in {
                "lead-acid-loqs-isothermal-v1", "split-ocvr-isothermal-v1"}
            else BATTERY_TYPE)


def battery_state_type(spec):
    return (LEAD_ACID_STATE_TYPE
            if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1"
            else BATTERY_STATE_TYPE)


def battery_record(state):
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type in {BATTERY_TYPE, LEAD_ACID_BATTERY_TYPE}
        and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("this battery system requires exactly one installed cell")
    return records[0] if records else None


def initial_battery_slot(record, spec):
    if record.equipment_type != battery_equipment_type(spec):
        raise ValueError("battery chemistry must match its declared equipment type")
    initial = spec.initial_state()
    declaration = StateSlotSpec(
        f"{record.equipment_id}:inventory",
        "equipment",
        record.equipment_id,
        battery_state_type(spec),
        (
            *(
                StateFieldSpec(key, "mol", species_id=species)
                for key, species in spec.inventory_species.items()
            ),
            StateFieldSpec("discharged_charge_C", "C", minimum=None),
            StateFieldSpec("delivered_work_J", "J", minimum=None),
            *(StateFieldSpec(key, "dimensionless", mixing="equal", minimum=-1.0, maximum=1.0)
              for key in spec.hysteresis_keys),
            *(StateFieldSpec(key, "K", mixing="equal", minimum=None)
              for key in getattr(spec, "thermal_coordinate_keys", ())),
        ),
        mechanism_id=spec.model_id,
        batch_reset="preserve",
    )
    return StateSlot(
        declaration, record.attached_vessel_id, {**initial.amounts_mol, **initial.coordinates}
    )


def unpack_battery(state):
    record = battery_record(state)
    if record is None:
        raise ValueError("no electrochemical battery is installed")
    spec = battery_spec_from_dict(record.settings["physical_model"])
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.owner_kind == "equipment"
        and s.spec.owner_id == record.equipment_id
        and s.spec.state_type == battery_state_type(spec)
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
        coordinates={key: slot.values[key] for key in spec.coordinate_keys},
    )
    programme = ScalarControlProgram(**record.settings["electrical_control"]["program"])
    return record, slot, spec, physical, programme


class BatteryRuntime:
    supported_operations = ("configure_electrical_control", "queue_electrical_stage", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return battery_record(state) is not None

    def temperature_bounds_k(self, state):
        _, _, spec, _, _ = unpack_battery(state)
        # A thickness model has real temperatures but no declared contact
        # for the laboratory's movable lumped heater.
        return None if (spec.thermal is None or getattr(spec, "model_family", None)
                        == "spm-spatial-thermal-v1") else (
            spec.thermal.minimum_temperature_K, spec.thermal.maximum_temperature_K)

    def reconcile_initialization(self, transition):
        # All cell inventory and any declared heat belong to the physical equipment;
        # replacing surrounding material/housing cannot cool or recharge it.
        state = transition.state
        locals_ = {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}
        changed = {}
        scope = transition.receipt["scope"]
        introduced = transition.receipt.get("introduced_device", {})
        for identity, local in tuple(locals_.items()):
            record = battery_record(local)
            if record is None:
                continue
            if (scope == "equipment" and introduced.get("equipment_id") == record.equipment_id):
                spec = battery_spec_from_dict(record.settings["physical_model"])
                if local.volume_L != spec.volume_l:
                    raise ValueError("replacement battery must fit its declared housing volume")
                temperature = spec.temperature_K
                source = "introduced_device_model_initial_temperature"
            elif scope in {"material", "vessel"} and identity in transition.receipt["vessels"]:
                temperature = transition.receipt["vessels"][identity]["previous_temperature_K"]
                source = "retained_device"
            else:
                continue
            locals_[identity] = local.replace(temperature_K=temperature)
            changed[identity] = {"temperature_K": temperature, "temperature_source": source}
        if not changed:
            return transition
        active = locals_.pop(state.vessel_id)
        return replace(transition, state=active.replace(inactive_vessels=locals_),
                       receipt={**transition.receipt, "battery_initialization": changed})

    def validate_world(self):
        declared = {c.vessel for c in self.composition.components
                    if c.kind in {"battery_spm", "battery_cell"}}
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
                        physics.models_of("thermal"),
                        physics.models_of("flash"),
                        physics.models_of("liquid"),
                        physics.models_of("column"),
                        physics.models_of("concentration"),
                    )
                )
                or any(
                    component.vessel == state.vessel_id
                    and self.composition.modules.components()[component.kind].processes
                    for component in self.composition.components
                )
            ):
                raise ValueError("SPM cannot bind another kinetic or thermal strategy")

    # Snapshot restoration asks the shared coordinator for this contract.
    # Keep one physical validator for initialization, restore and advancement.
    validate_state = validate

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
                diagnostics={},
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
                "energy_scope": result.solver_diagnostic.get(
                    "energy_scope", "synthetic OCV chemical energy and lumped heat capacity"
                    if spec.thermal
                    else f"isothermal {spec.model_id}; full cell heat balance not modelled"),
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
