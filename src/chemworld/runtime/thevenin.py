"""Equipment-owned empirical circuit on the existing shared physical clock."""

from dataclasses import replace

from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CURRENT,
    THEVENIN_ID,
    VOLTAGE,
    WORK,
    TheveninSpec,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.battery_thevenin_control import thevenin_control_program
from chemworld.physchem.coupled_system import CoupledEvent, CoupledState
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

DEVICE_TYPE = "thevenin_equivalent_circuit"
STATE_TYPE = "thevenin-rc-node-temperature-state-v1"
STRATEGY = "thevenin-empirical-circuit"
METER = "thevenin_meter"


def thevenin_record(state):
    records = [r for r in state.equipment.equipment.values()
               if r.equipment_type == DEVICE_TYPE and r.attached_vessel_id == state.vessel_id]
    if len(records) > 1:
        raise ValueError("a Thevenin workspace requires one unambiguous circuit")
    return records[0] if records else None


def initial_thevenin_slot(record, spec):
    declaration = StateSlotSpec(
        f"{record.equipment_id}:circuit", "equipment", record.equipment_id, STATE_TYPE,
        tuple(StateFieldSpec(c.name, c.unit, minimum=c.minimum)
              for c in spec.coordinate_declarations),
        mechanism_id=THEVENIN_ID, batch_reset="preserve")
    return StateSlot(declaration, record.attached_vessel_id, spec.initial_state().coordinates)


def unpack_thevenin(state):
    record = thevenin_record(state)
    if record is None:
        raise ValueError("no Thevenin circuit is installed")
    spec = TheveninSpec.from_dict(record.settings["physical_model"])
    slots = [s for s in state.state_slots.slots
             if s.spec.owner_kind == "equipment" and s.spec.owner_id == record.equipment_id
             and s.spec.state_type == STATE_TYPE]
    if len(slots) != 1 or slots[0].spec != initial_thevenin_slot(record, spec).spec:
        raise ValueError("Thevenin state layout and equipment ownership must match the model")
    p = spec.native_parameters
    if (state.phase != "formal" or state.volume_L != 0. or state.species_amounts
            or state.liquid_heat_capacity_J_L_K is not None
            or state.temperature_K != p["reference_frame_temperature_K"]):
        raise ValueError("Thevenin workspace cannot invent bulk inventory or cell heat capacity")
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("Thevenin circuit must own its connected control source")
    slot = slots[0]
    physical = CoupledState({}, p["reference_frame_temperature_K"], p["housing_volume_L"],
                            state.ledger.time_s, coordinates=slot.values)
    program = thevenin_control_program(record.settings["electrical_control"]["program"])
    return record, slot, spec, physical, program


class TheveninRuntime:
    supported_operations = ("configure_electrical_control", "queue_electrical_stage", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return thevenin_record(state) is not None

    def validate_state(self, state):
        _, _, spec, physical, program = unpack_thevenin(state)
        for coordinate in spec.coordinate_declarations:
            coordinate.validate(physical.coordinates[coordinate.name])
        thevenin_system(spec, program).advance(physical, 0.)

    validate = validate_state

    def validate_world(self) -> None:
        if self.composition is None:
            return
        physics = self.composition.physics
        declared = {c.vessel for c in self.composition.components if c.kind == "battery_thevenin"}
        installed = {v for v, s in physics.initial_states.items() if self.supports(s)}
        if declared != installed:
            raise ValueError("Thevenin declarations and unique state owners differ")
        if not installed:
            return
        if physics.lab_clock is None:
            raise ValueError("Thevenin workspaces require the shared laboratory clock")
        for vessel in installed:
            if (any(vessel in models for models in physics.models.values())
                    or any(c.vessel == vessel and self.composition.modules.components()[
                        c.kind].processes for c in self.composition.components)):
                raise ValueError("Thevenin cannot also bind an undeclared kinetic/thermal model")
            self.validate_state(physics.initial_states[vessel])

    def reconcile_initialization(self, transition):
        # Lifecycle owns replacement and retained slots. Recheck the resulting
        # apparatus, never regenerate its SOC or node temperatures on new material.
        for local in (transition.state, *transition.state.inactive_vessels.values()):
            if self.supports(local):
                self.validate_state(local)
        return transition

    def advance_controlled(self, state, action, control_stage, **kwargs):
        stage, index = control_stage

        def distance(candidate):
            current, voltage = (candidate.algebraic_values[k] for k in (CURRENT, VOLTAGE))
            if "stop_voltage_below_V" in stage:
                return voltage-stage["stop_voltage_below_V"]
            if "stop_voltage_above_V" in stage:
                return stage["stop_voltage_above_V"]-voltage
            return abs(current)-stage["stop_abs_current_A"]

        events = ((CoupledEvent(f"control:stage:{index}", distance),)
                  if any(key.startswith("stop_") for key in stage) else ())
        return self.advance(state, action, control_events=events, **kwargs)

    def advance(self, state, action, *, heat=False, jacket_program=None, control_events=()):
        if heat or jacket_program is not None:
            return ModelProviderResult({}, {"energy_scope": "empirical node heat only"},
                success=False, outcome="rejected", failure_reason="thevenin_boundary_only")
        try:
            self.validate_state(state)
            _, slot, spec, physical, program = unpack_thevenin(state)
            result = thevenin_system(spec, program, control_events=control_events).advance(
                physical, float(action["duration_s"]))
            final = result.final_state
            updated = replace(slot, values=final.coordinates)
            candidate = state.replace(
                state_slots=StateSlots(tuple(updated if s.spec.slot_id == slot.spec.slot_id else s
                                             for s in state.state_slots.slots)),
                process=replace(state.process, last_observation_record=None),
                ledger=advance_ledger(state.ledger, result.actual_duration_s, energy_deltas={
                    "flow_electrical_work_J": physical.coordinates[WORK]-final.coordinates[WORK],
                    "heat_loss_J": final.coordinates[AMBIENT_LOSS]-physical.coordinates[
                        AMBIENT_LOSS],
                }))
            fields = thevenin_fields(spec, final, program)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult({}, {"solver_error": str(exc)}, success=False,
                outcome="numerical_failure", failure_reason="thevenin_advance_failed")
        return ModelProviderResult({}, {
            "solver": dict(result.solver_diagnostic), "stopping_event": result.stopping_event,
            "circuit_conservation": dict(result.conservation_residuals),
            "energy_scope": "empirical two-node heat equation; no chemical total energy",
            "energy_balance_residual_J": None,
            "electrical_control": {"unit": program.unit, "target": program.value_at(final.time_s),
                "current_A": fields["current_A"], "voltage_V": fields["voltage_V"],
                "charge_C": final.coordinates["discharged_charge_C"],
                "work_J": final.coordinates[WORK]},
        }, state=candidate, outcome=result.outcome, actual_duration_s=result.actual_duration_s)
