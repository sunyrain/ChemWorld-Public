"""An owned electrode strategy for the common advance coordinator."""

from dataclasses import replace
from math import isclose

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.transient_electrode import (
    ELECTRICAL_COORDINATES,
    TransientElectrodeSpec,
    advance_electrode,
    electrode_readout,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

ELECTRODE_TYPE = "finite_planar_electrode"
CHARGE_STATE_TYPE = "electrode_charge"


def electrode_record(state):
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == ELECTRODE_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("one finite layer supports one working electrode")
    return records[0] if records else None


def charge_slot(state, record):
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.owner_kind == "equipment"
        and s.spec.owner_id == record.equipment_id
        and s.spec.state_type == CHARGE_STATE_TYPE
    ]
    if len(slots) != 1:
        raise ValueError("electrode requires its uniquely owned charge state")
    return slots[0]


def initial_charge_slot(state, record, coordinates):
    spec = StateSlotSpec(
        f"{record.equipment_id}:charge",
        "equipment",
        record.equipment_id,
        CHARGE_STATE_TYPE,
        tuple(
            StateFieldSpec(
                key, "J" if key in {"electrical_work_J", "ohmic_loss_J"} else "C", minimum=None
            )
            for key in ELECTRICAL_COORDINATES
        ),
        mechanism_id="planar-BV-double-layer-v1",
        batch_reset="preserve",
    )
    return StateSlot(spec, state.vessel_id, dict(coordinates))


def material_phases(spec, amounts, vessel_id):
    return PhaseLedger(
        {
            f"electrolyte:{i}": PhaseRecord(
                f"electrolyte:{i}",
                vessel_id,
                "liquid",
                float(volume) * 1000,
                {
                    species: amounts[spec.keys(species)[i]]
                    for species in (spec.oxidized_species, spec.reduced_species)
                },
                metadata={"spatial_cell": i, "spatial_model": "finite_planar_electrolyte"},
            )
            for i, volume in enumerate(spec.mesh.cell_volumes_m3)
        }
    )


def unpack_electrode(state):
    record = electrode_record(state)
    if record is None:
        raise ValueError("no finite planar electrode is installed")
    spec = TransientElectrodeSpec(**record.settings["physical_model"])
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("working electrode must be connected to its declared electrical source")
    slot = charge_slot(state, record)
    if set(slot.values) != set(ELECTRICAL_COORDINATES):
        raise ValueError("electrode charge layout differs from the installed model")
    if len(state.phases.phases) != spec.cells:
        raise ValueError("finite electrolyte layer has lost its cell topology")
    amounts = {}
    for i, volume in enumerate(spec.mesh.cell_volumes_m3):
        phase = state.phases.phases.get(f"electrolyte:{i}")
        if (
            phase is None
            or phase.metadata.get("spatial_cell") != i
            or phase.phase_type != "liquid"
            or phase.vessel_id != state.vessel_id
            or not isclose(phase.volume_L, volume * 1000, rel_tol=1e-10, abs_tol=1e-18)
            or set(phase.species_amounts_mol) != {spec.oxidized_species, spec.reduced_species}
        ):
            raise ValueError("invalid finite electrolyte compartment")
        for species in (spec.oxidized_species, spec.reduced_species):
            amounts[spec.keys(species)[i]] = phase.species_amounts_mol[species]
    physical = CoupledState(
        amounts, state.temperature_K, state.volume_L, state.ledger.time_s, coordinates=slot.values
    )
    program = ScalarControlProgram(**record.settings["electrical_control"]["program"])
    return record, slot, spec, physical, program


def read_electrical_state(state):
    _, _, spec, physical, program = unpack_electrode(state)
    return electrode_readout(spec, physical, program.value_at(physical.time_s))


class TransientElectrodeRuntime:
    supported_operations = ("configure_electrical_control", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return electrode_record(state) is not None

    def validate_world(self) -> None:
        if self.composition is None or self.composition.physics is None:
            return
        declared = {
            c.vessel for c in self.composition.components if c.kind == "transient_electrode"
        }
        installed = {
            name
            for name, state in self.composition.physics.initial_states.items()
            if self.supports(state)
        }
        if declared != installed:
            raise ValueError("electrode components and installed physical equipment must agree")
        for vessel in installed:
            self.validate(self.composition.physics.initial_states[vessel])

    def validate(self, state):
        _, _, spec, physical, program = unpack_electrode(state)
        # A zero-duration validation uses exactly the same accepted-state rules.
        advance_electrode(spec, physical, program, 0)
        if self.composition is not None:
            physics = self.composition.physics
            others = (
                physics.models_of("thermal"),
                physics.models_of("flash"),
                physics.models_of("liquid"),
                physics.models_of("column"),
                physics.models_of("concentration"),
            )
            if any(state.vessel_id in model for model in others):
                raise ValueError("isothermal electrode cannot share a competing physical strategy")
            if physics.network.reactions:
                raise ValueError("this electrode slice has no homogeneous kinetic coupling")

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="electrode_isothermal_boundary",
            )
        try:
            self.validate(state)
            _, slot, spec, physical, program = unpack_electrode(state)
            result = advance_electrode(spec, physical, program, float(action["duration_s"]))
            final = result.final_state
            slots = StateSlots(
                tuple(
                    replace(s, values=dict(final.coordinates))
                    if s.spec.slot_id == slot.spec.slot_id
                    else s
                    for s in state.state_slots.slots
                )
            )
            candidate = state.replace(
                phases=material_phases(spec, final.amounts_mol, state.vessel_id),
                state_slots=slots,
                process=replace(state.process, last_observation_record=None),
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    energy_deltas={
                        "flow_electrical_work_J": final.coordinates["electrical_work_J"]
                        - physical.coordinates["electrical_work_J"]
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="electrode_solver_failed",
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
                "electrical_conservation": dict(result.conservation_residuals),
                "energy_scope": "isothermal working electrode; complete cell U not modelled",
            },
        )
