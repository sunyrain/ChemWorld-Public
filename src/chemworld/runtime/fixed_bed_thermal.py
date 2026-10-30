"""Atomic double-temperature fixed-bed advancement on original WorldState owners."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import fsum, isclose, isfinite
from typing import Any, cast

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, VesselLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.coupled_system import CoupledState, CoupledStateView
from chemworld.physchem.fixed_bed import fixed_bed_gas_key, fixed_bed_pressure
from chemworld.physchem.fixed_bed_thermal import (
    THERMAL_FIXED_BED_POLICY,
    bed_gas_temperature,
    bed_solid_temperature,
    build_thermal_fixed_bed_system,
)
from chemworld.physchem.fixed_bed_thermal_specs import FixedBedThermalSpec, bed_thermal_key
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.sorbent_kinetics import (
    accepted_sorbent_slot,
    sorbent_inventory,
    sorbent_key,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.fixed_bed import _fixed_bed_world_layout
from chemworld.runtime.kernel_contracts import ModelProviderResult


def advance_thermal_fixed_bed(
    state: WorldState, spec: FixedBedThermalSpec, duration_s: float, *,
    inlet_phase_id: str | None, outlet_phase_id: str | None,
    evaluation_times_s: Sequence[float] | None = None,
    policy: ODESolverPolicy = THERMAL_FIXED_BED_POLICY,
) -> ModelProviderResult:
    """Never initialize a clean sorbent lot or replace an absent thermal field."""
    diagnostic: dict[str, Any] = {
        "scope": "finite_ideal_gas; separate_gas_and_solid_U; original_Ergun_and_LDF",
        "boundary_scope": "selected_original_finite_gas_compartments",
    }
    try:
        if not isinstance(state, WorldState) or not isinstance(spec, FixedBedThermalSpec):
            raise ValueError("thermal bed needs its original world and thermal declaration")
        bed = spec.bed
        phases, hardware, phase_ids, records, slots, volumes, occupied = _fixed_bed_world_layout(
            state, bed, duration_s, inlet_phase_id=inlet_phase_id, outlet_phase_id=outlet_phase_id)
        local_phases = {p.phase_id: p for p in phases.phases.values()
                        if p.vessel_id == state.vessel_id}
        if (set(local_phases) != set(spec.gas_phase_ids)
                or any(p.phase_type != "gas" or set(p.species_amounts_mol) != set(bed.component_ids)
                       or p.solvents.volume_L != 0 or not isfinite(p.volume_L) or p.volume_L <= 0
                       or any(type(n) not in {int, float} or not isfinite(n) or n < 0
                              for n in p.species_amounts_mol.values())
                       for p in local_phases.values())):
            raise ValueError("every gas compartment must retain its authored thermal field")
        all_slots = {slot.spec.slot_id: slot for slot in state.state_slots.slots}
        thermal_slot = all_slots[spec.slot_id]
        spec.validate_slot(thermal_slot, state.vessel_id)
        if any(thermal_slot.values[bed_thermal_key(kind, phase)] > hardware.max_temperature_K+1e-7
               for kind, identities in (("gas_K", spec.gas_phase_ids),
                                         ("solid_K", bed.cell_phase_ids)) for phase in identities):
            raise ValueError("an original thermal field exceeds the apparatus temperature limit")
        anchor = spec.anchor_phase_id
        anchor_temperature = thermal_slot.values[bed_thermal_key("gas_K", anchor)]
        if not isclose(state.temperature_K, anchor_temperature, rel_tol=0, abs_tol=1e-10):
            raise ValueError("World temperature must identify the original anchor gas sensor")
        system, momentum = build_thermal_fixed_bed_system(
            spec, volumes, slots, inlet_phase_id=inlet_phase_id, outlet_phase_id=outlet_phase_id,
            policy=policy, maximum_pressure_Pa=hardware.max_pressure_Pa,
            maximum_temperature_K=hardware.max_temperature_K)
        amounts = {fixed_bed_gas_key(p, k): n for p, record in records.items()
                   for k, n in record.species_amounts_mol.items()}
        for slot in slots:
            amounts.update(sorbent_inventory(slot, bed.sorbent))
        initial = CoupledState(
            amounts, anchor_temperature, fsum(volumes.values()),
            state.vessel_elapsed_s.get(state.vessel_id, state.ledger.time_s),
            coordinates={c.name: thermal_slot.values[c.name] for c in system.coordinates})
        if not isclose(state.pressure_Pa, fixed_bed_pressure(bed, initial, anchor, volumes[anchor]),
                       rel_tol=1e-9, abs_tol=1e-4):
            raise ValueError("World pressure must identify the original anchor gas sensor")
        result = system.advance(initial, duration_s, evaluation_times_s=evaluation_times_s)
        final = result.final_state

        def pressure(view: CoupledStateView, phase: str) -> float:
            return fixed_bed_pressure(bed, replace(view, temperature_K=bed_gas_temperature(
                spec, view, phase)), phase, volumes[phase])

        updates = {
            p: replace(record, species_amounts_mol={
                k: final.amounts_mol[fixed_bed_gas_key(p, k)] for k in bed.component_ids},
                metadata={**record.metadata, "fixed_bed_pressure_Pa": pressure(final, p),
                          "fixed_bed_temperature_K": bed_gas_temperature(spec, final, p)})
            for p, record in records.items()}
        updated = PhaseLedger({**phases.phases, **updates})
        slot_updates = {slot.spec.slot_id: accepted_sorbent_slot(slot, bed.sorbent, final)
                        for slot in slots}
        thermal_values = {**dict(final.coordinates),
                          bed_thermal_key("gas_K", anchor): final.temperature_K}
        slot_updates[spec.slot_id] = spec.project_slot(thermal_slot, thermal_values)
        updated_slots = StateSlots(tuple(slot_updates.get(slot.spec.slot_id, slot)
                                         for slot in state.state_slots.slots))
        before, after = dict(phases.total_amounts_mol()), dict(updated.total_amounts_mol())
        for key, amount in state.state_slots.species_amounts().items():
            before[key] = before.get(key, 0.)+amount
        for key, amount in updated_slots.species_amounts().items():
            after[key] = after.get(key, 0.)+amount
        if any(not isclose(before.get(key, 0.), after.get(key, 0.), abs_tol=1e-10, rel_tol=1e-9)
               for key in before.keys() | after.keys()):
            raise ValueError("thermal bed original material conservation failed")
        final_pressure = pressure(final, anchor)
        candidate = state.replace(
            phases=updated, species_amounts=updated.total_amounts_mol(), state_slots=updated_slots,
            temperature_K=final.temperature_K, pressure_Pa=final_pressure,
            vessels=VesselLedger({**cast(VesselLedger, state.vessels).vessels,
                                 state.vessel_id: replace(
                hardware, temperature_K=final.temperature_K, pressure_Pa=final_pressure)}),
            ledger=advance_ledger(state.ledger, result.actual_duration_s,
                                  energy_deltas={"energy_jacket_J": final.boundary_heat_J}),
            vessel_elapsed_s={**state.vessel_elapsed_s, state.vessel_id: final.time_s},
            metadata={**state.metadata, "fixed_bed_pressure_sensor_phase_id": anchor,
                      "fixed_bed_temperature_sensor_phase_id": anchor,
                      "fixed_bed_thermal_owner_slot": spec.slot_id})

        def profile(view: CoupledStateView) -> dict[str, Any]:
            return {
                "time_s": view.time_s,
                "heat_import_J": view.coordinates[bed_thermal_key("heat_import_J")],
                "heat_export_J": view.coordinates[bed_thermal_key("heat_export_J")],
                "gas_amounts_mol": {p: {k: view.amounts_mol[fixed_bed_gas_key(p, k)]
                                       for k in bed.component_ids} for p in phase_ids},
                "gas_temperature_K": {p: bed_gas_temperature(spec, view, p) for p in phase_ids},
                "solid_temperature_K": {p: bed_solid_temperature(view, p)
                                        for p in bed.cell_phase_ids},
                "pressure_Pa": {p: pressure(view, p) for p in phase_ids},
                "bound_amounts_mol": [{k: view.amounts_mol[sorbent_key(slot, k)]
                                       for k in bed.sorbent.component_ids} for slot in slots],
            }

        diagnostic.update(
            momentum=momentum, stopping_event=result.stopping_event,
            pressure_sensor_phase_id=anchor, temperature_sensor_phase_id=anchor,
            solver=dict(result.solver_diagnostic), occupied_volume_L=occupied,
            conservation_residuals_mol=dict(result.conservation_residuals),
            energy_balance_residual_J=result.energy_balance_residual_J,
            external_heat_J=final.boundary_heat_J,
            heat_import_J=(final.coordinates[bed_thermal_key("heat_import_J")]
                           - initial.coordinates[bed_thermal_key("heat_import_J")]),
            heat_export_J=(final.coordinates[bed_thermal_key("heat_export_J")]
                           - initial.coordinates[bed_thermal_key("heat_export_J")]),
            binding_energy_change_J=fsum(
                bed.sorbent.adsorption_energy_J_mol[key]
                * (final.amounts_mol[sorbent_key(slot, key)]
                   - initial.amounts_mol[sorbent_key(slot, key)])
                for slot in slots for key in bed.sorbent.component_ids))
        return ModelProviderResult(
            outputs={"trajectory": [profile(view) for view in result.trajectory],
                     "final_profile": profile(final)}, diagnostics=diagnostic, state=candidate,
            actual_duration_s=result.actual_duration_s, outcome=result.outcome)
    except (ValueError, ArithmeticError, RuntimeError, KeyError, TypeError) as exc:
        return ModelProviderResult(
            outputs={}, diagnostics={**diagnostic, "error": str(exc)}, success=False,
            failure_reason="thermal_fixed_bed_advance_failed", outcome="numerical_failure")
