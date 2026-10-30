"""Atomic fixed-bed provider over the original WorldState, PhaseLedger and slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from math import fsum, isclose, isfinite
from typing import Any, cast

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    PhaseLedger,
    PhaseRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlot, StateSlots
from chemworld.physchem.coupled_system import CoupledState, CoupledStateView
from chemworld.physchem.fixed_bed import (
    FIXED_BED_POLICY,
    FixedBedSpec,
    build_fixed_bed_system,
    fixed_bed_gas_key,
    fixed_bed_pressure,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.sorbent_kinetics import (
    accepted_sorbent_slot,
    sorbent_inventory,
    sorbent_key,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


def _fixed_bed_world_layout(
    state: WorldState, spec: FixedBedSpec, duration_s: float, *,
    inlet_phase_id: str | None, outlet_phase_id: str | None,
) -> tuple[PhaseLedger, VesselRecord, tuple[str, ...], dict[str, PhaseRecord],
           tuple[StateSlot, ...], dict[str, float], float]:
    """Validate original ownership and geometry without advancing any model."""
    if (
        state.terminated
        or state.quenched
        or state.inactive_vessels
        or not isfinite(duration_s)
        or duration_s < 0
    ):
        raise ValueError("fixed bed requires an active assembly World and valid duration")
    phases = cast(PhaseLedger, state.phases)
    vessels = cast(VesselLedger, state.vessels)
    equipment = cast(EquipmentLedger, state.equipment)
    if (
        spec.bed_id not in equipment.equipment
        or equipment.equipment[spec.bed_id].attached_vessel_id != state.vessel_id
        or equipment.equipment[spec.bed_id].equipment_type != "fixed_bed"
        or equipment.equipment[spec.bed_id].batch_reset != "preserve"
    ):
        raise ValueError("fixed bed requires its original persistent equipment owner")
    phase_ids = (
        *spec.cell_phase_ids,
        *(() if inlet_phase_id is None else (inlet_phase_id,)),
        *(() if outlet_phase_id is None else (outlet_phase_id,)),
    )
    if len(set(phase_ids)) != len(phase_ids) or any(p not in phases.phases for p in phase_ids):
        raise ValueError("fixed-bed cell and boundary identities must be existing and distinct")
    records = {p: phases.phases[p] for p in phase_ids}
    for p, record in records.items():
        if (
            record.phase_id != p
            or record.vessel_id != state.vessel_id
            or record.phase_type != "gas"
            or record.solvents.volume_L > 0
            or not isfinite(record.volume_L)
            or record.volume_L <= 0
            or set(record.species_amounts_mol) != set(spec.component_ids)
            or any(
                isinstance(n, bool) or not isfinite(n) or n < 0
                for n in record.species_amounts_mol.values()
            )
        ):
            raise ValueError("fixed bed requires finite owned gas stock in every cell/tank")
    local_phases = [p for p in phases.phases.values() if p.vessel_id == state.vessel_id]
    hardware = vessels.vessels[state.vessel_id]
    if state.temperature_K > hardware.max_temperature_K or not isclose(
        state.volume_L, fsum(p.volume_L for p in local_phases), rel_tol=1e-12
    ):
        raise ValueError("assembly volume or bath temperature disagrees with original hardware")
    occupied = (
        state.volume_L + (1 - spec.void_fraction) * spec.area_m2 * spec.faces_m[-1] * 1000
    )
    if occupied > hardware.max_volume_L:
        raise ValueError("gas compartments plus solid bed exceed original hardware volume")
    if state.species_amounts != phases.total_amounts_mol():
        raise ValueError("World aggregate must be derived from its original phase inventories")
    all_slots = {s.spec.slot_id: s for s in state.state_slots.slots}
    slots = tuple(all_slots[spec.slot_id(i)] for i in range(len(spec.cell_phase_ids)))
    if any(s.location_id != state.vessel_id for s in slots):
        raise ValueError("bed sorbent slot belongs to another location")
    volumes = {p: record.volume_L for p, record in records.items()}
    return phases, hardware, phase_ids, records, slots, volumes, occupied


def advance_fixed_bed(
    state: WorldState,
    spec: FixedBedSpec,
    duration_s: float,
    *,
    inlet_phase_id: str | None,
    outlet_phase_id: str | None,
    evaluation_times_s: Sequence[float] | None = None,
    policy: ODESolverPolicy = FIXED_BED_POLICY,
) -> ModelProviderResult:
    """Advance existing finite gas compartments and persistent bed loading once.

    All compartments belong to one explicitly declared apparatus/vessel under
    a common bath. The World pressure scalar names the last bed-cell sensor;
    every local pressure is separately derived and checked. No cells, tanks or
    clean sorbent lots are synthesized by this advance operation.
    """
    diagnostic: dict[str, Any] = {
        "scope": "neutral_ideal_gas; common_isothermal_bath; Ergun_upwind_LDF",
        "pressure_sensor_phase_id": spec.cell_phase_ids[-1],
        "boundary_scope": "selected_original_finite_gas_compartments",
    }
    try:
        phases, hardware, phase_ids, records, slots, volumes, occupied = _fixed_bed_world_layout(
            state, spec, duration_s, inlet_phase_id=inlet_phase_id, outlet_phase_id=outlet_phase_id)
        system, momentum = build_fixed_bed_system(
            spec,
            volumes,
            slots,
            inlet_phase_id=inlet_phase_id,
            outlet_phase_id=outlet_phase_id,
            policy=policy,
            maximum_pressure_Pa=hardware.max_pressure_Pa,
        )
        diagnostic["momentum"] = momentum
        amounts = {
            fixed_bed_gas_key(p, k): n
            for p, record in records.items()
            for k, n in record.species_amounts_mol.items()
        }
        for slot in slots:
            amounts.update(sorbent_inventory(slot, spec.sorbent))
        initial = CoupledState(
            amounts,
            state.temperature_K,
            fsum(volumes.values()),
            state.vessel_elapsed_s.get(state.vessel_id, state.ledger.time_s),
        )
        sensor = spec.cell_phase_ids[-1]
        sensor_pressure = fixed_bed_pressure(spec, initial, sensor, volumes[sensor])
        if not isclose(state.pressure_Pa, sensor_pressure, rel_tol=1e-9, abs_tol=1e-4):
            raise ValueError("World pressure must identify the actual last bed-cell sensor")
        result = system.advance(initial, duration_s, evaluation_times_s=evaluation_times_s)
        final = result.final_state
        updates = {
            p: replace(
                record,
                species_amounts_mol={
                    k: final.amounts_mol[fixed_bed_gas_key(p, k)] for k in spec.component_ids
                },
                metadata={
                    **record.metadata,
                    "fixed_bed_pressure_Pa": fixed_bed_pressure(spec, final, p, volumes[p]),
                },
            )
            for p, record in records.items()
        }
        updated = PhaseLedger({**phases.phases, **updates})
        slot_updates = {
            slot.spec.slot_id: accepted_sorbent_slot(slot, spec.sorbent, final) for slot in slots
        }
        updated_slots = StateSlots(
            tuple(slot_updates.get(s.spec.slot_id, s) for s in state.state_slots.slots)
        )
        before = phases.total_amounts_mol()
        after = updated.total_amounts_mol()
        for k, n in state.state_slots.species_amounts().items():
            before[k] = before.get(k, 0.0) + n
        for k, n in updated_slots.species_amounts().items():
            after[k] = after.get(k, 0.0) + n
        if any(
            not isclose(before.get(k, 0.0), after.get(k, 0.0), abs_tol=1e-10, rel_tol=1e-9)
            for k in set(before) | set(after)
        ):
            raise ValueError("fixed-bed original aggregate material balance failed")
        candidate = state.replace(
            phases=updated,
            species_amounts=updated.total_amounts_mol(),
            state_slots=updated_slots,
            pressure_Pa=fixed_bed_pressure(spec, final, sensor, volumes[sensor]),
            ledger=advance_ledger(
                state.ledger,
                result.actual_duration_s,
                energy_deltas={"energy_jacket_J": final.boundary_heat_J},
            ),
            vessel_elapsed_s={**state.vessel_elapsed_s, state.vessel_id: final.time_s},
            metadata={**state.metadata, "fixed_bed_pressure_sensor_phase_id": sensor},
        )

        def profile(view: CoupledStateView) -> dict[str, Any]:
            return {
                "time_s": view.time_s,
                "gas_amounts_mol": {
                    p: {k: view.amounts_mol[fixed_bed_gas_key(p, k)] for k in spec.component_ids}
                    for p in phase_ids
                },
                "pressure_Pa": {
                    p: fixed_bed_pressure(spec, view, p, volumes[p]) for p in phase_ids
                },
                "loading_mol_kg": [
                    {
                        k: view.amounts_mol[sorbent_key(slot, k)] / slot.values["mass_kg"]
                        for k in spec.sorbent.component_ids
                    }
                    for slot in slots
                ],
                "bound_amounts_mol": [
                    {k: view.amounts_mol[sorbent_key(slot, k)] for k in spec.sorbent.component_ids}
                    for slot in slots
                ],
                "bath_heat_J": getattr(view, "boundary_heat_J", 0.0),
            }

        diagnostic.update(
            momentum=momentum,
            stopping_event=result.stopping_event,
            solver=dict(result.solver_diagnostic),
            conservation_residuals_mol=dict(result.conservation_residuals),
            energy_balance_residual_J=result.energy_balance_residual_J,
            bath_heat_J=final.boundary_heat_J,
            occupied_volume_L=occupied,
            capacity_mol=[s.values["sites_mol"] for s in slots],
            carrier_mass_kg=[s.values["mass_kg"] for s in slots],
        )
        return ModelProviderResult(
            outputs={
                "trajectory": [profile(s) for s in result.trajectory],
                "final_profile": profile(final),
            },
            diagnostics=diagnostic,
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome=result.outcome,
        )
    except (ValueError, ArithmeticError, RuntimeError, KeyError) as exc:
        return ModelProviderResult(
            outputs={},
            diagnostics={**diagnostic, "error": str(exc)},
            success=False,
            failure_reason="fixed_bed_advance_failed",
            outcome="numerical_failure",
        )
