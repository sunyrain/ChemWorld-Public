"""Batch and event-driven dynamic batch reactor models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_shared import (
    HeatTransferSpec,
    JacketTemperatureProgram,
    PressureBoundarySpec,
    ReactorResult,
    ReactorState,
    ReactorValidityDomain,
    SamplingEventSpec,
)
from chemworld.physchem.reactor_solvers import (
    _amount_vector,
    _integrate_result,
    _material_balance_error,
    _segment_evaluation_times,
    _solve_interval,
    _validated_amount,
    make_batch_rhs,
)
from chemworld.physchem.solver_backend import ODESolveReport
from chemworld.physchem.thermochemistry import NASA7SpeciesThermo

BatchOperationType = Literal["configure", "heat", "wait", "reaction_advance"]


@dataclass(frozen=True)
class BatchOperationRecord:
    operation_id: str
    operation_type: BatchOperationType
    start_time_s: float
    end_time_s: float
    applied: bool
    state_before: ReactorState
    state_after: ReactorState


@dataclass
class BatchReactorSession:
    """Stateful, idempotent operating facade over the dynamic batch model.

    Every time-advancing operation routes through one integration path.  A
    repeated operation id with the same payload is a no-op; reusing it for a
    different payload is rejected.  This prevents UI/runtime retries from
    adding reaction time or heat duty twice.
    """

    model: DynamicBatchReactorModel
    state: ReactorState
    heat_transfer: HeatTransferSpec = field(default_factory=HeatTransferSpec)
    pressure_boundary: PressureBoundarySpec | None = None
    validity_domain: ReactorValidityDomain | None = None
    records: list[BatchOperationRecord] = field(default_factory=list)
    _fingerprints: dict[str, tuple[object, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def create(
        cls,
        model: DynamicBatchReactorModel,
        initial_amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        heat_transfer: HeatTransferSpec | None = None,
        pressure_boundary: PressureBoundarySpec | None = None,
        validity_domain: ReactorValidityDomain | None = None,
    ) -> BatchReactorSession:
        amounts = dict(
            zip(
                model.network.species_ids,
                _amount_vector(model.network, initial_amounts_mol),
                strict=True,
            )
        )
        return cls(
            model=model,
            state=ReactorState(
                amounts_mol=amounts,
                volume_L=volume_L,
                temperature_K=temperature_K,
                time_s=0.0,
                pressure_Pa=(
                    pressure_boundary.pressure_Pa if pressure_boundary is not None else None
                ),
            ),
            heat_transfer=heat_transfer or HeatTransferSpec(),
            pressure_boundary=pressure_boundary,
            validity_domain=validity_domain,
        )

    def configure(
        self,
        operation_id: str,
        *,
        heat_transfer: HeatTransferSpec | None = None,
        pressure_boundary: PressureBoundarySpec | None = None,
    ) -> BatchOperationRecord:
        resolved_heat = heat_transfer or self.heat_transfer
        resolved_pressure = pressure_boundary or self.pressure_boundary
        fingerprint = ("configure", resolved_heat, resolved_pressure)
        duplicate = self._check_operation_id(operation_id, fingerprint)
        if duplicate is not None:
            return duplicate
        before = self.state
        self.heat_transfer = resolved_heat
        self.pressure_boundary = resolved_pressure
        if resolved_pressure is not None and before.pressure_Pa != resolved_pressure.pressure_Pa:
            self.state = replace(before, pressure_Pa=resolved_pressure.pressure_Pa)
        record = BatchOperationRecord(
            operation_id=operation_id,
            operation_type="configure",
            start_time_s=before.time_s,
            end_time_s=self.state.time_s,
            applied=True,
            state_before=before,
            state_after=self.state,
        )
        return self._commit(operation_id, fingerprint, record)

    def advance(
        self,
        operation_id: str,
        *,
        duration_s: float,
        operation_type: Literal["heat", "wait", "reaction_advance"] = "reaction_advance",
        evaluation_points: int = 5,
    ) -> BatchOperationRecord:
        if duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        if evaluation_points < 2:
            raise ValueError("evaluation_points must be at least two")
        fingerprint = (
            operation_type,
            float(duration_s),
            self.heat_transfer,
            self.pressure_boundary,
        )
        duplicate = self._check_operation_id(operation_id, fingerprint)
        if duplicate is not None:
            return duplicate
        before = self.state
        evaluation_times = tuple(np.linspace(0.0, duration_s, evaluation_points))
        segment = self.model.simulate(
            before.amounts_mol,
            initial_volume_L=before.volume_L,
            temperature_K=before.temperature_K,
            duration_s=duration_s,
            heat_transfer=self.heat_transfer,
            pressure_boundary=self.pressure_boundary,
            validity_domain=self.validity_domain,
            evaluation_times_s=evaluation_times,
        )
        self.state = ReactorState(
            amounts_mol=segment.final_state.amounts_mol,
            volume_L=segment.final_state.volume_L,
            temperature_K=segment.final_state.temperature_K,
            time_s=before.time_s + duration_s,
            pressure_Pa=segment.final_state.pressure_Pa,
            energy_jacket_J=before.energy_jacket_J + segment.final_state.energy_jacket_J,
            heat_reaction_J=before.heat_reaction_J + segment.final_state.heat_reaction_J,
            heat_loss_J=before.heat_loss_J + segment.final_state.heat_loss_J,
            material_in_mol=_sum_ledgers(
                before.material_in_mol,
                segment.final_state.material_in_mol,
            ),
            material_out_mol=_sum_ledgers(
                before.material_out_mol,
                segment.final_state.material_out_mol,
            ),
        )
        record = BatchOperationRecord(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time_s=before.time_s,
            end_time_s=self.state.time_s,
            applied=True,
            state_before=before,
            state_after=self.state,
        )
        return self._commit(operation_id, fingerprint, record)

    def _check_operation_id(
        self,
        operation_id: str,
        fingerprint: tuple[object, ...],
    ) -> BatchOperationRecord | None:
        if not operation_id.strip():
            raise ValueError("operation_id cannot be empty")
        previous = self._fingerprints.get(operation_id)
        if previous is None:
            return None
        if previous != fingerprint:
            raise ValueError(f"operation_id {operation_id!r} was reused with a different payload")
        original = next(record for record in self.records if record.operation_id == operation_id)
        return replace(original, applied=False)

    def _commit(
        self,
        operation_id: str,
        fingerprint: tuple[object, ...],
        record: BatchOperationRecord,
    ) -> BatchOperationRecord:
        self._fingerprints[operation_id] = fingerprint
        self.records.append(record)
        return record


@dataclass(frozen=True)
class BatchReactorModel:
    network: ReactionNetworkSpec
    reactor_id: str = "batch"

    def simulate(
        self,
        initial_amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        duration_s: float,
        heat_transfer: HeatTransferSpec | None = None,
        pressure_boundary: PressureBoundarySpec | None = None,
        validity_domain: ReactorValidityDomain | None = None,
        evaluation_times_s: Sequence[float] | None = None,
    ) -> ReactorResult:
        if duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        if volume_L <= 0:
            raise ValueError("volume_L must be positive")
        if temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        thermal = HeatTransferSpec() if heat_transfer is None else heat_transfer
        y0 = np.array(
            [
                *(_amount_vector(self.network, initial_amounts_mol)),
                temperature_K,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        rhs = make_batch_rhs(self.network, volume_getter=lambda: volume_L, thermal=thermal)

        return _integrate_result(
            model_id="batch",
            reactor_id=self.reactor_id,
            network=self.network,
            initial_amounts_mol=initial_amounts_mol,
            initial_volume_L=volume_L,
            initial_temperature_K=temperature_K,
            y0=y0,
            duration_s=duration_s,
            rhs=rhs,
            volume_getter=lambda _time_s, _y: volume_L,
            evaluation_times_s=evaluation_times_s,
            temperature_index=len(self.network.species_ids),
            energy_jacket_index=len(self.network.species_ids) + 1,
            heat_reaction_index=len(self.network.species_ids) + 2,
            heat_loss_index=len(self.network.species_ids) + 3,
            pressure_boundary=pressure_boundary,
            validity_domain=validity_domain,
        )


@dataclass(frozen=True)
class DynamicBatchReactorModel:
    """Event-driven batch reactor with thermochemical heat release.

    The model is intentionally compact but follows the professional 0D reactor
    separation used by Cantera/IDAES: reaction rates update species amounts,
    reaction enthalpies define heat release, wall/jacket terms define heat
    exchange, and destructive sampling is an explicit material-out event.
    """

    network: ReactionNetworkSpec
    reactor_id: str = "dynamic_batch"

    def simulate(
        self,
        initial_amounts_mol: Mapping[str, float],
        *,
        initial_volume_L: float,
        temperature_K: float,
        duration_s: float,
        heat_transfer: HeatTransferSpec | None = None,
        species_thermo: Mapping[str, NASA7SpeciesThermo] | None = None,
        jacket_program: JacketTemperatureProgram | None = None,
        sampling_events: Sequence[SamplingEventSpec] = (),
        pressure_boundary: PressureBoundarySpec | None = None,
        validity_domain: ReactorValidityDomain | None = None,
        evaluation_times_s: Sequence[float] | None = None,
    ) -> ReactorResult:
        if duration_s < 0:
            raise ValueError("duration_s cannot be negative")
        if initial_volume_L <= 0:
            raise ValueError("initial_volume_L must be positive")
        if temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        thermal = HeatTransferSpec() if heat_transfer is None else heat_transfer
        events = tuple(sorted(sampling_events, key=lambda event: event.time_s))
        if any(event.time_s > duration_s for event in events):
            raise ValueError("sampling events cannot occur after duration_s")
        if evaluation_times_s is not None:
            for time_s in evaluation_times_s:
                if time_s < 0 or time_s > duration_s:
                    raise ValueError("evaluation_times_s must lie inside the simulation")

        n_species = len(self.network.species_ids)
        y_current = np.array(
            [
                *_amount_vector(self.network, initial_amounts_mol),
                temperature_K,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        current_volume = float(initial_volume_L)
        current_time = 0.0
        material_out = dict.fromkeys(self.network.species_ids, 0.0)
        sample_records: list[dict[str, object]] = []
        solver_diagnostics: list[dict[str, object]] = []
        times: list[float] = []
        states: list[np.ndarray] = []
        volumes: list[float] = []

        def append_solution(segment: ODESolveReport) -> None:
            for idx, time_value in enumerate(segment.t):
                if times and abs(float(time_value) - times[-1]) < 1e-12:
                    times[-1] = float(time_value)
                    states[-1] = np.array(segment.y[:, idx], dtype=float)
                    volumes[-1] = current_volume
                else:
                    times.append(float(time_value))
                    states.append(np.array(segment.y[:, idx], dtype=float))
                    volumes.append(current_volume)

        rhs = make_batch_rhs(
            self.network, volume_getter=lambda: current_volume, thermal=thermal,
            species_thermo=species_thermo,
            jacket_program=jacket_program.temperature_at if jacket_program is not None else None,
        )

        transition_times = sorted({event.time_s for event in events} | {duration_s})
        for next_time in transition_times:
            if next_time > current_time:
                segment = _solve_interval(
                    rhs,
                    y_current,
                    start_s=current_time,
                    end_s=next_time,
                    evaluation_times_s=_segment_evaluation_times(
                        current_time,
                        next_time,
                        evaluation_times_s,
                    ),
                )
                append_solution(segment)
                solver_diagnostics.append(segment.diagnostic.to_dict())
                y_current = np.array(segment.y[:, -1], dtype=float)
                current_time = next_time
            elif not times:
                times.append(current_time)
                states.append(y_current.copy())
                volumes.append(current_volume)

            events_at_time = [event for event in events if abs(event.time_s - current_time) < 1e-12]
            for event in events_at_time:
                if event.volume_L >= current_volume:
                    raise ValueError("sampling volume cannot remove all reactor volume")
                fraction = event.volume_L / current_volume
                sampled: dict[str, float] = {}
                for index, species_id in enumerate(self.network.species_ids):
                    removed = max(float(y_current[index]), 0.0) * fraction
                    y_current[index] = max(float(y_current[index]) - removed, 0.0)
                    material_out[species_id] += removed
                    sampled[species_id] = removed
                current_volume -= event.volume_L
                sample_records.append(
                    {
                        **event.to_dict(),
                        "fraction_removed": fraction,
                        "sampled_amounts_mol": sampled,
                        "remaining_volume_L": current_volume,
                    }
                )
                if times and abs(times[-1] - current_time) < 1e-12:
                    states[-1] = y_current.copy()
                    volumes[-1] = current_volume

        if not times:
            times.append(0.0)
            states.append(y_current.copy())
            volumes.append(current_volume)

        state_matrix = np.vstack(states).T
        amounts_timeseries = {
            species_id: tuple(
                _validated_amount(float(value), species_id) for value in state_matrix[index]
            )
            for index, species_id in enumerate(self.network.species_ids)
        }
        initial_state = ReactorState(
            amounts_mol={
                species_id: float(initial_amounts_mol.get(species_id, 0.0))
                for species_id in self.network.species_ids
            },
            volume_L=initial_volume_L,
            temperature_K=temperature_K,
            time_s=0.0,
            pressure_Pa=(
                pressure_boundary.pressure_Pa if pressure_boundary is not None else None
            ),
        )
        final_state = ReactorState(
            amounts_mol={
                species_id: amounts_timeseries[species_id][-1]
                for species_id in self.network.species_ids
            },
            volume_L=current_volume,
            temperature_K=max(float(state_matrix[n_species, -1]), 1.0),
            time_s=duration_s,
            pressure_Pa=(
                pressure_boundary.pressure_Pa if pressure_boundary is not None else None
            ),
            energy_jacket_J=float(state_matrix[n_species + 1, -1]),
            heat_reaction_J=float(state_matrix[n_species + 2, -1]),
            heat_loss_J=float(state_matrix[n_species + 3, -1]),
            material_out_mol=material_out,
        )
        return ReactorResult(
            reactor_id=self.reactor_id,
            model_id="dynamic_batch",
            network_id=self.network.network_id,
            initial_state=initial_state,
            final_state=final_state,
            times_s=tuple(times),
            amounts_mol=amounts_timeseries,
            temperatures_K=tuple(float(value) for value in state_matrix[n_species]),
            material_balance_error_mol=_material_balance_error(
                self.network,
                initial_state,
                final_state,
            ),
            volumes_L=tuple(volumes),
            pressures_Pa=(
                tuple(pressure_boundary.pressure_Pa for _ in times)
                if pressure_boundary is not None
                else ()
            ),
            validity_domain=validity_domain,
            metadata={
                "heat_source": (
                    "nasa7_reaction_enthalpy" if species_thermo is not None else "reaction_delta_h"
                ),
                "sample_events": sample_records,
                "volume_L_timeseries": tuple(volumes),
                "solver_diagnostics": solver_diagnostics,
                "jacket_program": (
                    jacket_program.to_dict() if jacket_program is not None else None
                ),
                "reference_contract": "cantera_idaes_zero_d_energy_balance_slice",
                "coupled_derivative_assembly": "assemble_coupled_rates",
                "general_internal_energy_covered": False,
                "pressure_boundary": (
                    {
                        "mode": pressure_boundary.mode,
                        "pressure_Pa": pressure_boundary.pressure_Pa,
                    }
                    if pressure_boundary is not None
                    else None
                ),
            },
        )


def _sum_ledgers(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(left.get(key, 0.0)) + float(right.get(key, 0.0))
        for key in set(left) | set(right)
    }


__all__ = [
    "BatchOperationRecord",
    "BatchOperationType",
    "BatchReactorModel",
    "BatchReactorSession",
    "DynamicBatchReactorModel",
]
