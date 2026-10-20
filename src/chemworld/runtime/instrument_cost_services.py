"""Instrument cost and destructive-sampling services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from chemworld.foundation import (
    PhysicalConstitution,
    WorldState,
    equipment_settings,
    instrument_completed,
    instrument_equipment_id,
    upsert_equipment_record,
)
from chemworld.runtime.full_process_contract import active, sample_domain, withdraw_sample
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.spectral_settings import spectral_settings
from chemworld.world.instruments import (
    INSTRUMENT_RUNTIME_MODEL_ID,
    INSTRUMENT_RUNTIME_PROVENANCE,
    INSTRUMENT_RUNTIME_PROVIDER_PATH,
    instrument_contracts,
    instrument_runtime_contract_hash,
)
from chemworld.world.operations import instrument_name
from chemworld.world.spectral_contract import (
    SPECTRAL_INSTRUMENTS,
    acquisition_cost,
    acquisition_seconds,
)


class ChemWorldInstrumentCostServices:
    """Apply measurement cost, sample consumption, and assay markers."""

    def __init__(self, constitution: PhysicalConstitution, *, modules=None, thermal=None) -> None:
        self.constitution = constitution
        self.modules = modules
        self.thermal = thermal

    def apply_measurement_cost(
        self, state: WorldState, action: dict[str, Any]
    ) -> ModelProviderResult:
        instrument_id = instrument_name(action.get("instrument", "hplc"), modules=self.modules)
        instrument = self.constitution.instruments.get(instrument_id)
        contract = instrument_contracts(
            include_particle_size=active(state), modules=self.modules
        ).get(instrument_id)
        if instrument is None or contract is None:
            raise ValueError(f"unsupported instrument: {instrument_id!r}")
        phase = contract.input_state_schema.get("required_phase")
        if phase is not None and state.phase != phase:
            return ModelProviderResult(
                outputs={}, diagnostics={"required_phase": phase},
                success=False, outcome="rejected", failure_reason="instrument_sample_phase",
            )
        if instrument.requires_terminated and instrument_completed(state.equipment, instrument_id):
            raise ValueError("final assay cannot be repeated")
        if not all(
            isfinite(value) and value >= 0.0
            for value in (state.volume_L, instrument.sample_volume_L, instrument.cost)
        ):
            raise ValueError("instrument cost and sampling domain must be finite and nonnegative")
        volume = float(instrument.sample_volume_L)
        if state.volume_L < volume:
            raise ValueError(
                f"insufficient sample volume for {instrument_id}: "
                f"required={volume}, available={state.volume_L}"
            )
        initial = state
        specimen = state
        duration = contract.physical_duration_s
        advancement = ModelProviderResult(outputs={}, diagnostics={}, state=state)
        if duration:
            if self.thermal is None or not (
                self.thermal.advancer.supports(state)
            ):
                return ModelProviderResult(
                    outputs={}, diagnostics={}, success=False, outcome="rejected",
                    failure_reason="measurement_requires_supported_physical_clock",
                )
            advancement = self.thermal.integrate(
                state, {"duration_s": duration, "operation": action["operation"]}, heat=False
            )
            if advancement.outcome in {"rejected", "numerical_failure"}:
                return advancement
            state = advancement.state
            if contract.clock_semantics == "in_situ_end":
                specimen = state
        complete = advancement.outcome == "success"
        sampled = withdraw_sample(state, volume)
        spectral = instrument_id in SPECTRAL_INSTRUMENTS
        configuration = spectral_settings(state, instrument_id) if spectral else {}
        cost = acquisition_cost(instrument_id, configuration) if spectral else instrument.cost
        seconds = acquisition_seconds(instrument_id, configuration) if spectral else (
            contract.analysis_time_s if complete or not contract.host_evolves_during_analysis
            else advancement.actual_duration_s
        )
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + cost,
            sample_consumed_L=state.ledger.sample_consumed_L + volume,
            analysis_time_s=state.ledger.analysis_time_s + seconds,
        )
        equipment_id = instrument_equipment_id(instrument_id)
        previous_settings = equipment_settings(state.equipment, equipment_id)
        use_count = int(previous_settings.get("use_count", 0)) + 1
        native_clock = contract.clock_semantics != "instantaneous_snapshot"
        local_time = initial.vessel_elapsed_s.get(initial.vessel_id, initial.ledger.time_s)
        sampled_time = local_time + (
            advancement.actual_duration_s if contract.clock_semantics == "in_situ_end" else 0.0
        )
        instrument_time = float(previous_settings.get("elapsed_time_s", 0.0)) + seconds + (
            advancement.actual_duration_s if contract.clock_semantics == "in_situ_end" else 0.0
        )
        sampled_at = {
            "clock_id": (
                f"vessel:{initial.vessel_id}"
                + (f":batch:{initial.samples.batch_generation}" if initial.samples.batch_generation
                   else "")
            ) if native_clock else "world",
            "time_s": sampled_time if native_clock else initial.ledger.time_s,
        }
        available_at = (
            {"clock_id": f"{state.vessel_id}:{equipment_id}", "time_s": instrument_time}
            if contract.clock_semantics == "offline_snapshot"
            and not contract.host_evolves_during_analysis
            else {
                **sampled_at,
                "time_s": sampled_at["time_s"] + (
                    advancement.actual_duration_s
                    if contract.host_evolves_during_analysis else 0.0
                ),
            }
        )
        acquisition = {
            "clock_semantics": contract.clock_semantics,
            "sampled_at": sampled_at,
            "available_at": available_at,
            "sample_id": f"{state.vessel_id}:{instrument_id}:acquisition:{use_count}",
            "acquisition_complete": complete,
            "host_evolves_during_analysis": contract.host_evolves_during_analysis,
            "reactor_advance_s": advancement.actual_duration_s,
            "analysis_effort_s": seconds,
        }
        execution: dict[str, Any] = {
            "measurement_index": use_count,
            "model_id": INSTRUMENT_RUNTIME_MODEL_ID,
            "provider_path": INSTRUMENT_RUNTIME_PROVIDER_PATH,
            "provider_contract_hash": instrument_runtime_contract_hash(
                include_particle_size=active(state), modules=self.modules
            ),
            "maturity": "reference_validated",
            "role": "runtime",
            "provenance": list(INSTRUMENT_RUNTIME_PROVENANCE),
            "diagnostics": {
                "instrument_contract_loaded": True,
                "instrument_id": instrument_id,
                "sample_volume_sufficient": True,
                "sample_volume_exact": True,
                "final_assay_repeat_guard": instrument.requires_terminated,
                "calibration_profile": contract.calibration_profile,
                "detection_contract": dict(contract.detection_contract),
                "saturation_contract": dict(contract.saturation_contract),
                "missingness_contract": dict(contract.missingness_contract),
            },
        }
        execution_history = list(previous_settings.get("execution_history", ()))
        if spectral:
            execution.update(
                model_id="finite-calibrated-reporter-spectra-v1",
                maturity="development",
                provider_path="chemworld.physchem.configurable_spectra.acquire_spectrum",
                provenance=["finite reporter calibration and noisy acquired-trace fit"],
                configuration=configuration,
                analysis_time_s=seconds,
            )
        if instrument_id == "particle_size":
            execution.update(
                model_id="synthetic-particle-summary-v1",
                maturity="development",
                provenance=["synthetic population-balance scalar observation"],
            )
        native_spec = (
            None if self.modules is None else self.modules.instrument_specs().get(instrument_id)
        )
        if native_spec is not None and native_spec.sample_transform is not None:
            sampled = native_spec.sample_transform(state, sampled, action, seconds)
        if native_spec is not None and native_spec.native_observer is not None:
            observer = native_spec.native_observer
            provider = getattr(observer, "__qualname__", type(observer).__qualname__)
            module = getattr(observer, "__module__", type(observer).__module__)
            execution.update(
                model_id=f"native-observer:{instrument_id}",
                provider_path=f"{module}.{provider}", maturity="development",
                provenance=["module-declared native observer; no reference qualification implied"],
            )
        execution["diagnostics"]["sample_domain"] = list(sample_domain(state))
        execution["diagnostics"]["acquisition"] = acquisition
        execution_history.append(execution)
        equipment = upsert_equipment_record(
            sampled.equipment,
            equipment_id=equipment_id,
            equipment_type="instrument",
            attached_vessel_id=state.vessel_id,
            status="completed" if instrument.requires_terminated else "used",
            batch_reset="preserve",
            settings={
                "instrument_id": instrument_id,
                "last_time_s": state.ledger.time_s,
                "last_cost": cost,
                "last_analysis_time_s": seconds,
                "last_sample_consumed_L": volume,
                "use_count": use_count,
                "elapsed_time_s": instrument_time,
                "model_id": execution["model_id"],
                "provider_path": execution["provider_path"],
                "provider_contract_hash": execution["provider_contract_hash"],
                "provenance": execution["provenance"],
                "diagnostics": execution["diagnostics"],
                "execution_history": execution_history,
            },
        )
        return replace(
            advancement,
            state=sampled.replace(ledger=ledger, equipment=equipment),
            measurement_state=specimen,
            diagnostics={**advancement.diagnostics, "acquisition": acquisition},
        )


__all__ = ["ChemWorldInstrumentCostServices"]
