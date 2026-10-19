"""Instrument cost and destructive-sampling services for the transactional runtime."""

from __future__ import annotations

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

    def __init__(self, constitution: PhysicalConstitution) -> None:
        self.constitution = constitution

    def apply_measurement_cost(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        instrument_id = instrument_name(action.get("instrument", "hplc"))
        instrument = self.constitution.instruments.get(instrument_id)
        contract = instrument_contracts(include_particle_size=active(state)).get(instrument_id)
        if instrument is None or contract is None:
            raise ValueError(f"unsupported instrument: {instrument_id!r}")
        if instrument_id == "final_assay" and instrument_completed(state.equipment, "final_assay"):
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
        sampled = withdraw_sample(state, volume)
        spectral = instrument_id in SPECTRAL_INSTRUMENTS
        configuration = spectral_settings(state, instrument_id) if spectral else {}
        cost = acquisition_cost(instrument_id, configuration) if spectral else instrument.cost
        seconds = acquisition_seconds(instrument_id, configuration) if spectral else 0.0
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + cost,
            sample_consumed_L=state.ledger.sample_consumed_L + volume,
            analysis_time_s=state.ledger.analysis_time_s + seconds,
        )
        equipment_id = instrument_equipment_id(instrument_id)
        previous_settings = equipment_settings(state.equipment, equipment_id)
        use_count = int(previous_settings.get("use_count", 0)) + 1
        execution: dict[str, Any] = {
            "measurement_index": use_count,
            "model_id": INSTRUMENT_RUNTIME_MODEL_ID,
            "provider_path": INSTRUMENT_RUNTIME_PROVIDER_PATH,
            "provider_contract_hash": instrument_runtime_contract_hash(
                include_particle_size=active(state)
            ),
            "maturity": "reference_validated",
            "role": "runtime",
            "provenance": list(INSTRUMENT_RUNTIME_PROVENANCE),
            "diagnostics": {
                "instrument_contract_loaded": True,
                "instrument_id": instrument_id,
                "sample_volume_sufficient": True,
                "sample_volume_exact": True,
                "final_assay_repeat_guard": instrument_id == "final_assay",
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
        execution["diagnostics"]["sample_domain"] = list(sample_domain(state))
        execution_history.append(execution)
        equipment = upsert_equipment_record(
            sampled.equipment,
            equipment_id=equipment_id,
            equipment_type="instrument",
            attached_vessel_id=state.vessel_id,
            status="completed" if instrument_id == "final_assay" else "used",
            settings={
                "instrument_id": instrument_id,
                "last_time_s": state.ledger.time_s,
                "last_cost": cost,
                "last_analysis_time_s": seconds,
                "last_sample_consumed_L": volume,
                "use_count": use_count,
                "model_id": execution["model_id"],
                "provider_path": execution["provider_path"],
                "provider_contract_hash": execution["provider_contract_hash"],
                "provenance": execution["provenance"],
                "diagnostics": execution["diagnostics"],
                "execution_history": execution_history,
            },
        )
        return sampled.replace(ledger=ledger, equipment=equipment)


__all__ = ["ChemWorldInstrumentCostServices"]
