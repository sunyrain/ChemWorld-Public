"""Instrument observation module for ChemWorld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chemworld.world.operations import DOWNSTREAM_OBSERVATION_KEYS, EQUILIBRIUM_OBSERVATION_KEYS
from chemworld.world.spectra import (
    final_assay_spectra,
    gc_chromatogram,
    hplc_chromatogram,
    ph_meter_signal,
    uvvis_spectrum,
)


def processed_estimate(
    values: dict[str, float | None],
    observed_mask: dict[str, bool],
) -> dict[str, float | None]:
    estimate_keys = (
        "yield",
        "selectivity",
        "conversion",
        "byproduct_signal",
        "degradation_warning",
        *DOWNSTREAM_OBSERVATION_KEYS,
        *EQUILIBRIUM_OBSERVATION_KEYS,
    )
    return {key: values.get(key) for key in estimate_keys if observed_mask.get(key, False)}


def raw_signal(
    instrument_id: str,
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Synthesize one bounded public signal packet.

    ``species_amounts_mol`` may contain only task-public aggregate channels at
    the runtime boundary.  The returned packet is explicitly synthetic and
    carries anonymous assignments rather than mechanism species identities.
    """

    if instrument_id == "uvvis":
        return uvvis_spectrum(
            values,
            species_amounts_mol=species_amounts_mol,
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    if instrument_id == "ph_meter":
        return ph_meter_signal(
            values,
            seed=seed,
            replicate_count=replicate_count,
        )
    if instrument_id == "hplc":
        return hplc_chromatogram(
            values,
            species_amounts_mol=species_amounts_mol,
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    if instrument_id == "gc":
        return gc_chromatogram(
            values,
            species_amounts_mol=species_amounts_mol,
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    if instrument_id == "final_assay":
        return final_assay_spectra(
            values,
            species_amounts_mol=species_amounts_mol,
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return {}


def observation_units() -> dict[str, str]:
    return {
        "yield": "dimensionless",
        "selectivity": "dimensionless",
        "conversion": "dimensionless",
        "byproduct_signal": "dimensionless",
        "degradation_warning": "dimensionless",
        "virtual_spectrum_summary": "dimensionless",
        **dict.fromkeys(DOWNSTREAM_OBSERVATION_KEYS, "dimensionless"),
        **dict.fromkeys(EQUILIBRIUM_OBSERVATION_KEYS, "dimensionless"),
        "cost": "currency",
        "safety_risk": "risk",
        "score": "dimensionless",
    }


def base_public_values(*, cost: float, safety_risk: float) -> dict[str, float | None]:
    return {
        "yield": None,
        "selectivity": None,
        "conversion": None,
        "byproduct_signal": None,
        "degradation_warning": None,
        "virtual_spectrum_summary": None,
        **dict.fromkeys(DOWNSTREAM_OBSERVATION_KEYS, None),
        "cost": cost,
        "safety_risk": safety_risk,
        "score": 0.0,
    }


def base_observed_mask() -> dict[str, bool]:
    return {
        "yield": False,
        "selectivity": False,
        "conversion": False,
        "byproduct_signal": False,
        "degradation_warning": False,
        "virtual_spectrum_summary": False,
        **dict.fromkeys(DOWNSTREAM_OBSERVATION_KEYS, False),
        "cost": True,
        "safety_risk": True,
        "score": True,
    }


@dataclass(frozen=True)
class ObservationModuleSpec:
    module_id: str = "instrument_observation"
    version: str = "0.4"
    layers: tuple[str, ...] = (
        "sample_state",
        "raw_signal",
        "peaks",
        "assignments",
        "processed_estimate",
        "uncertainty",
        "calibration",
        "missingness",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "layers": list(self.layers),
            "partial_observation": True,
            "history_access": "on_demand_by_public_spectrum_id",
            "history_delivery": "catalog_first_then_requested_packet",
            "measurement_accounting": "cost_sample_failure_and_operation_ledger",
            "masking_policy": "mask_spectral_evidence_only_without_world_state_mutation",
            "synthetic_boundary": (
                "not a real-instrument emulator and not an empirical sample-spectrum predictor"
            ),
            "downstream_keys": list(DOWNSTREAM_OBSERVATION_KEYS),
            "equilibrium_keys": list(EQUILIBRIUM_OBSERVATION_KEYS),
        }


__all__ = [
    "ObservationModuleSpec",
    "base_observed_mask",
    "base_public_values",
    "observation_units",
    "processed_estimate",
    "raw_signal",
]
