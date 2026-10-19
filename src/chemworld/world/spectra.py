"""Virtual spectroscopy and chromatography signal synthesis for ChemWorld."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from chemworld.physchem.spectroscopy import (
    build_signal_spec,
    potentiometric_ph,
    synthesize_signal,
)


@dataclass(frozen=True)
class SignalPeak:
    center: float
    width: float
    area: float
    assignment: str

    def to_dict(self, *, center_key: str, width_key: str) -> dict[str, Any]:
        return {
            center_key: round(float(self.center), 6),
            width_key: round(float(self.width), 6),
            "area": round(float(self.area), 6),
            "assignment": self.assignment,
        }


def _observed(values: dict[str, float | None], key: str) -> float:
    value = values.get(key)
    return 0.0 if value is None else float(np.clip(value, 0.0, 1.0))


def _axis(start: float, stop: float, points: int) -> np.ndarray:
    return np.linspace(start, stop, points, dtype=float)


def _gaussian(axis: np.ndarray, peak: SignalPeak) -> np.ndarray:
    width = max(float(peak.width), 1.0e-6)
    height = float(peak.area) / (width * np.sqrt(2.0 * np.pi))
    return height * np.exp(-0.5 * ((axis - float(peak.center)) / width) ** 2)


def _rounded(values: np.ndarray | list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _chromatogram(
    *,
    kind: str,
    time_min: np.ndarray,
    peaks: tuple[SignalPeak, ...],
    baseline: float,
) -> dict[str, Any]:
    intensity = np.full_like(time_min, fill_value=baseline, dtype=float)
    for peak in peaks:
        intensity += _gaussian(time_min, peak)
    max_intensity = float(np.max(intensity))
    if max_intensity > 0.0:
        intensity = intensity / max_intensity
    return {
        "kind": kind,
        "time_min": _rounded(time_min),
        "intensity": _rounded(intensity),
        "peaks": [
            peak.to_dict(center_key="retention_time_min", width_key="width_min") for peak in peaks
        ],
        "baseline": round(float(baseline), 6),
        "normalization": "max_intensity",
    }


def _species_signal(
    instrument_id: str,
    species_amounts_mol: dict[str, float] | None,
    *,
    volume_L: float,
    seed: int,
    replicate_count: int,
) -> dict[str, Any] | None:
    if not species_amounts_mol:
        return None
    species_ids = tuple(species_amounts_mol)
    target_species = tuple(
        species_id for species_id in species_ids if _public_species_role(species_id) == "target"
    )
    impurity_species = tuple(
        species_id
        for species_id in species_ids
        if _public_species_role(species_id) in {"byproduct", "degradation", "impurity"}
    )
    spec = build_signal_spec(
        instrument_id,
        species_ids,
        target_species=target_species,
        impurity_species=impurity_species,
    )
    if instrument_id == "uvvis":
        detector_gain = 10.0
        spec = replace(
            spec,
            detector_gain=detector_gain,
            features=tuple(
                replace(feature, response_factor=feature.response_factor * detector_gain)
                for feature in spec.features
            ),
        )
    packet = synthesize_signal(
        spec,
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    ).to_public_dict(sample_volume_L=volume_L)
    packet["source"] = "species_amounts_with_calibration"
    return packet


def _require_signal_packet(packet: dict[str, Any] | None, instrument_id: str) -> dict[str, Any]:
    """Keep fallback synthesis fail-closed under optimized Python execution."""

    if packet is None:
        raise RuntimeError(f"{instrument_id} signal synthesis produced no packet")
    return packet


def _public_species_role(species_id: str) -> str:
    """Return the task-visible role encoded in a public aggregate species key."""

    if species_id == "target_public":
        return "target"
    if species_id == "impurity_public":
        return "impurity"
    if species_id == "degradation_public":
        return "degradation"
    if species_id == "reactant_public":
        return "reactant"
    return "unassigned"


def _public_proxy_amounts(
    values: dict[str, float | None],
    *,
    volume_L: float,
    include_reactant: bool = True,
) -> dict[str, float]:
    """Map public scalar evidence to anonymous aggregate signal channels."""

    scale = max(float(volume_L), 1.0e-9)
    conversion = max(
        _observed(values, "conversion"),
        _observed(values, "flow_conversion"),
    )
    target = max(
        _observed(values, "yield"),
        _observed(values, "purity"),
        _observed(values, "crystal_purity"),
        _observed(values, "distillate_purity"),
        _observed(values, "selective_product_yield"),
        _observed(values, "electrochemical_selectivity"),
    )
    impurity = max(
        _observed(values, "byproduct_signal"),
        _observed(values, "impurity_signal"),
    )
    degradation = _observed(values, "degradation_warning")
    amounts: dict[str, float] = {}
    if include_reactant:
        amounts["reactant_public"] = max(1.0 - conversion, 0.0) * scale
    if target > 0.0:
        amounts["target_public"] = target * scale
    if impurity > 0.0:
        amounts["impurity_public"] = impurity * scale
    if degradation > 0.0:
        amounts["degradation_public"] = degradation * scale
    return amounts


def hplc_chromatogram(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a compact HPLC chromatogram with reactant, product, and impurity peaks."""

    species_packet = _species_signal(
        "hplc",
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    )
    if species_packet is None:
        species_packet = _species_signal(
            "hplc",
            _public_proxy_amounts(values, volume_L=volume_L),
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return _require_signal_packet(species_packet, "hplc")


def gc_chromatogram(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a compact GC chromatogram for volatile byproducts and distillate quality."""

    species_packet = _species_signal(
        "gc",
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    )
    if species_packet is None:
        species_packet = _species_signal(
            "gc",
            _public_proxy_amounts(values, volume_L=volume_L, include_reactant=False),
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    if species_packet is None:
        species_packet = _species_signal(
            "gc",
            {"impurity_public": 0.0},
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return _require_signal_packet(species_packet, "gc")


def uvvis_spectrum(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a Beer-Lambert species spectrum or fallback broad proxy bands."""

    species_packet = _species_signal(
        "uvvis",
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    )
    if species_packet is None:
        species_packet = _species_signal(
            "uvvis",
            _public_proxy_amounts(values, volume_L=volume_L),
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return _require_signal_packet(species_packet, "uvvis")


def ph_meter_signal(
    values: dict[str, float | None],
    *,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a compact public pH-meter packet.

    The scalar Gym key is normalized to [0, 1], while this raw packet exposes
    the chemically meaningful pH and electrode response as an instrument
    observation. It does not expose hidden equilibrium constants or species
    amounts.
    """

    rng = np.random.default_rng(seed)
    normalized = _observed(values, "pH_normalized")
    hydrogen_activity = 10.0 ** (-float(np.clip(14.0 * normalized, 0.0, 14.0)))
    pH = potentiometric_ph(hydrogen_activity_mol_L=hydrogen_activity)
    pH_replicates = [
        round(float(np.clip(pH + rng.normal(0.0, 0.02), 0.0, 14.0)), 3)
        for _ in range(max(int(replicate_count), 1))
    ]
    electrode_mV = -59.16 * (pH - 7.0)
    return {
        "kind": "ph_meter_signal",
        "instrument_id": "ph_meter",
        "signal_type": "potentiometric_ph",
        "pH": round(pH, 3),
        "hydrogen_mol_L": round(10.0 ** (-pH), 12),
        "electrode_mV": round(electrode_mV, 3),
        "replicate_pH": pH_replicates,
        "sample_state": {
            "physical_state": "virtual_liquid_sample",
            "temperature_K": 298.15,
            "activity_basis": "dimensionless_relative_to_standard_state",
        },
        "axis": {"key": "replicate_index", "unit": "index", "point_count": len(pH_replicates)},
        "raw_signal": {"key": "electrode_mV", "unit": "mV"},
        "processed_estimates": {
            "pH_normalized": round(normalized, 6),
            "acid_dissociation_fraction": round(
                _observed(values, "acid_dissociation_fraction"),
                6,
            ),
            "precipitation_signal": round(_observed(values, "precipitation_signal"), 6),
            "equilibrium_residual": round(_observed(values, "equilibrium_residual"), 6),
            "equilibrium_confidence": round(_observed(values, "equilibrium_confidence"), 6),
        },
        "uncertainty": {
            "pH_std": 0.02,
            "normalized_pH_std": round(0.02 / 14.0, 6),
        },
        "peaks": [],
        "assignments": [],
        "calibration": {
            "method": "Nernstian hydrogen-activity response at 298.15 K",
            "slope_mV_per_pH": -59.16,
            "pH_range": [0.0, 14.0],
            "lod_pH": 0.02,
            "loq_pH": 0.06,
        },
        "missingness": {
            "policy": "failed_measurement_has_no_signal_packet",
            "entries": [],
        },
        "metadata": {
            "noise_model": "seeded_independent_gaussian_pH",
            "baseline": 0.0,
            "baseline_drift": 0.0,
            "synthetic": True,
            "real_sample_prediction": False,
            "boundary_note": (
                "Synthetic potentiometric benchmark evidence; not a real electrode reading."
            ),
        },
        "visibility": "public_processed_observation",
    }


def ir_spectrum(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a low-resolution IR-like transmittance spectrum."""

    species_packet = _species_signal(
        "ir",
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    )
    if species_packet is None:
        species_packet = _species_signal(
            "ir",
            _public_proxy_amounts(values, volume_L=volume_L),
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return _require_signal_packet(species_packet, "ir")


def nmr_spectrum(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return a toy 1H NMR-like spectrum for product and impurity proxies."""

    species_packet = _species_signal(
        "nmr",
        species_amounts_mol,
        volume_L=volume_L,
        seed=seed,
        replicate_count=replicate_count,
    )
    if species_packet is None:
        species_packet = _species_signal(
            "nmr",
            _public_proxy_amounts(values, volume_L=volume_L),
            volume_L=volume_L,
            seed=seed,
            replicate_count=replicate_count,
        )
    return _require_signal_packet(species_packet, "nmr")


def final_assay_spectra(
    values: dict[str, float | None],
    *,
    species_amounts_mol: dict[str, float] | None = None,
    volume_L: float = 1.0,
    seed: int = 0,
    replicate_count: int = 1,
) -> dict[str, Any]:
    """Return the multi-instrument spectral packet used by final assay records."""

    return {
        "kind": "final_assay_packet",
        "quality": "synthetic_reference_assay",
        "channels": [
            "hplc",
            "gc",
            "uvvis",
            "ph_meter",
            "ir",
            "nmr",
            "calibrated_mass_balance",
        ],
        "spectra": {
            "hplc": hplc_chromatogram(
                values,
                species_amounts_mol=species_amounts_mol,
                volume_L=volume_L,
                seed=seed,
                replicate_count=replicate_count,
            ),
            "gc": gc_chromatogram(
                values,
                species_amounts_mol=species_amounts_mol,
                volume_L=volume_L,
                seed=seed + 1,
                replicate_count=replicate_count,
            ),
            "uvvis": uvvis_spectrum(
                values,
                species_amounts_mol=species_amounts_mol,
                volume_L=volume_L,
                seed=seed + 2,
                replicate_count=replicate_count,
            ),
            "ph_meter": ph_meter_signal(
                values,
                seed=seed + 5,
                replicate_count=replicate_count,
            ),
            "ir": ir_spectrum(
                values,
                species_amounts_mol=species_amounts_mol,
                volume_L=volume_L,
                seed=seed + 3,
                replicate_count=replicate_count,
            ),
            "nmr": nmr_spectrum(
                values,
                species_amounts_mol=species_amounts_mol,
                volume_L=volume_L,
                seed=seed + 4,
                replicate_count=replicate_count,
            ),
        },
        "mass_balance": {
            "process_mass_balance_error": round(
                _observed(values, "process_mass_balance_error"),
                6,
            ),
            "recovery": round(_observed(values, "recovery"), 6),
            "purity": round(_observed(values, "purity"), 6),
        },
        "metadata": {
            "synthetic": True,
            "real_sample_prediction": False,
            "boundary_note": (
                "Synthetic multi-channel benchmark evidence; not an empirical assay report."
            ),
        },
    }


def raw_signal_schema(instrument_id: str) -> dict[str, Any]:
    """Return the JSON-friendly raw-signal schema advertised by an instrument."""

    if instrument_id in {"nmr", "ir", "ms"}:
        return {
            "type": "object",
            "required": [
                "kind",
                "axis",
                "raw_signal",
                "calibration",
                "processed_estimates",
                "noise",
                "settings",
            ],
            "properties": {"kind": {"const": instrument_id + "_spectrum"}},
            "additionalProperties": True,
        }
    if instrument_id == "particle_size":
        return {
            "type": "object",
            "required": ["kind", "d50_um", "fines_number_fraction"],
            "properties": {"kind": {"const": "particle_size_signal"}},
            "additionalProperties": True,
        }
    if instrument_id in {"hplc", "gc"}:
        axis = "time_min"
        signal = "intensity"
    elif instrument_id == "uvvis":
        axis = "wavelength_nm"
        signal = "absorbance"
    elif instrument_id == "ph_meter":
        return {
            "type": "object",
            "required": [
                "kind",
                "pH",
                "electrode_mV",
                "sample_state",
                "axis",
                "raw_signal",
                "processed_estimates",
                "uncertainty",
                "calibration",
                "missingness",
                "metadata",
            ],
            "properties": {
                "kind": {"const": "ph_meter_signal"},
                "pH": {"type": "number", "minimum": 0.0, "maximum": 14.0},
                "hydrogen_mol_L": {"type": "number", "minimum": 0.0},
                "electrode_mV": {"type": "number"},
                "replicate_pH": {"type": "array", "items": {"type": "number"}},
                "processed_estimates": {"type": "object"},
                "uncertainty": {"type": "object"},
                "sample_state": {"type": "object"},
                "axis": {"type": "object"},
                "raw_signal": {"type": "object"},
                "peaks": {"type": "array"},
                "assignments": {"type": "array"},
                "calibration": {"type": "object"},
                "missingness": {"type": "object"},
                "metadata": {"type": "object"},
                "visibility": {"type": "string"},
                "source": {"type": "string"},
            },
            "additionalProperties": True,
        }
    elif instrument_id == "final_assay":
        return {
            "type": "object",
            "required": ["kind", "channels", "spectra", "mass_balance", "metadata"],
            "properties": {
                "kind": {"const": "final_assay_packet"},
                "channels": {"type": "array", "items": {"type": "string"}},
                "spectra": {"type": "object"},
                "mass_balance": {"type": "object"},
                "metadata": {"type": "object"},
                "visibility": {"type": "string"},
                "source": {"type": "string"},
            },
            "additionalProperties": True,
        }
    else:
        axis = "x"
        signal = "y"
    return {
        "type": "object",
        "required": [
            "schema_version",
            "kind",
            "instrument_id",
            axis,
            signal,
            "sample_state",
            "axis",
            "raw_signal",
            "peaks",
            "assignments",
            "processed_estimates",
            "uncertainty",
            "calibration",
            "missingness",
            "metadata",
        ],
        "properties": {
            "kind": {"type": "string"},
            axis: {"type": "array", "items": {"type": "number"}},
            signal: {"type": "array", "items": {"type": "number"}},
            "replicate_signals": {"type": "array"},
            "replicate_count": {"type": "integer", "minimum": 1},
            "peaks": {"type": "array"},
            "bands": {"type": "array"},
            "processed_estimates": {"type": "object"},
            "uncertainty": {"type": "object"},
            "metadata": {"type": "object"},
            "sample_state": {"type": "object"},
            "axis": {"type": "object"},
            "raw_signal": {"type": "object"},
            "assignments": {"type": "array"},
            "calibration": {"type": "object"},
            "missingness": {"type": "object"},
            "instrument_id": {"const": instrument_id},
            "schema_version": {"const": "chemworld-public-synthetic-signal-0.2"},
            "source": {"type": "string"},
            "visibility": {"type": "string"},
        },
        "additionalProperties": True,
    }


__all__ = [
    "SignalPeak",
    "final_assay_spectra",
    "gc_chromatogram",
    "hplc_chromatogram",
    "ir_spectrum",
    "nmr_spectrum",
    "ph_meter_signal",
    "raw_signal_schema",
    "uvvis_spectrum",
]
