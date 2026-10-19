"""Finite calibrated reporter spectra, with estimates fitted from acquired traces.

Channels are anonymous synthetic assay reporters. They are not predictions of
the private chemical mechanism's molecular structure or experimental spectra.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.optimize import nnls

from chemworld.physchem.mass_spectrometry import (
    FragmentIonSpec,
    MassSpectrumAnalyteSpec,
    simulate_mass_spectrum,
)
from chemworld.physchem.nmr import (
    NMRMultiplicity,
    ProtonNMRMethodSpec,
    ProtonNMRSignalSpec,
    simulate_proton_nmr,
)
from chemworld.physchem.spectroscopy import IRFunctionalGroupBandSpec

CHANNELS = ("reactant_reporter", "target_reporter", "impurity_reporter")
SPECTRAL_MODEL_ID = "finite-calibrated-reporter-spectra-v1"


def calibration(instrument: str, config: dict[str, float]) -> tuple[np.ndarray, dict[str, Any]]:
    scale = config["resolution_factor"]
    profiles: list[dict[str, Any]] = []
    if instrument == "nmr":
        axis = np.linspace(0, 10, 2048)
        unit, width, shape = "ppm", 0.035 * scale, "lorentzian"
        method = ProtonNMRMethodSpec("finite-400MHz", 400, "anonymous", (), "reference", 0, 0)
        signals = tuple(
            ProtonNMRSignalSpec(
                name,
                name,
                shift,
                protons,
                cast(NMRMultiplicity, multiplicity),
                couplings,
                width * 400,
                "synthetic calibration reporter",
                SPECTRAL_MODEL_ID,
            )
            for name, shift, protons, multiplicity, couplings in (
                (CHANNELS[0], 2.1, 3, "t", (7.0,)),
                (CHANNELS[1], 4.2, 2, "q", (7.0,)),
                (CHANNELS[2], 7.1, 1, "d", (8.0,)),
            )
        )
        report = simulate_proton_nmr(
            signals, species_amounts_mol=dict.fromkeys(CHANNELS, 1.0), method=method
        )
        for signal in report.signals:
            profiles.append(
                {
                    "channel": signal.species_id,
                    "shape": shape,
                    "fwhm": width,
                    "lines": [
                        [line.chemical_shift_ppm, line.relative_intensity * signal.proton_count]
                        for line in signal.lines
                    ],
                }
            )
    elif instrument == "ir":
        axis = np.linspace(400, 4000, 1024)
        unit, width, shape = "cm^-1", 25.0 * scale, "gaussian"
        for name, centers in zip(CHANNELS, ((1710, 1150), (1650, 1050), (1735, 1220)), strict=True):
            bands = [
                IRFunctionalGroupBandSpec(
                    name, "finite reporter band", float(center), width, strength
                )
                for center, strength in zip(centers, (1.0, 0.45), strict=True)
            ]
            profiles.append(
                {
                    "channel": name,
                    "shape": shape,
                    "fwhm": width,
                    "lines": [[b.center_cm_inv, b.relative_intensity] for b in bands],
                }
            )
    elif instrument == "ms":
        axis = np.linspace(10, 65, 2048)
        unit, width, shape = "m/z", 0.12 * scale, "gaussian"
        for name, intensities in zip(
            CHANNELS, ((100, 15, 30), (15, 100, 25), (30, 20, 100)), strict=True
        ):
            analyte = MassSpectrumAnalyteSpec(
                name,
                "C3H6O",
                1,
                "finite-fragment-library",
                tuple(
                    FragmentIonSpec(str(i), formula, 1, float(intensity), "calibration fragment")
                    for i, (formula, intensity) in enumerate(
                        zip(("CH3", "CHO", "C2H3O"), intensities, strict=True)
                    )
                ),
                1.0,
                0.0,
                SPECTRAL_MODEL_ID,
            )
            mass_report = simulate_mass_spectrum(analyte, analyte_amount_mol=1.0)
            profiles.append(
                {
                    "channel": name,
                    "shape": shape,
                    "fwhm": width,
                    "lines": [
                        [peak.mass_to_charge, 0.25 * peak.relative_intensity / 100]
                        for peak in mass_report.isotope_envelope
                    ]
                    + [
                        [fragment.mass_to_charge, fragment.relative_intensity / 100]
                        for fragment in mass_report.fragments
                    ],
                }
            )
    else:
        raise ValueError("Unknown configurable spectrum")
    for profile in profiles:
        profile["reference_fwhm"] = profile["fwhm"] / scale
    return axis, {
        "catalog_version": SPECTRAL_MODEL_ID,
        "axis_unit": unit,
        "profiles": profiles,
        "response_unit": "signal per mol/L after dilution",
        "baseline_intercept": 0.005,
        "baseline_drift": 0.002,
        "dilution_factor": config["dilution_factor"],
        "identities": "finite anonymous reporter catalog; not private mechanism species",
    }


def design_matrix(axis: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    columns = []
    for profile in contract["profiles"]:
        signal = np.zeros_like(axis)
        width = profile["fwhm"]
        for center, intensity in profile["lines"]:
            x = (axis - center) / width
            peak = (
                1 / (1 + 4 * x * x)
                if profile["shape"] == "lorentzian"
                else np.exp(-4 * np.log(2) * x * x)
            )
            signal += intensity * peak * profile["reference_fwhm"] / width
        columns.append(signal)
    return np.column_stack(columns)


def fit_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Recompute estimates from the public trace and public calibration only."""
    axis = np.asarray(packet["axis"]["values"], dtype=float)
    y = np.asarray(packet["raw_signal"]["values"], dtype=float)
    contract = packet["calibration"]
    matrix = design_matrix(axis, contract)
    baseline = contract["baseline_intercept"] + contract["baseline_drift"] * (axis - axis[0]) / (
        axis[-1] - axis[0]
    )
    fit, residual = nnls(matrix, y - baseline)
    dilution = contract["dilution_factor"]
    noise = packet["noise"]["standard_deviation"]
    covariance = np.linalg.pinv(matrix.T @ matrix)
    std = noise * np.sqrt(np.maximum(np.diag(covariance), 0)) * dilution
    condition = float(np.linalg.cond(matrix))
    saturated = packet["raw_signal"]["saturated_points"] > 0
    estimates, uncertainty, missingness, lod = {}, {}, {}, {}
    for index, name in enumerate(CHANNELS):
        value = float(fit[index] * dilution)
        limit = max(1e-6 * dilution, 3.3 * float(std[index]))
        reason = (
            "saturation"
            if saturated
            else "unidentifiable"
            if condition > 1e8
            else "below_lod"
            if value < limit
            else None
        )
        estimates[name] = None if reason else value
        uncertainty[name], missingness[name], lod[name] = float(std[index]), reason, limit
    return {
        "concentration_mol_L": estimates,
        "standard_error_mol_L": uncertainty,
        "missingness": missingness,
        "detection_limit_mol_L": lod,
        "fit_residual_norm": float(residual),
        "calibration_condition_number": condition,
        "method": "nonnegative least squares after declared baseline subtraction",
        "uncertainty_method": "linear covariance approximation, not empirical device uncertainty",
    }


def acquire_spectrum(
    instrument: str,
    concentrations: tuple[float, float, float],
    config: dict[str, float],
    rng: np.random.Generator,
    *,
    noise_multiplier: float = 1.0,
) -> dict[str, Any]:
    if any(not np.isfinite(value) or value < 0 for value in concentrations):
        raise ValueError("Spectral concentrations must be finite and nonnegative")
    axis, contract = calibration(instrument, config)
    matrix = design_matrix(axis, contract)
    mean = matrix @ (np.asarray(concentrations) / config["dilution_factor"])
    baseline = contract["baseline_intercept"] + contract["baseline_drift"] * (axis - axis[0]) / (
        axis[-1] - axis[0]
    )
    sigma = (
        {"nmr": 0.003, "ir": 0.002, "ms": 0.004}[instrument]
        * noise_multiplier
        / np.sqrt(config["scan_count"])
    )
    measured = mean + baseline + rng.normal(0, sigma, len(axis))
    saturated = int(np.count_nonzero(measured > 2.0))
    packet = {
        "kind": instrument + "_spectrum",
        "instrument": instrument,
        "axis": {"unit": contract["axis_unit"], "values": axis.tolist()},
        "raw_signal": {
            "unit": "relative_signal",
            "values": np.minimum(measured, 2.0).tolist(),
            "saturated_points": saturated,
            "upper_signal_limit": 2.0,
        },
        "settings": dict(config),
        "calibration": contract,
        "noise": {
            "distribution": "independent Gaussian",
            "standard_deviation": float(sigma),
            "scan_scaling": "inverse square root",
        },
        "metadata": {
            "synthetic": True,
            "catalog_version": SPECTRAL_MODEL_ID,
            "boundary": (
                "finite reporter calibration; no prediction of molecular structure or real spectra"
            ),
        },
    }
    packet["processed_estimates"] = fit_packet(packet)
    return packet
