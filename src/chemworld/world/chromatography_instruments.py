"""Finite programmed instruments: one trace and signal-derived estimates."""

import numpy as np

from chemworld.foundation import ChannelAxis, Instrument, Observation, ObservationChannel
from chemworld.runtime.chromatography import (
    ANALYSIS_S,
    COST,
    INSTRUMENT_IDS,
    SAMPLE_L,
    model,
    programme,
    sample_outlet,
    sample_transform,
    settings,
)
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract
from chemworld.world.spectra import SignalPeak, _gaussian


def trace_basis(program, analytes, axis):
    """Unit-area peaks reuse the existing Gaussian and plate-width equation."""
    columns, resolved = [], []
    centers, widths = [], []
    for i, analyte in enumerate(analytes):
        rt = program.retention_time_s(analyte)
        centers.append(rt)
        sigma = 0 if rt is None else rt / np.sqrt(analyte.theoretical_plates)
        widths.append(sigma)
        columns.append(
            np.zeros_like(axis)
            if rt is None
            else _gaussian(axis, SignalPeak(rt, sigma, 1, f"peak_{i + 1}"))
        )
        resolved.append(rt is not None and rt + 4 * sigma <= axis[-1])
    for i, rt in enumerate(centers):
        if rt is not None:
            for j in range(i):
                if centers[j] is not None and abs(rt - centers[j]) < 3 * (widths[i] + widths[j]):
                    resolved[i] = resolved[j] = False
    return np.column_stack(columns), resolved


def public_trace_basis(axis, profile):
    """Reconstruct the declared method basis without access to simulator state."""
    return np.column_stack(
        [
            np.zeros_like(axis)
            if peak["retention_s"] is None
            else _gaussian(axis, SignalPeak(peak["retention_s"], peak["width_s"], 1, peak["id"]))
            for peak in profile["peaks"]
        ]
    )


def fit_public_trace(axis, signal, basis, resolved, calibration, *, saturated=False):
    """No material, equipment truth or physical gain enters this inverse fit."""
    # Finite three-peak nonnegative least squares, with unrestricted affine
    # baseline: enumerate the eight active sets and choose the lowest residual.
    # No hidden physical baseline or gain is used by the inverse problem.
    best = None
    for active in range(8):
        indices = [i for i in range(3) if active & (1 << i)]
        matrix = np.column_stack((basis[:, indices], np.ones_like(axis), axis / axis[-1]))
        solution = np.linalg.lstsq(matrix, signal, rcond=None)[0]
        if any(value < 0 for value in solution[: len(indices)]):
            continue
        residual = np.sum((np.sum(matrix * solution, axis=1) - signal) ** 2)
        if best is None or residual < best[0]:
            areas = np.zeros(3)
            areas[indices] = solution[: len(indices)]
            best = residual, areas
    if best is None:
        raise ValueError("Public chromatogram fit did not converge")
    areas = best[1]
    estimates = [
        max(0.0, (a - c["intercept"]) / c["slope"]) for a, c in zip(areas, calibration, strict=True)
    ]
    valid = [r and not saturated for r in resolved]
    return areas, estimates, valid


def read_chromatography(state, action, rng):
    instrument = action["instrument"]
    config = settings(state, instrument)
    params, analytes = model(state, instrument)
    program = programme(state, instrument)
    receipt = config["chromatography_acquisition"]
    axis = np.linspace(0, ANALYSIS_S, 2401)
    basis, resolved = trace_basis(program, analytes, axis)
    count = config["use_count"]
    gain = params["initial_gain"] + (count - 1) * params["gain_per_analysis"]
    baseline = params["baseline"] + (count - 1) * params["baseline_per_analysis"]
    amplitudes = [
        gain
        * (
            a.detector_response_slope * receipt["effluent_mol"][a.analyte_id] / SAMPLE_L
            + a.detector_response_intercept
        )
        for a in analytes
    ]
    signal = np.sum(basis * np.asarray(amplitudes), axis=1)
    signal += baseline + params["baseline_drift"] * axis / ANALYSIS_S
    signal += rng.normal(0, params["noise_sd"], len(axis))
    saturated = bool(np.any(signal > params["saturation_AU"]))
    signal = np.minimum(signal, params["saturation_AU"])
    calibration = config.get(
        "calibration",
        [
            {"slope": a.detector_response_slope, "intercept": a.detector_response_intercept}
            for a in analytes
        ],
    )
    # These nominal method templates and response calibrations are explicitly
    # public, not a hidden truth-assisted estimate. The finite instrument assumes
    # known peak identities/shapes; it does not discover unknown chromatographic
    # mechanisms. Gain, baseline, sample composition and residue remain private.
    profile = {
        "peaks": [
            {
                "id": f"peak_{i}",
                "retention_s": receipt["retention_s"][a.analyte_id],
                "width_s": (
                    None
                    if receipt["retention_s"][a.analyte_id] is None
                    else receipt["retention_s"][a.analyte_id] / np.sqrt(a.theoretical_plates)
                ),
            }
            for i, a in enumerate(analytes, 1)
        ],
        "resolved": [bool(value) for value in resolved],
        "response": calibration,
    }
    areas, concentrations, valid = fit_public_trace(
        axis,
        signal,
        public_trace_basis(axis, profile),
        profile["resolved"],
        profile["response"],
        saturated=saturated,
    )
    channels = [
        ObservationChannel(
            "chromatogram",
            "AU",
            tuple(signal),
            quality="saturated" if saturated else "valid",
            axis=ChannelAxis("elapsed", "s", tuple(axis)),
            uncertainty=params["noise_sd"],
        )
    ]
    for i, (area, concentration, ok) in enumerate(
        zip(areas, concentrations, valid, strict=True), 1
    ):
        quality = "valid" if ok else "invalid"
        channels.extend(
            (
                ObservationChannel(
                    f"peak_{i}_area",
                    "AU*s",
                    float(area) if ok else None,
                    quality,
                    role="processed",
                    derived_from=("chromatogram",),
                    acquisition_relation="acquired_trace_fit",
                ),
                ObservationChannel(
                    f"peak_{i}_concentration",
                    "mol/L",
                    float(concentration) if ok else None,
                    quality,
                    role="processed",
                    derived_from=(f"peak_{i}_area",),
                    acquisition_relation="acquired_trace_fit",
                ),
            )
        )
    return Observation(
        channels=tuple(channels),
        raw_signal={"processing_calibration": profile},
        instrument_id=instrument,
        cost=COST,
        sample_consumed_L=SAMPLE_L,
    )


def chromatography_instrument_specs():
    result = []
    for name in INSTRUMENT_IDS:
        definition = Instrument(name, "Finite programmed chromatograph", (), COST, SAMPLE_L, {})
        contract = InstrumentContract(
            instrument_id=name,
            observable_keys=(),
            input_state_schema={"required_phase": "liquid", "sample_consumption_L": SAMPLE_L},
            axis_contract={"unit": "s", "points": 2401},
            raw_signal_schema={"chromatogram": {"unit": "AU"}},
            processed_estimate_schema={"peak_area": "AU*s", "concentration": "mol/L"},
            uncertainty_model="Gaussian detector noise; no fitted uncertainty claim",
            noise_model={},
            calibration_contract={"method": "public standard trace regression"},
            detection_contract={},
            saturation_contract={"policy": "clip trace, invalidate fits"},
            baseline_drift_contract={"owner": "instrument"},
            missingness_contract={"overlap_or_uneluted": "invalid concentration"},
            cost=COST,
            latency_s=0,
            sample_consumption_L=SAMPLE_L,
            destructive=True,
            requires_terminated=False,
            calibration_profile="finite-programmed-chromatography-v1",
            synthetic_boundary="Uniform programme, symmetric peaks, lumped carryover; no PDE",
            clock_semantics="offline_snapshot",
            analysis_time_s=ANALYSIS_S,
        )
        result.append(
            InstrumentSpec(
                definition,
                contract,
                native_observer=read_chromatography,
                sample_transform=sample_transform,
                sample_outlet=sample_outlet,
            )
        )
    return tuple(result)
