"""The predeclared eight-unit nonlinear OCV block; preserves all outcomes."""

import argparse
import json
import platform
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from battery_dfn_pybamm import build_dfn_reference
from battery_frequency_pybamm import reference_frequency
from battery_ocv_cases import models, released_functions, source_curve
from battery_spme_pybamm import build_spme_reference
from run_battery_frequency_block import native_frequency, values
from scipy.integrate import quad

from chemworld.physchem.battery_electrolyte import REGIONS
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_models import advance_battery, battery_readout
from chemworld.physchem.reactor_shared import ScalarControlProgram

PROTOCOL = ((600.0, 0.3), (200.0, 0.0), (300.0, -0.2))


def material_comparison(electrode):
    import pybamm

    curve, table = source_curve(electrode)
    function = released_functions()[electrode]
    symbolic = pybamm.InputParameter("theta")
    expression = function(symbolic)
    derivatives = (expression, expression.diff(symbolic), expression.diff(symbolic).diff(symbolic))
    points = []
    for theta in np.linspace(0.01, 0.99, 99):
        original = curve.derivatives(float(theta), order=2)
        expected = [float(d.evaluate(inputs={"theta": float(theta)})) for d in derivatives]
        actual_integral = curve.integral(0.01, float(theta))
        expected_integral, integration_error = quad(function, 0.01, float(theta), epsabs=1e-12)
        actual = np.r_[original, actual_integral]
        reference = np.r_[expected, expected_integral]
        bounds = np.r_[1e-10, 1e-8 + 1e-8 * np.abs(reference[1:3]), 1e-9]
        errors = np.abs(actual - reference)
        points.append(
            {
                "theta": float(theta),
                "native": actual.tolist(),
                "reference": reference.tolist(),
                "errors": errors.tolist(),
                "bounds": bounds.tolist(),
                "reference_quad_error_V": integration_error,
                "passed": bool(np.all(np.isfinite(actual)) and np.all(errors <= bounds)),
            }
        )
    return {
        "passed": all(p["passed"] for p in points),
        "points": points,
        "curve": curve.to_dict(),
        "parameter_table": table.to_dict(),
        "response_order": ["U_V", "dU_dtheta_V", "d2U_dtheta2_V", "integral_V"],
    }


def time_comparison(spec, family, progress):
    functions = released_functions()
    if family == "DFN":
        simulation = build_dfn_reference(
            spec, spatial_cells=64, radial_cells=240, ocp_functions=functions
        )
    else:
        simulation = build_spme_reference(
            spec,
            cells=120,
            radial_cells=240,
            radial_mesh="exponential-right",
            family=family,
            ocp_functions=functions,
        )
    state, rows = spec.initial_state(), []
    for index, (duration, current) in enumerate(PROTOCOL, 1):
        progress(f"{family} native segment {index}/3")
        clock = np.linspace(0.0, duration, 31)
        begin = state.time_s
        result = advance_battery(
            spec,
            state,
            ScalarControlProgram(((0.0, current),), "A"),
            duration,
            evaluation_times_s=clock,
        )
        progress(f"{family} reference segment {index}/3")
        solution = simulation.step(duration, t_eval=clock, inputs={"Current function [A]": current})
        ref_times = begin + clock
        if index > 1:
            ref_times[0] += 1e-8  # Read the new input just after the shared segment boundary.
        native_rows = [battery_readout(spec, s, current) for s in result.trajectory]
        actual = np.array(
            [
                [
                    r.voltage_V,
                    r.negative_surface_fraction,
                    r.positive_surface_fraction,
                    s.coordinates["discharged_charge_C"],
                ]
                for r, s in zip(native_rows, result.trajectory, strict=True)
            ]
        )
        reference = np.column_stack(
            [
                np.asarray(solution[v](ref_times)).reshape(-1)
                for v in (
                    "Voltage [V]",
                    "X-averaged negative particle surface concentration",
                    "X-averaged positive particle surface concentration",
                    "Discharge capacity [A.h]",
                )
            ]
        )
        reference[:, 3] *= 3600
        salt_actual, salt_reference = np.zeros((31, 0)), np.zeros((31, 0))
        if spec.electrolyte is not None:
            e = spec.electrolyte
            salt_actual = np.array(
                [
                    [e.region_average(e.concentrations(s), r) for r in REGIONS]
                    for s in result.trajectory
                ]
            )
            salt_reference = np.column_stack(
                [
                    np.asarray(
                        solution[f"X-averaged {r} electrolyte concentration [mol.m-3]"](ref_times)
                    ).reshape(-1)
                    for r in REGIONS
                ]
            )
        error = np.abs(actual - reference)
        bounds = np.column_stack(
            [
                np.full(31, 0.005),
                np.full(31, 0.005),
                np.full(31, 0.005),
                1e-4 + 0.005 * np.abs(reference[:, 3]),
            ]
        )
        conservation = dict(result.conservation_residuals)
        conserved = all(
            abs(v) <= (1e-6 if k.endswith("_C") else 1e-10) for k, v in conservation.items()
        )
        complete = (
            result.outcome == "success"
            and result.actual_duration_s == duration
            and solution.termination == "final time"
            and len(actual) == 31
        )
        salt_error = np.abs(salt_actual - salt_reference)
        rows.append(
            {
                "segment": index,
                "duration_s": duration,
                "current_A": current,
                "native_outcome": result.outcome,
                "native_event": result.stopping_event,
                "reference_termination": solution.termination,
                "times_s": (begin + clock).tolist(),
                "native": actual.tolist(),
                "reference": reference.tolist(),
                "errors": error.tolist(),
                "bounds": bounds.tolist(),
                "native_salt_mol_m3": salt_actual.tolist(),
                "reference_salt_mol_m3": salt_reference.tolist(),
                "maximum_salt_error_mol_m3": float(np.max(salt_error, initial=0)),
                "conservation": conservation,
                "passed": bool(
                    complete
                    and conserved
                    and np.all(np.isfinite(actual))
                    and np.all(error <= bounds)
                    and np.all(salt_error <= 20)
                ),
            }
        )
        state = result.final_state
    return {
        "passed": all(r["passed"] for r in rows),
        "segments": rows,
        "response_order": ["V", "negative_surface_fraction", "positive_surface_fraction", "C"],
        "reference_grid": {
            "radial_cells": 240,
            "radial_mesh": "exponential-right",
            "stretch": 2.3,
            "space_cells_per_region": 64 if family == "DFN" else 120,
        },
    }


def frequency_comparison(spec, family, progress):
    frequencies = np.logspace(-5, -1, 13)
    progress(f"{family} native frequencies")
    if spec.electrolyte is not None:
        native = native_frequency(spec, frequencies)
    else:
        state = spec.initial_state()
        response = battery_impedance(spec, frequencies, state=state)
        native = {
            "success": response.success and state == spec.initial_state(),
            "points": [
                {
                    "frequency_Hz": float(f),
                    "response_real": [z.real, 0.0, 0.0, 0.0],
                    "response_imag": [z.imag, 0.0, 0.0, 0.0],
                }
                for f, z in zip(frequencies, response.impedance_ohm, strict=True)
            ],
            "diagnostics": response.response.diagnostics,
        }
    progress(f"{family} reference frequencies")
    reference = reference_frequency(
        spec, family, frequencies, refined=True, ocp_functions=released_functions()
    )
    result = {"passed": False, "native": native, "reference": reference}
    if not native["success"] or not reference["success"]:
        return result
    actual, expected = values(native), values(reference)
    bound = np.array([0.002, 1.0, 1.0, 1.0]) + np.array([0.005, 0.01, 0.01, 0.01]) * np.abs(
        expected
    )
    errors = np.abs(actual - expected)
    result.update(
        passed=bool(np.all(errors <= bound)), errors=errors.tolist(), bounds=bound.tolist()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pybamm

    data = {
        "mode": "development",
        "total": 8,
        "completed": 0,
        "units": [],
        "block": "sourced nonlinear OCV consumption",
        "native_conv45_complete": False,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pybamm": pybamm.__version__,
        },
    }
    begin = time.monotonic()
    status, stop = {"stage": "initialization"}, threading.Event()

    def progress(stage):
        status["stage"] = stage
        print(stage, flush=True)

    def heartbeat():
        while not stop.wait(20):
            elapsed, count = time.monotonic() - begin, data["completed"]
            eta = "unknown" if count == 0 else f"{elapsed / count * (8 - count):.1f}s"
            print(
                f"{status['stage']}: {count}/8, elapsed={elapsed:.1f}s, "
                f"rate={60 * count / elapsed:.2f}/min, ETA={eta}",
                flush=True,
            )

    # Instantiate/validate the complete fixed matrix before executing its first solve.
    model_cases = models()
    units = [("negative", None), ("positive", None), *model_cases, *model_cases]
    with args.output.open("x") as output:
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            for index, (name, spec) in enumerate(units, 1):
                started = time.monotonic()
                progress(f"M{index:02} {name} starting")
                try:
                    result = (
                        material_comparison(name)
                        if index <= 2
                        else time_comparison(spec, name, progress)
                        if index <= 5
                        else frequency_comparison(spec, name, progress)
                    )
                except Exception:
                    result = {"passed": False, "exception": traceback.format_exc()}
                data["units"].append(
                    {
                        "unit": f"M{index:02}",
                        "name": name,
                        "parameters": None if spec is None else spec.to_dict(),
                        **result,
                        "elapsed_s": time.monotonic() - started,
                    }
                )
                data["completed"] = index
                data["elapsed_s"] = time.monotonic() - begin
                data["passed"] = sum(r["passed"] for r in data["units"])
                serialized = json.dumps(data, indent=2, allow_nan=False)
                output.seek(0)
                output.write(serialized)
                output.truncate()
                output.flush()
                print(f"M{index:02}: passed={result['passed']}, {index}/8 complete", flush=True)
        finally:
            stop.set()
            worker.join()
    print(f"Completed: {data['passed']}/8", flush=True)
    return int(data["passed"] != 8)


if __name__ == "__main__":
    raise SystemExit(main())
