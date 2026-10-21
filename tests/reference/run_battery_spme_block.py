"""Execute the predeclared 13-unit development block; retain all failures."""

import argparse
import json
import platform
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pybamm
import scipy
from battery_spme_pybamm import reference_spme

from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import BatterySPMSpec, advance_battery, battery_readout
from chemworld.physchem.reactor_shared import ScalarControlProgram


def constant(current):
    return ScalarControlProgram(((0, current),), "A")


def compare(spec, segments, *, samples=31, include_spm=False, required_events=None):
    # First locate both physical endpoints, then request the reference solver at
    # the prescribed common times. PyBaMM's Solution linearly interpolates its
    # saved states; that is not BDF dense output and can miss the initial transient.
    references = reference_spme(spec, segments)
    state = spec.initial_state()
    native_results, clocks = [], []
    for index, (duration, current) in enumerate(segments):
        if index >= len(references):
            raise RuntimeError("reference stopped before a required protocol segment")
        ref_start, ref_end, _, _ = references[index]
        native = advance_battery(spec, state, constant(current), duration)
        common = min(native.actual_duration_s, ref_end - ref_start)
        if common <= 0:
            raise RuntimeError("no positive common physical interval")
        clocks.append(np.linspace(0, common, samples))
        native_results.append(native)
        state = native.final_state
    references = reference_spme(spec, segments, evaluation_times=clocks)
    state = spec.initial_state()
    rows = []
    for index, (_duration, current) in enumerate(segments):
        if index >= len(references):
            raise RuntimeError("reference stopped before a required protocol segment")
        ref_start, ref_end, _, solution = references[index]
        native = native_results[index]
        common = min(native.actual_duration_s, ref_end - ref_start)
        if common <= 0:
            raise RuntimeError("no positive common physical interval")
        sampled = advance_battery(spec, state, constant(current), common,
                                  evaluation_times_s=np.linspace(0, common, samples))
        points = sampled.trajectory[:samples]
        elapsed = np.array([s.time_s - state.time_s for s in points])
        # A step's input changes immediately AFTER the previous segment's end.
        ref_times = np.clip(ref_start + elapsed, ref_start + (1e-8 if index else 0), ref_end)
        native_v = np.array([battery_readout(spec, s, current).voltage_V for s in points])
        ref_v = np.asarray(solution["Voltage [V]"](ref_times)).reshape(-1)
        native_c = np.array([spec.electrolyte.concentrations(s) for s in points]).T
        ref_c = np.asarray(solution["Electrolyte concentration [mol.m-3]"](
            ref_times, x=spec.electrolyte.mesh.centers_m)).reshape(native_c.shape)
        surface_errors = []
        for name in ("negative", "positive"):
            values = np.array([getattr(battery_readout(spec, s, current),
                                       f"{name}_surface_fraction") for s in points])
            reference = np.asarray(solution[
                f"X-averaged {name} particle surface concentration"](ref_times)).reshape(-1)
            surface_errors.append(float(np.max(abs(values - reference))))
        potential_errors = {}
        for field, variable in (
            ("concentration_overpotential_V", "X-averaged concentration overpotential [V]"),
            ("electrolyte_ohmic_V", "X-averaged electrolyte ohmic losses [V]"),
            ("solid_ohmic_V", "X-averaged solid phase ohmic losses [V]"),
        ):
            actual = np.array([getattr(battery_readout(spec, s, current), field) for s in points])
            expected = np.asarray(solution[variable](ref_times)).reshape(-1)
            potential_errors[field] = float(np.max(abs(actual - expected)))
        voltage_error = float(np.max(abs(native_v - ref_v)))
        concentration_error = float(np.max(abs(native_c - ref_c)))
        time_error = abs(native.actual_duration_s - (ref_end - ref_start))
        conserved = all(abs(value) <= (1e-6 if name.endswith("_C") else 1e-10)
                        for name, value in native.conservation_residuals.items())
        events_match = (required_events is None or
                        (native.stopping_event, solution.termination) == required_events)
        passed = bool(voltage_error <= 0.005 and concentration_error <= 20
                      and max(surface_errors) <= 0.003 and conserved
                      and time_error <= max(1e-4, 0.01 * (ref_end - ref_start))
                      and len(points) == samples and events_match)
        row = {
            "segment": index + 1, "passed": passed, "current_A": current,
            "native_duration_s": native.actual_duration_s,
            "reference_duration_s": ref_end - ref_start,
            "native_event": native.stopping_event, "reference_event": solution.termination,
            "voltage_max_error_V": voltage_error,
            "electrolyte_max_error_mol_m3": concentration_error,
            "surface_max_errors": surface_errors, "time_error_s": time_error,
            "potential_component_max_errors_V": potential_errors,
            "required_events_match": events_match,
            "conservation_residuals": dict(native.conservation_residuals),
            "elapsed_s": elapsed.tolist(), "native_voltage_V": native_v.tolist(),
            "reference_voltage_V": ref_v.tolist(),
            "native_final_concentration_mol_m3": native_c[:, -1].tolist(),
            "reference_final_concentration_mol_m3": ref_c[:, -1].tolist(),
        }
        if include_spm and index == 0:
            spm = replace(spec, electrolyte=None)
            base = advance_battery(spm, spm.initial_state(), constant(current), common,
                                    evaluation_times_s=elapsed)
            base_v = np.array([battery_readout(spm, s, current).voltage_V
                               for s in base.trajectory[:samples]])
            row["unchanged_parameter_spm_max_error_V"] = float(np.max(abs(base_v - ref_v)))
        rows.append(row)
        state = native.final_state
    return {"passed": all(row["passed"] for row in rows), "segments": rows}


def limit(spec, multiplier):
    e = spec.electrolyte
    fast = replace(spec, electrolyte=replace(
        e, diffusivity_m2_s=e.diffusivity_m2_s * multiplier,
        conductivity_S_m=e.conductivity_S_m * multiplier,
        solid_conductivity_S_m=tuple(s * multiplier for s in e.solid_conductivity_S_m),
    ))
    base = replace(spec, electrolyte=None)
    current = spec.current_at_c_rate(0.5)
    times = np.linspace(0, 100, 31)
    a = advance_battery(fast, fast.initial_state(), constant(current), 100,
                        evaluation_times_s=times)
    b = advance_battery(base, base.initial_state(), constant(current), 100,
                        evaluation_times_s=times)
    error = max(abs(battery_readout(fast, x, current).voltage_V
                    - battery_readout(base, y, current).voltage_V)
                for x, y in zip(a.trajectory, b.trajectory, strict=True))
    return {"passed": bool(error <= 1 / multiplier), "multiplier": multiplier,
            "maximum_voltage_difference_V": float(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier block results")
    base = BatterySPMSpec(electrolyte=BatteryElectrolyteSpec())
    rate = base.current_at_c_rate
    cases = [
        (f"discharge-{r}C", lambda r=r: compare(base, [(3960 / r, rate(r))], include_spm=True))
        for r in (0.1, 1.0, 3.0)
    ]
    cases.extend([
        ("charge", lambda: compare(base, [(5000, rate(-0.5))])),
        ("discharge-rest", lambda: compare(base, [(150, rate(0.5)), (500, 0)])),
        ("pulse", lambda: compare(base, [(40, rate(1)), (60, 0), (30, rate(-0.5))])),
        ("salt-cutoff", lambda: compare(replace(base, electrolyte=replace(
            base.electrolyte, diffusivity_m2_s=1e-12, minimum_concentration_mol_m3=990)),
            [(100, 0.3)], required_events=("battery:electrolyte:depleted",
                                          "event: native:electrolyte:depleted"))),
        ("voltage-cutoff", lambda: compare(replace(base, minimum_voltage_V=3.6),
            [(5000, rate(0.5))], required_events=("battery:minimum-voltage",
                                                 "event: Minimum voltage [V]"))),
    ])
    for cells in (10, 20, 40):
        spec = replace(base, negative=replace(base.negative, cells=2 * cells),
                       positive=replace(base.positive, cells=2 * cells),
                       electrolyte=replace(base.electrolyte, cells=(cells,) * 3))
        cases.append((f"grid-{cells}", lambda spec=spec: compare(
            spec, [(200, rate(0.5))], samples=21)))
    for multiplier in (1e4, 1e6):
        cases.append((f"spm-limit-{multiplier:g}", lambda m=multiplier: limit(base, m)))
    results = []
    started = time.monotonic()
    for i, (identity, run) in enumerate(cases, 1):
        print(f"START {i}/{len(cases)} {identity}", flush=True)
        unit_start = time.monotonic()
        try:
            outcome = run()
        except Exception as exc:
            outcome = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
        results.append({"id": identity, "elapsed_s": time.monotonic() - unit_start, **outcome})
        print(f"END {i}/{len(cases)} {identity}: {outcome['passed']} "
              f"({results[-1]['elapsed_s']:.1f}s)", flush=True)
    grid = [row for row in results if row["id"].startswith("grid-")]
    convergence = False
    if all("segments" in row for row in grid):
        coarse, fine = grid[0]["segments"][0], grid[-1]["segments"][0]
        convergence = bool(fine["voltage_max_error_V"] <= coarse["voltage_max_error_V"] + 1e-7
                           and fine["electrolyte_max_error_mol_m3"]
                           <= coarse["electrolyte_max_error_mol_m3"] + 1e-5)
        grid[-1]["passed"] = grid[-1]["passed"] and convergence
    payload = {
        "block": "NATIVE-CONV-43", "units": len(cases),
        "passed": sum(bool(row["passed"]) for row in results),
        "failed": sum(not row["passed"] for row in results), "skipped": 0,
        "convergence": convergence, "elapsed_s": time.monotonic() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "pybamm": pybamm.__version__},
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}), flush=True)
    return bool(payload["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
