"""Run all 15 predeclared DFN development units and retain every failure."""

import argparse
import json
import platform
import threading
import time
import traceback
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pybamm
import scipy
from battery_dfn_pybamm import reference_dfn

from chemworld.physchem.battery_dfn import BatteryDFNSpec, dfn_fields
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    ParticleSpec,
    advance_battery,
    battery_readout,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram


def program(value, unit="A"):
    return ScalarControlProgram(((0, value),), unit)


def conserved(result):
    return all(abs(v) <= (1e-6 if key.endswith("_C") else 1e-10)
               for key, v in result.conservation_residuals.items())


def compare(spec, segments, event=None):
    references = reference_dfn(spec, segments)
    state = spec.initial_state()
    native_results, clocks = [], []
    for index, (duration, value, unit) in enumerate(segments):
        start, end, _ = references[index]
        result = advance_battery(spec, state, program(value, unit), duration)
        common = min(result.actual_duration_s, end - start)
        if common <= 0:
            raise RuntimeError("no positive common physical interval")
        native_results.append(result)
        clocks.append(np.linspace(0, common, 31))
        state = result.final_state
    references = reference_dfn(spec, segments, evaluation_times=clocks)
    rows, state = [], spec.initial_state()
    e = spec.electrolyte
    nn = e.cells[0]
    for index, (_, value, unit) in enumerate(segments):
        start, end, ref = references[index]
        original = native_results[index]
        clock = clocks[index]
        sampled = advance_battery(spec, state, program(value, unit), float(clock[-1]),
                                  evaluation_times_s=clock)
        points = sampled.trajectory[:31]
        times = np.clip(start + np.array([s.time_s - state.time_s for s in points]),
                        start + (1e-8 if index else 0), end)
        fields = [dfn_fields(spec, s, program=program(value, unit)) for s in points]

        def error(actual, variable, x=None, ref=ref, times=times):
            expected = np.asarray(ref[variable](times, **({} if x is None else {"x": x})))
            actual = np.asarray(actual)
            expected = expected.reshape(actual.shape)
            return float(np.max(abs(actual - expected))), expected

        ve, rv = error([f["voltage_V"] for f in fields], "Voltage [V]")
        ie, _ = error([f["current_A"] for f in fields], "Current [A]")
        ce, rc = error(np.array([f["electrolyte_concentration_mol_m3"] for f in fields]).T,
                       "Electrolyte concentration [mol.m-3]", e.mesh.centers_m)
        pe, _ = error(np.array([f["electrolyte_potential_V"] for f in fields]).T,
                       "Electrolyte potential [V]", e.mesh.centers_m)
        details, passed = {}, True
        for name, selection in (("negative", slice(0, nn)), ("positive", slice(nn, None))):
            x = e.mesh.centers_m[e.region_slices[name]]
            for field, variable, tolerance in (
                ("surface_fraction", f"{name.capitalize()} particle surface concentration", 0.005),
                ("solid_potential_V", f"{name.capitalize()} electrode potential [V]", 0.005),
                ("interfacial_current_A_m2", f"{name.capitalize()} electrode interfacial current "
                 "density [A.m-2]", None),
            ):
                err, expected = error(
                    np.array([f[field][selection] for f in fields]).T, variable, x)
                bound = tolerance if tolerance is not None else 0.005 + 0.05 * float(
                    np.max(abs(expected)))
                details[f"{name}:{field}"] = {"max_error": err, "bound": bound}
                passed &= err <= bound
        actual_q = np.array([s.coordinates["discharged_charge_C"] for s in points])
        expected_q = np.asarray(ref["Discharge capacity [A.h]"](times)).reshape(-1) * 3600
        qerrors = abs(actual_q - expected_q)
        time_error = abs(original.actual_duration_s - (end - start))
        event_ok = event is None or (original.stopping_event == event[0]
                                     and ref.termination == event[1])
        passed &= (len(points) == 31 and ve <= 0.005 and ce <= 20 and pe <= 0.005 and ie <= 0.005
                   and bool(np.all(qerrors <= 1e-4 + 0.005 * abs(expected_q)))
                   and time_error <= max(1e-4, 0.01 * (end - start))
                   and conserved(original) and event_ok)
        rows.append({"segment": index + 1, "passed": bool(passed), "voltage_max_error_V": ve,
            "current_max_error_A": ie, "salt_max_error_mol_m3": ce,
            "electrolyte_potential_max_error_V": pe, "field_errors": details,
            "charge_max_error_C": float(max(qerrors)), "time_error_s": time_error,
            "native_duration_s": original.actual_duration_s, "reference_duration_s": end - start,
            "native_event": original.stopping_event, "reference_event": ref.termination,
            "required_events_match": event_ok,
            "conservation": dict(original.conservation_residuals),
            "native_voltage_V": [f["voltage_V"] for f in fields],
            "reference_voltage_V": rv.tolist(),
            "native_final_salt": fields[-1]["electrolyte_concentration_mol_m3"].tolist(),
            "reference_final_salt": rc[:, -1].tolist(), "times_s": times.tolist()})
        state = original.final_state
    return {"passed": all(r["passed"] for r in rows), "segments": rows}


def limit(spec, kind):
    values = spec.to_dict()
    values.pop("model_family")
    rate = 0.02 if kind == "spme" else 0.5
    if kind == "spm":
        spec = replace(spec, electrolyte=replace(spec.electrolyte,
            diffusivity_m2_s=spec.electrolyte.diffusivity_m2_s * 1e6,
            conductivity_S_m=spec.electrolyte.conductivity_S_m * 1e6,
            solid_conductivity_S_m=tuple(s * 1e6 for s in spec.electrolyte.solid_conductivity_S_m)))
        values["electrolyte"] = None
    control = BatterySPMSpec(**values)
    current = spec.current_at_c_rate(rate)
    clock = np.linspace(0, 100, 31)
    a = advance_battery(spec, spec.initial_state(), program(current), 100, evaluation_times_s=clock)
    b = advance_battery(control, control.initial_state(), program(current), 100,
                        evaluation_times_s=clock)
    av = np.array([battery_readout(spec, s, current).voltage_V for s in a.trajectory])
    bv = np.array([battery_readout(control, s, current).voltage_V for s in b.trajectory])
    ve = float(max(abs(av - bv)))
    ce = 0.0
    if kind == "spme":
        ce = float(np.max(abs(np.array([spec.electrolyte.concentrations(s) for s in a.trajectory])
                              - np.array([control.electrolyte.concentrations(s)
                                          for s in b.trajectory]))))
    return {"passed": bool(ve <= (0.003 if kind == "spme" else 1e-5) and ce <= 2
                            and a.actual_duration_s == b.actual_duration_s == 100 and conserved(a)),
            "voltage_max_error_V": ve, "salt_max_error_mol_m3": ce,
            "conservation": dict(a.conservation_residuals)}


def units():
    base = BatteryDFNSpec(negative=ParticleSpec(cells=24), positive=ParticleSpec(cells=24),
                          electrolyte=BatteryElectrolyteSpec(cells=(8, 8, 8)))
    for i, rate, duration in ((1, 0.1, 300), (2, 1, 100), (3, 3, 100)):
        yield f"D{i:02}", base, [(duration, base.current_at_c_rate(rate), "A")], None, None
    variant = replace(base, electrolyte=replace(base.electrolyte, area_m2=0.03,
        lengths_m=(8e-5, 3e-5, 12e-5), porosities=(0.35, 0.55, 0.45),
        diffusivity_m2_s=3e-10, conductivity_S_m=0.8, solid_conductivity_S_m=(70, 120)),
        negative=replace(base.negative, radius_m=8e-6, diffusivity_m2_s=2e-14,
                         exchange_current_scale_A_m2=0.8),
        positive=replace(base.positive, radius_m=12e-6, diffusivity_m2_s=8e-15,
                         exchange_current_scale_A_m2=1.3))
    yield "D04", variant, [(100, variant.current_at_c_rate(0.7), "A")], None, None
    yield "D05", base, [(100, base.current_at_c_rate(-0.5), "A")], None, None
    yield "D06", base, [(100, base.current_at_c_rate(0.7), "A"), (100, 0, "A")], None, None
    yield "D07", base, [(30, base.current_at_c_rate(r), "A") for r in (0.7, -0.7)], None, None
    yield "D08", base, [(50, 3.75, "V")], None, None
    voltage = battery_readout(base, base.initial_state(), 0.3).voltage_V - 0.01
    yield "D09", replace(base, minimum_voltage_V=voltage), [(500, 0.3, "A")], (
        "battery:minimum-voltage", "event: Minimum voltage [V]"), None
    salt = replace(base, electrolyte=replace(base.electrolyte,
        diffusivity_m2_s=1e-12, minimum_concentration_mol_m3=990))
    yield "D10", salt, [(100, 0.3, "A")], (
        "battery:electrolyte:depleted", "event: native:electrolyte:depleted"), None
    for i, nx, nr in ((11, 4, 12), (12, 8, 24), (13, 16, 48)):
        grid = replace(base, negative=replace(base.negative, cells=nr),
            positive=replace(base.positive, cells=nr), electrolyte=replace(base.electrolyte,
                                                                          cells=(nx, nx, nx)))
        yield f"D{i}", grid, [(100, base.current_at_c_rate(1), "A")], None, None
    yield "D14", base, [], None, "spme"
    yield "D15", base, [], None, "spm"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, rows = time.monotonic(), []
    active, stop = ["initializing"], threading.Event()

    def progress():
        while not stop.wait(30):
            elapsed = time.monotonic() - start
            print(json.dumps({"completed": len(rows), "total": 15, "active": active[0],
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(elapsed * (15 - len(rows)) / len(rows)) if rows else None}),
                flush=True)

    with args.output.open("x") as output:
        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        for identity, spec, segments, event, kind in units():
            active[0] = identity
            started = time.monotonic()
            print(f"{identity}: starting ({len(rows)}/15 complete)", flush=True)
            try:
                result = limit(spec, kind) if kind else compare(spec, segments, event)
            except Exception:
                result = {"passed": False, "exception": traceback.format_exc()}
            rows.append({"unit": identity, "parameters": spec.to_dict(), "protocol": segments,
                          "limit": kind, "elapsed_s": time.monotonic() - started, **result})
            print(f"{identity}: passed={result['passed']} ({len(rows)}/15 complete)", flush=True)
        grid = [next(r for r in rows if r["unit"] == f"D{i}") for i in (11, 12, 13)]
        convergence = False
        if all("segments" in r for r in grid):
            convergence = all(
                b["segments"][0][key] <= a["segments"][0][key] + tolerance
                for a, b in pairwise(grid)
                for key, tolerance in (
                    ("voltage_max_error_V", 1e-6), ("salt_max_error_mol_m3", 0.01)))
        if not convergence:
            for row in grid:
                row["passed"] = False
                row["grid_convergence_failed"] = True
        result = {"mode": "development", "total": 15,
            "passed": sum(r["passed"] for r in rows), "grid_converged": convergence,
            "elapsed_s": time.monotonic() - start, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pybamm": pybamm.__version__,
            "units": rows}
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
        stop.set()
        thread.join()
        print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["passed"] == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
