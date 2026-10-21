"""All 18 original models against independent RoadRunner plus five closed forms.

The unchanged native third-run trajectories are reused in full. Original
COPASI failures remain failures, irrespective of this separate block's result.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
import traceback
from pathlib import Path


def load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trajectory_block = load_sibling("run_sbml_trajectory_block")
cases = trajectory_block.cases
TIMES, QUANTITIES = trajectory_block.TIMES, trajectory_block.QUANTITIES
ANALYTICAL = {2, 3, 4, 11, 17}


def closed_form(index, target, mp):
    """Fixed original mol/L/s models, not evaluated by any production function."""
    t, v0, w, c0, r = map(mp.mpf, (str(target), ".01", ".002", "1", ".1"))
    event_time = mp.mpf(2 if index == 11 else "2.5")
    after = t >= event_time
    volume = (v0+w*t if index in {2, 3} else v0*(1+mp.mpf(".1")*t)
              if index == 4 else mp.mpf(".04") if index == 11 and after else v0)
    if index == 3:
        def integral(c, v, u):
            return c*v*u+(c*w+r*v)*u*u/2+r*w*u**3/3

        ca = mp.mpf(2)+r*(t-event_time) if after else c0+r*t
        na = ca*volume
        nb = mp.mpf(".3")*(integral(c0, v0, event_time)
            + integral(mp.mpf(2), v0+w*event_time, t-event_time) if after else integral(c0, v0, t))
    else:
        k = mp.mpf(".15" if index == 17 else ".3")
        before_a = v0*mp.exp(-k*event_time)
        jump_amount = mp.mpf({2: ".03", 4: ".025", 11: ".02", 17: ".02"}[index])
        na = jump_amount*mp.exp(-k*(t-event_time)) if after else v0*mp.exp(-k*t)
        nb = v0-before_a+jump_amount-na if after else v0-na
    p = mp.mpf(5)+r*(t-event_time) if after else mp.mpf(3)+r*t
    vb = mp.mpf(".02")
    return [na, nb, volume, vb, na/volume, nb/vb, p]


def reference(directory, output):
    import importlib.metadata
    import sys

    import mpmath as mp
    import roadrunner

    versions = {name: importlib.metadata.version(name)
                for name in ("libroadrunner", "mpmath", "numpy")}
    if versions["libroadrunner"] != "2.8.0" or versions["mpmath"] != "1.3.0":
        raise ValueError("independent reference version differs from the fixed block")
    mp.mp.dps = 60
    report = {"expected_inputs": 18, "versions": versions, "units": []}
    started = time.perf_counter()
    selections = ["time", "A", "B", "c1", "c2", "[A]", "[B]", "p"]
    for name, scales, index, _ in cases():
        unit = {"input": name, "passed": False, "failures": []}
        report["units"].append(unit)
        try:
            text = (directory/name).read_text(encoding="utf-8")
            unit["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rr = roadrunner.RoadRunner(text)
            rr.setIntegrator("cvode")
            rr.integrator.setValue("relative_tolerance", 1e-11)
            rr.integrator.setValue("absolute_tolerance", 1e-13)
            rr.integrator.setValue("maximum_time_step", .02)
            rr.resetAll()
            rr.timeCourseSelections = selections
            data = rr.simulate(0., 4., 401)
            unit["CVODE_settings"] = {key: rr.integrator.getValue(key) for key in
                ("relative_tolerance", "absolute_tolerance", "maximum_time_step")}
            unit["raw_columns"], unit["raw_source_units_rows"] = list(data.colnames), data.tolist()
            if list(data.colnames) != selections:
                raise ValueError("RoadRunner output selections differ from the declared quantities")
            samples = []
            for target in TIMES:
                rows = [row for row in data if abs(float(row[0])-target) <= 1e-9]
                if not rows:
                    raise ValueError("RoadRunner omitted a fixed output time")
                _, na, nb, va, vb, ca, cb, p = map(float, rows[-1])
                q, v = scales["amount_mol"], scales["volume_L"]
                values = [na*q, nb*q, va*v, vb*v, ca*q/v, cb*q/v, p]
                if not all(math.isfinite(x) for x in values):
                    raise ValueError("nonfinite RoadRunner quantity")
                samples.append({"time_source": target, "values": values})
            unit["samples"] = samples
            if index in ANALYTICAL:
                unit["analytical_samples"] = []
                for target in TIMES:
                    values = closed_form(index, target, mp)
                    unit["analytical_samples"].append({"time_source": target,
                        "values": list(map(float, values)), "mpmath_60": list(map(str, values))})
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
        if "chemworld" in sys.modules or "COPASI" in sys.modules:
            raise ValueError("independent reference imported a native or other SDK module")
        driver.write(output, report)
        trajectory_block.progress("reference", report, started)


def comparisons(actual, reference, label):
    result = []
    if len(actual) != len(TIMES) or len(reference) != len(TIMES):
        raise ValueError("full fixed trajectory nodes are required")
    for target, left, right in zip(TIMES, actual, reference, strict=True):
        if left["time_source"] != target or abs(right["time_source"]-target) > 1e-9:
            raise ValueError("trajectory time nodes differ")
        for key, a, b in zip(QUANTITIES, left["values"], right["values"], strict=True):
            absolute, relative = ((2e-10, 2e-6) if key.startswith("N:") else
                (1e-12, 2e-8) if key.startswith("V:") else
                (2e-8, 2e-6) if key.startswith("C:") else (1e-8, 2e-6))
            tolerance = absolute+relative*abs(b)
            result.append({"comparison": label, "time_source": target, "quantity": key,
                "actual": a, "reference": b, "error": abs(a-b), "tolerance": tolerance,
                "passed": math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance})
    return result


def native(directory, reference_report, output):
    baseline = Path(os.environ["CHEMWORLD_SBML_NATIVE_BASELINE"]).resolve()
    original = json.loads((baseline/"native.json").read_text(encoding="utf-8"))
    original_reference = json.loads((baseline/"reference.json").read_text(encoding="utf-8"))
    if original["executed_inputs"] != 18 or original["comparison_count"] != 1386:
        raise ValueError("reuse requires the full original 18-source native execution")
    report = {"expected_inputs": 18, "expected_comparisons": 2156, "units": [],
              "native_baseline": str(baseline), "executed_inputs": 0,
              "original_COPASI_passed_inputs": original["passed_inputs"],
              "original_COPASI_complete_and_passed": original["complete_and_passed"]}
    started = time.perf_counter()
    for name, _, index, _ in cases():
        unit = {"input": name, "passed": False, "failures": [], "comparisons": []}
        report["units"].append(unit)
        try:
            ref = next(u for u in reference_report["units"] if u["input"] == name)
            prior_ref = next(u for u in original_reference["units"] if u["input"] == name)
            prior = next(u for u in original["units"] if u["input"] == name)
            source_hash = hashlib.sha256((directory/name).read_bytes()).hexdigest()
            if not ref["passed"] or not source_hash == ref["sha256"] == prior_ref["sha256"]:
                raise ValueError("reference/source invalid or different from original native input")
            rows = comparisons(prior["samples"], ref["samples"], "native_RoadRunner")
            if index in ANALYTICAL:
                rows.extend(comparisons(prior["samples"], ref["analytical_samples"],
                                        "native_closed_form"))
                rows.extend(comparisons(ref["samples"], ref["analytical_samples"],
                                        "RoadRunner_closed_form"))
            unit["comparisons"] = rows
            unit["failures"] = [row for row in rows if not row["passed"]]
            unit["passed"] = not unit["failures"]
            report["executed_inputs"] += 1
        except Exception:
            unit["failures"].append(traceback.format_exc())
        driver.write(output, report)
        trajectory_block.progress("native", report, started)
    report["passed_inputs"] = sum(u["passed"] for u in report["units"])
    report["unexecuted_inputs"] = 18-report["executed_inputs"]
    report["comparison_count"] = sum(len(u["comparisons"]) for u in report["units"])
    report["complete_and_passed"] = (report["passed_inputs"] == 18
                                      and report["comparison_count"] == 2156)
    driver.write(output, report)


if __name__ == "__main__":
    driver = load_sibling("run_sbml_math_block")
    driver.__file__ = __file__
    driver.inputs, driver.reference, driver.native = cases, reference, native
    raise SystemExit(driver.main())
