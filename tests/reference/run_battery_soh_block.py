"""Execute all twelve fixed SOH units, retain failures, never overwrite an attempt."""

import argparse
import json
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from battery_soh_pybamm import case, reference
from scipy.integrate import quad

from chemworld.physchem.battery_soh import HalfCellSOHSpec, solve_electrode_soh


def numerical(spec, un, up, index):
    native = solve_electrode_soh(spec)
    expected, method = reference(spec, un, up, index)
    actual = list(native.solutions)
    checks = {
        "complete": native.status in {"unique", "multiple"},
        "branch_count": len(actual) == len(expected) if index != 6 else len(actual) >= 1,
    }
    comparisons = []
    for target in expected:
        keys = tuple(k for k in ("x_0", "x_100", "y_0", "y_100") if k in target)
        if not actual:
            checks["has_reference_branch"] = False
            continue
        s = min(actual, key=lambda b: max(abs(b[k] - target[k]) for k in keys))
        errors = {k: abs(s[k] - target[k]) for k in keys}
        capacity_error = abs(s["cell_capacity_C"] - target["cell_capacity_C"])
        comparisons.append(
            {
                "reference": target,
                "native": s,
                "endpoint_errors": errors,
                "capacity_error_C": capacity_error,
            }
        )
        checks[f"reference_branch_{len(comparisons)}"] = max(
            errors.values()
        ) <= 1e-7 and capacity_error <= 1e-4 + 1e-7 * abs(target["cell_capacity_C"])
    path_errors = []
    for s in actual:
        volts = [
            float(up(s[y]) - (0 if isinstance(spec, HalfCellSOHSpec) else un(s[x])))
            for x, y in (("x_0", "y_0"), ("x_100", "y_100"))
        ]
        residuals = np.abs(np.array(volts) - [spec.minimum_voltage_V, spec.maximum_voltage_V])
        # These extra interior evaluations check the accepted physical path with
        # independent functions; completeness is the native bounded search, not
        # an assertion that sampling establishes absence of roots.
        path = [
            float(
                up(s["y_0"] + soc * (s["y_100"] - s["y_0"]))
                - (
                    0
                    if isinstance(spec, HalfCellSOHSpec)
                    else un(s["x_0"] + soc * (s["x_100"] - s["x_0"]))
                )
            )
            for soc in np.linspace(0, 1, 101)
        ]
        path_errors.append(
            {
                "endpoint_voltage_errors_V": residuals.tolist(),
                "minimum_V": min(path),
                "maximum_V": max(path),
            }
        )
        checks[f"physical_path_{len(path_errors)}"] = (
            bool(np.max(residuals) <= 1e-8)
            and min(path) >= spec.minimum_voltage_V - 1e-8
            and max(path) <= spec.maximum_voltage_V + 1e-8
        )
    maps = []
    if index <= 8 and actual:
        for branch, s in enumerate(actual):
            for soc in (0.0, 0.25, 0.5, 0.75, 1.0):
                mapped = native.at_soc(soc, branch=branch)
                recovered = native.soc_at_voltage(mapped["voltage_V"], branch=branch)
                passed = recovered.complete and any(
                    abs(r["value"][0] - soc) <= 1e-7 for r in recovered.roots
                )
                maps.append(
                    {
                        "branch": branch,
                        "soc": soc,
                        "mapped": mapped,
                        "inverse": recovered.to_dict(),
                        "passed": passed,
                    }
                )
            energy, error = quad(
                lambda soc, s=s: (
                    float(
                        up(s["y_0"] + soc * (s["y_100"] - s["y_0"]))
                        - (
                            0
                            if isinstance(spec, HalfCellSOHSpec)
                            else un(s["x_0"] + soc * (s["x_100"] - s["x_0"]))
                        )
                    )
                    * s["cell_capacity_C"]
                ),
                0,
                1,
                epsabs=1e-9,
                epsrel=1e-11,
            )
            actual_energy = native.theoretical_energy_j(branch=branch)
            maps.append(
                {
                    "branch": branch,
                    "reference_energy_J": energy,
                    "quad_error_J": error,
                    "native_energy_J": actual_energy,
                    "passed": abs(actual_energy - energy) <= 1e-6 + 1e-8 * abs(energy),
                }
            )
        checks["soc_voltage_energy"] = all(r["passed"] for r in maps)
    sensitivity = []
    if index in (1, 3, 5) and len(actual) == 1:
        s = actual[0]
        local = s["local_parameter_response"]
        if local["status"] != "local_full_rank":
            checks["sensitivity"] = False
        else:
            names, coords = local["parameters"], local["coordinates"]
            output = "cyclable_lithium_C" if spec.known == "cell_capacity" else "cell_capacity_C"
            for col, name in enumerate(names):
                h = 1e-5 if name.endswith("_V") else abs(getattr(spec, name)) * 1e-4
                plus = reference(replace(spec, **{name: getattr(spec, name) + h}), un, up, index)[
                    0
                ][0]
                minus = reference(replace(spec, **{name: getattr(spec, name) - h}), un, up, index)[
                    0
                ][0]
                ref = np.array([(plus[k] - minus[k]) / (2 * h) for k in (*coords, output)])
                observed = np.r_[
                    np.array(local["coordinate_derivatives"])[:, col],
                    local["output_derivatives"][output][col],
                ]
                errors = np.abs(observed - ref)
                sensitivity.append(
                    {
                        "parameter": name,
                        "step": h,
                        "reference": ref.tolist(),
                        "native": observed.tolist(),
                        "errors": errors.tolist(),
                        "passed": bool(np.all(errors <= 1e-5 + 2e-3 * np.abs(ref))),
                    }
                )
            checks["sensitivity"] = all(r["passed"] for r in sensitivity)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "native": native.to_dict(),
        "reference_method": method,
        "comparisons": comparisons,
        "paths": path_errors,
        "soc_voltage_energy": maps,
        "sensitivity": sensitivity,
    }


def unit(index):
    spec, un, up = case(index)
    if index <= 10:
        return numerical(spec, un, up, index)
    native = solve_electrode_soh(spec)
    checks = {
        "expected_status": native.status == ("infeasible" if index == 11 else "unresolved"),
        "no_selected_solution": not native.solutions,
        "reason_retained": bool(native.infeasible_reason if index == 11 else native.unresolved),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "native": native.to_dict(),
        "reference_method": "inventory domain exclusion"
        if index == 11
        else "constant rank-one analytic voltage Jacobian; continuum of endpoint pairs",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pybamm
    import scipy

    data = {
        "mode": "development",
        "denominator": 12,
        "numerical_denominator": 10,
        "semantic_boundary_denominator": 2,
        "reference_version": pybamm.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "units": [],
    }
    start, stop = time.monotonic(), threading.Event()

    def progress():
        while not stop.wait(20):
            done = len(data["units"])
            elapsed = time.monotonic() - start
            print(
                f"SOH active {done}/12; elapsed={elapsed:.1f}s; "
                f"ETA={elapsed / done * (12 - done):.1f}s"
                if done
                else f"SOH active 0/12; elapsed={elapsed:.1f}s",
                flush=True,
            )

    with args.output.open("x", encoding="utf8") as output:
        worker = threading.Thread(target=progress, daemon=True)
        worker.start()
        try:
            for index in range(1, 13):
                unit_start = time.monotonic()
                try:
                    answer = unit(index)
                except Exception as exc:
                    answer = {
                        "passed": False,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                data["units"].append(
                    {"id": f"S{index:02}", "elapsed_s": time.monotonic() - unit_start, **answer}
                )
                data["elapsed_s"] = time.monotonic() - start
                data["passed"] = sum(r["passed"] for r in data["units"])
                data["numerical_passed"] = sum(r["passed"] for r in data["units"][:10])
                data["semantic_boundary_passed"] = sum(r["passed"] for r in data["units"][10:])
                payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                output.seek(0)
                output.write(payload)
                output.truncate()
                output.flush()
                print(
                    f"S{index:02} passed={answer['passed']}; {index}/12; "
                    f"elapsed={data['elapsed_s']:.2f}s",
                    flush=True,
                )
        finally:
            stop.set()
            worker.join()
    return int(data["passed"] != 12)


if __name__ == "__main__":
    raise SystemExit(main())
