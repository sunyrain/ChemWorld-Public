# ruff: noqa: B023 -- callbacks are consumed synchronously in their creating iteration
"""Fixed 126-card / 1380-state / 11040-comparison INCOMP block.

Every unavailable/domain result is retained separately from numerical agreement.
CoolProp is used only in this independent reference runner.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from chemworld.data.incompressible import (
    SOURCE_COMMIT,
    incompressible_catalog,
    incompressible_fluid,
)

PROPERTIES = (
    ("density", "rhomass", 1e-7), ("specific_heat", "cpmass", 1e-5),
    ("enthalpy_J_kg", "hmass", 1e-4), ("entropy_J_kg_K", "smass", 1e-6),
    ("internal_energy_J_kg", "umass", 1e-4), ("viscosity", "viscosity", 1e-12),
    ("conductivity", "conductivity", 1e-10), ("drho_dT_kg_m3_K", "rho_T", 1e-8),
)


def attempt(function):
    try:
        return function()
    except Exception as exc:
        return exc


def extract(value, function):
    if isinstance(value, Exception):
        raise value
    return function(value)


def compare(inputs, name, native, reference, atol, rtol=2e-8):
    left, right = attempt(native), attempt(reference)
    row = {"inputs": inputs, "quantity": name, "atol": atol, "rtol": rtol}
    for key, value in (("native", left), ("reference", right)):
        if isinstance(value, Exception):
            row[key+"_error"] = f"{type(value).__name__}: {value}"
        elif not math.isfinite(value):
            row[key+"_error"] = "nonfinite result: " + str(value)
        else:
            row[key] = float(value)
    if "native" in row and "reference" in row:
        error = abs(left-right)
        row.update(absolute_error=error,
                   status="numeric_pass" if error <= atol+rtol*abs(right) else "numeric_failure")
    else:
        row["status"] = ("both_unavailable" if "native_error" in row and "reference_error" in row
                         else "native_unavailable" if "native_error" in row
                         else "reference_unavailable")
    return row


def run():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("qualification requires locked CoolProp 8.0.0")
    cards = incompressible_catalog()
    rows, states, derivative_rows, caloric_diagnostic = [], [], [], []
    assert len(cards) == 126
    started = time.monotonic()

    def reference(fluid, T, p, x):
        backend = CP.AbstractState("INCOMP", fluid.fluid_id)
        setters = {"pure": "set_mass_fractions", "mass": "set_mass_fractions",
                   "mole": "set_mole_fractions", "volume": "set_volu_fractions"}
        setter = setters[fluid.fraction_basis]
        getattr(backend, setter)([x])
        backend.update(CP.PT_INPUTS, p, T)
        return backend

    for card_index, fluid in enumerate(cards, 1):
        xmin, xmax = fluid.fraction_bounds
        concentrations = (1.,) if fluid.fraction_basis == "pure" else tuple(
            xmin+q*(xmax-xmin) for q in (.1, .5, .9))
        lower, upper = fluid.temperature_bounds_K
        for x in concentrations:
            for q in (.1, .5, .9):
                T = lower+q*(upper-lower)
                for p in (1e5, 2e6):
                    inputs = {"fluid": fluid.fluid_id, "T_K": T, "p_Pa": p, "x": x,
                              "fraction_basis": fluid.fraction_basis}
                    native = attempt(lambda: fluid.state(T, p, x))
                    ref = attempt(lambda: reference(fluid, T, p, x))
                    states.append({"inputs": inputs, "native": str(native)
                                   if isinstance(native, Exception) else native.to_dict()})
                    for key, method, atol in PROPERTIES:
                        def native_value(key=key):
                            if key in {"density", "specific_heat", "viscosity", "conductivity"}:
                                return fluid.property_value(key, T, p, x)
                            return extract(native, lambda s: getattr(s, key))

                        def reference_value(method=method):
                            if method == "rho_T":
                                return extract(ref, lambda s: s.first_partial_deriv(
                                    CP.iDmass, CP.iT, CP.iP))
                            return extract(ref, lambda s: getattr(s, method)())

                        rows.append(compare(inputs, key, native_value, reference_value, atol))
        elapsed = time.monotonic()-started
        print(f"INCOMP cards {card_index}/126; states {len(states)}/1380; "
              f"checks {len(rows)}/11040; {card_index/max(elapsed, .001):.1f} cards/s; "
              f"ETA {elapsed*(126-card_index)/card_index:.1f}s", flush=True)
    for name in ("MEG", "MCA", "APG", "DowQ"):
        fluid = incompressible_fluid(name)
        xmin, xmax = fluid.fraction_bounds
        concentrations = (1.,) if fluid.fraction_basis == "pure" else tuple(
            xmin+q*(xmax-xmin) for q in (.25, .5, .75))
        T = sum(fluid.temperature_bounds_K)/2
        for x in concentrations:
            for p in (1e5, 2e6):
                inputs = {"fluid": name, "T_K": T, "p_Pa": p, "x": x}
                state = attempt(lambda: fluid.state(T, p, x))
                fields = (("mass_density_kg_m3", "drho_dT_kg_m3_K", "drho_dx_kg_m3", 1e-7),
                          ("enthalpy_J_kg", "cp_J_kg_K", "dh_dx_J_kg", .02),
                          ("entropy_J_kg_K", None, "ds_dx_J_kg_K", 1e-4))
                for field, t_field, x_field, atol in fields:
                    for axis, step in (("T", .001), ("x", 1e-5)):
                        def analytic(axis=axis, t_field=t_field, x_field=x_field):
                            return extract(state, lambda s: getattr(s, x_field) if axis == "x"
                                           else s.cp_J_kg_K/T if t_field is None
                                           else getattr(s, t_field))

                        def difference(axis=axis, step=step, field=field):
                            if axis == "x" and fluid.fraction_basis == "pure":
                                return 0.0
                            values = [getattr(fluid.state(T+i*step if axis == "T" else T, p,
                                                         x+i*step if axis == "x" else x), field)
                                      for i in (1, -1)]
                            return (values[0]-values[1])/(2*step)

                        derivative_rows.append(compare(inputs, field+"_"+axis, analytic,
                                                       difference, atol, 2e-5))
                def diagnostic():
                    ref = reference(fluid, T, p, x)
                    difference = (reference(fluid, T+.001, p, x).hmass()
                                  - reference(fluid, T-.001, p, x).hmass())/.002
                    return {"cp_fit": ref.cpmass(), "exported_h_T": ref.first_partial_deriv(
                        CP.iHmass, CP.iT, CP.iP), "finite_difference_h_T": difference,
                        "native_h_T": extract(state, lambda s: s.cp_J_kg_K)}
                diagnostic_result = attempt(diagnostic)
                caloric_diagnostic.append({"inputs": inputs, "result": str(diagnostic_result)
                                           if isinstance(diagnostic_result, Exception)
                                           else diagnostic_result})
    assert len(states) == 1380 and len(rows) == 11040 and len(derivative_rows) == 120
    return {"evidence_mode": "development", "reference": "CoolProp 8.0.0",
            "parameter_source_commit": SOURCE_COMMIT, "cards": [f.fluid_id for f in cards],
            "states": states, "comparisons": rows, "derivatives": derivative_rows,
            "caloric_diagnostic": caloric_diagnostic,
            "comparison_counts": dict(Counter(r["status"] for r in rows)),
            "derivative_counts": dict(Counter(r["status"] for r in derivative_rows)),
            "elapsed_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite an earlier incompressible result")
    result = run()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("comparison_counts", "derivative_counts")}),
          flush=True)


if __name__ == "__main__":
    main()
