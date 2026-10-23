"""Fixed 125-unit development block; NATIVE_CONV22D1_EXPERIMENT.md."""

import argparse
import json
import math
import time
import traceback
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import chemicals
import numpy as np
from chemicals.critical import Hekayati_Raeissi

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.critical_kernels import hekayati_raeissi_critical
from chemworld.physchem.critical_properties import (
    CriticalEstimate,
    eos_component_from_critical_inputs,
    estimate_critical_property,
)
from chemworld.physchem.eos import CubicEOSSpec, evaluate_cubic_eos
from chemworld.physchem.specs import ComponentSpec

FIXTURES = (
    (18.01528, .000018, .000056, 647.096, 22064000),
    (46.06844, .000058, .000168, 514.71, 6268000),
    (92.13842, .00010686, .000316, 591.75, 4108000),
    (142.286, .000195, .000624, 617.7, 2110000),
    (250., .0003, .0009, 800., 1500000),
    (400., .00045, .0013, 1000., 800000),
)
R = 8.31446261815324
METHOD = "Hekayati_Raeissi"


def decimal_reference(inputs):
    """Independent 50-digit expansion with the documented dimensional scales."""
    with localcontext() as ctx:
        ctx.prec = 50
        d = Decimal
        values = {k: d(str(v)) for k, v in inputs.items()}
        m, r = values["MW"], d("8.31446261815324")
        supplied_volume = "Vc" in values
        if supplied_volume:
            c = values["Vc"] * 1000
            factor = (-d(".001822904")*m*c*c + d(".00001947108")*m*m*c
                      + d("3.794091")*c - d(".00000002472981")*m*m*m)
        else:
            v = values["V_sat"] * 1000
            values["Vc"] = (
                -d("2.636518")*v*v*v + d(".01048744")*m*v + d("1.222756")*v
                + d(".000000006924349")*m*m*m - d(".00001102846")*m*m
                + d(".001871249")*m
            ) / 1000
            factor = (-d("143.4646")*v*v*v + d(".1781802")*m*v*v
                      + d("41.96576")*v*v - d(".00003443955")*m*m
                      + d(".009812262")*m + d(".1106396"))
        if "Tc" not in values:
            values["Tc"] = values["Pc"] * factor / (1000*r)
        if "Pc" not in values:
            values["Pc"] = 1000*r*values["Tc"] / factor
        return tuple(float(values[key]) for key in ("Tc", "Pc", "Vc"))


def compare(row, label, actual, expected):
    error = abs(actual-expected)
    tolerance = max(1e-13, 3e-12*abs(expected))
    row.setdefault("comparisons", []).append({
        "label": label, "actual": actual, "expected": expected,
        "absolute_error": error, "tolerance": tolerance,
    })
    assert math.isfinite(actual) and error <= tolerance, (label, actual, expected, tolerance)


def params(fixture, volume, known):
    all_inputs = dict(zip(("MW", "V_sat", "Vc", "Tc", "Pc"), fixture, strict=True))
    names = ["MW", *volume, *known]
    return {key: all_inputs[key] for key in names}


def combination(row, inputs):
    actual = hekayati_raeissi_critical(**inputs)
    expected = Hekayati_Raeissi(**inputs)
    decimal = decimal_reference(inputs)
    component = ComponentSpec("synthetic", "C7H8")
    reports = {}
    for i, key in enumerate(("Tc", "Pc", "Vc")):
        compare(row, "upstream/"+key, actual[i], expected[i])
        compare(row, "decimal50/"+key, actual[i], decimal[i])
        if key in inputs:
            assert actual[i] == inputs[key], "supplied critical value was changed"
        else:
            report = estimate_critical_property(component, method=METHOD,
                                                parameters=inputs, output_property=key)
            assert report.value == actual[i]
            assert CriticalEstimate.from_dict(json.loads(json.dumps(report.to_dict()))) == report
            reports[key] = report.to_dict()
    row["reports"] = reports


def cubic_reference(model, Tc, Pc, T, P):
    # Independent PR/SRK polynomial assembly in reduced variables, omega=.2.
    tr, pr, omega = T/Tc, P/Pc, .2
    if model == "peng_robinson":
        A = .45724*pr/tr**2*(1+(.37464+1.54226*omega-.26992*omega**2)*(1-tr**.5))**2
        B = .07780*pr/tr
        poly = [1, B-1, A-3*B**2-2*B, -A*B+B**2+B**3]
    else:
        A = .42747*pr/tr**2*(1+(.480+1.574*omega-.176*omega**2)*(1-tr**.5))**2
        B = .08664*pr/tr
        poly = [1, -1, A-B-B**2, -A*B]
    z = max(float(root.real) for root in np.roots(poly) if abs(root.imag) < 1e-10)
    return z, z*R*T/P


def consumer(row, inputs, model):
    component = ComponentSpec("synthetic", "C7H8")
    converted = {}
    for key, value in inputs.items():
        scale, unit = ({"MW": (.001, "kg/mol"), "Vc": (1000, "L/mol"),
                        "V_sat": (1000, "L/mol"), "Pc": (1e-5, "bar"),
                        "Tc": (1, "K")}[key])
        converted[key] = InputValue(value*scale, unit, None, "Fixed synthetic input")
    target = "Tc" if "Pc" in inputs else "Pc"
    report = estimate_critical_property(component, method=METHOD,
                                       parameters=converted, output_property=target)
    report = CriticalEstimate.from_dict(json.loads(json.dumps(report.to_dict())))
    fields = {key: report if key == target else converted[key] for key in ("Tc", "Pc")}
    spec = eos_component_from_critical_inputs(component, **fields,
                                              omega=InputValue(.2, "dimensionless", None,
                                                               "Fixed synthetic omega"))
    assert report.inputs["MW"]["unit"] == "kg/mol"
    assert "not directly verified" in report.citation
    expected_Tc, expected_Pc, _ = decimal_reference(inputs)
    Tc, Pc = spec.critical_temperature_K, spec.critical_pressure_Pa
    assert math.isclose(Tc, expected_Tc, rel_tol=3e-12)
    assert math.isclose(Pc, expected_Pc, rel_tol=3e-12)
    back_inputs = {key: value for key, value in inputs.items() if key not in {"Tc", "Pc"}}
    back_inputs[target] = report.value
    reversed_values = hekayati_raeissi_critical(**back_inputs)
    for i, key in enumerate(("Tc", "Pc")):
        if key in inputs:
            assert math.isclose(reversed_values[i], inputs[key], rel_tol=3e-12)
    row["report"] = report.to_dict()
    for tr, pr in ((1.05, .05), (1.5, .4)):
        T, P = tr*expected_Tc, pr*expected_Pc
        actual = evaluate_cubic_eos(CubicEOSSpec("Hekayati", model, (spec,)),
                                   {"synthetic": 1}, temperature_K=T, pressure_Pa=P)
        z, volume = cubic_reference(model, expected_Tc, expected_Pc, T, P)
        compare(row, f"{tr}/{pr}/Z", actual.compressibility_factor, z)
        compare(row, f"{tr}/{pr}/Vm", actual.molar_volume_m3_mol, volume)


def rejections():
    baseline = {"MW": 92., "V_sat": .00010686, "Tc": 600., "Pc": 4108000., "Vc": .000316}
    yield "no-Tc-Pc", {k: v for k, v in baseline.items() if k not in {"Tc", "Pc"}}
    yield "no-volumes", {k: v for k, v in baseline.items() if k not in {"Vc", "V_sat"}}
    for key in baseline:
        for name, invalid in (("nan", float("nan")), ("zero", 0.), ("negative", -1.)):
            yield f"{key}/{name}", {**baseline, key: invalid}
    yield "wrong-saturation-temperature", {**baseline, "saturation_temperature_K": 300.}
    yield "negative-volume-fit", {"MW": 92., "V_sat": .01, "Tc": 600.}
    yield "negative-critical-volume-factor", {"MW": 92., "Vc": .1, "Tc": 600.}
    yield "negative-saturation-factor", {"MW": 92., "V_sat": .00085, "Tc": 600.}
    yield "no-subcritical-liquid-known", {"MW": 92., "V_sat": .00010686, "Tc": 298.15}
    yield "no-subcritical-liquid-estimated", {"MW": 92., "V_sat": .00010686, "Pc": 1000.}


def units():
    for i, fixture in enumerate(FIXTURES):
        for volume in (("V_sat",), ("Vc",), ("V_sat", "Vc")):
            for known in (("Tc",), ("Pc",), ("Tc", "Pc")):
                inputs = params(fixture, volume, known)
                yield f"combination/{i}/{'+'.join(volume)}/{'+'.join(known)}", inputs, None
        for volume in (("V_sat",), ("Vc",)):
            for known in (("Tc",), ("Pc",)):
                for model in ("peng_robinson", "srk"):
                    inputs = params(fixture, volume, known)
                    yield f"consumer/{i}/{volume[0]}/{known[0]}/{model}", inputs, model
    for identity, inputs in rejections():
        yield "reject/"+identity, inputs, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier attempts; select a new output")
    assert chemicals.__version__ == "1.5.2"
    cases = list(units())
    assert len(cases) == 125 and len({c[0] for c in cases}) == 125
    start, rows = time.perf_counter(), []
    for identity, inputs, model in cases:
        # Strict JSON preserves nonfinite rejection inputs as explicit strings.
        row = {"unit": identity, "inputs": {
            k: v if math.isfinite(v) else str(v) for k, v in inputs.items()}}
        try:
            if identity.startswith("combination/"):
                combination(row, inputs)
            elif identity.startswith("consumer/"):
                consumer(row, inputs, model)
            else:
                try:
                    hekayati_raeissi_critical(**inputs)
                except ValueError as exc:
                    row["rejection"] = str(exc)
                else:
                    raise AssertionError("invalid input was accepted")
            row["status"] = "passed"
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"Hekayati-Raeissi: {len(rows)}/{len(cases)} units", flush=True)
    result = {"mode": "development", "reference_version": chemicals.__version__,
              "reference_provenance": (
                  "public implementation expressions; paper not directly verified"),
              "total": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "comparisons": sum(len(r.get("comparisons", ())) for r in rows),
              "families": dict(Counter(r["unit"].split('/')[0] for r in rows)),
              "seconds": time.perf_counter()-start, "results": rows}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    for row in rows:
        if row["status"] == "failed":
            print(row["unit"], row["failure"])
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
