"""Fixed public pressure method/derivative/Jacobian/recovery development block."""
# ruff: noqa: C408 -- explicit named scientific parameters in fixed cases

import argparse
import json
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import vapor_pressure as vp
from pressure_reference import ReferencePressure, finite_difference
from run_pressure_data_block import check_inverse, near

from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.vapor_pressure import (
    antoine_coefficients_from_point,
    arrhenius_pressure_extrapolation,
    dippr101_coefficients_from_point,
    vapor_pressure_parameter_jacobian,
    vapor_pressure_temperature_derivative,
)
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)

CASES = (
    ("antoine", "Antoine", dict(A=8.5, B=1500.0, C=-30.0, base=10.0), (300.0, 350.0, 400.0)),
    (
        "trc_antoine_extended",
        "TRC_Antoine_extended",
        dict(Tc=227.51, to=-120.0, A=8.95894, B=510.595, C=-15.95, n=2.41377, E=-93.74, F=7425.9),
        (180.0, 190.0, 200.0),
    ),
    (
        "wagner",
        "Wagner_original",
        dict(Tc=600.0, Pc=3e6, a=-7.0, b=1.0, c=-1.0, d=-2.0),
        (300.0, 350.0, 400.0),
    ),
    (
        "wagner25",
        "Wagner",
        dict(Tc=600.0, Pc=3e6, a=-7.0, b=1.0, c=-1.0, d=-2.0),
        (300.0, 350.0, 400.0),
    ),
    (
        "yaws_psat",
        "Yaws_Psat",
        dict(A=30.7129, B=-2469.0, C=-7.351, D=2.8025e-10, E=2.7361e-6),
        (300.0, 350.0, 400.0),
    ),
    (
        "tde_pv_expansion",
        "TDE_PVExpansion",
        dict(a1=20.0, a2=-3000.0, a3=0.25, a4=0.001, a5=1e-6, a6=100.0, a7=1e-18, a8=1e4),
        (300.0, 350.0, 400.0),
    ),
    (
        "boiling_critical_pressure",
        "boiling_critical_relation",
        dict(Tb=409.3, Tc=617.1, Pc=36e5),
        (300.0, 350.0, 400.0),
    ),
    (
        "lee_kesler_pressure",
        "Lee_Kesler",
        dict(Tc=617.1, Pc=36e5, omega=0.299),
        (300.0, 350.0, 400.0),
    ),
    (
        "ambrose_walton_pressure",
        "Ambrose_Walton",
        dict(Tc=617.1, Pc=36e5, omega=0.299),
        (300.0, 350.0, 400.0),
    ),
    ("sanjari_pressure", "Sanjari", dict(Tc=617.1, Pc=36e5, omega=0.299), (300.0, 350.0, 400.0)),
    ("edalat_pressure", "Edalat", dict(Tc=617.1, Pc=36e5, omega=0.299), (300.0, 350.0, 400.0)),
    (
        "clapeyron_sublimation",
        "Psub_Clapeyron",
        dict(Tt=273.15, Pt=611.0, Hsub_t=51100.0),
        (200.0, 225.0, 250.0),
    ),
    (
        "arrhenius_pressure",
        "Arrhenius_extrapolation",
        dict(T_ref=400.0, P_ref=1e5, slope=-1600.0),
        (300.0, 350.0, 400.0),
    ),
    ("iapws_saturation", "Psat_IAPWS", {}, (300.0, 373.15, 600.0)),
    ("dippr101", "EQ101", dict(A=70.0, B=-7000.0, C=-7.0, D=1e-6, E=2.0), (300.0, 350.0, 400.0)),
    ("dippr115", "EQ115", dict(A=70.0, B=-7000.0, C=-7.0, D=1e-6, E=100.0), (300.0, 350.0, 400.0)),
)
JACOBIANS = {
    "Antoine": ("A", "B", "C"),
    "TRC_Antoine_extended": ("to", "A", "B", "C", "n", "E", "F"),
    "Wagner_original": tuple("abcd"),
    "Wagner": tuple("abcd"),
    "Yaws_Psat": tuple("ABCDE"),
}


def curve_for(equation, params, Ts):
    return PropertyCorrelation(
        equation,
        equation,
        params,
        {"temperature": "K"},
        "Pa",
        "sublimation_pressure" if equation == "clapeyron_sublimation" else "vapor_pressure",
        {"temperature": (min(Ts) * 0.99, max(Ts) * 1.01)},
        "Fixed independent method qualification",
    )


def method_point(curve, ref, T):
    pressure = ref.evaluate(T)
    near(evaluate_correlation(curve, temperature_K=T).value, pressure)
    measures = {"pressure": pressure, "derivatives": []}
    for order in range(1, 4 if curve.equation_id == "arrhenius_pressure" else 3):
        actual = vapor_pressure_temperature_derivative(curve, temperature_K=T, order=order).value
        try:
            expected = ref.evaluate(T, order)
        except AttributeError:
            first, second = finite_difference(ref.evaluate, T, order)
            near(first, second, 2e-5, 1e-9)
            near(actual, second, 2e-5, 1e-9)
            kind = "independent_five_point_h_and_h_over_2"
            expected = second
        else:
            near(actual, expected)
            kind = "public_analytic"
        measures["derivatives"].append(
            {"order": order, "reference_kind": kind, "actual": actual, "expected": expected}
        )
    check_inverse(curve, T, pressure, (T * 0.999, T * 1.001))
    return measures


def jacobian(curve, method, params, T):
    names = JACOBIANS[method]
    expected = getattr(vp, method + "_fitting_jacobian")([T], **params)[0]
    actual = vapor_pressure_parameter_jacobian(curve, (T,), parameters=names).values[0]
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        near(a, b, 2e-9, 1e-9)
    return {"parameters": names, "actual": actual, "expected": expected}


def recover(method):
    T, params = 350.0, dict(A=8.5, B=1500.0, C=-30.0, base=10.0)
    P = vp.Antoine(T, **params)
    first, second = vp.dAntoine_dT(T, **params), vp.d2Antoine_dT2(T, **params)
    point = dict(temperature_K=T, pressure_Pa=P, first_derivative_Pa_K=first)
    if method == "AB":
        actual = antoine_coefficients_from_point(**point)
        expected, keys = vp.Antoine_AB_coeffs_from_point(T, P, first), "AB"
    elif method == "ABC":
        actual = antoine_coefficients_from_point(**point, second_derivative_Pa_K2=second)
        expected, keys = vp.Antoine_coeffs_from_point(T, P, first, second), "ABC"
    elif method == "DIPPR":
        actual = dippr101_coefficients_from_point(**point, second_derivative_Pa_K2=second)
        expected, keys = vp.DIPPR101_ABC_coeffs_from_point(T, P, first, second), "ABC"
    else:
        curve = curve_for("antoine", params, (300, 400))
        actual = arrhenius_pressure_extrapolation(
            curve, anchor_temperature_K=T, temperature_bounds_K=(200, 500)
        ).coefficients
        expected, keys = vp.Arrhenius_parameters(T, P, first), ("T_ref", "P_ref", "slope")
    for k, value in zip(keys, expected, strict=True):
        near(actual[k], value, 2e-9, 1e-9)
    return {"actual": actual, "expected": expected}


def water_point(T):
    pressure = vp.Psat_IAPWS(T)
    near(water_saturation_pressure(T), pressure)
    inverse = water_saturation_temperature(pressure)
    assert abs(inverse - T) <= 2e-8
    assert abs(inverse - vp.Tsat_IAPWS(pressure)) <= 2e-8
    return {"temperature_K": T, "pressure_Pa": pressure, "inverse_K": inverse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert chemicals.__version__ == "1.5.2"
    units = []
    for equation, method, params, temperatures in CASES:
        curve = curve_for(equation, params, temperatures)
        ref = ReferencePressure(equation, method, params)
        for T in temperatures:
            units.append((f"{equation}:{T}", lambda c=curve, r=ref, T=T: method_point(c, r, T)))
        if method in JACOBIANS:
            for T in temperatures:
                units.append(
                    (
                        f"jacobian:{method}:{T}",
                        lambda c=curve, m=method, p=params, T=T: jacobian(c, m, p, T),
                    )
                )
    for method in ("AB", "ABC", "DIPPR", "Arrhenius"):
        units.append(("recover:" + method, lambda m=method: recover(m)))
    for T in (273.15, 300.0, 373.15, 600.0, 647.096):
        units.append((f"water:{T}", lambda T=T: water_point(T)))
    assert len(units) == 72
    summary = {"mode": "development", "expected": len(units), "units": []}
    start = time.perf_counter()
    with args.output.open("x", encoding="utf-8") as output:
        for name, execute in units:
            item = {"unit": name}
            try:
                item.update(status="passed", measurements=execute())
            except Exception:
                item.update(status="failed", error=traceback.format_exc())
            summary["units"].append(item)
        summary.update(
            elapsed_seconds=time.perf_counter() - start,
            passed=sum(u["status"] == "passed" for u in summary["units"]),
        )
        summary["failed"] = len(units) - summary["passed"]
        json.dump(summary, output, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "units"}), flush=True)
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
