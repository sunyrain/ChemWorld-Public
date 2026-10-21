"""Fixed 63-unit phase-change formula qualification; upstream is test-only."""
# ruff: noqa: C408 -- named scientific parameters

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import phase_change as pc
from pressure_reference import finite_difference

from chemworld.physchem.phase_change_kernels import boiling_vaporization_enthalpy, watson_exponent
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.specs import PropertyCorrelation


def near(actual, expected, rtol=3e-10, atol=1e-8):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return abs(actual - expected)


def compare_derivative(curve, function, temperature):
    values = finite_difference(function, temperature, 1)
    native = correlation_temperature_derivative(curve, temperature_K=temperature).value
    near(*values, rtol=5e-7, atol=3e-7)
    return max(near(native, value, rtol=5e-7, atol=3e-7) for value in values)


def cases():
    for index, Tc in enumerate((600.0, 750.0)):
        csp = dict(Tc=Tc, omega=(0.2, 0.45)[index])
        for equation, reference, params in (
            (
                "watson_hvap",
                "Watson",
                dict(
                    Tc=Tc,
                    T_ref=0.5 * Tc,
                    Hvap_ref=(30000.0, 45000.0)[index],
                    exponent=(0.38, 0.5)[index],
                ),
            ),
            (
                "clapeyron_hvap",
                "Clapeyron",
                dict(Tc=Tc, Pc=(3e6, 5e6)[index], Psat=(1e5, 2e5)[index], dZ=(0.9, 0.8)[index]),
            ),
            ("pitzer_hvap", "Pitzer", csp),
            ("smk_hvap", "SMK", csp),
            ("mk_hvap", "MK", csp),
            ("velasco_hvap", "Velasco", csp),
            ("alibakhshi_hvap", "Alibakhshi", dict(Tc=Tc, C=(-16.0, -25.0)[index])),
            (
                "ppds12_hvap",
                "PPDS12",
                dict(
                    Tc=Tc,
                    **dict(
                        zip(
                            "ABCDE",
                            (
                                (4.60584, 13.97224, -10.592315, 2.120205, 4.277128),
                                (1.71484, 25.8604, -24.3564, 7.40881, 6.80068),
                            )[index],
                            strict=True,
                        )
                    ),
                ),
            ),
        ):
            for fraction in (0.5, 0.65, 0.8):
                T = Tc * fraction
                yield f"{equation}/{index}/{fraction}", (equation, reference, params, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain earlier attempts; choose a new output path")
    assert chemicals.__version__ == "1.5.2"
    start, rows = time.perf_counter(), []

    def run(identity, function):
        try:
            metrics = function()
            rows.append({"unit": identity, "status": "passed", "metrics": metrics})
        except Exception:
            rows.append({"unit": identity, "status": "failed", "failure": traceback.format_exc()})

    for identity, (equation, reference, params, T) in cases():

        def compare(identity=identity, equation=equation, params=params, T=T, reference=reference):
            curve = PropertyCorrelation(
                identity, equation, params, {"temperature": "K"}, "J/mol", "heat_of_vaporization"
            )
            function = lambda T: getattr(pc, reference)(T, **params)  # noqa: E731
            return {
                "value_error": near(
                    evaluate_correlation(curve, temperature_K=T).value, function(T)
                ),
                "derivative_error": compare_derivative(curve, function, T),
            }

        run(identity, compare)
    for method in ("Riedel", "Chen", "Liu", "Vetere"):
        for Tb, Tc, Pc in (
            (294.0, 466.0, 5.55e6),
            (388.4, 620.0, 56.3e5),
            (373.15, 647.14, 22.064e6),
        ):
            run(
                f"anchor/{method}/{Tb}",
                lambda method=method, Tb=Tb, Tc=Tc, Pc=Pc: {
                    "error": near(
                        boiling_vaporization_enthalpy(
                            method=method,
                            boiling_temperature_K=Tb,
                            critical_temperature_K=Tc,
                            critical_pressure_Pa=Pc,
                        ),
                        getattr(pc, method)(Tb, Tc, Pc),
                    )
                },
            )
    for exponent in (0.2, 0.38, 0.65):
        h2 = 40000 * (200 / 300) ** exponent
        run(
            f"exponent/{exponent}",
            lambda h2=h2: {
                "error": near(
                    watson_exponent(
                        temperature_1_K=300,
                        temperature_2_K=400,
                        critical_temperature_K=600,
                        enthalpy_1_J_mol=40000,
                        enthalpy_2_J_mol=h2,
                    ),
                    pc.Watson_n(300, 400, 40000, h2, 600),
                    rtol=0,
                    atol=2e-12,
                )
            },
        )
    passed = sum(row["status"] == "passed" for row in rows)
    result = {
        "mode": "development",
        "reference": chemicals.__version__,
        "expected": 63,
        "executed": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "seconds": time.perf_counter() - start,
        "units": rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return int(passed != 63 or len(rows) != 63)


if __name__ == "__main__":
    raise SystemExit(main())
