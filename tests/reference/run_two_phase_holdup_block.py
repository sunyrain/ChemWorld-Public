"""Fixed 232 inputs / 237 comparisons for holdup/viscosity families."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.two_phase_voidage as reference

from chemworld.physchem.two_phase_holdup import (
    TwoPhaseProperties,
    martinelli_parameter,
    two_phase_mixture_density,
    two_phase_mixture_viscosity,
    two_phase_void_fraction,
    void_fraction_from_density,
)

PROPERTIES = TwoPhaseProperties(900, 10, 0.0005, 1.5e-5, 0.05, 1e6, 5e6)


def inputs():
    for method, quality, flux in product(
        reference.two_phase_voidage_correlations, (0.1, 0.5, 0.9), (100, 400)
    ):
        yield {
            "family": "voidage",
            "method": method,
            "x": quality,
            "G": flux,
            "D": 0.02,
            "angle": 0,
        }
    for method, quality in product(
        reference.liquid_gas_viscosity_correlations, (0, 0.1, 0.5, 0.9, 1)
    ):
        yield {"family": "viscosity", "method": method, "x": quality}
    for alpha in (0, 0.1, 0.5, 0.9, 1):
        yield {"family": "density", "alpha": alpha}
    for quality, exponent in product((0.1, 0.5, 0.9), (None, 0.2, 1)):
        yield {"family": "martinelli", "x": quality, "n": exponent}
    for diameter in (5e-5, 1e-4, 1e-3):
        yield {
            "family": "voidage",
            "method": "Kawahara",
            "x": 0.5,
            "G": 100,
            "D": diameter,
            "angle": 0,
        }
    for method, values in (("Graham", (0.005, 0.1)), ("Kopte_Newell_Chato", (0.02, 0.1))):
        for target in values:
            flux = target * sqrt((1 - 0.5) * 10**2 * 9.80665 * 0.02 / 0.5**3)
            yield {
                "family": "voidage",
                "method": method,
                "x": 0.5,
                "G": flux,
                "D": 0.02,
                "angle": 0,
                "target_Ft": target,
            }
    for target in (9.9, 10.1):
        factor = (10 / 900) ** 0.5 * (0.0005 / 1.5e-5) ** 0.1
        quality = 1 / (1 + (target / factor) ** (1 / 0.9))
        yield {
            "family": "voidage",
            "method": "Domanski Didion",
            "x": quality,
            "G": 100,
            "D": 0.02,
            "angle": 0,
            "target_Xtt": target,
        }
    for reynolds in (100, 2000):
        yield {
            "family": "voidage",
            "method": "Tandon Varma Gupta",
            "x": 0.5,
            "G": reynolds * 0.0005 / 0.02,
            "D": 0.02,
            "angle": 0,
            "target_Re": reynolds,
        }
    for angle in (-30, 0, 60):
        yield {
            "family": "voidage",
            "method": "Woldesemayat Ghajar",
            "x": 0.5,
            "G": 100,
            "D": 0.02,
            "angle": angle,
        }


def compare(name, native, expected):
    return {
        "quantity": name,
        "native": native,
        "reference": expected,
        "relative_error": abs(native - expected) / max(abs(expected), 1e-300),
        "relative_tolerance": 2e-11,
        "absolute_tolerance": 1e-12,
        "passed": isclose(native, expected, rel_tol=2e-11, abs_tol=1e-12),
    }


def run(case, values):
    if case["family"] == "voidage":
        expected = reference.liquid_gas_voidage(
            x=case["x"],
            rhol=900,
            rhog=10,
            D=case["D"],
            m=case["G"] * pi * case["D"] ** 2 / 4,
            mul=0.0005,
            mug=1.5e-5,
            sigma=0.05,
            P=1e6,
            Pc=5e6,
            angle=case["angle"],
            g=9.80665,
            Method=case["method"],
        )
        native = two_phase_void_fraction(
            PROPERTIES,
            case["x"],
            method=case["method"],
            mass_flux_kg_m2_s=case["G"],
            diameter_m=case["D"],
            inclination_degrees=case["angle"],
        )
        values.append(compare("alpha", native, expected))
    elif case["family"] == "viscosity":
        expected = reference.gas_liquid_viscosity(
            case["x"], 0.0005, 1.5e-5, 900, 10, Method=case["method"]
        )
        values.append(
            compare(
                "viscosity_Pa_s",
                two_phase_mixture_viscosity(PROPERTIES, case["x"], method=case["method"]),
                expected,
            )
        )
    elif case["family"] == "density":
        expected = reference.density_two_phase(case["alpha"], 900, 10)
        values.append(
            compare("density_kg_m3", two_phase_mixture_density(PROPERTIES, case["alpha"]), expected)
        )
        values.append(
            compare(
                "alpha",
                void_fraction_from_density(PROPERTIES, expected),
                reference.two_phase_voidage_experimental(expected, 900, 10),
            )
        )
    else:
        values.append(
            compare(
                "Xtt",
                martinelli_parameter(PROPERTIES, case["x"], friction_exponent=case["n"]),
                reference.Lockhart_Martinelli_Xtt(case["x"], 900, 10, 0.0005, 1.5e-5, n=case["n"]),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if fluids.__version__ != "1.3.1":
        raise ValueError("requires fluids 1.3.1")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records = []
    for index, case in enumerate(inputs(), 1):
        row = {"unit": index, "input": case, "measurements": []}
        try:
            run(case, row["measurements"])
            row["passed"] = all(m["passed"] for m in row["measurements"])
        except Exception as exc:
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(row)
        if index % 25 == 0 or not row["passed"]:
            print(f"Holdup {index}/232", flush=True)
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 232,
        "expected_comparisons": 237,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))
    return 0 if summary["passed_inputs"] == 232 else 1


if __name__ == "__main__":
    raise SystemExit(main())
