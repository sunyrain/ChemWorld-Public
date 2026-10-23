"""Fixed 215 inputs/531 comparisons, retaining the known source area-sign issue."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.two_phase as reference
from scipy.interpolate import splev

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.two_phase_balances import (
    two_phase_acceleration_gradient,
    two_phase_acceleration_pressure_drop,
    two_phase_gravity_gradient,
    two_phase_gravity_pressure_drop,
)
from chemworld.physchem.two_phase_holdup import TwoPhaseProperties
from chemworld.physchem.two_phase_regimes import (
    LogRegimeBoundary,
    TaitelDuklerBoundaries,
    mandhane_gregory_aziz_regime,
    taitel_dukler_regime,
)

P = TwoPhaseProperties(900, 10, 0.0005, 1.5e-5)
OUT = TwoPhaseProperties(850, 8)


def source_curves():
    curves, raw = [], []
    source = InputSource(
        DataSourceProvenance(
            "local-taitel-chart",
            0,
            "Taitel-Dukler chart splines in locally installed fluids1.3.1",
            "fluids1.3.1",
            "Independent chart redistribution rights not established; local only",
        ),
        "literature",
    )
    for label in ("A", "C", "D"):
        tck = getattr(reference, f"Dukler_X{label}_tck")
        knots = sorted({float(x) for x in tck[0]})
        low, high = knots[0], knots[-1]
        # The source has one internal cubic knot (A/C), or a cubic polynomial (D).
        # Five not-a-knot samples with the third at that knot preserve its spline.
        if len(knots) == 3:
            mid = knots[1]
            xs = [low, (low + mid) / 2, mid, (mid + high) / 2, high]
        else:
            xs = [low + (high - low) * index / 3 for index in range(4)]
        columns = tuple(
            QuantityColumn(
                name, "dimensionless", "parameter", (-100, 100), "Explicit log10 chart coordinates"
            )
            for name in ("log10_X", "log10_value")
        )
        rows = tuple(
            ScientificRow(
                str(index),
                "local-taitel-chart",
                label,
                str(index),
                "parameters",
                {
                    "log10_X": InputValue(value, "dimensionless", None, "No uncertainty supplied"),
                    "log10_value": InputValue(
                        float(splev(value, tck)),
                        "dimensionless",
                        None,
                        "Source chart interpolation, not measured uncertainty",
                    ),
                },
            )
            for index, value in enumerate(xs)
        )
        curve = LogRegimeBoundary(
            TabulatedScalar(
                ScientificTable(f"chart-{label}", "parameters", columns, (source,), rows),
                (("log10_X", "log10_X", "dimensionless"),),
                "log10_value",
                "dimensionless",
            )
        )
        curves.append(curve)
        raw.append(tck)
    return TaitelDuklerBoundaries(*curves), raw


def inputs():
    for xi, xo, flux in product((0.1, 0.5, 0.9), (0.1, 0.5, 0.9), (50, 500)):
        yield {"family": "discrete", "xi": xi, "xo": xo, "G": flux}
    for angle, alpha, length in product((-90, -10, 0, 10, 90), (0, 0.2, 0.8, 1), (0, 2)):
        yield {"family": "gravity", "angle": angle, "alpha": alpha, "L": length}
    for x, flux, derivative in product((0, 0.2, 0.8, 1), (50, 500), (-0.001, 0, 0.001)):
        yield {"family": "differential", "x": x, "G": flux, "dA": derivative}
    for x, flux, derivative in product((0, 0.2, 0.8, 1), (50, 500), (-0.001, 0.001)):
        yield {"family": "independent-area", "x": x, "G": flux, "dA": derivative}
    for vl, vg in product(
        (0.01, 0.1, 0.2, 0.3, 0.56, 1, 1.15, 2.5, 4.8, 14, 30), (0.1, 1, 5, 20, 80, 500)
    ):
        yield {"family": "Mandhane", "vl_ft_s": vl, "vg_ft_s": vg}
    for x, flux, angle in product((0.01, 0.1, 0.5, 0.9), (10, 100, 1000), (-5, 0, 5)):
        yield {"family": "Taitel", "x": x, "G": flux, "angle": angle}
    for curve, position in product(range(3), (0, 0.25, 0.5, 0.75, 1)):
        yield {"family": "curve", "curve": curve, "position": position}


def compare(quantity, native, expected):
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "passed": native == expected
        if isinstance(native, str)
        else isclose(native, expected, rel_tol=2e-9, abs_tol=1e-9),
        "relative_tolerance": 2e-9,
        "absolute_tolerance": 1e-9,
    }


def alpha(p, x):
    return (x / p.vapor_density_kg_m3) / (
        x / p.vapor_density_kg_m3 + (1 - x) / p.liquid_density_kg_m3
    )


def evaluate(case, boundaries, raw_curves):
    family = case["family"]
    if family == "discrete":
        xi, xo, flux = case["xi"], case["xo"], case["G"]
        native = two_phase_acceleration_pressure_drop(
            P,
            OUT,
            mass_flux_kg_m2_s=flux,
            inlet_quality=xi,
            outlet_quality=xo,
            inlet_void_fraction=alpha(P, xi),
            outlet_void_fraction=alpha(OUT, xo),
        )
        expected = reference.two_phase_dP_acceleration(
            flux * pi * 0.05**2 / 4, 0.05, xi, xo, alpha(P, xi), alpha(OUT, xo), 900, 10, 850, 8
        )
        return [compare("acceleration_Pa", native, expected)]
    if family == "gravity":
        angle, a, length = case["angle"], case["alpha"], case["L"]
        local = two_phase_gravity_gradient(P, void_fraction=a, inclination_degrees=angle)
        segment = two_phase_gravity_pressure_drop(
            P,
            inlet_void_fraction=a,
            length_m=length,
            inclination_degrees=angle,
            outlet=OUT,
            outlet_void_fraction=0.8 * a,
        )
        return [
            compare(
                "gradient_Pa_m", local, reference.two_phase_dP_dz_gravitational(angle, a, 900, 10)
            ),
            compare(
                "segment_Pa",
                segment,
                reference.two_phase_dP_gravitational(angle, length, a, 900, 10, 0.8 * a, 850, 8),
            ),
        ]
    if family in {"differential", "independent-area"}:
        x, flux, derivative = case["x"], case["G"], case["dA"]
        independent = family == "independent-area"
        pl, vg, q = (0, 0, 0) if independent else (-5e-12, -4e-7, -2e-7)
        native = two_phase_acceleration_gradient(
            P,
            mass_flux_kg_m2_s=flux,
            vapor_quality=x,
            pressure_drop_gradient_Pa_m=120,
            liquid_volume_pressure_derivative_m3_kg_Pa=pl,
            vapor_volume_pressure_derivative_m3_kg_Pa=vg,
            quality_pressure_derivative_per_Pa=q,
            area_m2=0.01,
            area_gradient_m=derivative,
        )
        if independent:
            volume = x / 10 + (1 - x) / 900
            mass = flux * 0.01
            step = 1e-6

            def kinetic(z):
                return (mass * volume / (0.01 + z * derivative)) ** 2 / (2 * volume)

            expected = (kinetic(step) - kinetic(-step)) / (2 * step)
        else:
            expected = reference.two_phase_dP_dz_acceleration(
                flux * 0.01, sqrt(0.04 / pi), x, 900, 10, pl, vg, q, 120, derivative
            )
        return [compare("acceleration_gradient_Pa_m", native, expected)]
    if family == "Mandhane":
        properties = TwoPhaseProperties(999.552, 1.294292, 0.001, 1.8e-5, 0.0724)
        vl, vg = case["vl_ft_s"] * 0.3048, case["vg_ft_s"] * 0.3048
        gas_flux, liquid_flux = vg * 1.294292, vl * 999.552
        mass, quality = (
            (gas_flux + liquid_flux) * pi * 0.05**2 / 4,
            gas_flux / (gas_flux + liquid_flux),
        )
        native = mandhane_gregory_aziz_regime(
            properties, mass_flow_kg_s=mass, diameter_m=0.05, vapor_quality=quality
        )
        expected = reference.Mandhane_Gregory_Aziz_regime(
            mass, quality, 999.552, 1.294292, 0.001, 1.8e-5, 0.0724, 0.05
        )
        return [
            compare("regime", native.regime, expected[0]),
            compare(
                "liquid_velocity_m_s", native.liquid_superficial_velocity_m_s, expected[1] * 0.3048
            ),
            compare(
                "vapor_velocity_m_s", native.vapor_superficial_velocity_m_s, expected[2] * 0.3048
            ),
        ]
    if family == "Taitel":
        mass, quality, angle = case["G"] * pi * 0.05**2 / 4, case["x"], case["angle"]
        native = taitel_dukler_regime(
            P,
            boundaries,
            mass_flow_kg_s=mass,
            diameter_m=0.05,
            vapor_quality=quality,
            inclination_degrees=angle,
            allow_chart_extrapolation=True,
        )
        expected = reference.Taitel_Dukler_regime(
            mass, quality, 900, 10, 0.0005, 1.5e-5, 0.05, angle
        )
        return [compare("regime", native.regime, expected[0])] + [
            compare(name, native.parameters[name], value)
            for name, value in zip(("X", "T", "F", "K"), expected[1:], strict=True)
        ]
    curve = (boundaries.annular, boundaries.wavy, boundaries.bubbly)[case["curve"]]
    lo, hi = curve.source_curve.domain["log10_X"]
    coordinate = lo + (hi - lo) * case["position"]
    return [
        compare(
            "boundary",
            curve.value(10**coordinate, allow_extrapolation=True),
            10 ** float(splev(coordinate, raw_curves[case["curve"]])),
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite an attempt")
    cases, records, failures = list(inputs()), [], []
    assert len(cases) == 215
    boundaries, raw_curves = source_curves()
    start = time.perf_counter()
    for index, case in enumerate(cases, 1):
        record = {"unit": index, "input": case}
        try:
            record["measurements"] = evaluate(case, boundaries, raw_curves)
            record["passed"] = all(m["passed"] for m in record["measurements"])
        except Exception as exc:
            record.update(measurements=[], passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        if not record["passed"]:
            failures.append(record)
        if index % 50 == 0 or index == 215:
            elapsed = time.perf_counter() - start
            print(
                f"regimes {index}/215 rate={index / max(elapsed, 1e-9):.1f}/s "
                f"ETA={elapsed / index * (215 - index):.2f}s failures={len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 215,
        "expected_comparisons": 531,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": failures,
        "records": records,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"failures", "records"}}
        )
    )


if __name__ == "__main__":
    main()
