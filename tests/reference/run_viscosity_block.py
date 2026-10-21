"""Fixed first 58-method / all-1908-source blocks; reference execution is test-only."""
# ruff: noqa: C408 -- compact fixed scientific parameter declarations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import dippr
from chemicals import viscosity as reference
from pressure_reference import finite_difference

from chemworld.data.viscosity import ViscosityDatabase
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.viscosity_kernels import (
    viscosity_derivatives,
    viscosity_yaws_parameter_jacobian,
)
from chemworld.physchem.viscosity_mixtures import (
    herning_zipperer_viscosity,
    wilke_prefactors,
    wilke_viscosity,
)
from chemworld.physchem.water_transport import (
    water_correlation_length_nm,
    water_viscosity_enhancement,
    water_viscosity_iapws,
)

METHOD_CASES = (
    ("PPDS5", (350, 450, 550), dict(Tc=470.008, a0=1.08003e-5, a1=0.19583, a2=0.811897)),
    ("PPDS9", (300, 400, 500), dict(A=1.74793, B=1.33728, C=482.347, D=41.78, E=9.963e-5)),
    ("mu_TDE", (300, 400, 500), dict(A=-10, B=800, C=-1000, D=4000)),
    ("mu_Yaws", (300, 400, 500), dict(A=-6, B=800, C=-0.001, D=1e-6)),
    ("Viswanath_Natarajan_2", (273.15, 348.15, 380), dict(A=-5.9719 - math.log(100), B=1007)),
    ("Viswanath_Natarajan_2_exponential", (283.15, 288.15, 349.65), dict(C=4900800, D=-3.8075)),
    ("Viswanath_Natarajan_3", (280, 298.15, 330), dict(A=-5.7173, B=-1071.18, C=-129.51)),
    ("Letsou_Stiel", (340, 370, 390), dict(MW=44, Tc=450, Pc=6e6, omega=0.2)),
    ("Yoon_Thodos", (400, 600, 900), dict(Tc=500, Pc=5e6, MW=40)),
    ("Stiel_Thodos", (400, 600, 900), dict(Tc=500, Pc=5e6, MW=40)),
    ("viscosity_gas_Gharagheizi", (200, 400, 600), dict(Tc=500, Pc=5e6, MW=40)),
    ("Lucas_gas", (400, 550, 700), dict(Tc=512.6, Pc=80.9e5, Zc=0.224, MW=32.042, dipole=1.7)),
    ("mu_air_lemmon", (200, 300, 600), dict(rho=40.10292351061862)),
)
SOURCES = (
    ("DUTT_PRASAD_VISCOSITY", "Dutt Prasad 3 term.tsv", 100),
    (
        "PERRY_GAS_VISCOSITY",
        "Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv",
        345,
    ),
    ("PERRY_LIQUID_VISCOSITY", "Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv", 337),
    ("VDI_GAS_VISCOSITY", "VDI PPDS Dynamic viscosity of gases polynomials.tsv", 274),
    (
        "VDI_LIQUID_VISCOSITY",
        "VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv",
        271,
    ),
    ("VN2E_VISCOSITY", "Viswanath Natarajan Dynamic 2 term Exponential.tsv", 14),
    ("VN2_VISCOSITY", "Viswanath Natarajan Dynamic 2 term.tsv", 135),
    ("VN3_VISCOSITY", "Viswanath Natarajan Dynamic 3 term.tsv", 432),
)


def compare(actual, expected, *, rtol=3e-10, atol=1e-14):
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)
    return {"actual": actual, "reference": expected, "error": abs(actual - expected)}


def compare_slope(actual, function, temperature):
    slopes = finite_difference(function, temperature, 1)
    compare(*slopes, rtol=3e-6, atol=1e-13)
    return [compare(actual, slope, rtol=3e-6, atol=1e-13) for slope in slopes]


def method_unit(method, temperature, parameters):
    function = lambda t: getattr(reference, method)(t, **parameters)  # noqa: E731
    value, slope = viscosity_derivatives(method, temperature, parameters)
    return {
        "value": compare(value, function(temperature)),
        "derivatives": compare_slope(slope, function, temperature),
    }


def method_units():
    for method, temperatures, parameters in METHOD_CASES:
        for temperature in temperatures:
            yield (
                f"{method}/{temperature}",
                {"T": temperature, **parameters},
                (lambda m=method, t=temperature, p=parameters: method_unit(m, t, p)),
            )
    for i, (x, mus, mw) in enumerate(
        (
            ([1.0], [1.34e-5], [64.06]),
            ([0.0, 1.0], [1.34e-5, 9.5029e-6], [64.06, 46.07]),
            ([0.05, 0.95], [1.34e-5, 9.5029e-6], [64.06, 46.07]),
            ([0.2, 0.3, 0.5], [1e-5, 2e-5, 3e-5], [16.0, 28.0, 44.0]),
        )
    ):
        for method in ("Wilke", "Wilke_large", "Wilke_prefactored", "Herning_Zipperer"):

            def run(x=x, mus=mus, mw=mw, method=method):
                if method == "Herning_Zipperer":
                    return compare(
                        herning_zipperer_viscosity(x, mus, mw),
                        reference.Herning_Zipperer(x, mus, mw),
                    )
                if method == "Wilke_prefactored":
                    actual = wilke_viscosity(x, mus, factors=wilke_prefactors(mw))
                    expected = reference.Wilke_prefactored(x, mus, *reference.Wilke_prefactors(mw))
                else:
                    actual = wilke_viscosity(x, mus, mw)
                    expected = getattr(reference, method)(x, mus, mw)
                return compare(actual, expected)

            yield f"mixture/{i}/{method}", {"x": x, "mu": mus, "MW": mw}, run
    for temperature in (300, 400, 500):

        def run(t=temperature):
            parameters = dict(A=-6, B=800, C=-0.001, D=1e-6)
            actual = viscosity_yaws_parameter_jacobian([t], **parameters)[0]
            expected = reference.mu_Yaws_fitting_jacobian([t], **parameters)[0]
            return [compare(a, b, atol=1e-12) for a, b in zip(actual, expected, strict=True)]

        yield f"Yaws-Jacobian/{temperature}", {"T": temperature}, run


def reference_curve(source, raw):
    if source.startswith("PERRY_"):
        n = 4 if source == "PERRY_GAS_VISCOSITY" else 5
        p = [float(raw[f"C{i}"]) for i in range(1, n + 1)]
        f = dippr.EQ102 if n == 4 else dippr.EQ101
        return lambda t: f(t, *p)
    if source == "VDI_GAS_VISCOSITY":
        p = [float(raw[k]) for k in "ABCDE"]
        return lambda t: dippr.EQ100(t, *p)
    if source == "VDI_LIQUID_VISCOSITY":
        p = [float(raw[k]) for k in "ABCDE"]
        return lambda t: reference.PPDS9(t, *p)
    if source == "VN2E_VISCOSITY":
        return lambda t: reference.Viswanath_Natarajan_2_exponential(
            t, float(raw["C"]), float(raw["D"])
        )
    if source == "VN2_VISCOSITY":
        return lambda t: reference.Viswanath_Natarajan_2(
            t, float(raw["A"]) - math.log(100), float(raw["B"])
        )
    return lambda t: reference.Viswanath_Natarajan_3(
        t, float(raw["A"]) - 3, float(raw["B"]), float(raw["C"])
    )


def source_unit(database, source, index, raw, *, application_bounds=None):
    record = database.records(source)[index]
    assert record.source_key == raw["CAS"]
    assert record in database.records_for(raw["CAS"], method=source)
    for key in record.field_names:
        assert record.raw(key) == raw[key]
    if source == "VDI_GAS_VISCOSITY":
        bounds = (300, 800)
    elif source == "VDI_LIQUID_VISCOSITY":
        low, high = sorted([float(raw["C"]), float(raw["D"])])
        bounds = (low + 0.2 * (high - low), low + 0.8 * (high - low))
    else:
        bounds = (float(raw["Tmin"]), float(raw["Tmax"]))
    if application_bounds is not None:
        bounds = application_bounds
    function = reference_curve(source, raw)
    if not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] < bounds[1]:
        try:
            record.correlation(temperature_bounds_K=bounds)
        except ValueError as exc:
            return {
                "bounds": bounds,
                "unbuilt": 1,
                "rejection": str(exc),
                "reason": "fixed synthetic application interval is not positive and ordered",
            }
        raise AssertionError("native accepted invalid application interval")
    curve = record.correlation(temperature_bounds_K=bounds)
    result = {
        "bounds": bounds,
        "points": [],
        "value_comparisons": 0,
        "derivative_comparisons": 0,
        "rejected": 0,
    }
    for fraction in (0.25, 0.5, 0.75):
        t = bounds[0] + fraction * (bounds[1] - bounds[0])
        point = {"T": t}
        try:
            expected = function(t)
            valid = math.isfinite(expected) and expected > 0
            point["reference"] = expected if math.isfinite(expected) else str(expected)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            valid = False
            point["reference_exception"] = repr(exc)
        if not valid:
            try:
                evaluate_correlation(curve, temperature_K=t, validity_policy="raise")
            except ValueError as exc:
                point["rejection"] = str(exc)
                result["rejected"] += 1
            else:
                raise AssertionError("native accepted invalid source prediction")
        else:
            actual = evaluate_correlation(curve, temperature_K=t, validity_policy="raise").value
            point["value"] = compare(actual, expected)
            slope = correlation_temperature_derivative(curve, temperature_K=t).value
            point["derivatives"] = compare_slope(slope, function, t)
            result["value_comparisons"] += 1
            result["derivative_comparisons"] += 2
        result["points"].append(point)
    return result


def source_units(root):
    database = ViscosityDatabase(root)
    for source, filename, count in SOURCES:
        with (root / "Viscosity" / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == len(database.records(source)) == count
        for index, raw in enumerate(rows):
            yield (
                f"{source}/{index}",
                raw,
                (lambda s=source, i=index, r=raw: source_unit(database, s, i, r)),
            )


def positive_source_units(root):
    database = ViscosityDatabase(root)
    source, filename, _ = next(row for row in SOURCES if row[0] == "VDI_LIQUID_VISCOSITY")
    with (root / "Viscosity" / filename).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for index, raw in enumerate(rows):
        lo, hi = sorted((float(raw["C"]), float(raw["D"])))
        if lo + 0.2 * (hi - lo) > 0:
            continue
        lo = max(1, lo)
        bounds = (lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo))
        yield (
            f"positive-VDI/{index}",
            raw,
            lambda i=index, r=raw, b=bounds: source_unit(
                database, source, i, r, application_bounds=b
            ),
        )


def water_units():
    from chemicals.iapws import iapws95_P, iapws95_properties

    for temperature, density, micropas in (
        (298.15, 998, 889.735100),
        (298.15, 1200, 1437.649467),
        (373.15, 1000, 307.883622),
        (433.15, 1, 14.538324),
        (433.15, 1000, 217.685358),
        (873.15, 1, 32.619287),
        (873.15, 100, 35.802262),
        (873.15, 600, 77.430195),
        (1173.15, 1, 44.217245),
        (1173.15, 100, 47.640433),
        (1173.15, 400, 64.154608),
    ):

        def run(t=temperature, rho=density, expected=micropas * 1e-6):
            actual = water_viscosity_iapws(t, rho)
            metrics = method_unit("mu_IAPWS", t, {"rho": rho})
            metrics["published"] = compare(actual, expected, rtol=0, atol=5.1e-13)
            return metrics

        yield f"water/background/{temperature}/{density}", {"T": temperature, "rho": density}, run
    for density, length, enhancement, micropas in (
        (122, 0.309247, 1.00000289, 25.520677),
        (222, 1.571405, 1.00375120, 31.337589),
        (272, 5.266522, 1.03416789, 36.228143),
        (322, 16.590209, 1.09190440, 42.961579),
        (372, 5.603768, 1.03665871, 45.688204),
        (422, 1.876244, 1.00596332, 49.436256),
    ):

        def run(rho=density, length=length, enhancement=enhancement, expected=micropas * 1e-6):
            t = 647.35
            current = iapws95_properties(t, iapws95_P(t, rho))
            reference_state = iapws95_properties(970.644, iapws95_P(970.644, rho))
            compare(current[0], rho)
            compare(reference_state[0], rho)
            a, b = current[-1], reference_state[-1]
            xi = water_correlation_length_nm(t, rho, a, b)
            factor = water_viscosity_enhancement(xi)
            actual = water_viscosity_iapws(t, rho, drho_dP=a, drho_dP_reference=b)
            return {
                "EOS_derivatives": {"current": a, "reference": b},
                "length": compare(xi, length, rtol=0, atol=5.1e-7),
                "enhancement": compare(factor, enhancement, rtol=0, atol=5.1e-9),
                "published": compare(actual, expected, rtol=0, atol=5.1e-13),
                "upstream": compare(actual, reference.mu_IAPWS(t, rho, a, b)),
            }

        yield f"water/scientific/{density}", {"T": 647.35, "rho": density}, run
    for density in (50, 150, 300, 500, 800):
        for temperature in (600, 647.35, 800):

            def run(t=temperature, rho=density):
                return compare(
                    water_viscosity_iapws(t, rho, drho_dP=0.001, reference_approximation=True),
                    reference.mu_IAPWS(t, rho, 0.001),
                )

            yield (
                f"water/approximation/{temperature}/{density}",
                {"T": temperature, "rho": density, "synthetic_drho_dP": 0.001},
                run,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=(
            "methods",
            "sources",
            "water",
            "scales",
            "index",
            "models",
            "scale-inverse",
            "index-standard",
            "scale-monotone",
            "branches",
            "sources-positive",
            "gharagheizi-published",
        ),
        required=True,
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    start = time.perf_counter()
    from viscosity_extended_blocks import branch_units, index_units, model_units, scale_units

    generators = {
        "methods": method_units,
        "sources": lambda: source_units(args.source_directory),
        "water": water_units,
        "scales": lambda: scale_units(compare),
        "index": lambda: index_units(compare),
        "scale-inverse": lambda: scale_units(compare, independent_inverse=True),
        "scale-monotone": lambda: scale_units(compare, monotone=True),
        "index-standard": lambda: index_units(compare, standard_rounding=True),
        "branches": lambda: branch_units(compare, compare_slope),
        "gharagheizi-published": lambda: branch_units(
            compare, compare_slope, published_gharagheizi=True
        ),
        "sources-positive": lambda: positive_source_units(args.source_directory),
        "models": lambda: model_units(compare, compare_slope),
    }
    units = list(generators[args.block]())
    assert (
        len(units)
        == {
            "methods": 58,
            "sources": 1908,
            "water": 32,
            "scales": 1011,
            "index": 1248,
            "models": 39,
            "scale-inverse": 1011,
            "index-standard": 1248,
            "scale-monotone": 1011,
            "branches": 42,
            "sources-positive": 40,
            "gharagheizi-published": 5,
        }[args.block]
    )
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    rows = []
    for identity, inputs, function in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=function(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 250 == 0 or len(rows) == len(units):
            seconds = time.perf_counter() - start
            print(
                f"{args.block}: {len(rows)}/{len(units)}; {len(rows) / seconds:.1f}/s; "
                f"ETA {(len(units) - len(rows)) * seconds / len(rows):.1f}s",
                flush=True,
            )
    result = {
        "mode": "development",
        "block": args.block,
        "reference_version": chemicals.__version__,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    if args.block in {"sources", "sources-positive"}:
        for key in ("value_comparisons", "derivative_comparisons", "rejected", "unbuilt"):
            result[key] = sum(r.get("metrics", {}).get(key, 0) for r in rows)
    if args.block == "scale-inverse":
        result["reference_reverse_failures"] = sum(
            "failure" in r.get("metrics", {}).get("reference_reverse", {}) for r in rows
        )
    if args.block == "index-standard":
        result["reference_rounding_disagreements"] = sum(
            r.get("metrics", {}).get("rounding_agrees") is False for r in rows
        )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:5]:
        print(row["unit"], row["failure"])
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
