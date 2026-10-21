"""Fixed E4 scientific comparisons; reference dependencies stay in tests."""
# ruff: noqa: C408 -- compact scientific parameters

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import dippr
from chemicals import interface as reference
from pressure_reference import finite_difference

from chemworld.data.interface import INTERFACE_SOURCES, InterfaceDatabase
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.interface_kernels import interface_derivatives
from chemworld.physchem.interfaces import (
    diguilio_teja,
    water_oil_interfacial_tension,
    weinaug_katz,
    winterfeld_scriven_davis,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec

METHODS = (
    ("sigma_IAPWS", (300, 450, 600), {}),
    (
        "REFPROP_sigma",
        (280, 350, 550),
        dict(Tc=647.096, sigma0=-0.1306, n0=2.471, sigma1=0.2151, n1=1.233),
    ),
    ("Somayajulu", (300, 450, 600), dict(Tc=647.126, A=232.713514, B=-140.18645, C=-4.890098)),
    ("Jasper", (280, 298.15, 350), dict(a=24, b=0.0773)),
    ("Brock_Bird", (300, 412.15, 550), dict(Tb=447.3, Tc=685, Pc=3.952e6)),
    ("Pitzer_sigma", (293, 400, 550), dict(Tc=633, Pc=4.53e6, omega=0.249)),
    ("Zuo_Stenby", (293, 400, 550), dict(Tc=633, Pc=4.53e6, omega=0.249)),
    *(
        (m, (293.15, 400, 550), dict(Tb=404.75, Tc=633, Pc=4.53e6))
        for m in ("Sastri_Rao", "Sastri_Rao_acid", "Sastri_Rao_alcohol")
    ),
    (
        "Hakim_Steinberg_Stiel",
        (298.15, 400, 500),
        dict(Tc=563, Pc=4.414e6, omega=0.59, StielPolar=-0.07872),
    ),
    ("Miqueu", (250, 300, 330), dict(Tc=340.1, Vc=0.000199, omega=0.1687)),
    ("sigma_Gharagheizi_1", (95, 125, 165), dict(Tc=190.564, MW=16.04, omega=0.012)),
    ("sigma_Gharagheizi_2", (95, 125, 165), dict(Tb=111.66, Tc=190.564, Pc=4.599e6, Vc=0.0986e-3)),
    ("Aleem", (90, 100, 120), dict(MW=16.04246, Tb=111.6, rhol=458.7, Hvap_Tb=510870, Cpl=2465)),
    ("Mersmann_Kind_sigma", (250, 298.15, 360), dict(Tm=164.15, Tb=328.25, Tc=497.1, Pc=3430000)),
    ("API10A32", (288.7055555555555, 400, 600), dict(Tc=741.1111111111111, K_W=12.4)),
    ("PPDS14", (280, 400, 500), dict(Tc=562.05, a0=0.0786269, a1=1.28646, a2=-0.112304)),
    (
        "Watson_sigma",
        (300, 350, 450),
        dict(Tc=543.836, a1=-3.02417, a2=1.21792, a3=-5.26877e-9, a4=5.62659e-9, a5=-2.27553e-9),
    ),
    ("ISTExpansion", (300, 400, 600), dict(Tc=776, a1=0.037545, a2=0.0363288)),
)


def compare(actual, expected, *, rtol=3e-10, atol=1e-12):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)
    return {"actual": actual, "reference": expected, "absolute_error": abs(actual - expected)}


def slope(actual, function, t):
    values = finite_difference(function, t, 1)
    compare(*values, atol=1e-11, rtol=3e-6)
    return [compare(actual, v, atol=1e-11, rtol=3e-6) for v in values]


def method_unit(method, t, p):
    def ref(T):
        if method.startswith("Sastri_Rao_"):
            return reference.Sastri_Rao(T, **p, chemicaltype=method.removeprefix("Sastri_Rao_"))
        return getattr(reference, method)(T, **p) * (0.001 if method == "API10A32" else 1)

    value, derivative = interface_derivatives(method, t, p)
    return {"value": compare(value, ref(t)), "derivatives": slope(derivative, ref, t)}


def method_units():
    for method, temperatures, p in METHODS:
        for t in temperatures:
            yield f"{method}/{t}", {"T": t, **p}, lambda m=method, t=t, p=p: method_unit(m, t, p)


def source_unit(database, source, i, raw):
    record = database.records(source)[i]
    assert record.source_key == raw["CAS"]
    assert record in database.records_for(raw["CAS"], method=source)
    assert all(record.raw(k) == raw[k] for k in record.field_names)
    p = {
        k: float(v) if v else None
        for k, v in raw.items()
        if k not in {"CAS", "Name", "Fluid", "Chemical"}
    }
    if source in {"JASPER_LANGE", "MULERO_CACHADINA"}:
        bounds = p["Tmin"], p["Tmax"]
    else:
        bounds = p["Tm" if source == "VDI_SURFACE_TENSION" else "Tt"], p["Tc"]
    if not (all(t is not None and math.isfinite(t) for t in bounds) and 0 < bounds[0] < bounds[1]):
        try:
            record.correlation()
        except ValueError as exc:
            return {"unbuilt": 1, "rejection": str(exc), "bounds": bounds}
        raise AssertionError("invalid source interval silently built")
    curve = record.correlation()

    def ref(T):
        if source == "JASPER_LANGE":
            return 0.001 * (p["a"] - p["b"] * (T - 273.15))
        if source == "MULERO_CACHADINA":
            return reference.REFPROP_sigma(
                T, **{k: v for k, v in p.items() if k not in {"Tmin", "Tmax"}}
            )
        if source == "VDI_SURFACE_TENSION":
            return dippr.EQ106(T, **{k: v for k, v in p.items() if k != "Tm"})
        return reference.Somayajulu(T, **{k: v for k, v in p.items() if k != "Tt"})

    result = {"points": [], "value_comparisons": 0, "derivative_comparisons": 0, "rejected": 0}
    for f in (0.25, 0.5, 0.75):
        t = bounds[0] + f * (bounds[1] - bounds[0])
        expected = ref(t)
        if not math.isfinite(expected) or expected < 0:
            try:
                evaluate_correlation(curve, temperature_K=t, validity_policy="raise")
            except ValueError as exc:
                result["rejected"] += 1
                result["points"].append({"T": t, "reference": expected, "rejection": str(exc)})
            else:
                raise AssertionError("accepted negative/nonfinite source prediction")
        else:
            actual = evaluate_correlation(curve, temperature_K=t, validity_policy="raise").value
            derivative = correlation_temperature_derivative(curve, temperature_K=t).value
            result["points"].append(
                {
                    "T": t,
                    "value": compare(actual, expected),
                    "derivatives": slope(derivative, ref, t),
                }
            )
            result["value_comparisons"] += 1
            result["derivative_comparisons"] += 2
    return result


def source_units(root):
    database = InterfaceDatabase(root)
    for source, count in zip(INTERFACE_SOURCES, (522, 115, 64, 64, 272), strict=True):
        with (root / source.relative_path).open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == len(database.records(source.source_id)) == count
        for i, raw in enumerate(rows):
            yield (
                f"{source.source_id}/{i}",
                raw,
                lambda s=source.source_id, i=i, r=raw: source_unit(database, s, i, r),
            )


def completion_unit(database, source, index):
    record = database.records(source)[index]
    raw = dict(zip(record.field_names, record.raw_fields, strict=True))
    lo, hi = (float(raw[k]) if raw[k] else None for k in ("Tmin", "Tmax"))
    point = lo is not None and lo == hi
    if point:
        curve = record.correlation(source_point=True)
        component = ComponentSpec(record.cas_number, "C13H20O", cas_number=record.cas_number)
        card = database.property_input(
            record.cas_number, component, method=source, source_point=True
        )
        restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
        result = restored.evaluate_state(
            temperature_K=lo, pressure_Pa=101325, phase="liquid", quantities=("surface_tension",)
        ).properties["surface_tension"]
        expected = reference.Jasper(lo, float(raw["a"]), float(raw["b"]))
        for run in (
            lambda: correlation_temperature_derivative(curve, temperature_K=lo),
            lambda: evaluate_correlation(curve, temperature_K=lo + 1, validity_policy="ignore"),
        ):
            try:
                run()
            except ValueError:
                pass
            else:
                raise AssertionError("single source observation was extrapolated/differentiated")
        return {
            "points": [{"T": lo, "value": compare(result.value, expected)}],
            "value_comparisons": 1,
            "derivative_comparisons": 0,
        }
    lower = lo if lo is not None and lo > 0 else hi / 4 if hi is not None and hi < 250 else 250
    upper = hi if hi is not None and hi > 0 else lower + 100
    curve = record.correlation(temperature_bounds_K=(lower, upper))
    assert curve.metadata["source_bounds_K"] == [lo, hi]

    def ref(T):
        if source == "JASPER_LANGE":
            return reference.Jasper(T, float(raw["a"]), float(raw["b"]))
        return reference.REFPROP_sigma(
            T, **{k: float(v) for k, v in raw.items() if k not in {"CAS", "Fluid", "Tmin", "Tmax"}}
        )

    points = []
    for f in (0.25, 0.5, 0.75):
        t = lower + f * (upper - lower)
        points.append(
            {
                "T": t,
                "value": compare(evaluate_correlation(curve, temperature_K=t).value, ref(t)),
                "derivatives": slope(
                    correlation_temperature_derivative(curve, temperature_K=t).value, ref, t
                ),
            }
        )
    return {
        "points": points,
        "bounds": [lower, upper],
        "value_comparisons": 3,
        "derivative_comparisons": 6,
    }


def completion_units(root):
    database = InterfaceDatabase(root)
    indices = (
        60,
        133,
        149,
        158,
        163,
        168,
        262,
        265,
        349,
        350,
        381,
        386,
        390,
        408,
        409,
        415,
        423,
        424,
        425,
        429,
        435,
        450,
        458,
        466,
        468,
        487,
        493,
        494,
        505,
        516,
        518,
        519,
    )
    for source, chosen in (("JASPER_LANGE", indices), ("MULERO_CACHADINA", (99,))):
        for i in chosen:
            record = database.records(source)[i]
            yield (
                f"{source}/{i}",
                dict(zip(record.field_names, record.raw_fields, strict=True)),
                lambda s=source, i=i: completion_unit(database, s, i),
            )


FRACTIONS = ((1,), (1, 0), (0.258, 0.742), (0, 1), (0.2, 0, 0.8), (0.05, 0.15, 0.25, 0.2, 0.35))


def mixture_unit(method, x, t=None, neglect=False):
    n = len(x)
    sigma, rho = ([0.015 + 0.004 * i for i in range(n)], [9000 + 1500 * i for i in range(n)])
    if method == "Winterfeld_Scriven_Davis":
        value = winterfeld_scriven_davis(x, sigma, rho)
        expected = reference.Winterfeld_Scriven_Davis(x, sigma, rho)
    elif method == "Diguilio_Teja":
        tb, tc = [310 + 10 * i for i in range(n)], [470 + 25 * i for i in range(n)]
        value = diguilio_teja(t, x, sigma, tb, tc)
        expected = reference.Diguilio_Teja(t, x, sigma, tb, tc)
    else:
        ps = [5.1e-5 + 2.1e-5 * i for i in range(n)]
        ys = tuple(reversed(x))
        value = weinaug_katz(ps, 0.000125, None if neglect else 0.02011, x, ys)
        expected = reference.Weinaug_Katz(ps, 0.000125, 1e100 if neglect else 0.02011, x, ys)
    return {"value": compare(value, expected)}


def mixture_units():
    for i, x in enumerate(FRACTIONS):
        yield f"Winterfeld/{i}", {"x": x}, lambda x=x: mixture_unit("Winterfeld_Scriven_Davis", x)
        for t in (280, 320, 360):
            yield (
                f"Diguilio/{i}/{t}",
                {"x": x, "T": t},
                lambda x=x, t=t: mixture_unit("Diguilio_Teja", x, t),
            )
        for neglect in (False, True):
            yield (
                f"Weinaug/{i}/{neglect}",
                {"x": x, "neglect_vapor": neglect},
                lambda x=x, b=neglect: mixture_unit("Weinaug_Katz", x, neglect=b),
            )


def water_oil_unit(t, water, oil):
    def ref(T):
        return reference.Meybodi_Daryasafar_Karimi(water, oil, T, 914)

    actual, derivative = derivative_values(
        lambda T: water_oil_interfacial_tension(
            T, water_density_kg_m3=water, oil_density_kg_m3=oil, oil_critical_temperature_K=914
        ),
        t,
    )
    return {"value": compare(actual, ref(t)), "derivatives": slope(derivative, ref, t)}


def water_oil_units():
    for w in (800, 980, 1100):
        for o in (650, 760, 1050):
            for t in (300, 580):
                yield (
                    f"water-oil/{w}/{o}/{t}",
                    {"rho_water": w, "rho_oil": o, "T": t, "Tc": 914},
                    lambda t=t, w=w, o=o: water_oil_unit(t, w, o),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=("methods", "sources", "source-completion", "mixtures", "water-oil"),
        required=True,
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    units = list(
        {
            "methods": method_units,
            "sources": lambda: source_units(args.source_directory),
            "source-completion": lambda: completion_units(args.source_directory),
            "mixtures": mixture_units,
            "water-oil": water_oil_units,
        }[args.block]()
    )
    assert (
        len(units)
        == {
            "methods": 60,
            "sources": 1037,
            "source-completion": 33,
            "mixtures": 36,
            "water-oil": 18,
        }[args.block]
    )
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    start, rows = time.perf_counter(), []
    for identity, inputs, run in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=run(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 200 == 0 or len(rows) == len(units):
            seconds = time.perf_counter() - start
            print(
                f"{len(rows)}/{len(units)}; {len(rows) / seconds:.1f}/s; "
                f"ETA {(len(units) - len(rows)) * seconds / len(rows):.1f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "block": args.block,
        "reference_version": chemicals.__version__,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    if args.block in {"sources", "source-completion"}:
        for key in ("value_comparisons", "derivative_comparisons", "rejected", "unbuilt"):
            summary[key] = sum(r.get("metrics", {}).get(key, 0) for r in rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:5]:
        print(row["unit"], row["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
