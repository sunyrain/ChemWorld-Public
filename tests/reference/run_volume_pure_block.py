"""Fixed D2A development qualification; reference code is never read or shipped."""
# ruff: noqa: C408 -- scientific parameter declarations

import argparse
import json
import math
import time
import traceback
from itertools import pairwise
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import volume as reference

from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    volume_correlation,
    volume_derivatives,
    volume_parameter_jacobian,
    volume_value,
)

# 33 parameter sets, 15 public methods. Temperatures are fixed independently
# of native validity metadata and do not change in response to outcomes.
CASES = (
    ("Rackett", (240, 280, 330), dict(Tc=369.83, Pc=4248000.0, Zc=0.2763)),
    ("Rackett", (300, 450, 600), dict(Tc=647.14, Pc=22048320.0, Zc=0.245)),
    ("ideal_gas", (200, 300, 600), dict(P=101325.0)),
    ("ideal_gas", (200, 300, 600), dict(P=2e6)),
    *(
        ("Yen_Woods_saturation", (300, 450, 600), dict(Tc=647.14, Vc=55.45e-6, Zc=z))
        for z in (0.245, 0.26, 0.30)
    ),
    *(
        ("Yamada_Gunn", (300, 450, 600), dict(Tc=647.14, Pc=22048320.0, omega=w))
        for w in (0.245, 0.0)
    ),
    *(
        ("Townsend_Hales", (300, 450, 600), dict(Tc=647.14, Vc=55.95e-6, omega=w))
        for w in (0.3449, 0.15)
    ),
    *(
        (
            "Bhirud_normal",
            tuple(
                f * 500 for f in (0.4, 0.7, 0.95, 0.979, 0.98, 0.9805, 0.985, 0.997, 0.9995, 1.0)
            ),
            dict(Tc=500.0, Pc=33.7e5, omega=w),
        )
        for w in (0.252, 0.0)
    ),
    ("COSTALD", (240, 280, 330), dict(Tc=369.83333, Vc=0.20008161e-3, omega=0.1532)),
    ("COSTALD", (300, 450, 600), dict(Tc=647.14, Vc=55.95e-6, omega=0.3449)),
    ("SNM0", (90, 121, 140), dict(Tc=150.8, Vc=7.49e-5, omega=-0.004)),
    ("SNM0", (90, 121, 140), dict(Tc=150.8, Vc=7.49e-5, omega=-0.004, delta_SRK=-0.03259620)),
    ("SNM0", (240, 280, 330), dict(Tc=369.83, Vc=0.0002, omega=0.1532)),
    ("Campbell_Thodos", (240, 280, 330), dict(Tb=231.04, Tc=369.83, Pc=4248000.0, MW=44.09562)),
    (
        "Campbell_Thodos",
        (260, 320, 390),
        dict(Tb=239.82, Tc=405.45, Pc=111.7 * 101325, MW=17.03, dipole=1.47),
    ),
    (
        "Campbell_Thodos",
        (300, 450, 600),
        dict(Tb=373.15, Tc=647.14, Pc=22048320.0, MW=18.01528, dipole=1.85, has_hydroxyl=True),
    ),
    ("Goodman", (100, 160, 220), dict(Tt=243.225, Vml=0.00023585)),
    ("Goodman", (120, 220, 320), dict(Tt=350, Vml=0.0001)),
    ("PPDS17", (300, 400, 500), dict(Tc=562.05, a0=0.0115508, a1=0.281004, a2=-0.00635447)),
    ("PPDS17", (300, 450, 550), dict(Tc=600, a0=0.01, a1=0.3, a2=0.02)),
    (
        "TDE_VDNS_rho",
        (400, 550, 700),
        dict(Tc=772.999, rhoc=320.037, a1=795.092, a2=-169.132, a3=448.929, a4=-102.931),
    ),
    ("TDE_VDNS_rho", (300, 450, 550), dict(Tc=600, rhoc=300, a1=500, a2=100, a3=-20, a4=10)),
    ("Rackett_fit", (400, 550, 700), dict(Tc=748.402, rhoc=314.629, b=0.257033, n=0.280338)),
    ("Rackett_fit", (250, 298.15, 390), dict(Tc=425.18, rhoc=228.3, b=0.2724, n=0.2863)),
    ("CRC_inorganic", (250, 300, 350), dict(rho0=2370.0, k=2.687, Tm=239.08)),
    ("CRC_inorganic", (900, 1000, 1100), dict(rho0=1800.0, k=0.4, Tm=850.0)),
    (
        "volume_VDI_PPDS",
        (70, 90, 110),
        dict(Tc=126.19, rhoc=313.0, a=470.922, b=493.251, c=-560.469, d=389.611),
    ),
    (
        "volume_VDI_PPDS",
        (300, 450, 550),
        dict(Tc=600.0, rhoc=300.0, a=500.0, b=100.0, c=-20.0, d=10.0),
    ),
)


def compare(actual, expected, *, rtol=3e-10, atol=1e-13):
    if not (math.isfinite(actual) and math.isfinite(expected)):
        raise AssertionError((actual, expected))
    error = abs(actual - expected)
    if error > atol + rtol * abs(expected):
        raise AssertionError(dict(actual=actual, expected=expected, absolute_error=error))
    return dict(actual=float(actual), reference=float(expected), absolute_error=float(error))


def difference(function, x, h):
    return (
        function(x - 2 * h) - 8 * function(x - h) + 8 * function(x + h) - function(x + 2 * h)
    ) / (12 * h)


def method_unit(method, t, p):
    def ref(temperature):
        return getattr(reference, method)(temperature, **p)

    result = {"value": compare(volume_value(method, t, **p), ref(t))}
    if method == "Bhirud_normal" and t / p["Tc"] in (0.98, 1.0):
        result["slope_scope"] = "join/critical value only; no central stencil across branch"
        return result
    h = 1e-3 * t
    if method == "Bhirud_normal" and t / p["Tc"] >= 0.979:
        h = 1e-5 * p["Tc"]  # Stay within each fixed table cell.
    slope = volume_derivatives(method, t, p)[1]
    result["slopes"] = [
        compare(slope, difference(ref, t, step), rtol=8e-5, atol=2e-11 * max(1, abs(ref(t))))
        for step in (h, h / 2)
    ]
    # Optional MW conversions are checked in addition to the raw quantity.
    if method in {"PPDS17", "TDE_VDNS_rho", "Rackett_fit", "CRC_inorganic", "volume_VDI_PPDS"}:
        result["molar_conversion"] = compare(
            volume_value(method, t, **p, MW=78.11), getattr(reference, method)(t, **p, MW=78.11)
        )
    return result


def jacobian_unit(method, t, p):
    names = [k for k in p if k != "has_hydroxyl"]
    result = {}
    if method == "Yen_Woods_saturation" and p["Zc"] == 0.26:
        try:
            volume_parameter_jacobian(method, t, p, names=("Zc",))
        except ValueError as exc:
            result["Zc_rejection"] = str(exc)
        else:
            raise AssertionError("accepted derivative across Zc branch")
        names.remove("Zc")
    actual = volume_parameter_jacobian(method, t, p, names=names)
    for key in names:
        x = p[key]

        def ref(value, key=key):
            return getattr(reference, method)(t, **{**p, key: value})

        h = max(abs(x), 0.01) * 1e-4
        result[key] = [
            compare(
                actual[key],
                difference(ref, x, step),
                rtol=8e-5,
                atol=2e-9 * max(1e-5, abs(ref(x))) / max(abs(x), 0.01),
            )
            for step in (h, h / 2)
        ]
    return result


def table_unit(tr):
    # Raw reference table, independent of native constants and spline objects.
    # At .98 the documented polynomial branch is the API's value convention.
    p = dict(Tc=500.0, Pc=33.7e5, omega=0.252)
    t = 500 * tr
    return {
        "value": compare(volume_value("Bhirud_normal", t, **p), reference.Bhirud_normal(t, **p))
    }


def higher_unit(method, t):
    # 60-digit independent equations; derivatives use mpmath, not native jets.
    mp.mp.dps = 60
    r = mp.mpf("8.31446261815324")
    functions = {
        "Rackett": (
            dict(Tc=600.0, Pc=4e6, Zc=0.27),
            lambda x: (
                r * 600 / mp.mpf(4e6) * mp.mpf(".27") ** (1 + (1 - x / 600) ** (mp.mpf(2) / 7))
            ),
        ),
        "Goodman": (
            dict(Tt=600.0, Vml=0.0002),
            lambda x: mp.mpf(".0002") / (mp.mpf("1.28") - mp.mpf(".16") * x / 600),
        ),
        "ideal_gas": (dict(P=4e6), lambda x: r * x / mp.mpf(4e6)),
        "volume_VDI_PPDS": (
            dict(Tc=600.0, rhoc=300.0, a=500.0, b=100.0, c=-20.0, d=10.0),
            lambda x: (
                300
                + 500 * (1 - x / 600) ** mp.mpf(".35")
                + 100 * (1 - x / 600) ** (mp.mpf(2) / 3)
                - 20 * (1 - x / 600)
                + 10 * (1 - x / 600) ** (mp.mpf(4) / 3)
            ),
        ),
        "COSTALD": (
            dict(Tc=600.0, Vc=0.0002, omega=0.2),
            lambda x: (
                mp.mpf(".0002")
                * (
                    1
                    - mp.mpf("1.52816") * (1 - x / 600) ** (mp.mpf(1) / 3)
                    + mp.mpf("1.43907") * (1 - x / 600) ** (mp.mpf(2) / 3)
                    - mp.mpf(".81446") * (1 - x / 600)
                    + mp.mpf(".190454") * (1 - x / 600) ** (mp.mpf(4) / 3)
                )
                * (
                    1
                    - mp.mpf(".2")
                    * (
                        -mp.mpf(".296123")
                        + mp.mpf(".386914") * (x / 600)
                        - mp.mpf(".0427258") * (x / 600) ** 2
                        - mp.mpf(".0480645") * (x / 600) ** 3
                    )
                    / (x / 600 - mp.mpf("1.00001"))
                )
            ),
        ),
    }
    p, function = functions[method]
    return {
        "orders_0_to_3": [
            compare(a, float(mp.diff(function, mp.mpf(t), i)), rtol=2e-9, atol=1e-15)
            for i, a in enumerate(volume_derivatives(method, t, p, order=3))
        ]
    }


def consumer_unit(method, t, p):
    parameters = dict(p)
    pressure = parameters.pop("P", 101325.0)
    # A synthetic component's MW is explicit; Campbell's MW is a physical input.
    mw = parameters.get("MW", 78.11184)
    natural_molar = method not in {
        "PPDS17",
        "TDE_VDNS_rho",
        "Rackett_fit",
        "CRC_inorganic",
        "volume_VDI_PPDS",
    }
    if natural_molar:
        parameters["MW"] = mw
    curve = volume_correlation(
        method,
        parameters=parameters,
        output="density",
        temperature_range_K=(t * 0.999, t * 1.001),
        source_note="D2A fixed synthetic/reference case",
        pressure_range_Pa=(pressure / 2, pressure * 2) if method == "ideal_gas" else None,
    )
    package = ComponentPropertyPackage(
        ComponentSpec(
            "fixture", "C3H8" if method == "Campbell_Thodos" else "C6H6", molecular_weight_g_mol=mw
        ),
        (curve,),
    )
    serialized = json.loads(json.dumps(package.to_dict()))
    restored = ComponentPropertyPackage(
        ComponentSpec.from_dict(serialized["component"]),
        tuple(PropertyCorrelation.from_dict(c) for c in serialized["correlations"]),
    )
    value = restored.evaluate(curve.property_id, temperature_K=t, pressure_Pa=pressure).value
    raw = getattr(reference, method)(t, **p)
    expected = mw / 1000 / raw if natural_molar else 1 / raw if method == "Rackett_fit" else raw
    return {"density": compare(value, expected), "relation": curve.metadata["physical_relation"]}


def units():
    assert len(CASES) == 33
    for i, (method, temperatures, p) in enumerate(CASES):
        for t in temperatures:
            yield (
                f"method/{i}/{method}/{t}",
                {"T": t, **p},
                lambda m=method, t=t, p=p: method_unit(m, t, p),
            )
        # Bhirud's middle of the ten-point grid is .9805: use .7 for parameter stencils.
        t = temperatures[1]
        yield (
            f"jacobian/{i}/{method}",
            {"T": t, **p},
            lambda m=method, t=t, p=p: jacobian_unit(m, t, p),
        )
    nodes = (0.980, 0.982, 0.984, 0.986, 0.988, 0.990, 0.992, 0.994, 0.996, 0.998, 0.999, 1.0)
    for tr in (*nodes, *(a + f * (b - a) for a, b in pairwise(nodes) for f in (0.25, 0.5, 0.75))):
        yield f"table/{tr}", {"Tr": tr}, lambda tr=tr: table_unit(tr)
    for method in ("Rackett", "Goodman", "ideal_gas", "volume_VDI_PPDS", "COSTALD"):
        for t in (300, 450, 550):
            yield f"higher/{method}/{t}", {"T": t}, lambda m=method, t=t: higher_unit(m, t)
    seen = set()
    for method, temperatures, p in CASES:
        if method not in seen:
            seen.add(method)
            t = temperatures[1]
            yield (
                f"consumer/{method}",
                {"T": t, **p},
                lambda m=method, t=t, p=p: consumer_unit(m, t, p),
            )


def convention_units():
    fluids = (
        (373.15, 647.14, 22048320.0, 18.01528, 1.85),
        (337.65, 512.6, 8090000.0, 32.04186, 1.70),
        (351.44, 514.0, 6137000.0, 46.06844, 1.69),
    )
    for i, (tb, tc, pc, mw, mu) in enumerate(fluids):
        p = dict(Tb=tb, Tc=tc, Pc=pc, MW=mw, dipole=mu, has_hydroxyl=True)
        for fraction in (0.55, 0.70, 0.90):
            for selector in (0, 1):
                t = fraction * tc
                yield (
                    f"convention/{i}/{fraction}/{selector}",
                    {"T": t, **p, "hydroxyl_polar_term": selector},
                    lambda t=t, p=p, s=selector: convention_unit(t, p, s),
                )


def convention_unit(t, p, selector):
    mp.mp.dps = 60
    tb, tc, pc, mw, mu = (mp.mpf(str(p[k])) for k in ("Tb", "Tc", "Pc", "MW", "dipole"))
    pc /= 101325
    tbr = tb / tc
    s = tbr * mp.log(pc) / (1 - tbr)
    theta = pc * mu**2 / tc**2
    lam = pc ** (mp.mpf(1) / 3) / (mw ** mp.mpf(".5") * tc ** (mp.mpf(5) / 6))
    alpha = (
        mp.mpf(".690") * tbr - mp.mpf(".3342") + mp.mpf("5.79e-10") / tbr ** mp.mpf("32.75")
    ) * pc ** mp.mpf(".145")
    beta = (
        mp.mpf(".00318") * s
        - mp.mpf(".0211")
        + mp.mpf(".625") * lam ** mp.mpf("1.35")
        + mp.mpf("5.90") * theta ** mp.mpf(".835")
        + selector * mp.mpf("9.74e6") * theta ** mp.mpf("3.38")
    )

    def equation(x):
        return (
            mp.mpf("8.31446261815324")
            * tc
            / (pc * 101325)
            * (alpha + beta * (1 - x / tc)) ** (1 + (1 - x / tc) ** (mp.mpf(2) / 7))
        )

    actual = volume_derivatives(
        "Campbell_Thodos", t, {**p, "hydroxyl_polar_term": selector}, order=3
    )
    metrics = {
        "equation": [
            compare(a, float(mp.diff(equation, mp.mpf(t), order)), rtol=2e-9, atol=1e-15)
            for order, a in enumerate(actual)
        ]
    }
    ref = reference.Campbell_Thodos(t, **p)
    if selector:
        metrics["public_api"] = compare(actual[0], ref)
    else:
        metrics["documented_minus_api"] = actual[0] - ref
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=("pure", "convention"), default="pure")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    cases = list(units() if args.block == "pure" else convention_units())
    assert len(cases) == (221 if args.block == "pure" else 18)
    print(f"D2A: {len(cases)} fixed units", flush=True)
    start, rows = time.perf_counter(), []
    for identity, inputs, run in cases:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(status="passed", metrics=run())
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == len(cases):
            seconds = time.perf_counter() - start
            print(
                f"{len(rows)}/{len(cases)}; {len(rows) / seconds:.1f}/s; "
                f"ETA {(len(cases) - len(rows)) * seconds / len(rows):.1f}s",
                flush=True,
            )
    summary = dict(
        block=args.block,
        mode="development",
        reference_version=chemicals.__version__,
        total=len(rows),
        passed=sum(r["status"] == "passed" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows),
        seconds=time.perf_counter() - start,
        results=rows,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:10]:
        print(row["unit"], row["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
