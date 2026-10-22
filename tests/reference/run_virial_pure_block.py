# ruff: noqa: C408 -- scientific parameter declarations
"""Fixed full D3A matrix; only public reference calls and independent math."""

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals
import chemicals.virial as ref
import mpmath as mp

from chemworld.physchem import (
    TSONOPOULOS_CLASSES,
    VIRIAL_METHODS,
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    meng_polar_parameter,
    tsonopoulos_polar_parameters,
    virial_correlation,
    virial_derivatives,
    virial_fast,
    virial_matrix,
    virial_value,
    virial_vector,
)

R = 8.31446261815324
FLUIDS = (
    (190.56, 4599200.0, 0.286, 0.011),
    (425.2, 3800000.0, 0.274, 0.193),
    (647.1, 22050000.0, 0.229, 0.344),
    (500.0, 5000000.0, 0.31, -0.05),
)
METHODS = (
    "Pitzer_Curl",
    "Abbott",
    "Tsonopoulos",
    "Tsonopoulos_extended",
    "Oconnell_Prausnitz",
    "Xiang",
    "Meng",
    "Orbey_Vera",
    "Liu_Xiang",
)


def inputs(method, fluid):
    tc, pc, zc, w = fluid
    p = dict(
        Tc=tc, Pc=pc, Vc=zc * R * tc / pc, omega=w, a=-0.02 if method == "Meng" else 0.03, b=0.01
    )
    return {k: p[k] for k in VIRIAL_METHODS[method].required.split()}


def reference(method, T, p, order):
    func = getattr(
        ref, ("CVirial_" if method in ("Orbey_Vera", "Liu_Xiang") else "BVirial_") + method
    )
    if method in METHODS[:4]:
        return func(T, **p, order=order)
    if method == "Meng":
        # The public signature retains an unused Vc; it is not a native input.
        p = {**p, "Vc": 0.0002}
    return func(T, **p)[order]


def independent_terms(method, p):
    """Scientific literals transcribed independently, no native private tables."""
    w = p["omega"]
    base = {0: mp.mpf(".1445"), -1: mp.mpf("-.33"), -2: mp.mpf("-.1385"), -3: mp.mpf("-.0121")}
    if method == "Abbott":
        return [
            (0, mp.mpf(".083") + w * mp.mpf(".139")),
            (-mp.mpf("1.6"), -mp.mpf(".422")),
            (-mp.mpf("4.2"), -w * mp.mpf(".172")),
        ]
    if method == "Meng":
        return [
            (n, mp.mpf(a) + w * mp.mpf(b))
            for n, a, b in zip(
                (0, -1, -2, -3, -8),
                (".13356", "-.30252", "-.15668", "-.00724", "-.00022"),
                (".17404", "-.15581", ".38183", "-.44044", "-.00541"),
                strict=True,
            )
        ] + [(-6, p["a"])]
    if method == "Orbey_Vera":
        return [
            (0, mp.mpf(".01407") - w * mp.mpf(".02676")),
            (-mp.mpf("2.8"), mp.mpf(".02432") + w * mp.mpf(".01770")),
            (-3, w * mp.mpf(".040")),
            (-6, -w * mp.mpf(".003")),
            (-mp.mpf("10.5"), -mp.mpf(".00313") - w * mp.mpf(".00228")),
        ]
    if method == "Liu_Xiang":
        q = (p["Pc"] * p["Vc"] / (mp.mpf(str(R)) * p["Tc"]) - mp.mpf(".29")) ** 2
        table = (
            (".1623538", "-.5390344", "34.22804"),
            (".308744", "1.783526", "-74.76559"),
            ("-.01790184", "-1.055391", "279.922"),
            ("-.02789157", ".09955867", "-62.85431"),
        )
        return [
            (n, mp.mpf(a) + w * mp.mpf(b) + q * mp.mpf(c))
            for n, (a, b, c) in zip((0, -3, -6, -11), table, strict=True)
        ]
    if method == "Pitzer_Curl":
        terms = ((0, ".073"), (-1, ".46"), (-2, "-.5"), (-3, "-.097"), (-8, "-.0073"))
    elif method == "Oconnell_Prausnitz":
        terms = ((0, ".073"), (-2, ".46"), (-3, "-.5"), (-8, "-.097"))
    else:
        base[-8] = -mp.mpf(".000607")
        terms = ((0, ".0637"), (-2, ".331"), (-3, "-.423"), (-8, "-.008"))
    result = list(base.items()) + [(n, w * mp.mpf(c)) for n, c in terms]
    if method == "Tsonopoulos_extended":
        result += [(-6, p["a"]), (-8, -p["b"])]
    return result


def independent(method, T, p, order=0):
    p = {k: mp.mpf(str(v)) for k, v in p.items()}
    r = mp.mpf(str(R))
    tc = p["Tc"]
    tr = T / tc
    if method == "Xiang":
        theta = (p["Pc"] * p["Vc"] / (r * tc) - mp.mpf(".29")) ** 2
        return p["Vc"] * (
            -(mp.mpf("4.553") + mp.mpf("4.172") * p["omega"])
            * tr ** (-mp.mpf(".75"))
            * mp.exp(
                (mp.mpf(".02644") + mp.mpf(".075") * p["omega"] + mp.mpf("16.5") * theta) / tr**3
            )
            + (mp.mpf("3.53") + mp.mpf("4.297") * p["omega"]) * tr ** (-mp.mpf(".5"))
        )
    scale = (
        p["Vc"] ** 2
        if method == "Liu_Xiang"
        else (r * tc / p["Pc"]) ** (2 if method == "Orbey_Vera" else 1)
    )
    result = mp.mpf(0)
    for n, c in independent_terms(method, p):
        if order == 0:
            result += c * tr**n
        else:
            if order == -1:
                v = mp.log(T) if n == -1 else T ** (n + 1) / (n + 1)
            elif n == -1:
                v = T * (mp.log(T) - 1)
            elif n == -2:
                v = -mp.log(T)
            else:
                v = T ** (n + 2) / ((n + 1) * (n + 2))
            result += c * tc ** (-n) * v
    return scale * result


def main(output, block):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    mp.mp.dps = 60

    def unit(kind, identity, callback):
        comparisons = []

        def compare(actual, expected, *, third=False, order=0, rtol=2e-9):
            a, b = float(actual), float(expected)
            atol = 1e-9 if order < 0 else (1e-17 if third else 1e-13)
            result = dict(
                actual=a,
                expected=b,
                order=order,
                absolute_error=abs(a - b),
                passed=abs(a - b) <= atol + rtol * abs(b),
            )
            comparisons.append(result)

        row = dict(kind=kind, inputs=identity, comparisons=comparisons)
        try:
            callback(compare)
            row["passed"] = all(c["passed"] for c in comparisons)
        except Exception:
            row["passed"] = False
            row["error"] = traceback.format_exc()
        rows.append(row)

    if block == "batch":
        for method in METHODS:
            ps = [inputs(method, fluid) for fluid in FLUIDS]
            prefix = "CVirial_" if VIRIAL_METHODS[method].third else "BVirial_"
            columns = {
                "Tcs": [f[0] for f in FLUIDS],
                "Pcs": [f[1] for f in FLUIDS],
                "omegas": [f[3] for f in FLUIDS],
            }
            if method in ("Xiang", "Liu_Xiang", "Meng"):
                columns["Vcs"] = [f[2] * R * f[0] / f[1] for f in FLUIDS]
            if method in ("Meng", "Tsonopoulos_extended"):
                columns["ais"] = [p["a"] for p in ps]
            if method == "Tsonopoulos_extended":
                columns["bs"] = [p["b"] for p in ps]

            def batch_check(c, method=method, ps=ps, prefix=prefix, columns=columns):
                expected = getattr(ref, prefix + method + "_vec")(600.0, **columns)
                actual = virial_vector(method, 600.0, ps)
                for i in range(4):
                    for order in range(4):
                        c(
                            actual[i][order],
                            expected[order][i],
                            third=VIRIAL_METHODS[method].third,
                            order=order,
                        )

            unit("public_vector", {"method": method, "T": 600.0, "parameters": ps}, batch_check)

            def matrix_check(c, method=method, ps=ps, prefix=prefix, columns=columns):
                columns = {k: [v[:2], v[2:]] for k, v in columns.items()}
                expected = getattr(ref, prefix + method + "_mat")(600.0, **columns)
                actual = virial_matrix(method, 600.0, [ps[:2], ps[2:]])
                for i in range(2):
                    for j in range(2):
                        for order in range(4):
                            c(
                                actual[i][j][order],
                                expected[order][i][j],
                                third=VIRIAL_METHODS[method].third,
                                order=order,
                            )

            unit("public_matrix", {"method": method, "T": 600.0, "parameters": ps}, matrix_check)
            if method in METHODS[:4]:

                def fast_check(c, method=method, ps=ps, prefix=prefix):
                    for p in ps:
                        expected = getattr(ref, prefix + method + "_fast")(600.0, **p)
                        for order, value in enumerate(virial_fast(method, 600.0, p)):
                            c(value, expected[order], order=order)

                unit("public_fast", {"method": method, "T": 600.0, "parameters": ps}, fast_check)
    else:
        for method in METHODS:
            for fluid in FLUIDS:
                p = inputs(method, fluid)
                for tr in (0.65, 0.9, 1.3, 2.0, 4.0):
                    T = tr * p["Tc"]

                    def check(c, method=method, p=p, T=T):
                        values = virial_derivatives(method, T, p)
                        for order, value in enumerate(values):
                            c(
                                value,
                                reference(method, T, p, order),
                                third=VIRIAL_METHODS[method].third,
                                order=order,
                            )
                        if VIRIAL_METHODS[method].integrals:
                            for order in (-1, -2):
                                c(
                                    virial_value(method, T, order=order, **p),
                                    reference(method, T, p, order),
                                    order=order,
                                )

                    unit("public", dict(method=method, T=T, parameters=p), check)
                T = mp.mpf(str(1.3 * p["Tc"]))

                def high(c, method=method, p=p, T=T):
                    native = virial_derivatives(method, float(T), p)
                    for order, value in enumerate(native):
                        c(
                            value,
                            mp.diff(lambda x: independent(method, x, p), T, order),
                            third=VIRIAL_METHODS[method].third,
                            order=order,
                            rtol=2e-10,
                        )
                    if VIRIAL_METHODS[method].integrals:
                        for order in (-1, -2):
                            c(
                                virial_value(method, float(T), order=order, **p),
                                independent(method, T, p, order),
                                order=order,
                                rtol=2e-10,
                            )
                            c(
                                native[0],
                                mp.diff(
                                    lambda x, order=order: independent(method, x, p, order),
                                    T,
                                    -order,
                                ),
                                rtol=2e-10,
                            )

                unit("independent_60_digits", dict(method=method, T=float(T), parameters=p), high)
        for species in TSONOPOULOS_CLASSES:
            for dipole in (0.0, 1.0, 2.0):
                for convention in ("documented", "chemicals_1_5_2"):

                    def polar(c, species=species, dipole=dipole, convention=convention):
                        ab = tsonopoulos_polar_parameters(
                            Tc=500,
                            Pc=5e6,
                            species_type=species,
                            dipole=dipole,
                            convention=convention,
                        )
                        # API compatibility uses class; documented methanol/zero-dipole
                        # alkanol is checked with its explicit published parameters.
                        api = dict(species_type=species, dipole=dipole)
                        if convention == "documented" and species == "methanol":
                            api = dict(a=0.0878, b=0.0525)
                        elif convention == "documented" and species == "alkanol" and dipole == 0:
                            api = dict(a=0.0878, b=0.00908)
                        for tr in (0.8, 1.2, 2.0):
                            for order in range(4):
                                c(
                                    virial_value(
                                        "Tsonopoulos_extended",
                                        tr * 500,
                                        order=order,
                                        Tc=500,
                                        Pc=5e6,
                                        omega=0.2,
                                        **ab,
                                    ),
                                    ref.BVirial_Tsonopoulos_extended(
                                        tr * 500, 500, 5e6, 0.2, order=order, **api
                                    ),
                                    order=order,
                                )

                    unit("polar_" + convention, dict(species=species, dipole=dipole), polar)
        for dipole in (0.0, 1.0, 2.0):
            for halo in (False, True):
                unit(
                    "meng_polar",
                    dict(dipole=dipole, haloalkane=halo),
                    lambda c, d=dipole, h=halo: c(
                        meng_polar_parameter(Tc=500, Pc=5e6, dipole=d, haloalkane=h),
                        ref.Meng_virial_a(500, 5e6, d, haloalkane=h),
                    ),
                )
        for method in METHODS:
            p = inputs(method, FLUIDS[0])
            T = 1.3 * p["Tc"]

            def consumer(c, method=method, p=p, T=T):
                curve = virial_correlation(
                    method,
                    parameters=p,
                    temperature_range_K=(0.6 * p["Tc"], 4.1 * p["Tc"]),
                    source_note="Fixed synthetic development parameters",
                )
                restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
                package = ComponentPropertyPackage(ComponentSpec("synthetic", "CH4"), (restored,))
                c(
                    package.evaluate(curve.property_id, temperature_K=T).value,
                    reference(method, T, p, 0),
                    third=VIRIAL_METHODS[method].third,
                )

            unit("native_consumer", dict(method=method, T=T, parameters=p), consumer)
    summary = dict(
        block=block,
        mode="development",
        reference_version=chemicals.__version__,
        units=len(rows),
        passed=sum(r["passed"] for r in rows),
        failed=sum(not r["passed"] for r in rows),
        comparisons=sum(len(r["comparisons"]) for r in rows),
        comparison_failures=sum(not c["passed"] for r in rows for c in r["comparisons"]),
        counts=dict(Counter(r["kind"] for r in rows)),
        elapsed_s=time.perf_counter() - started,
        failures=[r for r in rows if not r["passed"]],
        complete_and_passed=all(r["passed"] for r in rows),
    )
    (output / "units.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["complete_and_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", choices=["main", "batch"], default="main")
    args = parser.parse_args()
    raise SystemExit(main(args.output, args.block))
