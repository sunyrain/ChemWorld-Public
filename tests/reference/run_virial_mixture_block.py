"""Fixed D3B coverage. Public reference calls and independent high precision math."""

import argparse
import json
import time
import traceback
from itertools import product
from math import log
from pathlib import Path

import chemicals
import chemicals.virial as ref
import mpmath as mp
import numpy as np
from tests.reference.run_virial_pure_block import FLUIDS, independent

from chemworld.physchem import (
    VIRIAL_METHODS,
    VirialRootError,
    compressibility_from_second_virial,
    density_to_pressure_virial,
    density_virial_state,
    lee_kesler_cross_volumes,
    meng_duan_interactions,
    pressure_to_density_virial,
    pressure_virial_state,
    pressure_virial_volume,
    second_virial_from_compressibility,
    tarakad_danner_interactions,
    virial_coefficient_matrices,
    virial_cross_parameters,
    virial_mixture_response,
    virial_pressure,
    virial_volume_composition_response,
)

R = 8.31446261815324


def run(output):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows, current = [], []
    mp.mp.dps = 60

    def compare(label, actual, expected, atol=1e-13, rtol=2e-8, reference_difference=False):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        passed = a.shape == b.shape and bool(np.allclose(a, b, rtol=rtol, atol=atol))
        current.append(
            {
                "label": label,
                "actual": a.tolist(),
                "reference": b.tolist(),
                "comparisons": int(a.size),
                "passed": passed,
                "reference_domain_difference": bool(reference_difference),
                "absolute_tolerance": atol,
                "relative_tolerance": rtol,
            }
        )

    def unit(name, inputs, function):
        current.clear()
        error = None
        try:
            function()
        except Exception:
            error = traceback.format_exc()
        row = {
            "id": name,
            "inputs": inputs,
            "checks": list(current),
            "error": error,
            "passed": error is None
            and all(c["passed"] or c["reference_domain_difference"] for c in current),
        }
        rows.append(row)
        print(
            f"D3B completed={len(rows)} unit={name} passed={row['passed']} "
            f"elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )

    for n in (1, 2, 3):
        t, p, v, w = (
            [200, 400, 600][:n],
            [4e6, 5e6, 6e6][:n],
            [1e-4, 2e-4, 4e-4][:n],
            [0.01, 0.2, 0.4][:n],
        )
        cross = virial_cross_parameters(t, p, v, w)
        rk = ref.Tarakad_Danner_virial_CSP_kijs(v)
        rt = ref.Tarakad_Danner_virial_CSP_Tcijs(t, rk)
        expected = {
            "kij": rk,
            "Tc": rt,
            "Pc": ref.Tarakad_Danner_virial_CSP_Pcijs(t, p, v, rt),
            "Vc": ref.Lee_Kesler_virial_CSP_Vcijs(v),
            "omega": ref.Tarakad_Danner_virial_CSP_omegaijs(w),
        }
        for name, value in expected.items():
            actual = (
                tarakad_danner_interactions(v)
                if name == "kij"
                else lee_kesler_cross_volumes(v)
                if name == "Vc"
                else [[cross[i][j][name] for j in range(n)] for i in range(n)]
            )
            unit(
                f"cross-{n}-{name}",
                {"Tc": t, "Pc": p, "Vc": v, "omega": w},
                lambda a=actual, b=value: compare("public", a, b, rtol=2e-10),
            )

    carbons = [*range(1, 17), 20, 24, 32]
    classes = [("alkane", n) for n in carbons] + [
        ("carbon_dioxide", 0),
        ("nitrogen", 0),
        ("other", 0),
    ]
    atoms = [{"C": n, "H": 2 * n + 2} for n in carbons] + [{"C": 1, "O": 2}, {"N": 2}, {"Fe": 1}]
    identifiers = [f"explicit-alkane-{n}" for n in carbons] + ["124-38-9", "7727-37-9", "7439-89-6"]
    unit(
        "meng-all-and-heldout",
        {"classes": classes, "CAS": identifiers, "atoms": atoms},
        lambda: compare(
            "public-484-including-fixed-holdouts",
            meng_duan_interactions(classes, unknown="zero"),
            ref.Meng_Duan_2005_virial_CSP_kijs(identifiers, atoms),
            rtol=2e-10,
        ),
    )
    # The paper documents inorganic log^(3/2); constants recovered on carbon 1.
    unit(
        "meng-inorganic-independent",
        {"carbons": carbons},
        lambda: compare(
            "documented-basis",
            [
                meng_duan_interactions([(name, 0), ("alkane", n)])[0][1]
                for name in ("carbon_dioxide", "nitrogen")
                for n in carbons
            ],
            [m * log(n + 1) ** 1.5 for m in (0.07475, 0.04311) for n in carbons],
            rtol=2e-10,
        ),
    )

    fractions = ([1.0], [0.25, 0.75], [0.2, 0.3, 0.5])
    for z in fractions:
        n = len(z)
        fluid = FLUIDS[:n]
        cross = virial_cross_parameters(
            [a[0] for a in fluid],
            [a[1] for a in fluid],
            [a[2] * R * a[0] / a[1] for a in fluid],
            [a[3] for a in fluid],
        )
        params = {
            method: [
                [
                    {key: cell[key] for key in VIRIAL_METHODS[method].required.split()}
                    for cell in row
                ]
                for row in cross
            ]
            for method in ("Tsonopoulos", "Orbey_Vera")
        }
        for temperature in (400.0, 700.0, 1200.0):

            def mixture_case(temperature=temperature, params=params, z=z, n=n):
                responses = {}
                for kind, method in [("B", "Tsonopoulos"), ("C", "Orbey_Vera")]:
                    matrices = virial_coefficient_matrices(method, temperature, params[method])
                    r = virial_mixture_response(z, matrices, kind=kind)
                    responses[kind] = r
                    atol = 1e-13 if kind == "B" else 1e-17
                    negative = kind == "C" and any(x < 0 for row in matrices[0] for x in row)
                    if kind == "B":
                        compare(
                            "B-public-T",
                            r.temperature_derivatives,
                            [ref.BVirial_mixture(z, m) for m in matrices],
                            atol,
                        )
                        calls = [
                            ref.dBVirial_mixture_dzs,
                            ref.d2BVirial_mixture_dzizjs,
                            ref.d3BVirial_mixture_dzizjzks,
                        ]
                    else:
                        calls_t = [
                            ref.CVirial_mixture_Orentlicher_Prausnitz,
                            ref.dCVirial_mixture_dT_Orentlicher_Prausnitz,
                            ref.d2CVirial_mixture_dT2_Orentlicher_Prausnitz,
                            ref.d3CVirial_mixture_dT3_Orentlicher_Prausnitz,
                        ]
                        for d, call in enumerate(calls_t):
                            compare(
                                f"C-public-T{d}",
                                r.temperature_derivatives[d],
                                call(z, *matrices[: d + 1]),
                                atol,
                                reference_difference=negative,
                            )
                        calls = [
                            ref.dCVirial_mixture_Orentlicher_Prausnitz_dzs,
                            ref.d2CVirial_mixture_Orentlicher_Prausnitz_dzizjs,
                            ref.d3CVirial_mixture_Orentlicher_Prausnitz_dzizjzks,
                        ]
                        compare(
                            "C-public-Tz",
                            r.temperature_gradient,
                            ref.d2CVirial_mixture_Orentlicher_Prausnitz_dTdzs(z, *matrices[:2]),
                            atol,
                            reference_difference=negative,
                        )
                    for label, native, call in zip(
                        ("z", "zz", "zzz"), (r.gradient, r.hessian, r.third), calls, strict=True
                    ):
                        compare(
                            kind + "-public-" + label,
                            native,
                            call(z, matrices[0]),
                            atol,
                            reference_difference=negative,
                        )

                    def math_formula(*q, method=method, kind=kind):
                        t, *ys = q
                        coefficients = [
                            [independent(method, t, cell) for cell in row] for row in params[method]
                        ]
                        if kind == "B":
                            return sum(
                                ys[i] * ys[j] * coefficients[i][j]
                                for i, j in product(range(n), repeat=2)
                            )

                        def root(i, j, k):
                            v = coefficients[i][j] * coefficients[j][k] * coefficients[i][k]
                            return mp.sign(v) * abs(v) ** (mp.mpf(1) / 3)

                        return sum(
                            ys[i] * ys[j] * ys[k] * root(i, j, k)
                            for i, j, k in product(range(n), repeat=3)
                        )

                    point = tuple(mp.mpf(str(x)) for x in (temperature, *z))

                    def derivative(axes, point=point):
                        orders = tuple(axes.count(i) for i in range(n + 1))
                        return float(mp.diff(math_formula, point, orders))

                    compare(
                        kind + "-mp-T",
                        r.temperature_derivatives,
                        [derivative([0] * d) for d in range(4)],
                        atol,
                    )
                    for prefix, tensor in [
                        ((), r.gradient),
                        ((), r.hessian),
                        ((), r.third),
                        ((0,), r.temperature_gradient),
                        ((0,), r.temperature_hessian),
                        ((0, 0), r.second_temperature_gradient),
                    ]:
                        shape = np.asarray(tensor).shape
                        values = np.array(
                            [
                                derivative([*prefix, *(i + 1 for i in index)])
                                for index in np.ndindex(shape)
                            ]
                        ).reshape(shape)
                        compare(kind + f"-mp-{prefix}-{len(shape)}", tensor, values, atol)
                    s = virial_mixture_response(z, matrices, kind=kind, coordinates="independent")
                    projection = np.vstack((np.eye(n - 1), -np.ones(n - 1)))
                    compare(kind + "-simplex-gradient", s.gradient, projection.T @ r.gradient, atol)
                    compare(
                        kind + "-simplex-hessian",
                        np.asarray(s.hessian).reshape(n - 1, n - 1),
                        projection.T @ r.hessian @ projection,
                        atol,
                    )
                b, c = responses["B"], responses["C"]
                answer = virial_volume_composition_response(temperature, 1e5, b, c)
                volume = answer["state"]["molar_volume_m3_mol"]
                g = ref.dV_dzs_virial(b.value, c.value, volume, b.gradient, c.gradient)
                h = ref.d2V_dzizjs_virial(
                    b.value, c.value, volume, b.gradient, c.gradient, g, b.hessian, c.hessian
                )
                compare("V-public-gradient", answer["gradient_m3_mol"], g, 1e-12)
                compare("V-public-hessian", answer["hessian_m3_mol"], h, 1e-12)
                # High precision implicit root of a local Taylor expression;
                # B/C derivatives above have independent full-function checks.
                zz = tuple(mp.mpf(str(a)) for a in z)

                def volume_at(*ys):
                    delta = [a - b for a, b in zip(ys, zz, strict=True)]

                    def local(r):
                        return (
                            mp.mpf(str(r.value))
                            + sum(mp.mpf(str(r.gradient[i])) * delta[i] for i in range(n))
                            + sum(
                                mp.mpf(str(r.hessian[i][j])) * delta[i] * delta[j] / 2
                                for i, j in product(range(n), repeat=2)
                            )
                        )

                    bb, cc = local(b), local(c)
                    return mp.findroot(
                        lambda v: (
                            mp.mpf("100000") * v / (mp.mpf(str(R)) * temperature)
                            - 1
                            - bb / v
                            - cc / v**2
                        ),
                        mp.mpf(str(volume)),
                    )

                compare(
                    "V-mp-gradient",
                    answer["gradient_m3_mol"],
                    [
                        float(mp.diff(volume_at, zz, tuple(int(k == i) for k in range(n))))
                        for i in range(n)
                    ],
                    1e-12,
                )
                compare(
                    "V-mp-hessian",
                    answer["hessian_m3_mol"],
                    [
                        [
                            float(
                                mp.diff(
                                    volume_at,
                                    zz,
                                    tuple(int(k == i) + int(k == j) for k in range(n)),
                                )
                            )
                            for j in range(n)
                        ]
                        for i in range(n)
                    ],
                    1e-12,
                )

            unit(
                f"mixture-{n}-{temperature:g}",
                {"fractions": z, "T": temperature, "cross": cross},
                mixture_case,
            )

    coefficients = ((), (1e-4,), (-1e-4, 2e-8), (1e-4, 1e-8, 1e-12), (1e-4, 1e-8, 1e-12, 1e-16))
    for t, p, c in product((300.0, 600.0), (1e4, 1e5, 1e6), coefficients):

        def density_case(t=t, p=p, c=c):
            state = density_virial_state(t, p, c)
            compare(
                "Z-public", state.compressibility_factor, ref.Z_from_virial_density_form(t, p, c)
            )
            compare(
                "P-inverse", virial_pressure(t, state.molar_volume_m3_mol, c) / p, 1.0, rtol=1e-10
            )
            compare("mechanical-stability", state.pressure_volume_slope < 0, True, atol=0, rtol=0)
            converted = density_to_pressure_virial(t, c)
            compare("series-roundtrip", pressure_to_density_virial(t, converted), c, atol=1e-25)
            b, cc, d, e = (*c, *([0.0] * (4 - len(c))))
            series = (
                b,
                cc - b * b,
                d - 3 * b * cc + 2 * b**3,
                e - 4 * b * d - 2 * cc * cc + 10 * b * b * cc - 5 * b**4,
            )
            compare(
                "independent-series-coefficients",
                converted,
                [x / (R * t) ** i for i, x in enumerate(series[: len(c)], 1)],
                atol=1e-30,
            )
            compare("B-to-Z", compressibility_from_second_virial(t, p, b), ref.B_to_Z(b, t, p))
            compare(
                "B-from-Z", second_virial_from_compressibility(t, p, 1.02), ref.B_from_Z(1.02, t, p)
            )

        unit(f"density-{t:g}-{p:g}-{len(c)}", {"T": t, "P": p, "coefficients": c}, density_case)

        def pressure_case(t=t, p=p, c=c):
            cp = density_to_pressure_virial(t, c)
            volume = pressure_virial_volume(t, p, cp)
            compare("Z-public", p * volume / (R * t), ref.Z_from_virial_pressure_form(p, cp))
            state = pressure_virial_state(t, volume, cp)
            compare("P-inverse", state.pressure_Pa / p, 1.0, rtol=1e-10)
            compare("mechanical-stability", state.pressure_volume_slope < 0, True, atol=0, rtol=0)

        unit(
            f"pressure-{t:g}-{p:g}-{len(c)}",
            {"T": t, "P": p, "density_coefficients": c},
            pressure_case,
        )

    for b in (-0.2, -0.3, -0.25):

        def root_case(b=b):
            try:
                state = density_virial_state(1 / R, 1, (b,))
            except VirialRootError as exc:
                compare("expected-rejection", b <= -0.25, True, atol=0, rtol=0)
                compare("candidate-diagnostics", bool(exc.candidates), True, atol=0, rtol=0)
            else:
                compare(
                    "analytic-gas-root", state.compressibility_factor, (1 + (1 + 4 * b) ** 0.5) / 2
                )
                compare("two-candidates", len(state.candidates), 2, atol=0, rtol=0)
                compare(
                    "one-stable", sum(c["accepted"] for c in state.candidates), 1, atol=0, rtol=0
                )

        unit(f"root-boundary-{b}", {"T": 1 / R, "P": 1, "B": b}, root_case)
    for sign in (-1, 0, 1):

        def sign_case(sign=sign):
            value = sign * 1e-8
            matrix = [[value, value], [value, value]]
            derivatives = (matrix, *([[0.0, 0.0], [0.0, 0.0]] for _ in range(3)))
            r = virial_mixture_response([0.3, 0.7], derivatives, kind="C")
            compare("signed-real-value", r.value, value, 1e-17)
            compare("homogeneous-gradient", r.gradient, [3 * value] * 2, 1e-17)
            compare("constant-temperature", r.temperature_derivatives[1:], [0, 0, 0], 1e-17)
            compare(
                "public-signed-domain",
                r.value,
                ref.CVirial_mixture_Orentlicher_Prausnitz([0.3, 0.7], matrix),
                1e-17,
                reference_difference=sign < 0,
            )

        unit(f"C-signed-{sign}", {"C": sign * 1e-8, "fractions": [0.3, 0.7]}, sign_case)

    result = {
        "block": "NATIVE-CONV-22D3B",
        "chemicals_version": chemicals.__version__,
        "units": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "comparisons": sum(c["comparisons"] for r in rows for c in r["checks"]),
        "failed_units": [r["id"] for r in rows if not r["passed"]],
        "reference_differences": [
            {"unit": r["id"], **c}
            for r in rows
            for c in r["checks"]
            if not c["passed"] and c["reference_domain_difference"]
        ],
        "elapsed_seconds": time.monotonic() - start,
        "results": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"results", "reference_differences"}}
        ),
        flush=True,
    )
    return not result["failed_units"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
