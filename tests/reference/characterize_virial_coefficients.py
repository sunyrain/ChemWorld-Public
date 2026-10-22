# ruff: noqa: C408 -- scientific parameter declarations
"""Recover omitted scientific constants from public calls, never source inspection.

Fixed design: NATIVE_CONV22D3A_EXPERIMENT.md. This is characterization data,
not the independent holdout qualification of the native implementation.
"""

import argparse
import json
from pathlib import Path

import chemicals
import chemicals.virial as ref
import numpy as np
from scipy.optimize import root

R = 8.31446261815324
TC, PC = 500.0, 5e6
ANCHORS = ((0.0, 0.29), (0.5, 0.29), (0.0, 0.24))


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    rows, fits = [], {}
    for method, ts in (("Xiang", (0.8, 1.2, 2.0)), ("Liu_Xiang", (0.7, 1.0, 1.5, 2.5))):
        coefficients = []
        for omega, z in ANCHORS:
            vc = z * R * TC / PC
            f = getattr(ref, ("BVirial_" if method == "Xiang" else "CVirial_") + method)
            values = np.array([f(t * TC, TC, PC, vc, omega)[0] for t in ts])
            if method == "Xiang":

                def residual(b, ts=ts, values=values, vc=vc):
                    tr = np.array(ts)
                    return (
                        -b[0] * tr ** (-0.75) * np.exp(b[1] * tr**-3)
                        + b[2] * tr ** (-0.5)
                        - values / vc
                    )

                solved = root(residual, (2.0, 0.2, 1.0), tol=1e-11)
                if np.max(np.abs(residual(solved.x))) > 1e-10:
                    raise ValueError(str(solved))
                c = solved.x
            else:
                c = np.linalg.solve(
                    np.array([[t**p for p in (0, -3, -6, -11)] for t in ts]), values / vc**2
                )
            coefficients.append(c)
            rows.append(
                dict(
                    method=method,
                    omega=omega,
                    Zc=z,
                    Tr=ts,
                    values=values.tolist(),
                    coefficients=c.tolist(),
                )
            )
        base, w, theta = (
            coefficients[0],
            (coefficients[1] - coefficients[0]) / 0.5,
            coefficients[2] - coefficients[0],
        )
        options = {}
        for kind, anchor in (("linear", -0.05), ("squared", 0.05**2)):
            table = np.array([base, w, theta / anchor]).T
            checks = []
            for z in (0.19, 0.34):
                vc = z * R * TC / PC
                coeffs = table @ np.array([1, 0.2, (z - 0.29) ** (1 if kind == "linear" else 2)])
                for t in (0.65, 1.1, 3.0):
                    expected = f(t * TC, TC, PC, vc, 0.2)[0]
                    if method == "Xiang":
                        actual = vc * (
                            -coeffs[0] * t ** (-0.75) * np.exp(coeffs[1] * t**-3)
                            + coeffs[2] * t ** (-0.5)
                        )
                    else:
                        actual = vc**2 * sum(
                            c * t**p for c, p in zip(coeffs, (0, -3, -6, -11), strict=True)
                        )
                    checks.append(
                        dict(
                            Zc=z,
                            Tr=t,
                            actual=actual,
                            expected=expected,
                            passed=abs(actual - expected)
                            <= 2e-8 * abs(expected) + (1e-13 if method == "Xiang" else 1e-16),
                        )
                    )
            options[kind] = dict(
                table=table.tolist(), holdouts=checks, passed=all(c["passed"] for c in checks)
            )
        fits[method] = options
    conventions = []
    classes = (
        "simple",
        "normal",
        "ketone",
        "aldehyde",
        "alkyl nitrile",
        "ether",
        "carboxylic acid",
        "ester",
        "alkyl halide",
        "mercaptan",
        "sulfide",
        "disulfide",
        "alkanol",
        "methanol",
        "water",
    )
    for name in classes:
        for dipole in (0.0, 1.0, 2.0):
            # Solve a/Tr^6-b/Tr^8 from two public calls; third point is held out.
            tr = np.array([0.8, 1.2])
            vals = [
                (
                    ref.BVirial_Tsonopoulos_extended(
                        t * TC, TC, PC, 0.2, species_type=name, dipole=dipole
                    )
                    - ref.BVirial_Tsonopoulos(t * TC, TC, PC, 0.2)
                )
                / (R * TC / PC)
                for t in tr
            ]
            a, b = np.linalg.solve(np.array([tr**-6, -(tr**-8)]).T, vals)
            t = 2.0
            expected = ref.BVirial_Tsonopoulos_extended(
                t * TC, TC, PC, 0.2, species_type=name, dipole=dipole
            )
            actual = ref.BVirial_Tsonopoulos(t * TC, TC, PC, 0.2) + (R * TC / PC) * (
                a / t**6 - b / t**8
            )
            conventions.append(
                dict(
                    species_type=name,
                    dipole=dipole,
                    a=a,
                    b=b,
                    heldout=expected,
                    passed=abs(actual - expected) < 1e-13 + 2e-8 * abs(expected),
                )
            )
    public = []
    for t in (0.8, 1.2, 2.0):
        public.append(
            dict(
                Tr=t,
                Abbott=ref.BVirial_Abbott(t * TC, TC, PC, 0.2),
                Tsonopoulos=ref.BVirial_Tsonopoulos(t * TC, TC, PC, 0.2),
                Oconnell=ref.BVirial_Oconnell_Prausnitz(t * TC, TC, PC, 0.2),
            )
        )
    meng = [
        dict(dipole=d, haloalkane=h, a=ref.Meng_virial_a(TC, PC, d, haloalkane=h))
        for d in (0.0, 1.0, 2.0)
        for h in (False, True)
    ]
    data = dict(
        reference_version=chemicals.__version__,
        anchors=rows,
        fits=fits,
        polar=conventions,
        public=public,
        meng=meng,
    )
    (output / "results.json").write_text(
        json.dumps(data, indent=2, default=lambda v: v.item()) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(fits=fits, polar=conventions, public=public, meng=meng),
            indent=2,
            default=lambda v: v.item(),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
