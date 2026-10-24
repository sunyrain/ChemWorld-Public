# ruff: noqa: B023 -- comparison closures execute in each fixed unit.
"""Independent high-precision equation discriminator for fixed CSP inputs."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import mpmath as mp
from run_corresponding_states import ORDERS, ecs_models, lkp_models, mbwr_parameters

from chemworld.physchem.corresponding_states import (
    ExtendedCorrespondingStatesModel,
    LeeKeslerPlockerModel,
    MBWRReference,
)
from chemworld.physchem.cpa import CPAModel
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.molecular_eos import (
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)


def m(value):
    return mp.mpf(str(value))


def alpha_parameter(p, t):
    tr = t / m(p.critical_temperature_K)
    cs = list(map(m, p.alpha_coefficients))
    if p.alpha_model == "soave":
        return (1 + cs[0] * (1 - mp.sqrt(tr))) ** 2
    ell, mm, n = cs
    return tr ** (n * (mm - 1)) * mp.exp(ell * (1 - tr ** (mm * n)))


def reference_alpha(ref, t, rho):
    if isinstance(ref, CPAModel):
        p = ref.components[0]
        a = (
            m(p.attraction_Pa_m6_mol2)
            * (1 + m(p.alpha_coefficient) * (1 - mp.sqrt(t / m(p.critical_temperature_K)))) ** 2
        )
        b = m(p.covolume_m3_mol)
        s1, s2 = (1 + mp.sqrt(2), 1 - mp.sqrt(2)) if ref.cubic_model == "peng_robinson" else (1, 0)
        return -mp.log(1 - b * rho) - a / (
            b * m(ref.gas_constant_J_mol_K) * t * (s1 - s2)
        ) * mp.log((1 + s1 * b * rho) / (1 + s2 * b * rho))
    if isinstance(ref, HelmholtzFluid):
        delta, tau = rho / m(ref.reducing_molar_density_mol_m3), m(ref.reducing_temperature_K) / t
        answer = mp.mpf(0)
        for term in ref.residual_terms:
            c = term.coefficients
            for i, n in enumerate(c["n"]):
                p = {key: m(values[i]) for key, values in c.items()}
                exponent = mp.mpf(0)
                if term.type == "ResidualHelmholtzPower":
                    if p["l"]:
                        exponent = -(delta ** p["l"])
                elif term.type == "ResidualHelmholtzGaussian":
                    exponent = (
                        -p["eta"] * (delta - p["epsilon"]) ** 2
                        - p["beta"] * (tau - p["gamma"]) ** 2
                    )
                else:
                    raise ValueError("independent reference only expects Power/Gaussian")
                answer += m(n) * delta ** int(p["d"]) * tau ** p["t"] * mp.exp(exponent)
        return answer
    # Independent pressure-polynomial exponent declarations, gamma integration.
    p32 = (
        [(2, j) for j in (1, 0.5, 0, -1, -2)]
        + [(3, j) for j in (1, 0, -1, -2)]
        + [(4, 1), (4, 0), (4, -1), (5, 0), (6, -1), (6, -2), (7, -1), (8, -1), (8, -2), (9, -2)]
    )
    p32 += [
        (i, j)
        for i, js in (
            (3, (-2, -3)),
            (5, (-2, -4)),
            (7, (-2, -3)),
            (9, (-2, -4)),
            (11, (-2, -3)),
            (13, (-2, -3, -4)),
        )
        for j in js
    ]
    p19 = (
        [(2, j) for j in (1, 0, -1, -2, -3)]
        + [(3, j) for j in (1, 0, -1)]
        + [(4, 1), (4, 0), (5, 1), (5, 0), (6, 0)]
    )
    p19 += [(i, j) for i in (3, 5) for j in (-2, -3, -4)]
    gamma, d = m(ref.gamma_L2_mol2), rho / 1000
    result = mp.mpf(0)
    for k, (c, (i, j)) in enumerate(
        zip(ref.coefficients, p32 if ref.variant == 32 else p19, strict=True)
    ):
        coefficient = m(c)
        if ref.variant == 19:
            coefficient *= -1000 if 1 <= k <= 4 else 1000
        integral = (
            d ** (i - 1) / (i - 1)
            if k < (19 if ref.variant == 32 else 13)
            else mp.gammainc((i - 1) // 2, 0, gamma * d * d) / (2 * gamma ** ((i - 1) // 2))
        )
        result += coefficient * t ** (m(j) - 1) * integral / (1000 * m(ref.gas_constant_J_mol_K))
    return result


def ecs_alpha(model, t, rho, x):
    cs, ref = model.components, model.reference_shape
    ai = [m(c.attraction_Pa_m6_mol2) * alpha_parameter(c, t) for c in cs]
    a = sum(
        x[i] * x[j] * mp.sqrt(ai[i] * ai[j]) * (1 - m(model.interaction[i][j]))
        for i in range(len(cs))
        for j in range(len(cs))
    )
    h = sum(xx * m(c.covolume_m3_mol) for xx, c in zip(x, cs, strict=True)) / m(ref.covolume_m3_mol)
    target = a / (h * m(ref.attraction_Pa_m6_mol2))
    if ref.alpha_model == "soave":
        mm = m(ref.alpha_coefficients[0])
        f = ((mm * mp.sqrt(t / m(ref.critical_temperature_K)) + mp.sqrt(target)) / (1 + mm)) ** 2
    else:
        f = mp.findroot(
            lambda f: f * alpha_parameter(ref, t / f) - target,
            (m(0.8), m(1.2)),
            tol=mp.eps * 128,
            maxsteps=80,
        )
    return reference_alpha(model.reference, t / f, rho * h)


def lkp_alpha(model, t, rho, x):
    cs = model.components
    v = sum(
        x[i]
        * x[j]
        * (
            (
                m(a.critical_molar_volume_m3_mol) ** (mp.mpf(1) / 3)
                + m(b.critical_molar_volume_m3_mol) ** (mp.mpf(1) / 3)
            )
            / 2
        )
        ** 3
        for i, a in enumerate(cs)
        for j, b in enumerate(cs)
    )
    tc = sum(
        x[i]
        * x[j]
        * (
            (
                (
                    m(a.critical_molar_volume_m3_mol) ** (mp.mpf(1) / 3)
                    + m(b.critical_molar_volume_m3_mol) ** (mp.mpf(1) / 3)
                )
                / 2
            )
            ** 3
        )
        ** m(0.25)
        * mp.sqrt(m(a.critical_temperature_K) * m(b.critical_temperature_K))
        * m(model.temperature_interaction[i][j])
        for i, a in enumerate(cs)
        for j, b in enumerate(cs)
    ) / v ** m(0.25)
    omega = sum(xx * m(c.acentric_factor) for xx, c in zip(x, cs, strict=True))
    tr, d, w = t / tc, rho * v / (m(0.2905) - m(0.085) * omega), omega / m(0.3978)

    def basis(tr, dd, reference):
        raw = (
            (
                ".2026579",
                ".331511",
                ".027655",
                ".203488",
                ".0313385",
                ".0503618",
                ".016901",
                ".041577",
                ".000048736",
                ".00000740336",
                "1.226",
                ".03754",
            )
            if reference
            else (
                ".1181193",
                ".265728",
                ".154790",
                ".030323",
                ".0236744",
                ".0186984",
                "0",
                ".042724",
                ".0000155488",
                ".0000623689",
                ".65392",
                ".060167",
            )
        )
        b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, beta, gamma = map(mp.mpf, raw)
        b, c, d5, e = (
            b1 - b2 / tr - b3 / tr**2 - b4 / tr**3,
            c1 - c2 / tr + c3 / tr**3,
            d1 + d2 / tr,
            c4 / tr**3,
        )
        ar = (
            b * dd
            + c * dd**2 / 2
            + d5 * dd**5 / 5
            + e / (2 * gamma) * (beta + 1 - (beta + 1 + gamma * dd * dd) * mp.exp(-gamma * dd * dd))
        )
        z = (
            1
            + b * dd
            + c * dd**2
            + d5 * dd**5
            + e * dd**2 * (beta + gamma * dd * dd) * mp.exp(-gamma * dd * dd)
        )
        return ar, z

    q0, q1 = mp.findroot(
        lambda q0, q1: (
            basis(tr, d / q0, False)[1] / q0 - basis(tr, d / q1, True)[1] / q1,
            (1 - w) * q0 + w * q1 - 1,
        ),
        (1, 1),
        tol=mp.eps * 128,
        maxsteps=80,
    )
    return (1 - w) * (basis(tr, d / q0, False)[0] - mp.log(q0)) + w * (
        basis(tr, d / q1, True)[0] - mp.log(q1)
    )


def run(output, source_dir):
    output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 80
    models = []
    for shape, base, case in product(
        ("SRK", "PR"), ("SRK", "PR", "NIST_MEOS", "MBWR19", "MBWR32"), (0, 5, 8)
    ):
        native, _, x = ecs_models(case, shape, base, source_dir)
        models.append(
            (f"{shape}/{base}/case{case}", native, x, (200.0, 300.0, 400.0), (0.0, 1000.0, 12000.0))
        )
    for shape, base, case in product(("SRK", "PR"), ("SRK", "PR"), (0, 5, 8)):
        native, _, x = ecs_models(case, shape, base, source_dir)
        native = replace(
            native,
            components=tuple(
                replace(c, alpha_model="twu91", alpha_coefficients=(0.65 + 0.1 * i, 0.9, 1.1))
                for i, c in enumerate(native.components)
            ),
            reference_shape=replace(
                native.reference_shape, alpha_model="twu91", alpha_coefficients=(0.8, 0.9, 1.1)
            ),
        )
        models.append(
            (
                f"Twu/{shape}/{base}/case{case}",
                native,
                x,
                (200.0, 300.0, 400.0),
                (0.0, 1000.0, 12000.0),
            )
        )
    for variant in (19, 32):
        cs = mbwr_parameters(source_dir, variant)
        native = MBWRReference(
            "C3", variant, tuple(cs[:-1]), cs[-1], (10.0, 3000.0), "fixed public data"
        )
        models.append(
            (
                f"MBWR{variant}",
                native,
                (1.0,),
                (200.0, 300.0, 400.0),
                (0.0, 1e-6, 1.0, 1000.0, 5000.0, 12000.0),
            )
        )
    for case in (0, 5, 8):
        native, _, x = lkp_models(case, "thermopack")
        models.append((f"LKP/case{case}", native, x, (250.0, 400.0, 600.0), (0.0, 1000.0, 5000.0)))
    grid = [
        (name, model, x, t, rho) for name, model, x, ts, rs in models for t, rho in product(ts, rs)
    ]
    assert len(grid) == 441
    failures, count, passed = [], 0, 0
    started = time.perf_counter()
    with (output / "records.jsonl").open("w") as stream:
        for unit, (name, native, xs, temperature, density) in enumerate(grid, 1):
            x, t, rho = tuple(map(m, xs)), m(temperature), m(density)
            equation = (
                ecs_alpha
                if isinstance(native, ExtendedCorrespondingStatesModel)
                else lkp_alpha
                if isinstance(native, LeeKeslerPlockerModel)
                else lambda model, t, rho, x: reference_alpha(model, t, rho)
            )
            records = []
            for i, j in ORDERS:
                records.append(
                    (
                        f"Ar{i}{j}",
                        lambda: molecular_scaled_derivative(
                            native,
                            temperature,
                            density,
                            xs,
                            inverse_temperature_order=i,
                            density_order=j,
                        ),
                        lambda: (
                            (1 / t) ** i
                            * rho**j
                            * mp.diff(
                                lambda beta, d: equation(native, 1 / beta, d, x),
                                (1 / t, rho),
                                (i, j),
                            )
                        ),
                    )
                )
            if not isinstance(native, MBWRReference):

                def mu(index):
                    def extensive(ni):
                        amounts = list(x)
                        amounts[index] = ni
                        total = sum(amounts)
                        return total * equation(
                            native, t, rho * total, tuple(v / total for v in amounts)
                        )

                    return m(native.gas_constant_J_mol_K) * t * mp.diff(extensive, x[index])

                for k in range(len(xs)):
                    records.append(
                        (
                            f"mu_{k}",
                            lambda k=k: molecular_residual_chemical_potentials(
                                native, temperature, density, xs
                            )[k],
                            lambda k=k: mu(k),
                        )
                    )
            results = []
            for quantity, actual, expected in records:
                atol = 2e-5 if quantity.startswith("mu") else 2e-9
                row = {"quantity": quantity, "rtol": 2e-7, "atol": atol}
                for role, fn in (("native", actual), ("reference", expected)):
                    try:
                        value = float(fn())
                        if not math.isfinite(value):
                            raise ValueError("nonfinite result")
                        row[role] = value
                    except Exception as exc:
                        row[role + "_exception"] = f"{type(exc).__name__}: {exc}"
                row["passed"] = (
                    "native" in row
                    and "reference" in row
                    and math.isclose(row["native"], row["reference"], rel_tol=2e-7, abs_tol=atol)
                )
                count += 1
                passed += row["passed"]
                results.append(row)
            record = {
                "unit": unit,
                "case": name,
                "T_K": temperature,
                "rho_mol_m3": density,
                "comparisons": results,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            failures.extend(
                {**{k: v for k, v in record.items() if k != "comparisons"}, **r}
                for r in results
                if not r["passed"]
            )
            if unit % 3 == 0 or unit == len(grid):
                elapsed = time.perf_counter() - started
                print(
                    f"CSP independent {unit}/{len(grid)}; {unit / elapsed:.2f} units/s; "
                    f"ETA {(len(grid) - unit) * elapsed / unit:.1f}s; failures {len(failures)}",
                    flush=True,
                )
    summary = {
        "mode": "development",
        "block": "independent-80digit",
        "units": len(grid),
        "comparisons": count,
        "expected_comparisons": 5220,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "model_inputs": [{"name": n, "model": v.to_dict()} for n, v, *_ in models],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("failures", "model_inputs")}))
    assert count == 5220
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.source_dir) else 1)
