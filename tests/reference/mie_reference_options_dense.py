# ruff: noqa: B023 -- every callback is consumed in the current fixed unit.
"""Independent 80-digit mixture Mie derivatives and zero-amount chemical potentials."""

import argparse
import json
import math
import time
from functools import cache
from itertools import product
from pathlib import Path

import mpmath as mp
from run_mie_high_precision import EFFECTIVE_ROWS, PHI_ROWS
from run_saftvrmie import CASES, ORDERS, PARAMETERS

from chemworld.physchem.molecular_eos import (
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


@cache
def diameter_series(component, temperature, parameters=PARAMETERS):
    _, _, sigma0, eps0, lr0 = parameters[component]
    sigma, eps, lr = (mp.mpf(str(v)) for v in (sigma0 * 1e10, eps0, lr0))
    t = mp.mpf(temperature)
    prefactor = lr / (lr - 6) * (lr / 6) ** (6 / (lr - 6))
    result = []
    for order in range(4):

        def integrand(radius, order=order):
            if radius == 0:
                return mp.mpf(1 if order == 0 else 0)
            potential = prefactor * eps / t * (radius**-lr - radius**-6)
            if order == 0:
                return -mp.expm1(-potential)
            polynomials = (
                -potential,
                2 * potential - potential**2,
                -6 * potential + 6 * potential**2 - potential**3,
            )
            return mp.exp(-potential) * polynomials[order - 1] / t**order

        result.append(sigma * mp.quad(integrand, [0, 0.5, 0.8, 0.95, 1]))
    return tuple(result)


def mixture_expression(case, temperature, *, cases=CASES, parameters=PARAMETERS, kij=0.02):
    ids, _ = cases[case]
    n = len(ids)
    components = [
        [
            mp.mpf(str(v))
            for v in (parameters[i][1], parameters[i][2] * 1e10, parameters[i][3], parameters[i][4])
        ]
        for i in ids
    ]
    t0 = mp.mpf(temperature)
    ds = [diameter_series(i, temperature, parameters) for i in ids]
    phi = [[mp.mpf(v) for v in row.split()] for row in PHI_ROWS]
    ceff = [[mp.mpf(v) for v in row.split()] for row in EFFECTIVE_ROWS]

    def expression(t, rho, amounts):
        total = sum(amounts)
        fractions = [v / total for v in amounts]
        mean_m = sum(x * c[0] for x, c in zip(fractions, components, strict=True))
        segments = [fractions[i] * components[i][0] / mean_m for i in range(n)]
        density = rho * mp.mpf("6.02214076e-7") * mean_m
        diameter = [
            sum(v * (t - t0) ** k / mp.factorial(k) for k, v in enumerate(series)) for series in ds
        ]
        moments = [sum(segments[i] * diameter[i] ** k for i in range(n)) for k in range(4)]
        eta = mp.pi * density / 6 * moments[3]
        zeta2 = mp.pi * density / 6 * moments[2]
        # BMCSL mixture free energy, written in normalized diameter moments.
        hard_spheres = (
            3 * moments[1] * zeta2 / (1 - eta)
            + moments[2] ** 2 / moments[3] * zeta2 / (1 - eta) ** 2
            + (moments[2] ** 3 / moments[3] ** 2 - 1) * mp.log(1 - eta)
        )
        z = (
            mp.pi
            * density
            / 6
            * sum(
                segments[i] * segments[j] * ((diameter[i] + diameter[j]) / 2) ** 3
                for i in range(n)
                for j in range(n)
            )
        )
        zb = (
            mp.pi
            * density
            / 6
            * sum(
                segments[i] * segments[j] * ((components[i][1] + components[j][1]) / 2) ** 3
                for i in range(n)
                for j in range(n)
            )
        )

        def compressibility(packing):
            return (1 - packing) ** 4 / (
                1 + 4 * packing + 4 * packing**2 - 4 * packing**3 + packing**4
            )

        mono, chain = hard_spheres, mp.mpf(0)
        for i, j in product(range(n), repeat=2):
            left, right = components[i], components[j]
            sigma = (left[1] + right[1]) / 2
            eps = (1 - (mp.mpf(str(kij)) if i != j else 0)) * mp.sqrt(left[2] * right[2])
            eps *= (left[1] * right[1]) ** mp.mpf("1.5") / sigma**3
            lr, la = 3 + mp.sqrt((left[3] - 3) * (right[3] - 3)), mp.mpf(6)
            d = (diameter[i] + diameter[j]) / 2
            ratio = sigma / d
            prefactor = lr / (lr - la) * (lr / la) ** (la / (lr - la))
            alpha = prefactor * (1 / (la - 3) - 1 / (lr - 3))
            fs = [
                sum(phi[k][q] * alpha**k for k in range(4))
                / (1 + sum(phi[k][q] * alpha ** (k - 3) for k in range(4, 7)))
                for q in range(6)
            ]

            def term(exponent, packing):
                effective = sum(
                    sum(ceff[k][q] / exponent**q for q in range(4)) * packing ** (k + 1)
                    for k in range(4)
                )
                integral_i = (1 - ratio ** (3 - exponent)) / (exponent - 3)
                integral_j = (
                    1
                    - (exponent - 3) * ratio ** (4 - exponent)
                    + (exponent - 4) * ratio ** (3 - exponent)
                )
                integral_j /= (exponent - 3) * (exponent - 4)
                return ratio**exponent * (
                    (1 - packing / 2) * integral_i / (1 - packing) ** 3
                    - mp.mpf("4.5") * packing * (1 + packing) * integral_j / (1 - packing) ** 3
                    - (1 - effective / 2) / ((exponent - 3) * (1 - effective) ** 3)
                )

            fa, fr = term(la, z), term(lr, z)
            faa, far, frr = term(2 * la, z), term(la + lr, z), term(2 * lr, z)
            chi = fs[0] * zb + fs[1] * zb**5 + fs[2] * zb**8
            mono += (
                segments[i]
                * segments[j]
                * (
                    2 * mp.pi * density * d**3 * eps * prefactor * (fa - fr) / t
                    + mp.pi
                    * density
                    * d**3
                    * eps**2
                    * prefactor**2
                    * compressibility(z)
                    * (1 + chi)
                    * (faa - 2 * far + frr)
                    / t**2
                    - eps**3 * fs[3] * zb * mp.exp(fs[4] * zb + fs[5] * zb**2) / t**3
                )
            )
            if i == j:
                k0 = -mp.log(1 - z) + (42 * z - 39 * z**2 + 9 * z**3 - 2 * z**4) / (
                    6 * (1 - z) ** 3
                )
                k1 = (z**4 + 6 * z**2 - 12 * z) / (2 * (1 - z) ** 3)
                k2 = -3 * z**2 / (8 * (1 - z) ** 2)
                k3 = (-(z**4) + 3 * z**2 + 3 * z) / (6 * (1 - z) ** 3)
                log_contact = k0 + k1 * ratio + k2 * ratio**2 + k3 * ratio**3
                g1 = prefactor * (
                    3 * mp.diff(lambda zz: zz * (term(la, zz) - term(lr, zz)), z)
                    - la * fa
                    + lr * fr
                )
                g2 = prefactor**2 * (
                    mp.mpf("1.5")
                    * mp.diff(
                        lambda zz: (
                            zz
                            * compressibility(zz)
                            * (term(2 * la, zz) - 2 * term(la + lr, zz) + term(2 * lr, zz))
                        ),
                        z,
                    )
                    - compressibility(z) * (la * faa - (la + lr) * far + lr * frr)
                )
                gamma = 10 * (1 - mp.tanh(10 * (mp.mpf(".57") - alpha))) * zb
                gamma *= mp.expm1(eps / t) * mp.exp(-mp.mpf("6.7") * zb - 8 * zb**2)
                chain -= (
                    fractions[i]
                    * (left[0] - 1)
                    * (
                        log_contact
                        + (eps / t * g1 + (eps / t) ** 2 * (1 + gamma) * g2) * mp.exp(-log_contact)
                    )
                )
        return mean_m * mono + chain

    return expression


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    failures, count, metadata = [], 0, []
    cache = {}
    with mp.workdps(80), (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, (case, temperature, density) in enumerate(
            product(range(4, 10), (180, 300, 600), (1000, 16000)), 1
        ):
            ids, raw_x = CASES[case]
            cs = tuple(
                MieSegment(*PARAMETERS[i][:4], "Fixed Mie input", PARAMETERS[i][4]) for i in ids
            )
            interaction = tuple(tuple(0.02 if i != j else 0 for j in ids) for i in ids)
            native = SAFTVRMieModel(
                cs, interaction, (100, 1000), "Original Mie mixture discriminator"
            )
            if (case, temperature) not in cache:
                cache[case, temperature] = mixture_expression(case, temperature)
            expression = cache[case, temperature]
            x = tuple(mp.mpf(str(v)) for v in raw_x)
            t, rho = mp.mpf(temperature), mp.mpf(density)
            errors, actual, expected = {}, {}, {}
            try:
                beta = 1 / t
                expected = {
                    f"Ar{i}{j}": float(
                        beta**i
                        * rho**j
                        * mp.diff(lambda b, r: expression(1 / b, r, x), (beta, rho), (i, j))
                    )
                    for i, j in ORDERS
                }
                for i in range(len(x)):

                    def extensive(amount, i=i):
                        amounts = tuple(amount if j == i else v for j, v in enumerate(x))
                        total = sum(amounts)
                        return total * expression(t, rho * total, amounts)

                    expected[f"mu_{i}"] = float(
                        mp.mpf(str(native.gas_constant_J_mol_K)) * t * mp.diff(extensive, x[i])
                    )
            except Exception as exc:
                errors["reference_exception"] = f"{type(exc).__name__}: {exc}"
            try:
                actual = {
                    f"Ar{i}{j}": molecular_scaled_derivative(
                        native,
                        temperature,
                        density,
                        raw_x,
                        inverse_temperature_order=i,
                        density_order=j,
                    )
                    for i, j in ORDERS
                }
                actual.update(
                    {
                        f"mu_{i}": value
                        for i, value in enumerate(
                            molecular_residual_chemical_potentials(
                                native, temperature, density, raw_x
                            )
                        )
                    }
                )
            except Exception as exc:
                errors["native_exception"] = f"{type(exc).__name__}: {exc}"
            metadata.append(
                {
                    "unit": unit,
                    "case": case,
                    "T_K": temperature,
                    "rho_mol_m3": density,
                    "fractions": raw_x,
                    "model": native.to_dict(),
                }
            )
            for quantity in (
                *(f"Ar{i}{j}" for i, j in ORDERS),
                *(f"mu_{i}" for i in range(len(x))),
            ):
                a, b = actual.get(quantity), expected.get(quantity)
                atol = 2e-5 if quantity.startswith("mu") else 2e-9
                passed = (
                    a is not None
                    and b is not None
                    and math.isfinite(a)
                    and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=2e-7, abs_tol=atol)
                )
                row = {
                    "unit": unit,
                    "case": case,
                    "T_K": temperature,
                    "rho_mol_m3": density,
                    "quantity": quantity,
                    "native": a,
                    "reference": b,
                    "passed": passed,
                    "rtol": 2e-7,
                    "atol": atol,
                    **errors,
                }
                stream.write(json.dumps(row) + "\n")
                count += 1
                if not passed:
                    failures.append(row)
            stream.flush()
            elapsed = time.perf_counter() - started
            print(
                f"Mie dense mixture {unit}/36; {unit / elapsed:.3f} units/s; "
                f"ETA {(36 - unit) * elapsed / unit:.1f}s; failures {len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "block": "independent-dense-mixture-80digit",
        "units": 36,
        "comparisons": count,
        "expected_comparisons": 444,
        "passed": count - len(failures),
        "failed": len(failures),
        "failures": failures,
        "model_inputs": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert count == 444
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
