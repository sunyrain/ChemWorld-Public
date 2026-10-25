"""Prospective classical public-option matrix and scalar-pressure discriminator."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

from molecular_reference import thermopack_residual_targets
from quantum_molecular_controls import FLAGS, HS
from run_quantum_molecular import reference_scaled
from run_saftvrmie import NAMES, ORDERS, models

from chemworld.physchem.mie_reference_options import MieReferenceOptions
from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)


def grid(block):
    if block in ("full", "pressure"):
        return [
            {
                "case": c,
                "hs": hs,
                "exact": exact,
                "extra": extra,
                "interaction": k,
                "t": t,
                "rho": r,
            }
            for c, (hs, exact, extra), k, t, r in product(
                range(10), HS, range(2), (300.0, 600.0), (100.0, 5000.0)
            )
        ]
    if block == "control":
        return [
            {
                "case": c,
                "hs": hs,
                "exact": exact,
                "extra": extra,
                "interaction": 1,
                "t": 600.0,
                "rho": 1000.0,
                "flags": flags,
            }
            for c, (hs, exact, extra), flags in product(
                (0, 6, 8), HS, product((False, True), repeat=5)
            )
        ]
    return [
        {
            "case": c,
            "hs": "Lafitte",
            "exact": False,
            "extra": False,
            "interaction": 0,
            "t": t,
            "rho": r,
            "cutoff": rc,
            "shift": shift,
        }
        for c, (rc, shift), t, r in product(
            range(4),
            ((None, False), (2.5, False), (2.5, True), (4.0, False), (4.0, True)),
            (300.0, 600.0),
            (100.0, 5000.0),
        )
    ]


def build(case):
    native, ref, x = models(case["case"], "Lafitte", "thermopack", False)
    n = len(x)
    k, ell, gamma = (0.04, 0.015, 0.025) if case["interaction"] else (0.02, 0.0, 0.0)

    def matrix(value):
        return tuple(tuple(value if i != j else 0.0 for j in range(n)) for i in range(n))

    options = MieReferenceOptions(
        hard_sphere_reference=case["hs"],
        exact_binary_dhs=case["exact"],
        enable_hs_extra=case["extra"],
        diameter_interaction=matrix(ell),
        repulsion_interaction=matrix(gamma),
    )
    native = replace(
        native, interaction=matrix(k), gas_constant_J_mol_K=ref.Rgas, reference_options=options
    )
    ref.set_hard_sphere_reference(
        {"Lafitte": "LAFITTE", "additive": "ADDITIVE", "nonadditive-Santos": "NON-ADDITIVE"}[
            case["hs"]
        ],
        case["exact"],
        case["extra"],
    )
    for i in range(n):
        for j in range(i + 1, n):
            ref.set_eps_kij(i + 1, j + 1, k)
            ref.set_sigma_lij(i + 1, j + 1, ell)
            ref.set_lr_gammaij(i + 1, j + 1, gamma)
    if "cutoff" in case:
        options = replace(
            options,
            enable_truncation_correction=case["cutoff"] is not None,
            enable_shift_correction=case["shift"],
            reduced_radius_cut=case["cutoff"] or 3.5,
        )
        native = replace(native, reference_options=options)
        ref.truncation_correction(case["cutoff"] is not None, case["shift"], case["cutoff"] or 3.5)
    return native, ref, x


def pressure_targets(ref, t, rho, x, step):
    r = ref.Rgas

    def responses(tt):
        h = rho * step
        p = [float(ref.pressure_tv(tt, 1 / (rho + k * h), x)[0]) for k in (-2, -1, 0, 1, 2)]
        slope = (p[0] - 8 * p[1] + 8 * p[3] - p[4]) / (12 * h)
        second = (-p[0] + 16 * p[1] - 30 * p[2] + 16 * p[3] - p[4]) / (12 * h * h)
        ar01 = p[2] / (rho * r * tt) - 1
        ar02 = slope / (r * tt) - 1 - 2 * ar01
        return slope, ar02, rho * second / (r * tt) - 2 * ar01 - 4 * ar02

    slope, ar02, ar03 = responses(t)
    h = t * step
    values = [responses(t + k * h)[1] for k in (-2, -1, 1, 2)]
    ar12 = -t * (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h)
    state, _ = thermopack_residual_targets(ref, t, rho, x)
    cp = state["residual_cv_J_mol_K"] + t * state["dP_dT_rho_Pa_K"] ** 2 / (rho * rho * slope) - r
    return {
        "dP_drho_T_Pa_m3_mol": slope,
        "Ar02": ar02,
        "Ar03": ar03,
        "Ar12": ar12,
        "residual_cp_J_mol_K": cp,
    }


def run(block, output, control_api="scalar"):
    output.mkdir(parents=True, exist_ok=False)
    cases = grid(block)
    totals = {
        "full": (960, 20928),
        "control": (1152, 2304),
        "truncation": (80, 1680),
        "pressure": (960, 9600),
    }
    assert len(cases) == totals[block][0]
    started = time.perf_counter()
    cache, inputs, failures, count = {}, [], [], 0
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, case in enumerate(cases, 1):
            key = json.dumps({k: v for k, v in case.items() if k not in ("t", "rho", "flags")})
            if key not in cache:
                cache[key] = build(case)
                inputs.append({"key": key, "model": cache[key][0].to_dict()})
            native, ref, x = cache[key]
            if "flags" in case:
                native = replace(
                    native,
                    reference_options=replace(
                        native.reference_options, **dict(zip(FLAGS, case["flags"], strict=True))
                    ),
                )
                for flag, setter in zip(
                    case["flags"],
                    (
                        ref.model_control_hard_sphere,
                        ref.model_control_a1,
                        ref.model_control_a2,
                        ref.model_control_a3,
                        ref.model_control_chain,
                    ),
                    strict=True,
                ):
                    setter(flag)
            t, rho = case["t"], case["rho"]
            names = (
                ("residual_helmholtz_J_mol", "pressure_Pa")
                if block == "control"
                else ("dP_drho_T_Pa_m3_mol", "Ar02", "Ar03", "Ar12", "residual_cp_J_mol_K")
                if block == "pressure"
                else (
                    *NAMES,
                    *(f"Ar{i}{j}" for i, j in ORDERS),
                    *(f"mu_{i}" for i in range(len(x))),
                )
            )
            actual, errors = {}, {}
            try:
                if block == "control":
                    actual = {
                        "residual_helmholtz_J_mol": native.alphar(t, rho, x) * ref.Rgas * t,
                        "pressure_Pa": rho
                        * ref.Rgas
                        * t
                        * (1 + molecular_scaled_derivative(native, t, rho, x, density_order=1)),
                    }
                else:
                    actual = dict(
                        molecular_eos_state(native, t, rho, x, require_stable=False).__dict__
                    )
                    actual.update(
                        {
                            f"Ar{i}{j}": molecular_scaled_derivative(
                                native, t, rho, x, inverse_temperature_order=i, density_order=j
                            )
                            for i, j in ORDERS
                        }
                    )
                    actual.update(
                        {
                            f"mu_{i}": v
                            for i, v in enumerate(
                                molecular_residual_chemical_potentials(native, t, rho, x)
                            )
                        }
                    )
            except Exception as exc:
                errors["native_exception"] = f"{type(exc).__name__}: {exc}"
            records = []
            for step in (1e-3, 3e-4) if block == "pressure" else (None,):
                target = {}
                try:
                    if block == "control":
                        target = (
                            {
                                "residual_helmholtz_J_mol": ref.helmholtz_tv(
                                    t, 1 / rho, x, property_flag="R"
                                )[0],
                                "pressure_Pa": ref.pressure_tv(t, 1 / rho, x)[0],
                            }
                            if control_api == "scalar"
                            else {
                                "residual_helmholtz_J_mol": ref.internal_energy_tv(
                                    t, 1 / rho, x, dedt=True, property_flag="R"
                                )[0]
                                - t
                                * ref.entropy_tv(t, 1 / rho, x, dsdt=True, property_flag="R")[0],
                                "pressure_Pa": ref.pressure_tv(t, 1 / rho, x, dpdv=True)[0],
                            }
                        )
                    elif block == "pressure":
                        target = pressure_targets(ref, t, rho, x, step)
                    else:
                        target, mu = thermopack_residual_targets(ref, t, rho, x)
                        target.update({f"mu_{i}": v for i, v in enumerate(mu)})
                        target.update(
                            {
                                f"Ar{i}{j}": v
                                for (i, j), v in reference_scaled(ref, t, rho, x).items()
                            }
                        )
                except Exception as exc:
                    errors["reference_exception"] = f"{type(exc).__name__}: {exc}"
                for name in names:
                    third = name.startswith("Ar") and int(name[-1]) + int(name[-2]) == 3
                    atol, rtol = (2e-6, 2e-5) if third else (2e-7, 3e-6)
                    a, b = actual.get(name), target.get(name)
                    a = float(a) if a is not None and math.isfinite(a) else None
                    b = float(b) if b is not None and math.isfinite(b) else None
                    passed = (
                        a is not None
                        and b is not None
                        and abs(a - b) <= atol + rtol * max(abs(a), abs(b))
                    )
                    row = dict(
                        quantity=name,
                        native=a,
                        reference=b,
                        atol=atol,
                        rtol=rtol,
                        step=step,
                        passed=passed,
                        **errors,
                    )
                    records.append(row)
                    count += 1
                    if not passed:
                        failures.append(dict(unit=unit, **case, **row))
            stream.write(json.dumps(dict(unit=unit, **case, comparisons=records)) + "\n")
            stream.flush()
            if unit % 12 == 0 or unit == len(cases):
                elapsed = time.perf_counter() - started
                print(
                    f"Classical {block}: {unit}/{len(cases)}, {unit / elapsed:.2f} units/s, "
                    f"ETA {(len(cases) - unit) * elapsed / unit:.1f}s, failed={len(failures)}",
                    flush=True,
                )
    summary = {
        "mode": "development",
        "block": block,
        "reference": "ThermoPack2.2.3",
        "control_reference_api": control_api,
        "units": len(cases),
        "comparisons": count,
        "expected_comparisons": totals[block][1],
        "passed": count - len(failures),
        "failed": len(failures),
        "all_failures": failures,
        "inputs": inputs,
        "seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert count == totals[block][1]
    print(json.dumps({k: v for k, v in summary.items() if k not in ("all_failures", "inputs")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block", choices=("full", "control", "truncation", "pressure"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-api", choices=("scalar", "full-derivative"), default="scalar")
    args = parser.parse_args()
    run(args.block, args.output, args.control_api)
