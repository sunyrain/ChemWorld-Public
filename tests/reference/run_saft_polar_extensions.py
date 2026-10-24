# ruff: noqa: B023 -- fixed-unit callbacks execute before advancing the loop.
"""Fixed correlation-polar comparisons; raw records remain outside Git."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import teqp

from chemworld.physchem.molecular_eos import molecular_scaled_derivative
from chemworld.physchem.saft_polar_extensions import CorrelationMultipole, CorrelationPolar
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel

PARAMETERS = ((1, 3.4e-10, 180, 12), (1.7, 3.8e-10, 230, 15), (2.2, 4e-10, 260, 18))
PROFILES = (
    (((1, 0),), (1,)),
    (((0, 1),), (1,)),
    (((1, 0.7),), (1,)),
    (((1, 0), (0.6, 0)), (0.3, 0.7)),
    (((1, 0), (0, 0.8)), (0.3, 0.7)),
    (((0, 1), (0, 0.8)), (0.3, 0.7)),
    (((1, 0.7), (0, 0)), (0.3, 0.7)),
    (((1, 0.3), (0.4, 0.7), (0, 0)), (0.2, 0.3, 0.5)),
    (((1, 0.7), (0, 0)), (0, 1)),
    (((1, 0.7), (0, 0)), (1, 0)),
)
CONTROLS = [
    ("GubbinsTwu", family, reduction, 1, 1)
    for family, reduction in product(
        ("Luckas", "GubbinsTwu", "Gottschalk"), ("packing_fraction", "Gubbins")
    )
    if (family, reduction) != ("Gottschalk", "Gubbins")
]
CONTROLS += [
    ("GrayGubbins", family, reduction, *scales)
    for family, reduction, scales in product(
        ("Luckas", "GubbinsTwu"), ("packing_fraction", "Gubbins"), ((1, 1), (0.8, 1.2))
    )
]
ORDERS = [(i, j) for i in range(4) for j in range(4 - i)]


def models(case, control, polarizable=False):
    moments, xs = PROFILES[case]
    cs = tuple(
        MieSegment(str(i), *PARAMETERS[i][:3], "fixed polar comparison", PARAMETERS[i][3])
        for i in range(len(moments))
    )
    spec = CorrelationPolar(
        tuple(
            CorrelationMultipole(
                str(i),
                d * 3.33564e-30,
                q * 3.33564e-40,
                "fixed polar comparison",
                (1 + 0.2 * i) * 1e-30 if polarizable else 0,
                (0.6 + 0.3 * i) * 1e-30 if polarizable else 0,
            )
            for i, (d, q) in enumerate(moments)
        ),
        "fixed polar comparison",
        *control,
    )
    interaction = tuple(
        tuple(0.02 if i != j else 0 for j in range(len(cs))) for i in range(len(cs))
    )
    native = SAFTVRMieModel(cs, interaction, (100, 1000), "fixed polar comparison", polar=spec)
    coeffs = [
        {
            "name": c.component_id,
            "m": c.segment_count,
            "sigma_m": c.diameter_m,
            "epsilon_over_k": c.well_depth_K,
            "lambda_a": c.attractive_exponent,
            "lambda_r": c.repulsive_exponent,
            "mu_Cm": p.dipole_C_m,
            "Q_Cm2": p.quadrupole_C_m2,
            "BibTeXKey": "fixed polar comparison",
        }
        for c, p in zip(cs, spec.moments, strict=True)
    ]
    name = control[0] + "+" + control[1]
    if control[0] == "GubbinsTwu" and control[2] == "Gubbins":
        name += "+GubbinsTwuRhostar"
    flags = {
        "approach": "use_packing_fraction"
        if control[2] == "packing_fraction"
        else "calculate_Gubbins_rhostar",
        "C3": control[3],
        "C3b": control[4],
    }
    if polarizable:
        flags["polarizable"] = {
            "alpha_symm / m^3": [p.alpha_parallel_m3 for p in spec.moments],
            "alpha_asymm / m^3": [p.alpha_perpendicular_m3 for p in spec.moments],
        }
    payload = {
        "coeffs": coeffs,
        "kmat": [list(r) for r in interaction],
        "polar_model": name,
        "polar_flags": flags,
        "SAFTVRMie_flags": {"epsilon_ij": "Lafitte"},
    }
    ref = teqp.make_model({"kind": "SAFT-VR-Mie", "model": payload}, validate=False)
    # Gross-Vrabec zero counts retain identical nonpolar mixing/flags while
    # avoiding the 0/0 zero-moment Padé in correlation-polar reference models.
    baseline = {
        **payload,
        "polar_model": "GrossVrabec",
        "polar_flags": {},
        "coeffs": [{**c, "mu_Cm": 0, "Q_Cm2": 0, "nmu": 0, "nQ": 0} for c in coeffs],
    }
    zero = teqp.make_model({"kind": "SAFT-VR-Mie", "model": baseline}, validate=False)
    return native, ref, zero, np.asarray(xs, dtype=float)


def run(output, polarizable):
    output.mkdir(parents=True, exist_ok=False)
    controls = CONTROLS[5:] if polarizable else CONTROLS
    cases = (0, 2, 3, 4, 6) if polarizable else range(len(PROFILES))
    temperatures = (350,) if polarizable else (280, 480)
    densities = (100, 1500, 4000) if polarizable else (100, 2500, 7000)
    grid = list(product(cases, range(len(controls)), temperatures, densities))
    (output / "units.json").write_text(json.dumps({"controls": controls, "units": grid}, indent=2))
    count = passed = 0
    cache, failures = {}, []
    started = last = time.monotonic()
    print(f"start units=0/{len(grid)} comparisons={len(grid) * len(ORDERS)}", flush=True)
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, (case, control, t, rho) in enumerate(grid, 1):
            key = case, control
            if key not in cache:
                cache[key] = models(case, controls[control], polarizable)
            native, ref, zero, xs = cache[key]
            polar_only = SimpleNamespace(
                alphar=lambda t, r, x: native.contributions(t, r, x)["correlation_polar"]
            )
            records = []
            for i, j in ORDERS:
                record = {"order": (i, j)}
                for role, callback in (
                    (
                        "native",
                        lambda: molecular_scaled_derivative(
                            polar_only, t, rho, xs, inverse_temperature_order=i, density_order=j
                        ),
                    ),
                    (
                        "reference",
                        lambda: ref.get_Arxy(i, j, t, rho, xs) - zero.get_Arxy(i, j, t, rho, xs),
                    ),
                ):
                    try:
                        value = float(callback())
                        if not math.isfinite(value):
                            raise ValueError("nonfinite value")
                        record[role] = value
                    except Exception as exc:
                        record[role + "_exception"] = f"{type(exc).__name__}: {exc}"
                record["passed"] = (
                    "native" in record
                    and "reference" in record
                    and math.isclose(
                        record["native"], record["reference"], rel_tol=2e-5, abs_tol=2e-7
                    )
                )
                count += 1
                passed += record["passed"]
                if not record["passed"]:
                    failures.append(
                        {
                            "unit": unit,
                            "case": case,
                            "control": control,
                            "T": t,
                            "rho": rho,
                            **record,
                        }
                    )
                records.append(record)
            stream.write(
                json.dumps(
                    {
                        "unit": unit,
                        "case": case,
                        "control": control,
                        "T": t,
                        "rho": rho,
                        "comparisons": records,
                    }
                )
                + "\n"
            )
            stream.flush()
            now = time.monotonic()
            if now - last > 20 or unit == len(grid):
                rate = unit / (now - started)
                print(
                    f"units={unit}/{len(grid)} passed={passed}/{count} rate={rate:.2f}/s "
                    f"eta={(len(grid) - unit) / rate:.0f}s",
                    flush=True,
                )
                last = now
    summary = {
        "mode": "development",
        "backend": "teqp " + teqp.__version__,
        "block": "polarizable-source-differences" if polarizable else "permanent",
        "units": len(grid),
        "comparisons": count,
        "passed": passed,
        "failed": count - passed,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--polarizable", action="store_true")
    args = parser.parse_args()
    run(args.output, args.polarizable)
