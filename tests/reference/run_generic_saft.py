"""Compact prospective GenericSAFT source/equation composition block."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path
from unittest.mock import patch

import mpmath as mp
import numpy as np
import teqp
from run_dufal import models as dufal_models

from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.generic_saft_import import generic_saft_from_teqp
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL

ORDERS = [(i, j) for i in range(4) for j in range(4 - i)]


def source_specification(kind, association):
    row = {
        "name": "A",
        "m": 1.0,
        "sigma_Angstrom": 3.7,
        "epsilon_over_k": 180.0,
        "lambda_r": 12.0,
        "lambda_a": 6.0,
        "BibTeXKey": "Fixed generic input",
    }
    base = {"kind": kind, "model": {"coeffs": [row], "kmat": [[0.0]]}}
    if kind == "softSAFT":
        base["model"] = {"m": [1.0], "sigma / m": [3.7e-10], "epsilon/kB / K": [180.0]}
    elif kind == "2CLJ":
        base["model"] = {"author": "2CLJF_Lisal", "L^*": 0.3}
    spec = {"kind": "genericSAFT", "model": {"nonpolar": base}}
    if association in ("CS", "KG"):
        spec["model"]["association"] = {
            "kind": "canonical",
            "model": {
                "Delta_rule": "CR1",
                "b / m^3/mol": [3e-5],
                "beta": [0.01],
                "epsilon / J/mol": [1000 * R_J_PER_MOL_K],
                "molecule_sites": [["e", "H"]],
            },
        }
    elif association == "Dufal":
        spec["model"]["association"] = {
            "kind": "Dufal",
            "model": {
                "Delta_rule": "Dufal",
                "sigma / m": [3.0555e-10],
                "epsilon/kB / K": [418.0],
                "lambda_r": [35.823],
                "kmat": [[0.0]],
                "epsilon_HB / J/mol": [1600 * R_J_PER_MOL_K],
                "K_HB / m^3": [496.66e-30],
                "molecule_sites": [["e", "e", "H", "H"]],
                "options": {"rtol": 1e-13, "atol": 1e-13, "max_iters": 1000},
            },
        }
    return spec


def canonical_derivatives(kind, temperature, density):
    with mp.workdps(60):
        beta, rho = 1 / mp.mpf(temperature), mp.mpf(density)

        def ar(inverse_t, r):
            eta = r * mp.mpf("3e-5") / 4
            g = (1 - eta / 2) / (1 - eta) ** 3 if kind == "CS" else 1 / (1 - mp.mpf("1.9") * eta)
            strength = r * mp.mpf("3e-5") * mp.mpf(".01") * mp.expm1(1000 * inverse_t) * g
            fraction = 2 / (1 + mp.sqrt(1 + 4 * strength))
            return 2 * (mp.log(fraction) - fraction / 2 + mp.mpf(".5"))

        return {
            (i, j): float(beta**i * rho**j * mp.diff(ar, (beta, rho), (i, j))) for i, j in ORDERS
        }


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dufal_reference = dufal_models(0, "teqp")[1]
    failures, metadata, count = [], [], 0
    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, (kind, association, state) in enumerate(
            product(
                ("PCSAFT", "SAFTVRMie", "softSAFT", "2CLJ"),
                ("none", "CS", "KG", "Dufal"),
                ((450.0, 10.0), (600.0, 1000.0)),
            ),
            1,
        ):
            temperature, density = state
            spec = source_specification(kind, association)
            imported = generic_saft_from_teqp(
                spec,
                component_ids=("A",),
                temperature_range_K=(100, 1000),
                source="Fixed GenericSAFT comparison",
                canonical_radial_distribution="Kontogeorgis"
                if association == "KG"
                else "Carnahan-Starling",
                two_center_scales=(3.7e-10, 180.0),
            )
            native = imported.model
            t_ref, rho_ref = (
                (temperature / 180, density * AVOGADRO_PER_MOL * (3.7e-10) ** 3)
                if kind == "2CLJ"
                else (temperature, density)
            )
            actual, expected, errors = {}, {}, {}
            try:
                ref_base = teqp.make_model(
                    {"kind": "genericSAFT", "model": {"nonpolar": spec["model"]["nonpolar"]}},
                    validate=False,
                )
                assoc = (
                    canonical_derivatives(association, temperature, density)
                    if association in ("CS", "KG")
                    else {
                        key: dufal_reference.get_Arxy(*key, temperature, density, np.array([1.0]))
                        if association == "Dufal"
                        else 0.0
                        for key in ORDERS
                    }
                )
                expected = {
                    f"Ar{i}{j}": ref_base.get_Arxy(i, j, t_ref, rho_ref, np.array([1.0]))
                    + assoc[i, j]
                    for i, j in ORDERS
                }
                expected["mu_0"] = (
                    R_J_PER_MOL_K * temperature * (expected["Ar00"] + expected["Ar01"])
                )
                expected["density_inverse"] = density
            except Exception as exc:
                errors["reference_exception"] = f"{type(exc).__name__}: {exc}"
            try:
                actual = {
                    f"Ar{i}{j}": molecular_scaled_derivative(
                        native,
                        temperature,
                        density,
                        (1.0,),
                        inverse_temperature_order=i,
                        density_order=j,
                    )
                    for i, j in ORDERS
                }
                actual["mu_0"] = molecular_residual_chemical_potentials(
                    native, temperature, density, (1.0,)
                )[0]
                if "Ar01" in expected:
                    pressure = density * R_J_PER_MOL_K * temperature * (1 + expected["Ar01"])
                    actual["density_inverse"] = molecular_density_from_pressure(
                        native,
                        temperature,
                        pressure,
                        (1.0,),
                        initial_density_mol_m3=0.95 * density,
                        density_bounds_mol_m3=(0.1 * density, 2 * density),
                    ).molar_density_mol_m3
            except Exception as exc:
                errors["native_exception"] = f"{type(exc).__name__}: {exc}"
            metadata.append(
                {
                    "unit": unit,
                    "base": kind,
                    "association": association,
                    "T_K": temperature,
                    "rho_mol_m3": density,
                    "model": native.to_dict(),
                    "import_notes": imported.notes,
                }
            )
            for name in (*(f"Ar{i}{j}" for i, j in ORDERS), "mu_0", "density_inverse"):
                a, b = actual.get(name), expected.get(name)
                atol = 2e-5 if name == "mu_0" else 2e-7 if name == "density_inverse" else 2e-9
                passed = (
                    a is not None
                    and b is not None
                    and math.isfinite(a)
                    and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=2e-7, abs_tol=atol)
                )
                row = {
                    "unit": unit,
                    "base": kind,
                    "association": association,
                    "T_K": temperature,
                    "rho_mol_m3": density,
                    "quantity": name,
                    "native": a,
                    "reference": b,
                    "rtol": 2e-7,
                    "atol": atol,
                    "passed": passed,
                    **errors,
                }
                stream.write(json.dumps(row) + "\n")
                count += 1
                if not passed:
                    failures.append(row)
            stream.flush()
            elapsed = time.perf_counter() - started
            print(
                f"GenericSAFT {unit}/32; {unit / elapsed:.2f}units/s; "
                f"ETA {(32 - unit) * elapsed / unit:.1f}s; failures {len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "units": 32,
        "comparisons": count,
        "expected_comparisons": 384,
        "passed": count - len(failures),
        "failed": len(failures),
        "failures": failures,
        "model_inputs": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert count == 384
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-diameter-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.source_diameter_diagnostic:
        from mie_quadrature_diagnostic import source_diameter

        with patch("chemworld.physchem.saftvrmie.mie_effective_diameter", source_diameter):
            result = run(args.output)
    else:
        result = run(args.output)
    raise SystemExit(0 if result else 1)
