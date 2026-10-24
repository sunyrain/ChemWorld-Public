# ruff: noqa: B023 -- callbacks execute within each fixed comparison.
"""Fixed 50-declaration teqp comparison for molecular potential extensions."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import teqp

from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.molecular_potential_extensions import (
    ChaparroMieModel,
    PohlMieModel,
    TwoCenterLJModel,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL

BASE = ("synthetic", 3.4e-10, 120.0, (50.0, 2000.0), "Fixed POTENTIAL_EXT comparison")
SCALE = AVOGADRO_PER_MOL * BASE[1] ** 3
ORDERS = [(i, j) for i in range(4) for j in range(4 - i)]
NAMES = (
    "pressure_Pa",
    "compressibility_factor",
    "residual_helmholtz_J_mol",
    "residual_internal_energy_J_mol",
    "residual_enthalpy_J_mol",
    "residual_entropy_J_mol_K",
    "residual_cv_J_mol_K",
    "residual_cp_J_mol_K",
    "dP_dT_rho_Pa_K",
    "dP_drho_T_Pa_m3_mol",
    "mu_J_mol",
    "lnphi",
)


def declarations():
    items = []
    for author, length, pole in product(
        ("Mecke", "Lisal"),
        (0.0, 0.4, 1.0),
        (
            ("none", 0.0),
            ("dipole", 0.0),
            ("dipole", 1.0),
            ("dipole", 4.0),
            ("quadrupole", 0.0),
            ("quadrupole", 1.0),
            ("quadrupole", 4.0),
        ),
    ):
        model = TwoCenterLJModel(
            *BASE, author=author, elongation=length, polar=pole[0], reduced_moment_squared=pole[1]
        )
        kind = {"none": "2CLJF", "dipole": "2CLJF-Dipole", "quadrupole": "2CLJF-Quadrupole"}[
            pole[0]
        ]
        params = {"author": "2CLJF_" + author, "L^*": length}
        if pole[0] != "none":
            params["(mu^*)^2" if pole[0] == "dipole" else "(Q^*)^2"] = pole[1]
        items.append((model, {"kind": kind, "model": params}, pole[1] == 0))
    for lr in (11.0, 12.0, 13.0):
        items.append(
            (
                PohlMieModel(*BASE, repulsive_exponent=lr),
                {"kind": "Mie_Pohl2023", "model": {"lambda_r": lr}},
                True,
            )
        )
    for lr, la in ((8.0, 6.0), (12.0, 6.0), (24.0, 6.0), (12.0, 4.0), (20.0, 8.0)):
        items.append(
            (
                ChaparroMieModel(*BASE, repulsive_exponent=lr, attractive_exponent=la),
                {"kind": "Mie_Chaparro2023", "model": {"lambda_r": lr, "lambda_a": la}},
                True,
            )
        )
    assert len(items) == 50
    return items


def targets(reference, native, tstar, rstar):
    a = {(i, j): reference.get_Arxy(i, j, tstar, rstar, np.ones(1)) for i, j in ORDERS}
    t, rho, R = tstar * 120, rstar / SCALE, native.gas_constant_J_mol_K
    z = 1 + a[0, 1]
    stiffness, thermal = 1 + 2 * a[0, 1] + a[0, 2], z - a[1, 1]
    return dict(
        zip(
            NAMES,
            (
                rho * R * t * z,
                z,
                R * t * a[0, 0],
                R * t * a[1, 0],
                R * t * (a[1, 0] + z - 1),
                R * (a[1, 0] - a[0, 0]),
                -R * a[2, 0],
                R * (-a[2, 0] + thermal**2 / stiffness - 1),
                rho * R * thermal,
                R * t * stiffness,
                R * t * (a[0, 0] + a[0, 1]),
                a[0, 0] + a[0, 1] - math.log(z),
            ),
            strict=True,
        )
    )


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    cases = declarations()
    grid = [
        ("derivative", c, t, r)
        for c, t, r in product(range(50), (1.2, 2.0, 4.0), (1e-6, 0.05, 0.3, 0.6))
    ]
    grid += [("state", c, 4.0, r) for c, r in product(range(50), (0.05, 0.3))]
    grid += [("inverse", c, 4.0, 0.05) for c in range(50)]
    grid += [("vacuum", c, 2.0, 0.0) for c in range(50)]
    grid += [("vacuum_derivative", c, 2.0, 0.0) for c, (_, _, ok) in enumerate(cases) if ok]
    assert len(grid) == 826
    failures, metadata, cache = [], [], {}
    count, passed = 0, 0
    started = time.perf_counter()

    def compare(quantity, actual, reference):
        nonlocal count, passed
        atol = (
            2e-9
            if quantity.startswith("Ar") or quantity in ("compressibility_factor", "lnphi")
            else 2e-5
            if quantity.endswith("_J_mol")
            else 1e-4
            if quantity == "pressure_Pa"
            else 2e-7
        )
        row = {"quantity": quantity, "rtol": 2e-7, "atol": atol}
        for label, fn in (("native", actual), ("reference", reference)):
            try:
                value = float(fn())
                if not math.isfinite(value):
                    raise ValueError("nonfinite result")
                row[label] = value
            except Exception as exc:
                row[label + "_exception"] = f"{type(exc).__name__}: {exc}"
        row["passed"] = (
            "native" in row
            and "reference" in row
            and math.isclose(row["native"], row["reference"], rel_tol=2e-7, abs_tol=atol)
        )
        count += 1
        passed += row["passed"]
        return row

    with (directory / "records.jsonl").open("w", encoding="utf8") as stream:
        for unit, (kind, case, tstar, rstar) in enumerate(grid, 1):
            native, spec, _ = cases[case]
            if case not in cache:
                try:
                    cache[case] = teqp.make_model(spec, validate=False)
                except Exception as exc:
                    cache[case] = exc
                metadata.append({"case": case, "native": native.to_dict(), "reference": spec})
            ref = cache[case]
            t, rho = tstar * 120, rstar / SCALE
            records = []
            if kind in ("derivative", "vacuum_derivative", "vacuum"):
                orders = [(0, 0)] if kind == "vacuum" else ORDERS
                for i, j in orders:
                    records.append(
                        compare(
                            f"Ar{i}{j}",
                            lambda: molecular_scaled_derivative(
                                native, t, rho, (1.0,), inverse_temperature_order=i, density_order=j
                            ),
                            lambda: ref.get_Arxy(i, j, tstar, rstar, np.ones(1)),
                        )
                    )
            else:
                values, expected, errors = {}, {}, {}
                try:
                    expected = targets(ref, native, tstar, rstar)
                except Exception as exc:
                    errors["reference"] = exc
                try:
                    if kind == "inverse":
                        root = molecular_density_from_pressure(
                            native,
                            t,
                            expected["pressure_Pa"],
                            (1.0,),
                            initial_density_mol_m3=rho * 1.1,
                            density_bounds_mol_m3=(rho * 0.8, rho * 1.2),
                        )
                        values["density_mol_m3"] = root.molar_density_mol_m3
                    else:
                        state = molecular_eos_state(native, t, rho, (1.0,), require_stable=False)
                        values.update(state.to_dict())
                        values["mu_J_mol"] = molecular_residual_chemical_potentials(
                            native, t, rho, (1.0,)
                        )[0]
                        values["lnphi"] = state.log_fugacity_coefficients[0]
                except Exception as exc:
                    errors["native"] = exc

                def lookup(role, quantity):
                    if role in errors:
                        raise errors[role]
                    return (values if role == "native" else expected)[quantity]

                if kind == "inverse":
                    records.append(
                        compare(
                            "density_mol_m3",
                            lambda: lookup("native", "density_mol_m3"),
                            lambda: rho,
                        )
                    )
                else:
                    records.extend(
                        compare(
                            name, lambda: lookup("native", name), lambda: lookup("reference", name)
                        )
                        for name in NAMES
                    )
            for record in records:
                row = {
                    "unit": unit,
                    "case": case,
                    "block": kind,
                    "Tstar": tstar,
                    "rhostar": rstar,
                    **record,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if not row["passed"]:
                    failures.append(row)
            if unit % 10 == 0 or unit == len(grid):
                elapsed = time.perf_counter() - started
                print(
                    f"potential extensions {unit}/{len(grid)}; {unit / elapsed:.2f} units/s; "
                    f"ETA {(len(grid) - unit) * elapsed / unit:.1f}s; checks {passed}/{count}",
                    flush=True,
                )
    assert count == 7560
    summary = {
        "mode": "development",
        "backend": "teqp",
        "version": teqp.__version__,
        "units": len(grid),
        "comparisons": count,
        "passed": passed,
        "failed": len(failures),
        "elapsed_s": time.perf_counter() - started,
        "declarations": metadata,
        "failures": failures,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("declarations", "failures")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
