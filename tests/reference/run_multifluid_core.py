"""Fixed 27A reference block. Run in a temporary teqp 0.23.2 overlay."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import teqp

from chemworld.data.multifluid import load_multifluid_model
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_scaled_derivative,
)
from chemworld.physchem.multifluid import MultifluidModel

R = 8.31446261815324
FAMILIES = (
    ("Methane", "Ethane"),
    ("Methane", "Nitrogen"),
    ("Ethane", "Nitrogen"),
    ("Methane", "Ethane", "Nitrogen"),
)


def source_input(names):
    root = Path(teqp.get_datapath()) / "dev"
    fluids = [json.loads((root / "fluids" / (name + ".json")).read_text()) for name in names]
    cas = {f["INFO"]["CAS"]: f["INFO"]["NAME"] for f in fluids}
    pairs = [
        p
        for p in json.loads((root / "mixtures/mixture_binary_pairs.json").read_text())
        if p["CAS1"] in cas and p["CAS2"] in cas
    ]
    # Preserve exactly the public record; no coefficients are fitted or changed.
    selected = {p.get("function", "") for p in pairs}
    deps = [
        p
        for p in json.loads((root / "mixtures/mixture_departure_functions.json").read_text())
        if p["Name"] in selected
    ]
    return {
        "fluids": fluids,
        "binary_pairs": pairs,
        "departure_functions": deps,
        "gas_constant_J_mol_K": R,
        "gas_constant_convention": "common_constant_override",
    }


def compositions(n):
    return (
        ((0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
        if n == 2
        else ((0.2, 0.3, 0.5), (0.6, 0.3, 0.1), (0.01, 0.49, 0.5))
    )


def reference_values(ref, t, rho, xs):
    z = np.array(xs)
    derivatives = {
        (i, j): float(ref.get_Arxy(i, j, t, rho, z)) for i in range(4) for j in range(4 - i)
    }
    a, a01, a02 = (derivatives[k] for k in ((0, 0), (0, 1), (0, 2)))
    a10, a20, a11 = (derivatives[k] for k in ((1, 0), (2, 0), (1, 1)))
    stiffness = 1 + 2 * a01 + a02
    thermal = 1 + a01 - a11
    values = {
        "pressure_Pa": rho * R * t * (1 + a01),
        "compressibility_factor": 1 + a01,
        "residual_helmholtz_J_mol": R * t * a,
        "residual_internal_energy_J_mol": R * t * a10,
        "residual_enthalpy_J_mol": R * t * (a10 + a01),
        "residual_entropy_J_mol_K": R * (a10 - a),
        "residual_cv_J_mol_K": -R * a20,
        "residual_cp_J_mol_K": R * (-a20 + thermal**2 / stiffness - 1),
        "dP_dT_rho_Pa_K": rho * R * thermal,
        "dP_drho_T_Pa_m3_mol": R * t * stiffness,
    }
    mu = ref.build_Psir_gradient_autodiff(t, rho * z)
    lnphi = np.log(ref.get_fugacity_coefficients(t, rho * z))
    return derivatives, values, mu, lnphi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite an earlier attempt")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows, errors, declarations = [], [], []

    def compare(block, case, quantity, actual, expected, atol=2e-8, rtol=2e-7):
        a, b = float(actual), float(expected)
        rows.append(
            {
                "block": block,
                "case": case,
                "quantity": quantity,
                "native": a,
                "reference": b,
                "atol": atol,
                "rtol": rtol,
                "passed": math.isfinite(a)
                and math.isfinite(b)
                and abs(a - b) <= atol + rtol * abs(b),
            }
        )

    for names in FAMILIES:
        try:
            raw = source_input(names)
            native = load_multifluid_model(
                raw, source="teqp 0.23.2 bundled source; explicit CODATA R"
            )
            ref = teqp.build_multifluid_model(
                list(names), teqp.get_datapath(), flags={"Rmodel": "CODATA"}
            )
            assert abs(ref.get_R(np.array(compositions(len(names))[0])) - R) < 1e-14
            declarations.append(
                {
                    "components": names,
                    "input_sha256": hashlib.sha256(
                        json.dumps(raw, sort_keys=True).encode()
                    ).hexdigest(),
                    "source_R": [f.gas_constant_J_mol_K for f in native.fluids],
                    "used_R": R,
                    "convention": native.gas_constant_convention,
                }
            )
        except Exception as exc:
            errors.append({"block": "model", "case": names, "error": repr(exc)})
            continue
        for xindex, xs in enumerate(compositions(len(names))):
            for t, rho in itertools.product((250.0, 300.0, 400.0), (100.0, 700.0, 1800.0)):
                case = [names, xindex, t, rho]
                try:
                    d, state, mu, lnphi = reference_values(ref, t, rho, xs)
                    actual = native.state(t, rho, xs, require_stable=False)
                    for (i, j), target in d.items():
                        compare(
                            "grid",
                            case,
                            f"Ar{i}{j}",
                            molecular_scaled_derivative(
                                native, t, rho, xs, inverse_temperature_order=i, density_order=j
                            ),
                            target,
                        )
                    for field, target in state.items():
                        compare("grid", case, field, getattr(actual, field), target)
                    for i, value in enumerate(native.residual_chemical_potentials(t, rho, xs)):
                        compare("grid", case, f"mu{i}", value, mu[i], atol=2e-5)
                        compare(
                            "grid", case, f"lnphi{i}", actual.log_fugacity_coefficients[i], lnphi[i]
                        )
                    tr, rhor = native.reducing(xs)
                    compare("grid", case, "Tr", tr, ref.get_Tr(np.array(xs)))
                    compare("grid", case, "rhor", rhor, ref.get_rhor(np.array(xs)))
                except Exception as exc:
                    errors.append({"block": "grid", "case": case, "error": repr(exc)})
            case = [names, xindex]
            try:
                actual = np.array(native.residual_chemical_potential_jacobian(300.0, 700.0, xs))
                expected = ref.build_Psir_Hessian_autodiff(300.0, 700.0 * np.array(xs))
                for i, j in itertools.product(range(len(names)), repeat=2):
                    compare("hessian", case, f"H{i}{j}", actual[i, j], expected[i, j], atol=2e-7)
                base = native.state(300.0, 700.0, xs)
                reverse = replace(native, fluids=native.fluids[::-1]).state(300.0, 700.0, xs[::-1])
                roundtrip = MultifluidModel.from_dict(
                    json.loads(json.dumps(native.to_dict()))
                ).state(300.0, 700.0, xs)
                for variant, result in (("reversed", reverse), ("codec", roundtrip)):
                    for field in (
                        "pressure_Pa",
                        "residual_enthalpy_J_mol",
                        "residual_entropy_J_mol_K",
                    ):
                        compare(
                            "invariants",
                            case,
                            variant + field,
                            getattr(result, field),
                            getattr(base, field),
                        )
                root = molecular_density_from_pressure(
                    native,
                    300.0,
                    base.pressure_Pa,
                    xs,
                    initial_density_mol_m3=650.0,
                    density_bounds_mol_m3=(100.0, 1500.0),
                )
                compare(
                    "invariants",
                    case,
                    "density_inverse",
                    root.molar_density_mol_m3,
                    700.0,
                    rtol=2e-8,
                )
                # H^r plus ideal R*T/rho_i is dmu/drho; use dP/drho_i=rhovec.H,
                # then project a composition change to the fixed-P tangent.
                rhovec = 700.0 * np.array(xs)
                full_h = actual + np.diag(R * 300.0 / rhovec)
                gradient_p = rhovec @ full_h
                for i in range(len(xs) - 1):
                    dx = np.zeros(len(xs))
                    dx[i] = 1
                    dx[-1] = -1
                    drho = -700.0 * np.dot(gradient_p, dx) / np.dot(gradient_p, xs)
                    dmu = full_h @ (700.0 * dx + np.array(xs) * drho)
                    compare(
                        "invariants",
                        case,
                        f"gibbs_duhem{i}",
                        np.dot(xs, dmu) / (R * 300.0),
                        0.0,
                        atol=2e-7,
                    )
                caloric = native.caloric_state(300.0, 700.0, xs)
                compare(
                    "invariants",
                    case,
                    "G_equals_H_TS",
                    caloric.gibbs_energy_J_mol,
                    caloric.enthalpy_J_mol - 300.0 * caloric.entropy_J_mol_K,
                )
                compare(
                    "invariants",
                    case,
                    "H_equals_U_PV",
                    caloric.enthalpy_J_mol,
                    caloric.internal_energy_J_mol + base.pressure_Pa / 700.0,
                )
            except Exception as exc:
                errors.append({"block": "central", "case": case, "error": repr(exc)})
            print(
                f"multifluid {names} composition {xindex + 1}/3: {len(rows)} comparisons, "
                f"{len(errors)} errors, {time.monotonic() - start:.1f}s",
                flush=True,
            )
        limits = (
            ((1.0, 0.0), (0.0, 1.0))
            if len(names) == 2
            else (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.4, 0.6, 0.0),
                (0.4, 0.0, 0.6),
                (0.0, 0.4, 0.6),
            )
        )
        for i, xs in enumerate(limits):
            case = [names, i]
            try:
                d, state, mu, lnphi = reference_values(ref, 300.0, 700.0, xs)
                actual = native.state(300.0, 700.0, xs, require_stable=False)
                compare("limits", case, "alphar", native.alphar(300.0, 700.0, xs), d[0, 0])
                for field in ("pressure_Pa", "residual_enthalpy_J_mol", "residual_entropy_J_mol_K"):
                    compare("limits", case, field, getattr(actual, field), state[field])
                for j, value in enumerate(native.residual_chemical_potentials(300.0, 700.0, xs)):
                    compare("limits", case, f"mu{j}", value, mu[j], atol=2e-5)
                    compare(
                        "limits", case, f"lnphi{j}", actual.log_fugacity_coefficients[j], lnphi[j]
                    )
            except Exception as exc:
                errors.append({"block": "limits", "case": case, "error": repr(exc)})
    counts = {
        block: {
            "executed": sum(r["block"] == block for r in rows),
            "passed": sum(r["block"] == block and r["passed"] for r in rows),
        }
        for block in ("grid", "hessian", "limits", "invariants")
    }
    result = {
        "mode": "development",
        "reference": "teqp " + teqp.__version__,
        "fixed_expected": {"grid": 2862, "hessian": 63, "limits": 108},
        "counts": counts,
        "errors": errors,
        "elapsed_s": time.monotonic() - start,
        "declarations": declarations,
        "comparisons": rows,
    }
    result["complete_and_passed"] = (
        not errors
        and all(r["passed"] for r in rows)
        and all(counts[k]["executed"] == n for k, n in result["fixed_expected"].items())
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"comparisons", "declarations"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
