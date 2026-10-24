"""Fixed isolation of native and wheel NIST conformal maps, retaining all rows."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from run_corresponding_states import ecs_models
from scipy.optimize import root
from thermopack.cubic import cubic
from thermopack.extended_csp import ext_csp
from thermopack.multiparameter import multiparam

from chemworld.physchem.caloric_math import partial_derivative


def raw_state(reference, temperature, density):
    ar, at = reference.helmholtz_tv(temperature, 1 / density, [1.0], dadt=True, property_flag="R")
    (pressure,) = reference.pressure_tv(temperature, 1 / density, [1.0])
    gas = reference.Rgas
    return (
        ar / (gas * temperature),
        pressure / (gas * temperature * density),
        (at / (gas * temperature) - ar / (gas * temperature**2)),
    )


def native_reference(model, temperature, density):
    ref = model.reference

    def ar(t, rho):
        return ref.alpha(
            "residual", rho / ref.reducing_molar_density_mol_m3, ref.reducing_temperature_K / t
        )

    alpha = ar(temperature, density)
    derivatives = {
        name: partial_derivative(
            lambda v: ar(v["T"], v["rho"]), {"T": temperature, "rho": density}, (name,)
        )
        for name in ("T", "rho")
    }
    return alpha, 1 + density * derivatives["rho"], derivatives["T"]


def inferred_map(source, pure, temperature, initial):
    target = np.asarray(raw_state(source, temperature, 1000.0)[:2])

    def residual(log_fh):
        f, h = np.exp(log_fh)
        actual = raw_state(pure, temperature / f, 1000 * h)
        return np.asarray(actual[:2]) - target

    result = root(residual, np.log(initial), method="hybr", tol=1e-11)
    f, h = np.exp(result.x)
    error = float(np.max(np.abs(residual(result.x))))
    if error > 1e-9:
        raise ValueError(f"conformal inverse failed: {result.message}, residual={error}")
    return (float(f), float(h)), {
        "solver_success": bool(result.success),
        "residual": error,
        "evaluations": int(result.nfev),
    }


def alternatives(native, temperature, shape):
    declared = native.shape_factors(temperature, (1.0,))
    component, ref = native.components[0], native.reference_shape
    target_m = replace(
        native, reference_shape=replace(ref, alpha_coefficients=component.alpha_coefficients)
    )
    reference_m = replace(
        native, components=(replace(component, alpha_coefficients=ref.alpha_coefficients),)
    )
    db = cubic("C3", shape)
    db_t, _, db_p = db.get_critical_parameters(1)
    true_t, true_p = 369.89, 4251200.0
    database = replace(
        native,
        reference_shape=replace(
            ref,
            critical_temperature_K=db_t,
            attraction_Pa_m6_mol2=ref.attraction_Pa_m6_mol2 * (db_t / true_t) ** 2 * true_p / db_p,
            covolume_m3_mol=ref.covolume_m3_mol * db_t / true_t * true_p / db_p,
        ),
    )
    ratio = 8.314472 / native.gas_constant_J_mol_K
    fit_r = replace(
        native,
        reference_shape=replace(
            ref,
            attraction_Pa_m6_mol2=ref.attraction_Pa_m6_mol2 * ratio**2,
            covolume_m3_mol=ref.covolume_m3_mol * ratio,
        ),
    )
    return {
        "declared": declared,
        "target_m_in_reference": target_m.shape_factors(temperature, (1.0,)),
        "reference_m_in_target": reference_m.shape_factors(temperature, (1.0,)),
        "database_reference_critical": database.shape_factors(temperature, (1.0,)),
        "reference_fit_R": fit_r.shape_factors(temperature, (1.0,)),
    }


def run(output, source_dir, binding):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pure = multiparam("C3", "NIST_MEOS")
    records, reference_rows, failures = [], [], []
    metadata = []
    for unit, (shape, case, temperature) in enumerate(
        product(("SRK", "PR"), range(3), (200.0, 300.0, 400.0)), 1
    ):
        native, source, _ = ecs_models(case, shape, "NIST_MEOS", source_dir)
        if binding == "original_python_wrapper":
            source = ext_csp(native.component_ids[0], shape, "Classic", "vdW", "NIST_MEOS", "C3")
        declared = native.shape_factors(temperature, (1.0,))
        row = {
            "unit": unit,
            "shape": shape,
            "case": case,
            "T_K": temperature,
            "declared_map": declared,
            "source_R": source.Rgas,
            "pure_R": pure.Rgas,
            "native_R": native.gas_constant_J_mol_K,
            "alternatives": alternatives(native, temperature, shape),
        }
        for density in (1.0, 1000.0, 10000.0):
            t0, rho0 = temperature / declared[0], density * declared[1]
            actual, expected = native_reference(native, t0, rho0), raw_state(pure, t0, rho0)
            for i, name in enumerate(("alphar", "Z", "d_alphar_dT")):
                match = math.isclose(actual[i], expected[i], rel_tol=2e-7, abs_tol=2e-9)
                item = {
                    "unit": unit,
                    "shape": shape,
                    "case": case,
                    "T_K": temperature,
                    "density": density,
                    "conformal_T": t0,
                    "conformal_density": rho0,
                    "quantity": name,
                    "native": actual[i],
                    "source": expected[i],
                    "passed": match,
                }
                reference_rows.append(item)
                if not match:
                    failures.append(item)
        try:
            inferred, solve = inferred_map(source, pure, temperature, declared)
            row["inferred_map"], row["solve"] = inferred, solve
            row["relative_alternative_errors"] = {
                name: [a / b - 1 for a, b in zip(value, inferred, strict=True)]
                for name, value in row["alternatives"].items()
            }
            held = []
            for density in (1.0, 10000.0):
                source_values = raw_state(source, temperature, density)
                mapped_values = raw_state(pure, temperature / inferred[0], density * inferred[1])
                held.append(
                    {
                        "density": density,
                        "source_alphar_Z": source_values[:2],
                        "inferred_alphar_Z": mapped_values[:2],
                    }
                )
            row["held_out"] = held
            minus, minus_solve = inferred_map(source, pure, temperature - 0.01, declared)
            plus, plus_solve = inferred_map(source, pure, temperature + 0.01, declared)
            row["temperature_maps"] = {
                "step_K": 0.01,
                "minus": minus,
                "plus": plus,
                "minus_solve": minus_solve,
                "plus_solve": plus_solve,
                "centered_derivative": [(a - b) / 0.02 for a, b in zip(plus, minus, strict=True)],
                "native_derivative": [
                    partial_derivative(
                        lambda v, i=i, native=native: native.shape_factors(v["T"], (1.0,))[i],
                        {"T": temperature},
                        ("T",),
                    )
                    for i in (0, 1)
                ],
            }
        except Exception as exc:
            row["diagnostic_exception"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
        metadata.append(native.to_dict())
        (output / "progress.json").write_text(json.dumps({"completed": unit, "total": 18}))
        elapsed = time.perf_counter() - started
        print(
            f"NIST map {unit}/18; {unit / elapsed:.2f} units/s; "
            f"ETA {(18 - unit) * elapsed / unit:.1f}s; reference failures {len(failures)}",
            flush=True,
        )
    report = {
        "mode": "development",
        "reference_binding": binding,
        "units": 18,
        "reference_comparisons": 162,
        "reference_passed": len(reference_rows) - len(failures),
        "reference_failures": failures,
        "reference_records": reference_rows,
        "map_records": records,
        "model_inputs": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    assert len(reference_rows) == 162
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    for row in records:
        print(
            row["shape"],
            row["case"],
            row["T_K"],
            row.get("inferred_map"),
            row.get("relative_alternative_errors"),
            row.get("diagnostic_exception"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-binding",
        default="original_python_wrapper",
        choices=("original_python_wrapper", "corrected_fortran_order"),
    )
    args = parser.parse_args()
    run(args.output, args.source_dir, args.reference_binding)
