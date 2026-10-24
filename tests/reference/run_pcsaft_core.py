# ruff: noqa: B023 -- compare invokes both callbacks synchronously before advancing the loop.
"""Fixed 720-unit / 7470-comparison nonpolar PC-SAFT development block."""

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
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.pcsaft_coefficients import PCSAFT_DISPERSION_COEFFICIENTS

PARAMETERS = (("Methane", 1, 3.7039e-10, 150.03),
              ("Ethane", 1.6069, 3.5206e-10, 191.42),
              ("Propane", 2.002, 3.6184e-10, 208.11))
CASES = [((i,), (1,)) for i in range(3)] + [
    ((0, 1), x) for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))] + [
    ((0, 1, 2), x) for x in ((.2, .3, .5), (0, .4, .6), (.999, .0005, .0005))]
ORDERS = [(i, j) for i in range(4) for j in range(4-i)]


def models(case, coefficient_set):
    indices, fractions = CASES[case]
    components = tuple(SAFTSegment(*PARAMETERS[i], "Gross-Sadowski 2001") for i in indices)
    interactions = tuple(tuple(.02 if i != j else 0 for j in indices) for i in indices)
    native = PCSAFTModel(components, interactions, (100, 1000), "Fixed development block",
                        coefficient_set)
    spec = {"coeffs": [{"name": c.component_id, "m": c.segment_count,
                        "sigma_Angstrom": c.diameter_m*1e10, "epsilon_over_k": c.well_depth_K,
                        "BibTeXKey": "Gross-IECR-2001"} for c in components],
            "kmat": [list(r) for r in interactions]}
    if coefficient_set != "Gross-Sadowski-2001":
        spec["ab"] = coefficient_set.replace("Liang-", "Liang-IECR-")
    reference = teqp.make_model({"kind": "PCSAFT", "model": spec})
    return native, reference, np.asarray(fractions, dtype=float)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    grid = [("derivative", c, s, t, r) for c, s, t, r in product(
        range(10), PCSAFT_DISPERSION_COEFFICIENTS, (180, 300, 500),
        (0, 1e-6, 1, 100, 1000, 8000, 16000))]
    grid += [("state", c, s, 500, r) for c, s, r in product(
        range(10), PCSAFT_DISPERSION_COEFFICIENTS, (1, 1000, 8000))]
    expected = sum(10 if row[0] == "derivative" else 11+len(CASES[row[1]][1]) for row in grid)
    assert len(grid) == 720 and expected == 7470
    started = time.perf_counter()
    failures, comparisons, passed = [], 0, 0
    cache = {}

    def compare(name, native, reference, *, rtol=2e-8, atol=2e-10):
        nonlocal comparisons, passed
        item = {"quantity": name, "rtol": rtol, "atol": atol}
        for role, function in (("native", native), ("reference", reference)):
            try:
                value = float(function())
                if not math.isfinite(value):
                    raise ValueError("nonfinite comparison")
                item[role] = value
            except Exception as exc:
                item[role+"_exception"] = f"{type(exc).__name__}: {exc}"
        item["passed"] = ("native" in item and "reference" in item and math.isclose(
            item["native"], item["reference"], rel_tol=rtol, abs_tol=atol))
        comparisons += 1
        passed += item["passed"]
        return item

    with (output/"records.jsonl").open("w", encoding="utf-8") as stream:
        for index, (kind, case, coefficients, t, rho) in enumerate(grid, 1):
            key = (case, coefficients)
            if key not in cache:
                cache[key] = models(*key)
            native, reference, xs = cache[key]
            records = []
            if kind == "derivative":
                for i, j in ORDERS:
                    records.append(compare(
                        f"Ar{i}{j}", lambda i=i, j=j: molecular_scaled_derivative(
                            native, t, rho, xs, inverse_temperature_order=i, density_order=j),
                        lambda i=i, j=j: reference.get_Arxy(i, j, t, rho, xs)))
            else:
                try:
                    state = molecular_eos_state(native, t, rho, xs)
                    error = None
                except Exception as exc:
                    state, error = None, exc

                def state_value(name):
                    if error:
                        raise error
                    return getattr(state, name)

                ar = {(i, j): reference.get_Arxy(i, j, t, rho, xs)
                      for i, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))}
                R = reference.get_R(xs)
                z = 1+ar[0, 1]
                stiffness = 1+2*ar[0, 1]+ar[0, 2]
                thermal = z-ar[1, 1]
                targets = {
                    "pressure_Pa": rho*R*t*z, "compressibility_factor": z,
                    "residual_helmholtz_J_mol": R*t*ar[0, 0],
                    "residual_internal_energy_J_mol": R*t*ar[1, 0],
                    "residual_enthalpy_J_mol": R*t*(ar[1, 0]+z-1),
                    "residual_entropy_J_mol_K": R*(ar[1, 0]-ar[0, 0]),
                    "residual_cv_J_mol_K": -R*ar[2, 0],
                    "residual_cp_J_mol_K": R*(-ar[2, 0]+thermal**2/stiffness-1),
                    "dP_dT_rho_Pa_K": rho*R*thermal,
                    "dP_drho_T_Pa_m3_mol": R*t*stiffness,
                }
                for name, target in targets.items():
                    atol = 1e-5 if name == "pressure_Pa" else 1e-6 if "_J_mol" in name else 2e-10
                    records.append(compare(name, lambda name=name: state_value(name),
                                           lambda target=target: target, atol=atol))
                for i in range(len(xs)):
                    records.append(compare(
                        f"log_phi_{i}", lambda i=i: state_value("log_fugacity_coefficients")[i],
                        lambda i=i: math.log(reference.get_fugacity_coefficients(t, rho*xs)[i])))
                records.append(compare(
                    "pressure_density_inverse", lambda: molecular_density_from_pressure(
                        native, t, targets["pressure_Pa"], xs, initial_density_mol_m3=1.1*rho,
                        density_bounds_mol_m3=(rho/2, rho*1.5)).molar_density_mol_m3,
                    lambda: rho, rtol=2e-9))
            unit = {"unit": index, "kind": kind, "case": case, "coefficient_set": coefficients,
                    "T_K": t, "rho_mol_m3": rho, "comparisons": records}
            stream.write(json.dumps(unit)+"\n")
            failures.extend({**{k: v for k, v in unit.items() if k != "comparisons"}, **r}
                            for r in records if not r["passed"])
            if index % 20 == 0 or index == 720:
                elapsed = time.perf_counter()-started
                print(f"PCSAFT {index}/720, {index/elapsed:.2f} units/s, "
                      f"ETA {(720-index)*elapsed/index:.1f}s; failures {len(failures)}", flush=True)
    summary = {"mode": "development", "reference": "teqp "+teqp.__version__,
               "inputs": len(grid), "comparisons": comparisons, "expected_comparisons": expected,
               "passed": passed, "failed": len(failures), "failures": failures,
               "elapsed_seconds": time.perf_counter()-started}
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)
    assert comparisons == expected
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
