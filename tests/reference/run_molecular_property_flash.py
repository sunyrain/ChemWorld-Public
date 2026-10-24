"""Fixed teqp residual / independent gamma-phi consumer comparison."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from run_saftvrmie import teqp_targets
from scipy.optimize import brentq

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.flash_units import ph_flash, tp_flash
from chemworld.physchem.molecular_properties import MolecularEOSSpec
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.property_packages import evaluate_property_state


def models():
    cs = (
        SAFTSegment("light", 1, 3.4e-10, 120, "explicit synthetic"),
        SAFTSegment("heavy", 2, 3.8e-10, 180, "explicit synthetic"),
    )
    residual = PCSAFTModel(cs, ((0.0, 0.02), (0.02, 0.0)), (100, 1000), "explicit synthetic")
    base = synthetic_vle_properties()
    native = replace(
        base,
        vapor_eos=MolecularEOSSpec("pcsaft-gas", residual, "vapor", (1e-8, 3000)),
        gas_references={k: replace(v, entropy_J_mol_K=4) for k, v in base.gas_references.items()},
    )
    reference = teqp.make_model(
        {
            "kind": "PCSAFT",
            "model": {
                "coeffs": [
                    {
                        "name": c.component_id,
                        "m": c.segment_count,
                        "sigma_Angstrom": c.diameter_m * 1e10,
                        "epsilon_over_k": c.well_depth_K,
                        "BibTeXKey": "synthetic",
                    }
                    for c in cs
                ],
                "kmat": [[0, 0.02], [0.02, 0]],
            },
        }
    )
    return native, reference


def gas(ref, t, p, x):
    x = np.asarray(x, dtype=float)
    rho = brentq(lambda d: d * R * t * (1 + ref.get_Ar01(t, d, x)) - p, 1e-8, 3000, xtol=1e-11)
    q = teqp_targets(ref, t, rho, x)
    z = q["compressibility_factor"]
    lnphi = np.asarray(ref.build_Psir_gradient_autodiff(t, rho * x)) / (R * t) - math.log(z)
    cp0 = float(x @ np.array([80.0, 100.0]))
    h0 = np.array([80.0, 100.0]) * (t - 298.15)
    s0 = 4 + np.array([80.0, 100.0]) * math.log(t / 298.15) - R * math.log(p / 101325)
    h = float(x @ h0) + q["residual_enthalpy_J_mol"]
    entropy = (
        float(x @ s0) - R * float(x @ np.log(x)) + q["residual_entropy_J_mol_K"] + R * math.log(z)
    )
    mu = h0 - t * s0 + R * t * (np.log(x) + lnphi)
    return {
        "enthalpy": h,
        "internal_energy": h - p / rho,
        "entropy": entropy,
        "cp": cp0 + q["residual_cp_J_mol_K"],
        "cv": cp0 - R + q["residual_cv_J_mol_K"],
        "molar_volume": 1 / rho,
        "dV_dT_P": q["dP_dT_rho_Pa_K"] / (rho * rho * q["dP_drho_T_Pa_m3_mol"]),
        "dV_dP_T": -1 / (rho * rho * q["dP_drho_T_Pa_m3_mol"]),
        "phi0": math.exp(lnphi[0]),
        "phi1": math.exp(lnphi[1]),
        "mu0": float(mu[0]),
        "mu1": float(mu[1]),
    }


def reference_flash(ref, t, p, fraction):
    z = np.array([fraction, 1 - fraction])
    psat = 1e5 * np.exp(np.array([30000.0, 40000.0]) / R * (1 / np.array([330.0, 370.0]) - 1 / t))
    y = z.copy()
    for _ in range(500):
        props = gas(ref, t, p, y)
        k = psat / p / np.array([props["phi0"], props["phi1"]])

        def rr(beta, k=k):
            return float(np.sum(z * (k - 1) / (1 + beta * (k - 1))))

        beta = 0.0 if rr(0) <= 0 else 1.0 if rr(1) >= 0 else brentq(rr, 0, 1, xtol=1e-14)
        x = z / (1 + beta * (k - 1))
        ynext = k * x
        x /= sum(x)
        ynext /= sum(ynext)
        if max(abs(ynext - y)) < 2e-13:
            y = ynext
            break
        y = 0.5 * (y + ynext)
    else:
        raise RuntimeError("independent flash iteration failed")
    nl, nv = (1 - beta) * x, beta * y
    liquid_h = float(x @ (np.array([80.0, 100.0]) * (t - 298.15) - np.array([30000.0, 40000.0])))
    h = (1 - beta) * liquid_h + beta * gas(ref, t, p, y)["enthalpy"]
    return {"beta": beta, "nL0": nl[0], "nL1": nl[1], "nV0": nv[0], "nV1": nv[1], "H": h}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    native, ref = models()
    records, failures, units = [], [], 0

    def compare(unit, actual, expected):
        keys = tuple(expected)
        for key in keys:
            atol = (
                2e-4
                if key in {"enthalpy", "internal_energy", "mu0", "mu1", "H", "T"}
                else 1e-12
                if key in {"molar_volume", "dV_dT_P", "dV_dP_T"}
                else 2e-6
                if key in {"entropy", "cp", "cv"}
                else 2e-7
            )
            row = {
                "unit": unit,
                "quantity": key,
                "reference": float(expected[key]),
                "rtol": 2e-6,
                "atol": atol,
            }
            if isinstance(actual, Exception):
                row.update(passed=False, error=repr(actual))
            else:
                value = float(actual[key])
                row.update(
                    native=value,
                    passed=math.isfinite(value)
                    and math.isclose(value, expected[key], rel_tol=2e-6, abs_tol=atol),
                )
            records.append(row)
            if not row["passed"]:
                failures.append(row)

    for fraction, t in product((0.2, 0.5, 0.8), (300.0, 350.0, 450.0)):
        feed = {"light": fraction, "heavy": 1 - fraction}
        try:
            report = evaluate_property_state(
                native.packages,
                feed,
                temperature_K=t,
                pressure_Pa=1e5,
                phase="gas",
                references=native.gas_references,
                eos=native.vapor_eos,
                quantities=(
                    "enthalpy",
                    "internal_energy",
                    "entropy",
                    "cp",
                    "cv",
                    "molar_volume",
                    "dV_dT_P",
                    "dV_dP_T",
                    "fugacity_coefficient",
                    "chemical_potential",
                ),
            )
            actual = {key: value.value for key, value in report.properties.items()}
            for i, key in enumerate(native.component_ids):
                actual[f"phi{i}"] = report.component_properties["fugacity_coefficient"][key].value
                actual[f"mu{i}"] = report.component_properties["chemical_potential"][key].value
        except Exception as exc:
            actual = exc
        compare(f"properties-{fraction}-{t}", actual, gas(ref, t, 1e5, [fraction, 1 - fraction]))
        units += 1
        target = reference_flash(ref, t, 1e5, fraction)
        try:
            result = tp_flash(native, feed, temperature_K=t, pressure_Pa=1e5)
            actual = {
                "beta": result.vapor_fraction,
                "H": result.total_enthalpy_j,
                **{f"nL{i}": result.liquid_amounts_mol[k] for i, k in enumerate(feed)},
                **{f"nV{i}": result.vapor_amounts_mol[k] for i, k in enumerate(feed)},
            }
        except Exception as exc:
            actual = exc
        compare(f"tp-{fraction}-{t}", actual, target)
        units += 1
        try:
            result = ph_flash(native, feed, pressure_Pa=1e5, enthalpy_J=target["H"])
            actual = {"T": result.temperature_K}
        except Exception as exc:
            actual = exc
        compare(f"ph-{fraction}-{t}", actual, {"T": t})
        units += 1
        elapsed = time.perf_counter() - started
        print(
            f"property/TP/PH {units}/27 units, {units / elapsed:.2f} units/s, "
            f"ETA {(27 - units) * elapsed / units:.1f}s",
            flush=True,
        )
    assert units == 27 and len(records) == 171
    summary = {
        "mode": "development",
        "units": units,
        "comparisons": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "model": native.to_dict(),
    }
    (output / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "model"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
