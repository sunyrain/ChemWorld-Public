"""Fixed22 native phase-ledger units, with independent binary teqp binodals."""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import teqp
from scipy.optimize import newton
from tests.reference.run_multifluid_catalog import NAMES, Progress, select, source_catalog
from tests.reference.run_multifluid_core import R

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
    equilibrate_molecular_phase_inventory,
    molecular_phase_stream,
)


def cases(*, local_feed_domain=False):
    values = []
    for names, t, pressures, gas, liquid in (
        (
            ("n-Propane", "n-Butane"),
            300.0,
            (3e5, 6e5, 9e5),
            (1.0, 3000.0, 500.0),
            (7000.0, 22000.0, 12000.0),
        ),
        (
            ("Methane", "Ethane"),
            200.0,
            (1e6, 2e6, 3e6),
            (1.0, 6000.0, 500.0),
            (12000.0, 30000.0, 20000.0),
        ),
    ):
        for pressure, z in itertools.product(pressures, (0.3, 0.5, 0.7)):
            values.append(
                {
                    "names": names,
                    "temperature": t,
                    "pressure": pressure,
                    "fractions": (z, 1 - z),
                    "gas": gas,
                    "liquid": liquid,
                }
            )
    for names, pressure in itertools.product((NAMES[:4], NAMES), (1e5, 1e6)):
        values.append(
            {
                "names": names,
                "temperature": 450.0,
                "pressure": pressure,
                "fractions": (1 / len(names),) * len(names),
                "gas": (1.0, 2000.0, 100.0),
            }
        )
    if local_feed_domain:
        values = values[:18]
        for case in values:
            lo, hi, original_seed = case["liquid"]
            case["reference_liquid_seed"] = original_seed
            case["liquid"] = (lo, hi, 10000.0 if case["names"][0] == "n-Propane" else 18000.0)
            case["composition_starts"] = (case["fractions"],)
    return values


def reference_binodal(case):
    model = teqp.build_multifluid_model(
        list(case["names"]), teqp.get_datapath(), flags={"Rmodel": "CODATA"}
    )
    options = teqp.MixVLETpFlags()
    options.maxiter = 120
    result = model.mix_VLE_Tp(
        case["temperature"],
        case["pressure"],
        np.array([0.2, 0.8]) * case.get("reference_liquid_seed", case["liquid"][2]),
        np.array([0.8, 0.2]) * case["gas"][2],
        options,
    )
    raw = {
        "success": bool(result.success),
        "message": result.message,
        "return_code": str(result.return_code),
        "iterations": int(result.num_iter),
        "rhovecL": result.rhovecL.tolist(),
        "rhovecV": result.rhovecV.tolist(),
        "residual": result.r.tolist(),
        "qualification": "explicit_positive_nontrivial_phases_and_original_P_mu_tolerances",
    }
    # teqp v0.23.2's Newton branch never updates success or return_code.
    # Preserve these flags above; qualify the returned endpoint by the fixed
    # independent equilibrium criteria, not that unpopulated status flag.
    rl, rv = np.asarray(result.rhovecL), np.asarray(result.rhovecV)
    if np.any(rl <= 0) or np.any(rv <= 0) or not np.all(np.isfinite(np.r_[rl, rv])):
        raw["failure_reason"] = "invalid_reference_phase_density"
        return model, raw, None
    rho_l, rho_v = float(rl.sum()), float(rv.sum())
    xl, xv = rl / rho_l, rv / rho_v
    t, p = case["temperature"], case["pressure"]
    actual_p = [
        rho * R * t * (1 + model.get_Arxy(0, 1, t, rho, x)) for rho, x in ((rho_l, xl), (rho_v, xv))
    ]
    mu = [
        np.log(x) + np.log(model.get_fugacity_coefficients(t, rho * x))
        for rho, x in ((rho_l, xl), (rho_v, xv))
    ]
    raw.update(
        pressure_relative_error=max(abs(value / p - 1) for value in actual_p),
        chemical_potential_error_RT=float(np.max(np.abs(mu[0] - mu[1]))),
    )
    if (
        rho_l <= rho_v
        or abs(xv[0] - xl[0]) <= 1e-6
        or not math.isfinite(raw["pressure_relative_error"])
        or not math.isfinite(raw["chemical_potential_error_RT"])
        or raw["pressure_relative_error"] > 2e-7
        or raw["chemical_potential_error_RT"] > 2e-7
    ):
        raw["failure_reason"] = "trivial_or_unqualified_reference_binodal"
        return model, raw, None
    beta = (case["fractions"][0] - float(xl[0])) / float(xv[0] - xl[0])
    expected = {
        "xL": float(xl[0]),
        "xV": float(xv[0]),
        "rhoL": rho_l,
        "rhoV": rho_v,
        "beta": beta,
        "phase_status": "two_phase" if 0 < beta < 1 else "single_phase",
        "single_basin": "liquid" if beta <= 0 else "vapor",
    }
    return model, raw, expected


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-feed-domain", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite earlier attempt")
    catalog = source_catalog()
    units, comparisons = [], []

    def compare(index, name, actual, reference, atol=2e-7, rtol=2e-5):
        a, b = float(actual), float(reference)
        row = {
            "unit": index,
            "quantity": name,
            "native": a,
            "reference": b,
            "atol": atol,
            "rtol": rtol,
            "passed": math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(b),
        }
        comparisons.append(row)

    planned_cases = cases(local_feed_domain=args.local_feed_domain)
    with Progress(len(planned_cases)) as progress:
        for index, case in enumerate(planned_cases):
            progress.current = (
                f"{index + 1}:{case['names']} T={case['temperature']} P={case['pressure']}"
            )
            unit = {
                "index": index,
                "case": case,
                "functional_checks": {},
                "source_sha256": catalog.declaration_sha256,
            }
            try:
                model = select(catalog, case["names"])
                basins = [MolecularDensityBasin("vapor", case["gas"][:2], case["gas"][2])]
                targets = [MolecularPhaseTarget("gas", "vapor", "gas", selected=True)]
                if "liquid" in case:
                    basins.append(
                        MolecularDensityBasin("liquid", case["liquid"][:2], case["liquid"][2])
                    )
                    targets.append(MolecularPhaseTarget("liquid", "liquid", "liquid"))
                spec = MolecularPhaseEquilibriumSpec(
                    model, tuple(basins), tuple(targets), model.caloric_model()
                )
                restored = MolecularPhaseEquilibriumSpec.from_dict(
                    json.loads(json.dumps(spec.to_dict()))
                )
                unit["functional_checks"]["source_spec_json_roundtrip"] = restored == spec
                stock = PhaseLedger(
                    {
                        "feed": PhaseRecord(
                            "feed",
                            "vessel",
                            "gas",
                            10.0,
                            dict(zip(model.component_ids, case["fractions"], strict=True)),
                        ),
                        "external": PhaseRecord(
                            "external", "other-vessel", "solid", 0.1, {"inert": 0.03}
                        ),
                    }
                )
                result = equilibrate_molecular_phase_inventory(
                    stock,
                    restored,
                    vessel_id="vessel",
                    source_phase_ids=("feed",),
                    temperature_K=case["temperature"],
                    pressure_Pa=case["pressure"],
                    composition_starts=case.get("composition_starts"),
                    max_iterations=120,
                    tpd_tolerance=1e-8,
                )
                unit["native"] = result.to_dict()
                unit["phase_resolved"] = result.success
                if not result.success:
                    unit["functional_checks"]["failure_exact_rollback"] = result.ledger is stock
                else:
                    before, after = stock.total_amounts_mol(), result.ledger.total_amounts_mol()
                    unit["functional_checks"].update(
                        material_balance=all(
                            math.isclose(before[k], after[k], rel_tol=5e-13, abs_tol=0)
                            for k in before
                        ),
                        untouched_ownership=result.ledger.phases["external"]
                        == stock.phases["external"],
                        phase_ledger_json_roundtrip=_phases(
                            json.loads(json.dumps(result.ledger.to_dict()))
                        )
                        == result.ledger,
                        phase_enthalpy_additivity=math.isclose(
                            result.total_enthalpy_J,
                            sum(result.phase_enthalpies_J.values()),
                            rel_tol=2e-10,
                            abs_tol=1e-9,
                        ),
                    )
                    for phase_id in result.installed_phase_ids:
                        phase = result.ledger.phases[phase_id]
                        stream = molecular_phase_stream(
                            result.ledger,
                            restored,
                            vessel_id="vessel",
                            phase_id=phase_id,
                            temperature_K=case["temperature"],
                            molar_rate_mol_s=0.02,
                        )
                        n = sum(phase.species_amounts_mol.values())
                        flux = stream.contribution({k: "port:" + k for k in model.component_ids})
                        unit["functional_checks"].update(
                            {
                                phase_id + "_live_composition": all(
                                    math.isclose(
                                        stream.state.mole_fractions[j],
                                        phase.species_amounts_mol[k] / n,
                                        rel_tol=2e-10,
                                        abs_tol=0,
                                    )
                                    for j, k in enumerate(model.component_ids)
                                ),
                                phase_id + "_pressure": math.isclose(
                                    stream.state.pressure_Pa, case["pressure"], rel_tol=2e-10
                                ),
                                phase_id + "_enthalpy_stream": math.isclose(
                                    stream.enthalpy_rate_W,
                                    0.02 * result.phase_enthalpies_J[phase_id] / n,
                                    rel_tol=2e-10,
                                ),
                                phase_id + "_material_flux": all(
                                    math.isclose(
                                        flux.port_rates_mol_s["port:" + k],
                                        0.02 * phase.species_amounts_mol[k] / n,
                                        rel_tol=2e-10,
                                        abs_tol=0,
                                    )
                                    for k in model.component_ids
                                ),
                            }
                        )
                if "liquid" in case:
                    ref, raw, expected = reference_binodal(case)
                    unit["reference_binodal"] = raw
                    unit["reference_expected"] = expected
                    if expected is not None and result.success:
                        compare(
                            index,
                            "phase_classification",
                            result.flash.phase_status == expected["phase_status"],
                            True,
                            0,
                            0,
                        )
                        if result.flash.phase_status == expected["phase_status"] == "two_phase":
                            liquid, vapor = sorted(
                                result.flash.phases, key=lambda p: -p.state.molar_density_mol_m3
                            )
                            for key, actual in (
                                ("xL", liquid.state.mole_fractions[0]),
                                ("xV", vapor.state.mole_fractions[0]),
                                ("rhoL", liquid.state.molar_density_mol_m3),
                                ("rhoV", vapor.state.molar_density_mol_m3),
                                ("beta", vapor.amount_mol / (liquid.amount_mol + vapor.amount_mol)),
                            ):
                                compare(index, key, actual, expected[key])
                        elif (
                            result.flash.phase_status == expected["phase_status"] == "single_phase"
                        ):
                            xs = np.array(case["fractions"])
                            t = case["temperature"]
                            p = case["pressure"]
                            basin = (
                                case["liquid"]
                                if expected["single_basin"] == "liquid"
                                else case["gas"]
                            )

                            def pressure_root(q, t=t, ref=ref, xs=xs, p=p):
                                r = math.exp(q)
                                return (r * R * t * (1 + ref.get_Arxy(0, 1, t, r, xs)) - p) / p

                            def pressure_slope(q, t=t, ref=ref, xs=xs, p=p):
                                r = math.exp(q)
                                return (
                                    r
                                    * R
                                    * t
                                    * (
                                        1
                                        + 2 * ref.get_Arxy(0, 1, t, r, xs)
                                        + ref.get_Arxy(0, 2, t, r, xs)
                                    )
                                    / p
                                )

                            log_rho, root = newton(
                                pressure_root,
                                math.log(basin[2]),
                                fprime=pressure_slope,
                                tol=1e-12,
                                maxiter=120,
                                full_output=True,
                            )
                            rho = math.exp(log_rho)
                            unit["single_reference_solver"] = {
                                "method": "external_EOS_analytic_Newton_in_log_density",
                                "initial_density": basin[2],
                                "bounds": basin[:2],
                                "converged": bool(root.converged),
                                "iterations": int(root.iterations),
                                "density": rho,
                                "pressure_relative_error": abs(pressure_root(log_rho)),
                                "dP_drho": pressure_slope(log_rho) * p / rho,
                            }
                            if (
                                not root.converged
                                or not basin[0] <= rho <= basin[1]
                                or pressure_slope(log_rho) <= 0
                                or abs(pressure_root(log_rho)) > 2e-7
                            ):
                                raise ValueError("unqualified independent single-phase root")
                            actual = result.flash.phases[0].state
                            compare(index, "single_density", actual.molar_density_mol_m3, rho)
                            compare(
                                index,
                                "single_reference_pressure",
                                actual.pressure_Pa,
                                actual.molar_density_mol_m3
                                * R
                                * t
                                * (1 + ref.get_Arxy(0, 1, t, actual.molar_density_mol_m3, xs)),
                            )
                            mus = ref.build_Psir_gradient_autodiff(t, rho * xs)
                            for j, mu in enumerate(
                                model.residual_chemical_potentials(
                                    t, actual.molar_density_mol_m3, xs
                                )
                            ):
                                compare(index, f"single_mu{j}", mu, mus[j], atol=2e-5)
            except Exception as exc:
                unit["error"] = repr(exc)
            units.append(unit)
            progress.done += 1
            print(
                f"phase unit{index + 1}/{len(planned_cases)}: "
                f"resolved={unit.get('phase_resolved', False)}; "
                f"checks={len(unit['functional_checks'])}; error={unit.get('error', '')}",
                flush=True,
            )
    result = {
        "mode": "development",
        "catalog_id": catalog.catalog_id,
        "revision": catalog.revision,
        "catalog_sha256": catalog.declaration_sha256,
        "domain": "explicit_feed_TPD_starts" if args.local_feed_domain else "default_TPD_starts",
        "planned_units": len(planned_cases),
        "binary_reference_requests": 18,
        "executed_units": len(units),
        "resolved_units": sum(u.get("phase_resolved", False) for u in units),
        "failed_or_unresolved_units": [
            u["index"] for u in units if not u.get("phase_resolved", False)
        ],
        "reference_qualified_binodals": sum(u.get("reference_expected") is not None for u in units),
        "independent_comparisons": {
            "executed": len(comparisons),
            "passed": sum(c["passed"] for c in comparisons),
        },
        "functional_checks": {
            "executed": sum(len(u["functional_checks"]) for u in units),
            "passed": sum(sum(u["functional_checks"].values()) for u in units),
        },
        "errors": [{"unit": u["index"], "error": u["error"]} for u in units if "error" in u],
        "units": units,
        "comparisons": comparisons,
        "elapsed_s": time.monotonic() - progress.start,
    }
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("units", "comparisons")}, indent=2)
    )


if __name__ == "__main__":
    main()
