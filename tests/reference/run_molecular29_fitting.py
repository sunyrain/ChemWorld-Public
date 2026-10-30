"""Frozen F: six native influence fits and sixty actual Rolo measurements.

Approximation: experiment contains saturated AIR; this model omits noncondensable
air and computes hydrocarbon vapor/liquid SGT. No same-boundary validation claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_interface import (
    MolecularInterfaceResult,
    PolynomialSquareGradientInfluence,
)
from chemworld.physchem.molecular_interface_fitting import (
    InfluenceFitParameter,
    MolecularInterfaceFitCase,
    SurfaceTensionObservation,
    fit_molecular_influence,
)

if __package__:
    from .molecular29_fitting_reference import CONSTANTS, DATA
else:
    from molecular29_fitting_reference import CONSTANTS, DATA


def density_bounds(endpoint_densities, covolumes, domain="endpoint-1.5"):
    """A numerical box; the PR joint packing domain still applies inside it."""
    if domain not in {"endpoint-1.5", "pr-covolume"}:
        raise ValueError("unknown fitting density domain")
    density = np.asarray(endpoint_densities)
    return tuple((float(.01*min(density[:, i])),
                  float(1.5*max(density[:, i])) if domain == "endpoint-1.5" else
                  float(np.nextafter(1/b, 0.))) for i, b in enumerate(covolumes))


def native_case(row, endpoint, density_domain="endpoint-1.5"):
    ids = tuple(row["component_ids"])
    r = endpoint["R_J_mol_K"]
    components = []
    for i, name in enumerate(ids):
        data = CONSTANTS[name]
        components.append(CPAComponent(name,
            endpoint["oma"]*r*r*data["Tc_K"]**2/data["Pc_Pa"],
            endpoint["omb"]*r*data["Tc_K"]/data["Pc_Pa"], data["Tc_K"],
            endpoint["kappa"][i], "HEOS constants; explicit PhasePy PR formula"))
    model = CPAModel(tuple(components), tuple(tuple(0. for _ in ids) for _ in ids),
        (290., 340.), "unshifted PR same-source saturation; noncondensable air omitted",
        cubic_model="peng_robinson", gas_constant_J_mol_K=r)
    states = [molecular_eos_state(model, row["temperature_K"], endpoint[f"{phase}_density_mol_m3"],
        tuple(endpoint[f"{phase}_mole_fractions"])) for phase in ("vapor", "liquid")]
    if len(ids) == 1:
        lo, hi = (state.molar_density_mol_m3 for state in states)
        return MolecularInterfaceFitCase(model, *states, interior_density_grid=(lo+.5*(hi-lo),))
    density = np.array([np.array(state.mole_fractions)*state.molar_density_mol_m3
                        for state in states])
    bounds = density_bounds(density, [c.covolume_m3_mol for c in components], density_domain)
    return MolecularInterfaceFitCase(model, *states, mesh=DiffusionMesh.planar(12e-9, 256),
                                    component_density_bounds=bounds)


def interpolated_profile(result, mesh, *, shift=0.):
    source_z = np.array(result.positions_m)+shift
    source_rho = np.array(result.component_densities_mol_m3)
    return tuple(tuple(float(v) for v in row) for row in np.column_stack([
        np.interp(mesh.faces_m, source_z, source_rho[:, i]) for i in range(source_rho.shape[1])]))


def evaluated(case, observation, influence):
    try:
        return case.evaluate(observation, influence)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return MolecularInterfaceResult(False, failure_reason=f"{type(exc).__name__}: {exc}")


def domain_record(observation_id, case, result=None):
    covolumes = [component.covolume_m3_mol for component in case.model.components]
    density = np.array([np.array(state.mole_fractions)*state.molar_density_mol_m3
                       for state in (case.left_state, case.right_state)])
    original = density_bounds(density, covolumes)
    record = {"observation_id": observation_id, "component_ids": case.model.component_ids,
              "component_covolumes_m3_mol": covolumes,
              "component_density_bounds_mol_m3": case.component_density_bounds,
              "original_endpoint_bounds_mol_m3": original if case.mesh is not None else None,
              "coexistence_component_densities_mol_m3": density.tolist(),
              "method": "scalar_first_integral" if case.mesh is None else "planar_free_density"}
    if result is not None and result.success and case.mesh is not None:
        profile = np.array(result.component_densities_mol_m3)
        lo, hi = np.min(profile, axis=0), np.max(profile, axis=0)
        bounds = np.array(case.component_density_bounds)
        record["profile_domain_diagnostics"] = {
            "minimum_densities_mol_m3": lo.tolist(), "maximum_densities_mol_m3": hi.tolist(),
            "lower_slack_mol_m3": (lo-bounds[:, 0]).tolist(),
            "upper_slack_mol_m3": (bounds[:, 1]-hi).tolist(),
            "maximum_to_original_upper": (hi/np.array(original)[:, 1]).tolist(),
            "maximum_to_actual_upper": (hi/bounds[:, 1]).tolist(),
            "maximum_joint_packing": float(np.max(profile@covolumes)),
            "role": "diagnostic only; no additional acceptance gate"}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--density-domain", choices=("endpoint-1.5", "pr-covolume"),
                        default="endpoint-1.5")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fitting output is create-only")
    args.output.mkdir(parents=True)
    dataset = json.loads(DATA.read_text())
    source = json.loads(args.source.read_text())
    endpoints = {row["observation_id"]: row for row in source["units"]}
    started = monotonic()
    status = {"stage": "initialization", "fits": 0, "forward": 0, "cached": 0,
              "refinements": 0, "last_seconds": None}
    stop = Event()

    def heartbeat():
        while not stop.wait(30):
            elapsed = monotonic()-started
            print(f"F live {json.dumps(status)} elapsed={elapsed:.1f}s "
                  f"rate={status['forward']/max(elapsed,1e-9):.3f}native/s", flush=True)

    Thread(target=heartbeat, daemon=True).start()
    fits, checks, failures = [], [], []
    pure_parameters = {}
    groups = list(dict.fromkeys(tuple(row["component_ids"]) for row in dataset["observations"]))
    raw = (args.output/"forward.jsonl").open("x", encoding="utf-8")
    starts = (args.output/"forward-starts.jsonl").open("x", encoding="utf-8")

    def write_forward(stage, observation, influence, result, duration):
        status["forward"] += 1
        status["last_seconds"] = duration
        raw.write(json.dumps({"stage": stage, "observation_id": observation.observation_id,
            "density_domain": args.density_domain,
            "influence": influence.to_dict(), "duration_s": duration,
            "result": result.to_dict()}, allow_nan=False)+"\n")
        raw.flush()

    def evaluate_logged(stage, case, observation, influence):
        initial = monotonic()
        status["in_flight"] = {"stage": stage, "observation_id": observation.observation_id}
        starts.write(json.dumps({**status["in_flight"], "started_elapsed_s": initial-started,
            "density_domain": args.density_domain,
            "domain": domain_record(observation.observation_id, case),
            "influence": influence.to_dict()}, allow_nan=False)+"\n")
        starts.flush()
        result = evaluated(case, observation, influence)
        write_forward(stage, observation, influence, result, monotonic()-initial)
        status["in_flight"] = None
        return result

    for group_index, ids in enumerate(groups, 1):
        status["stage"] = "+".join(ids)
        rows = [row for row in dataset["observations"] if tuple(row["component_ids"]) == ids]
        observations = [SurfaceTensionObservation(
            row["observation_id"], row["temperature_K"], row["surface_tension_N_m"],
            row["standard_uncertainty_N_m"], row["partition"],
            dataset["doi"]+" Table"+str(row["source_table"])+" measured this-work value")
            for row in rows]
        try:
            cases = {row["observation_id"]: native_case(
                        row, endpoints[row["observation_id"]], args.density_domain)
                     for row in rows}
            if not all(endpoints[row["observation_id"]]["success"] for row in rows):
                raise ValueError("independent source endpoint qualification failed")
            pure = len(ids) == 1
            coefficients = ((1e-19, 0.),) if pure else tuple(pure_parameters[name] for name in ids)
            template = PolynomialSquareGradientInfluence(ids, coefficients,
                ((0.,),) if pure else ((0., .1), (.1, 0.)), (290., 340.),
                "Rolo2002 approximate PR calibration; saturated-air effects omitted", 298.15, 100.)
            parameters = ((InfluenceFitParameter("c0", "pure_coefficient", (0, 0),
                            1e-22, 1e-17, 1e-19),
                           InfluenceFitParameter("c1", "pure_coefficient", (0, 1),
                            -2e-19, 2e-19, 1e-19)) if pure else
                          (InfluenceFitParameter("k01", "interaction", (0, 1), .001, 1.5, 1.),))
            last_profiles, cache, pure_base = {}, {}, {}

            def predictor(observation, influence, cases=cases, pure=pure, pure_base=pure_base,
                          cache=cache, last_profiles=last_profiles):
                case = cases[observation.observation_id]
                if pure:
                    if observation.observation_id not in pure_base:
                        constant = replace(influence, pure_coefficients_J_m5_mol2=((1e-19,),))
                        result = evaluate_logged("pure_base", case, observation, constant)
                        pure_base[observation.observation_id] = result
                    result = pure_base[observation.observation_id]
                    if not result.success:
                        return result
                    factor = np.sqrt(influence.matrix_at(observation.temperature_K)[0][0]/1e-19)
                    return replace(result, surface_tension_N_m=result.surface_tension_N_m*factor,
                                   positions_m=tuple(z*factor for z in result.positions_m))
                key = (observation.observation_id, json.dumps(influence.to_dict(), sort_keys=True))
                if key in cache:
                    status["cached"] += 1
                    return cache[key]
                previous = last_profiles.get(observation.observation_id)
                if previous is not None:
                    case = replace(case, initial_profile=previous.component_densities_mol_m3)
                result = evaluate_logged("binary_fit", case, observation, influence)
                cache[key] = result
                if result.success:
                    last_profiles[observation.observation_id] = result
                return result

            fit = fit_molecular_influence(template, parameters, observations, predictor,
                maximum_evaluations=30 if pure else 16, difference_step=.005,
                optimality_tolerance=1e-5,
                progress=lambda p, j=group_index, ids=ids: print(
                    f"F fit {j}/6 {ids} {p}", flush=True))
            result_dict = fit.to_dict()
            result_dict["component_ids"] = ids
            result_dict["density_domain"] = args.density_domain
            result_dict["numerical_domains"] = [domain_record(o.observation_id,
                cases[o.observation_id], None if pure or fit.influence is None else cache.get(
                    (o.observation_id, json.dumps(fit.influence.to_dict(), sort_keys=True))))
                for o in observations]
            fits.append(result_dict)
            (args.output/f"fit-{group_index}.json").write_text(
                json.dumps(result_dict, indent=2, allow_nan=False)+"\n")
            status["fits"] += 1
            if fit.influence is None:
                raise ValueError("fit did not produce a valid influence declaration")
            if pure:
                pure_parameters[ids[0]] = fit.influence.pure_coefficients_J_m5_mol2[0]
            for row, observation in zip(rows, observations, strict=True):
                base = predictor(observation, fit.influence)
                if pure:
                    direct = evaluate_logged("pure_direct_validation",
                        cases[observation.observation_id], observation, fit.influence)
                    error = None if not direct.success or not base.success else float(abs(
                        direct.surface_tension_N_m/base.surface_tension_N_m-1))
                    checks.append({"kind": "pure_scaling", "id": observation.observation_id,
                                   "relative_change": error, "pass": bool(error is not None
                                   and error <= 1e-8)})
                    continue
                for length, cells, kind in [(12e-9, 512, "grid")]+(
                    [(24e-9, 1024, "length")] if row["temperature_K"] == 313.15
                    and row["liquid_mole_fractions"][1] in (.5, .501) else []):
                    mesh = DiffusionMesh.planar(length, cells)
                    case = replace(cases[observation.observation_id], mesh=mesh,
                        initial_profile=interpolated_profile(base, mesh,
                            shift=6e-9 if kind == "length" else 0.) if base.success else None)
                    refined = evaluate_logged("binary_"+kind, case, observation, fit.influence)
                    status["refinements"] += 1
                    error = None if not refined.success or not base.success else float(abs(
                        refined.surface_tension_N_m/base.surface_tension_N_m-1))
                    checks.append({"kind": kind, "id": observation.observation_id,
                                   "relative_change": error, "pass": bool(error is not None
                                   and error <= .02), "failure_reason": refined.failure_reason})
                    base = refined
        except Exception as exc:
            failures.append({"component_ids": ids, "exception": f"{type(exc).__name__}: {exc}",
                             "affected_observation_ids": [row["observation_id"] for row in rows]})
            print(f"F group failure {ids}: {exc}", flush=True)
        print(f"F completed group {group_index}/6 elapsed={monotonic()-started:.1f}s", flush=True)
    raw.close()
    starts.close()
    stop.set()
    applicability = []
    for fit in fits:
        metric = fit["diagnostics"].get("metrics", {}).get("holdout", {})
        applicability.append({"component_ids": fit["component_ids"],
            "pass": bool(metric.get("complete") and
                         metric["mean_absolute_relative_error"] <= .05 and
                         metric["maximum_absolute_relative_error"] <= .1), "metrics": metric})
    expected = {"pure_scaling": 15, "grid": 45, "length": 3}
    summary = {"scope": "development measured-data fitting; saturated AIR omitted in PR-SGT",
        "density_domain": args.density_domain,
        "domain_constraint": "Per-component box only; original EOS joint packing remains required",
        "source_doi": dataset["doi"], "boundary_caveat": dataset["model_boundary_approximation"],
        "expected_observations": 60, "expected_training": 36, "expected_holdout": 24,
        "expected_fits": 6, "fits_completed": len(fits),
        "algorithmically_successful_fits": sum(f["success"] for f in fits),
        "predictions_recorded": sum(len(f["observations"]) for f in fits),
        "fits": fits, "applicability": applicability, "checks": checks, "failures": failures,
        "numerical_denominators": {kind: {"expected": number,
            "attempted": sum(c["kind"] == kind for c in checks),
            "passed": sum(c["kind"] == kind and c["pass"] for c in checks)}
            for kind, number in expected.items()}, "status": status,
        "elapsed_s": monotonic()-started}
    summary["success"] = bool(not failures and len(fits) == 6 and all(f["success"] for f in fits)
        and all(a["pass"] for a in applicability)
        and all(v["attempted"] == v["expected"] == v["passed"]
                for v in summary["numerical_denominators"].values()))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"F finished success={summary['success']} elapsed={summary['elapsed_s']:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
