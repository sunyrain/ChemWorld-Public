"""Five fixed single-phase potential/relaxed-Cp units, with isolated upstream."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import traceback
from pathlib import Path

CASES = (
    {"name": "AlMg-fcc", "database": "Al-Mg_Zhong.tdb", "phase": "FCC_A1", "T": 1000,
     "target": {"AL": .3, "MG": .7}, "sites": [{"AL": .5, "MG": .5}, {"VA": 1}]},
    {"name": "AlMg-liquid", "database": "Al-Mg_Zhong.tdb", "phase": "LIQUID", "T": 1000,
     "target": {"AL": .4, "MG": .6}, "sites": [{"AL": .5, "MG": .5}]},
    {"name": "CrFe-bcc", "database": "crfe_bcc_magnetic.tdb", "phase": "BCC_A2", "T": 1200,
     "target": {"CR": .4, "FE": .6}, "sites": [{"CR": .5, "FE": .5}, {"VA": 1}]},
    {"name": "AlNi-ordered", "database": "alni_dupin_2001.tdb", "phase": "FCC_L12", "T": 1000,
     "target": {"AL": .25, "NI": .75},
     "sites": [{"AL": .15, "NI": .85}, {"AL": .55, "NI": .45}, {"VA": 1}]},
    {"name": "Al2Fe-underdetermined", "database": "Al-Fe_sundman2009.tdb", "phase": "AL2FE",
     "T": 1000, "target": {"AL": 2/3, "FE": 1/3}, "sites": [{"AL": 1}, {"FE": 1}],
     "expected_status": "underdetermined"},
)
PRESSURE = 101325.0


def write(path, report, *, initial=False):
    with path.open("x" if initial else "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")


def reference(directory, output):
    import numpy as np
    import pycalphad
    import pycalphad.variables as v

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("the reference must be pycalphad 0.11.2")
    report = {"complete": False, "expected_units": 5, "units": [],
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    write(output, report, initial=True)
    for case in CASES:
        path = directory / case["database"]
        unit = {"case": case, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            db = pycalphad.Database(str(path))
            components = sorted(case["target"])
            if "VA" in db.elements:
                components.append("VA")
            model = pycalphad.Model(db, components, case["phase"])
            if case.get("expected_status") == "underdetermined":
                substitutions = {v.T: case["T"], v.P: PRESSURE}
                substitutions.update({v.SiteFraction(case["phase"], i, s): value
                                      for i, subl in enumerate(case["sites"])
                                      for s, value in subl.items()})
                unit.update(G=float(model.GM.subs(substitutions)),
                            expected_matrix_rank=3, expected_unknowns=4)
            else:
                def solve(temperature, case=case, db=db, components=components):
                    first = sorted(case["target"])[0]
                    equilibrium = pycalphad.equilibrium(
                        db, components, [case["phase"]],
                        {v.T: temperature, v.P: PRESSURE, v.N: 1,
                         v.X(first): case["target"][first]},
                        output=["HM"], calc_opts={"pdens": 500},
                    )
                    phases = np.asarray(equilibrium.Phase).reshape(-1)
                    amounts = np.asarray(equilibrium.NP).reshape(-1)
                    active = np.flatnonzero((phases == case["phase"]) & (amounts > 1e-8))
                    if len(active) != 1 or abs(float(amounts[active[0]])-1) > 1e-8:
                        raise ValueError("upstream reference is not one positive single phase")
                    index = active[0]
                    result = {"T": temperature,
                              "G": float(np.asarray(equilibrium.GM).reshape(-1)[0]),
                              "H": float(np.asarray(equilibrium.HM).reshape(-1)[0]),
                              "mu": {e: float(np.asarray(
                                  equilibrium.MU.sel(component=e),
                              ).reshape(-1)[0])
                                     for e in case["target"]},
                              "composition": {
                                  e: float(np.asarray(
                                      equilibrium.X.sel(component=e),
                                  ).reshape(-1)[index])
                                  for e in case["target"]},
                              "Y": np.asarray(equilibrium.Y).reshape(
                                  -1, equilibrium.sizes["internal_dof"],
                              )[index].tolist()}
                    numbers = [result["G"], result["H"], *result["mu"].values(),
                               *result["composition"].values()]
                    if not all(math.isfinite(x) for x in numbers):
                        raise ValueError("upstream equilibrium returned nonfinite properties")
                    return result

                baseline = solve(case["T"])
                samples = {str(offset): solve(case["T"]+offset) for offset in (-.1, .1, -.05, .05)}
                coarse = (samples["0.1"]["H"]-samples["-0.1"]["H"])/.2
                fine = (samples["0.05"]["H"]-samples["-0.05"]["H"])/.1
                cp = (4*fine-coarse)/3
                if abs(fine-coarse) > 1e-3+1e-5*abs(cp):
                    raise ValueError("upstream relaxed-Cp difference steps do not agree")
                unit.update(baseline=baseline, temperature_samples=samples,
                            Cp=cp, Cp_coarse=coarse, Cp_fine=fine)
        except Exception:
            unit["failure"] = traceback.format_exc()
        report["units"].append(unit)
        write(output, report)
        print(f"upstream single-phase units {len(report['units'])}/5: {case['name']}", flush=True)
    report["complete"] = True
    write(output, report)


def native(directory, upstream, output):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.tdb import TdbDatabase
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    report = {"complete_and_passed": False, "expected_units": 5, "units": []}
    write(output, report, initial=True)
    for ref in upstream["units"]:
        case = ref["case"]
        unit = {"name": case["name"], "passed": False, "failures": []}
        try:
            if "failure" in ref:
                raise ValueError("upstream failed: " + ref["failure"])
            path = directory / case["database"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
                raise ValueError("TDB changed between upstream and native evaluation")
            source = InputSource(DataSourceProvenance(
                "pycalphad-test-tdb:" + case["database"], 0,
                "https://github.com/pycalphad/pycalphad/blob/0.11.2/"
                "pycalphad/tests/databases/" + case["database"], "0.11.2",
                "Database-specific terms not independently established; external only",
                ref["sha256"],
            ), "literature")
            model = CefPhaseModel(TdbDatabase.from_file(path, source=source), case["phase"],
                                  extrapolation="extend", unspecified_endmembers="zero")
            initial = json.loads(json.dumps(case["sites"]))
            result = model.stationary_constitution(case["target"], initial, case["T"], PRESSURE)
            unit["solver_diagnostics"] = result.diagnostics
            assert result.success, result.failure_reason
            assert initial == case["sites"], "initial sites mutated"
            p, mu = result.properties, result.chemical_potentials
            unit.update(G=p.gibbs_J_mol, H=p.enthalpy_J_mol,
                        site_fractions=[dict(s) for s in result.site_fractions],
                        composition=dict(p.element_fractions), mu_status=mu.status,
                        mu=None if mu.values_J_mol is None else dict(mu.values_J_mol),
                        rank=mu.matrix_rank, unknowns=mu.unknowns,
                        mu_residual_J_mol=mu.maximum_residual_J_mol,
                        Cp=result.relaxed_heat_capacity_J_mol_K)
            assert all(abs(p.element_fractions[e]-x) <= 1e-9 for e, x in case["target"].items())
            if case.get("expected_status") == "underdetermined":
                assert mu.status == "underdetermined" and mu.values_J_mol is None
                assert mu.matrix_rank == ref["expected_matrix_rank"]
                assert mu.unknowns == ref["expected_unknowns"]
                assert abs(p.gibbs_J_mol-ref["G"]) <= 1e-4+1e-8*abs(ref["G"])
            else:
                baseline = ref["baseline"]
                assert mu.status == "determined" and mu.internal_minimum
                errors = {e: abs(mu.values_J_mol[e]-baseline["mu"][e]) for e in case["target"]}
                unit["mu_absolute_errors_J_mol"] = errors
                assert all(error <= 1e-4+1e-8*abs(baseline["mu"][e])
                           for e, error in errors.items()), "elemental mu difference"
                assert abs(p.gibbs_J_mol-baseline["G"]) <= 1e-4+1e-8*abs(baseline["G"])
                assert result.relaxed_heat_capacity_J_mol_K is not None, "relaxed Cp unavailable"
                unit["reference_Cp"] = ref["Cp"]
                unit["Cp_absolute_error_J_mol_K"] = abs(
                    result.relaxed_heat_capacity_J_mol_K-ref["Cp"],
                )
                assert unit["Cp_absolute_error_J_mol_K"] <= 1e-3+1e-5*abs(ref["Cp"]), "relaxed Cp"
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
        report["units"].append(unit)
        write(output, report)
        print(f"native single-phase units {len(report['units'])}/5: "
              f"{case['name']} passed={unit['passed']}", flush=True)
    report.update(passed_units=sum(u["passed"] for u in report["units"]),
                  complete_and_passed=len(report["units"]) == 5
                  and all(u["passed"] for u in report["units"]))
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only:
        reference(args.source_directory, args.output)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("CALPHAD evidence must be outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    ref_path = args.output / "reference.json"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()), "--reference-only",
               "--source-directory", str(args.source_directory), "--output", str(ref_path)]
    with (args.output / "reference.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process:
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"upstream single-phase preparation alive: pid={process.pid}, "
                          f"log_bytes={log.tell()}", flush=True)
        if process.returncode:
            write(args.output / "summary.json", {"complete_and_passed": False,
                  "reference_exit_code": process.returncode}, initial=True)
            return 1
    result = native(args.source_directory, json.loads(ref_path.read_text(encoding="utf-8")),
                    args.output / "summary.json")
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
