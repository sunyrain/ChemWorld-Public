"""Fixed 21-source-status / 18-material independent MQMQA thermal block."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True), encoding="utf-8")


def reference_module():
    spec = importlib.util.spec_from_file_location("isolated_mqmqa_material_reference",
        Path(__file__).with_name("mqmqa_material_reference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(directory, output, native_report=None):
    module = reference_module()
    report = {"expected_source_statuses": 21, "expected_materials": 18,
              "expected_nodes": 1170, "units": [],
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    started = time.perf_counter()
    completed_nodes = 0
    native_units = {} if native_report is None else {
        (u["database"], u["phase"]): u for u in native_report["units"]}
    for name, phase, raw, header, database in module.phases(directory):
        unit = {"database": name, "phase": phase, "points": [], "failures": [],
                "sha256": hashlib.sha256((directory/name).read_bytes()).hexdigest()}
        report["units"].append(unit)
        try:
            issues = module.source_issues(raw, header)
            if issues:
                unit.update(status="source_inconsistent", source_issues=issues)
            else:
                processor = module.ReferencePhase(database, phase, header)
                if native_report is None:
                    for i in range(65):
                        unit["points"].append(processor.at(1050+100*i/64))
                        completed_nodes += 1
                        if i % 8 == 0 or i == 64:
                            write(output, report)
                            elapsed = max(1e-9, time.perf_counter()-started)
                            print(f"reference nodes {completed_nodes}/1170 "
                                  f"rate={completed_nodes/elapsed:.2f}/s "
                                  f"eta={(1170-completed_nodes)*elapsed/completed_nodes:.1f}s "
                                  f"{name} {phase}", flush=True)
                    unit["fine_T"] = module.trajectory(unit["points"], 1)
                    unit["coarse_T"] = module.trajectory(unit["points"], 2)
                    unit["convergence_K"] = max(abs(a-b) for a, b in zip(
                        unit["fine_T"], unit["coarse_T"], strict=True))
                    if unit["convergence_K"] > 1e-4:
                        raise ValueError("independent 65/33-node trajectory convergence failed")
                else:
                    native = native_units[name, phase]
                    if native["failures"] or len(native["endpoints"]) != 2:
                        raise ValueError("native material did not complete both endpoints")
                    unit["points"] = [processor.at(p["T"]) for p in native["endpoints"]]
                unit["status"] = "computed"
        except Exception:
            unit.update(status="failed")
            unit["failures"].append(traceback.format_exc())
        write(output, report)
        print(f"reference materials {len(report['units'])}/21 {phase} {unit['status']}", flush=True)


def native(directory, curves, output):
    from chemworld.data.chem_sage import ChemSageDatabase
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.calphad_material import CefMaterialSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.physchem.mqmqa import MQMQAPhaseModel
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.solid_solutions import solid_enthalpy, solid_model
    from chemworld.world.solid_world import calphad_world_spec

    report = {"expected_source_statuses": 21, "expected_materials": 18,
              "expected_endpoints": 36, "units": []}
    databases = {}
    started = time.perf_counter()
    for ref in curves["units"]:
        name, phase = ref["database"], ref["phase"]
        unit = {"database": name, "phase": phase, "endpoints": [], "failures": []}
        report["units"].append(unit)
        try:
            if name not in databases:
                source = InputSource(DataSourceProvenance("public-chemsage-"+name, 0,
                    "Original public DAT "+name, "0.11.2", "database terms not established"),
                    "literature")
                databases[name] = ChemSageDatabase.from_file(directory/name, source=source)
            try:
                phase_model = MQMQAPhaseModel(databases[name], phase, activate_suppressed=True)
            except ValueError as exc:
                unit["source_rejection"] = traceback.format_exc()
                if ref["status"] != "source_inconsistent":
                    raise
                kinds = {issue["kind"] for issue in ref["source_issues"]}
                expected = ("pair source atoms" if "pair_element_identity" in kinds
                            else "coordination violates charge")
                assert expected in str(exc), (
                    "native refusal differs from independent source diagnosis")
                unit["status"] = "source_rejected"
                continue
            if ref["status"] != "computed":
                raise ValueError("reference did not establish a usable source and caloric curve")
            sites = (dict.fromkeys(phase_model.constituents[0],
                                   1/len(phase_model.constituents[0])),)
            stock = {e: .001*x for e, x in phase_model._composition(sites)[1].items()}
            model = CefMaterialSpec(phase_model, sites, 1050, 1150, 1100, .01,
                                    phase_type="liquid")
            world = calphad_world_spec(model, element_amounts_mol=stock, temperature_K=1100)
            runtime = ChemWorldRuntime(composition=world)
            runtime.reset(seed=40)
            split = ChemWorldRuntime.from_snapshot(runtime.snapshot())
            initial_h = solid_enthalpy(runtime._state, model)
            for target in (1130., 1080.):
                action = {"operation": "heat", "target_temperature_K": target, "duration_s": 3}
                result = runtime.execute(action)
                first = split.execute(dict(action, duration_s=1.5))
                second = split.execute({"operation": "wait", "duration_s": 1.5})
                state = runtime._state
                selected = solid_model(state, world)
                current = selected.at_tp(state.species_amounts, state.temperature_K)
                p, mu = current.constitution.properties, current.constitution.chemical_potentials
                h = solid_enthalpy(state, selected)
                point = {"T": state.temperature_K, "H_total": h,
                    "split_T": split._state.temperature_K,
                    "split_H": solid_enthalpy(split._state, model),
                    "G": p.gibbs_J_mol, "H": p.enthalpy_J_mol,
                    "x": dict(p.element_fractions), "mu_status": mu.status,
                    "mu": None if mu.values_J_mol is None else dict(mu.values_J_mol),
                    "energy_residual": h-initial_h-state.ledger.energy_jacket_J
                        +state.ledger.heat_loss_J,
                    "sites": [dict(s) for s in current.constitution.site_fractions],
                    "diagnostics": result.info}
                unit["endpoints"].append(point)
                write(output, report)
                assert all(r.info["transaction_status"] == "committed"
                           for r in (result, first, second)), "physical operation did not commit"
                assert state.species_amounts == split._state.species_amounts == stock, "inventory"
                assert abs(point["energy_residual"]) <= 1e-7, "physical heat balance"
                assert abs(point["T"]-point["split_T"]) <= 1e-4, "split temperature"
                assert abs(h-point["split_H"]) <= 1e-5, "split enthalpy"
                print(f"native MQMQA {len(report['units'])}/21 endpoint="
                      f"{len(unit['endpoints'])}/2 {phase}", flush=True)
            unit["status"] = "computed"
        except Exception:
            unit["status"] = "failed"
            unit["failures"].append(traceback.format_exc())
        finally:
            write(output, report)
            elapsed = max(1e-9, time.perf_counter()-started)
            n = len(report["units"])
            print(f"native materials {n}/21 rate={n/elapsed:.3f}/s "
                  f"eta={(21-n)*elapsed/n:.1f}s {phase} {unit['status']}", flush=True)


def compare(output):
    curves, native_report, reference_endpoints = [json.loads((output/name).read_text(
        encoding="utf-8")) for name in ("curves.json", "native.json", "endpoints.json")]
    result = {"expected_source_statuses": 21, "expected_materials": 18,
              "expected_endpoints": 36, "units": []}
    for curve, native_unit, ref_unit in zip(curves["units"], native_report["units"],
                                           reference_endpoints["units"], strict=True):
        unit = {"database": curve["database"], "phase": curve["phase"], "passed": False,
                "errors": [], "failures": []}
        result["units"].append(unit)
        try:
            assert all(u["phase"] == unit["phase"] and u["database"] == unit["database"]
                       for u in (native_unit, ref_unit)), "source identity"
            if curve["status"] == "source_inconsistent":
                assert native_unit["status"] == "source_rejected", "source accepted incorrectly"
                assert ref_unit["status"] == "source_inconsistent", "source diagnosis changed"
                unit.update(passed=True, source_rejection=True, issues=curve["source_issues"])
                continue
            assert all(u["status"] == "computed" for u in (curve, native_unit, ref_unit)), (
                "one or more stages failed")
            pairs = zip(native_unit["endpoints"], ref_unit["points"], strict=True)
            for i, (p, r) in enumerate(pairs):
                errors = {"T": abs(p["T"]-curve["fine_T"][i]),
                          "G": abs(p["G"]-r["G"]), "H": abs(p["H"]-r["H"]),
                          "x": max(abs(p["x"][e]-x) for e, x in r["x"].items())}
                unit["errors"].append(errors)
                assert errors["T"] <= 2e-4, "independent thermal trajectory"
                assert errors["x"] <= 1e-8, "independent composition"
                assert all(errors[k] <= 1e-4+1e-8*abs(r[k]) for k in ("G", "H")), "G/H"
                assert p["mu_status"] == r["mu_status"], "element potential determinacy"
                if r["mu"] is None:
                    assert p["mu"] is None
                else:
                    errors["mu"] = max(abs(p["mu"][e]-value)
                                       for e, value in r["mu"].items())
                    assert all(abs(p["mu"][e]-value) <= 1e-4+1e-8*abs(value)
                               for e, value in r["mu"].items()), "independent mu"
            assert len(unit["errors"]) == 2
            unit["passed"] = True
        except Exception:
            unit["failures"].append(traceback.format_exc())
    result["matched_source_rejections"] = sum(u.get("source_rejection", False)
                                             for u in result["units"])
    result["passed_materials"] = sum(u["passed"] and not u.get("source_rejection", False)
                                     for u in result["units"])
    result["passed_endpoints"] = sum(len(u["errors"]) for u in result["units"]
                                     if u["passed"] and not u.get("source_rejection", False))
    result["complete_classification_and_valid_materials"] = (
        len(result["units"]) == 21 and result["passed_materials"] == 18
        and result["matched_source_rejections"] == 3 and result["passed_endpoints"] == 36)
    result["all_original_21_numerical_phases_qualified"] = False
    write(output/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--stage", choices=("curves", "native", "endpoints"))
    args = parser.parse_args()
    if args.stage:
        if args.stage == "curves":
            reference(args.source_directory, args.output/"curves.json")
        elif args.stage == "native":
            native(args.source_directory, json.loads((args.output/"curves.json").read_text(
                encoding="utf-8")), args.output/"native.json")
        else:
            reference(args.source_directory, args.output/"endpoints.json", json.loads(
                (args.output/"native.json").read_text(encoding="utf-8")))
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("raw qualification data must stay outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for stage in ("curves", "native", "endpoints"):
        interpreter = sys.executable if stage == "native" else str(args.reference_python)
        command = [interpreter, *([] if stage == "native" else ["-I"]),
                   str(Path(__file__).resolve()), "--stage", stage,
                   "--source-directory", str(args.source_directory), "--output", str(args.output)]
        path = args.output/(stage+".log")
        with (path.open("x", encoding="utf-8") as log,
              subprocess.Popen(command, stdout=log, stderr=log, env=environment) as process):
            while True:
                try:
                    process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    print(f"{stage} alive pid={process.pid} bytes={path.stat().st_size} "
                          f"last={lines[-1] if lines else 'startup'}", flush=True)
        print(f"{stage} exit={process.returncode}", flush=True)
        if process.returncode:
            write(args.output/"stage_failure.json", {"stage": stage, "exit": process.returncode,
                  "expected_source_statuses": 21, "expected_materials": 18,
                  "expected_endpoints": 36, "log": str(path)})
            return 1
    result = compare(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["complete_classification_and_valid_materials"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
