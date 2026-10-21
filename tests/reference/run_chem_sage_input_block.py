"""Seven complete ChemSage declaration comparisons, no upstream native dependency."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

FILES = ("2026-Dixon-Na-K-Cl-I.dat", "Kaye_Pd-Ru-Tc-Mo.dat", "KF-NIF2_switched.dat",
         "MQMQA-tern-tests.dat", "Ocadiz-Flores.dat", "Shishin_Fe-Sb-O-S_slag.dat", "Viitala.dat")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def reference_signature(header, solutions, stoichiometric):
    def member(m):
        result = {"name": m.species_name, "option": m.gibbs_eq_type,
                  "stoichiometry": m.stoichiometry_pure_elements, "intervals": []}
        for interval in m.intervals:
            pressure = interval.PTVm_terms
            result["intervals"].append({
                "upper_K": interval.T_max,
                "coefficients": getattr(interval, "coefficients", getattr(
                    interval, "CP_coefficients", None)),
                "extra_powers": [[p.coefficient, p.exponent]
                                 for p in interval.additional_coeff_pairs],
                "pressure_volume_terms": [] if not pressure else pressure.terms,
                "transition_enthalpy_J_mol": getattr(interval, "H_trans", 0),
            })
        for field, origin in (("enthalpy_298_J_mol", "H298"),
                              ("entropy_298_J_mol_K", "S298")):
            result[field] = getattr(m.intervals[0], origin, None)
        for field, origin, default in (
            ("curie_temperature_K", "curie_temperature", None),
            ("magnetic_moment", "magnetic_moment", None),
            ("stoichiometric_factor", "stoichiometric_factor", None),
            ("chemical_group", "chemical_group", None),
            ("pair_stoichiometry", "stoichiometry_quadruplet", []), ("zeta", "zeta", None),
        ):
            result[field] = getattr(m, origin, default)
        return result

    phases = []
    for p in (*solutions, *stoichiometric):
        item = {"name": p.phase_name, "phase_type": p.phase_type or "PURE",
                "endmembers": [member(m) for m in p.endmembers],
                "site_ratios": getattr(p, "subl_ratios", []),
                "constituents": getattr(p, "constituent_array", None) or [],
                "endmember_indices": getattr(p, "endmember_constituent_idxs", None) or [],
                "magnetic_afm_factor": getattr(p, "magnetic_afm_factor", None),
                "magnetic_structure_factor": getattr(p, "magnetic_structure_factor", None),
                "interactions": [], "overrides": getattr(p, "chemical_group_overrides", []),
                "mqmqa": None}
        for ex in getattr(p, "excess_parameters", []):
            if type(ex).__name__ == "SUBQExcessQuadruplet":
                item["interactions"].append({
                    "kind": "MQMX", "indices": ex.mixing_const,
                    "coefficients": ex.excess_coeffs, "order": 0,
                    "exponents": ex.mixing_exponents, "mixing_type": ex.mixing_type,
                    "mixing_code": ex.mixing_code, "metadata": ex.metadata,
                    "additional_constituent": ex.additional_cation_mixing_const,
                    "additional_anion_constituent": ex.additional_anion_mixing_const,
                })
            else:
                magnetic = type(ex).__name__ == "ExcessRKMMagnetic"
                item["interactions"].append({
                    "kind": "MAGNETIC" if magnetic else
                    "QKT" if type(ex).__name__ == "ExcessQKTO" else "RKM",
                    "indices": ex.interacting_species_idxs,
                    "coefficients": [ex.curie_temperature, ex.magnetic_moment] if magnetic
                    else ex.coefficients, "order": getattr(ex, "parameter_order", 0),
                    "exponents": getattr(ex, "exponents", []), "mixing_type": 0,
                    "mixing_code": "", "metadata": [], "additional_constituent": 0,
                    "additional_anion_constituent": 0,
                })
        if type(p).__name__ == "Phase_SUBQ":
            item["mqmqa"] = {
                "cations": p.subl_1_const, "anions": p.subl_2_const,
                "cation_charges": p.subl_1_charges,
                "anion_charge_magnitudes": p.subl_2_charges,
                "cation_groups": p.subl_1_chemical_groups, "anion_groups": p.subl_2_chemical_groups,
                "pair_indices": p.subl_const_idx_pairs,
                "coordinations": [[q.quadruplet_idxs, q.quadruplet_coordinations]
                                   for q in p.quadruplets],
            }
        phases.append(item)
    return {"header": {"solution_sizes": header.list_soln_species_count,
                       "stoichiometric_phases": header.num_stoich_phases,
                       "components": header.pure_elements,
                       "masses_g_mol": header.pure_elements_mass,
                       "gibbs_indices": header.gibbs_coefficient_idxs,
                       "excess_indices": header.excess_coefficient_idxs}, "phases": phases}


def native_signature(parsed):
    result = asdict(parsed)
    del result["trailer"]
    del result["header"]["title"]
    for phase in result["phases"]:
        for name in ("status", "gas_slot", "declared_size"):
            del phase[name]
        if phase["phase_type"] == "PURE" or phase["mqmqa"]:
            phase["site_ratios"] = ()
        for member in phase["endmembers"]:
            for name in ("status", "real_gas_parameters", "aqueous_charge"):
                del member[name]
    return json.loads(json.dumps(result))


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                result.append({"field": prefix+"."+key, "native": left.get(key),
                               "reference": right.get(key)})
            else:
                result.extend(differences(left[key], right[key], prefix+"."+key))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [value for i, (a, b) in enumerate(zip(left, right, strict=True))
                for value in differences(a, b, prefix+f"[{i}]")]
    return [] if left == right else [{"field": prefix, "native": left, "reference": right}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only:
        import pycalphad
        from pycalphad.io.cs_dat import parse_cs_dat

        if pycalphad.__version__ != "0.11.2":
            raise ValueError("reference must be pycalphad 0.11.2")
        result = {"files": [], "dependencies": dict(sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))}
        for name in FILES:
            row = {"name": name}
            try:
                row["signature"] = reference_signature(*parse_cs_dat(
                    (args.source_directory/name).read_text(encoding="utf-8-sig").upper()))
            except Exception:
                row["failure"] = traceback.format_exc()
            result["files"].append(row)
            write(args.output, result)
        return 0
    if not args.reference_python:
        parser.error("--reference-python is required")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("ChemSage evidence must stay outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [str(args.reference_python), "-I", str(Path(__file__).resolve()), "--reference-only",
               "--source-directory", str(args.source_directory), "--output",
               str(args.output/"reference.json")]
    with (args.output/"reference.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=log, env=environment, timeout=60)
    if completed.returncode:
        write(args.output/"summary.json", {"expected_files": 7, "complete_and_passed": False,
                                          "reference_exit": completed.returncode})
        return 1
    from chemworld.data.chem_sage_parser import parse_chem_sage

    reference = json.loads((args.output/"reference.json").read_text(encoding="utf-8"))
    summary = {"expected_files": 7, "files": []}
    for name, ref in zip(FILES, reference["files"], strict=True):
        path = args.source_directory/name
        row = {"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "passed": False, "failures": [], "differences": []}
        try:
            parsed = parse_chem_sage(path.read_text(encoding="utf-8-sig"))
            native = native_signature(parsed)
            write(args.output/(name+".native.json"), native)
            assert "failure" not in ref, ref.get("failure")
            row["differences"] = differences(native, ref["signature"])
            row["phases"] = len(parsed.phases)
            row["endmembers"] = sum(len(p.endmembers) for p in parsed.phases)
            row["parameters"] = sum(len(p.interactions) for p in parsed.phases)
            row["passed"] = not row["differences"]
        except Exception:
            row["failures"].append(traceback.format_exc())
        summary["files"].append(row)
        write(args.output/"summary.json", summary)
        print(f"ChemSage input {len(summary['files'])}/7 {name} passed={row['passed']}", flush=True)
    summary["passed_files"] = sum(f["passed"] for f in summary["files"])
    summary["complete_and_passed"] = summary["passed_files"] == 7
    write(args.output/"summary.json", summary)
    return 0 if summary["complete_and_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
