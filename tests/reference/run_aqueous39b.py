"""Fixed 48-unit / 292-comparison 39B development block. Keep every attempt."""

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from aqueous39b_cases import DIELECTRIC, TP, inventory, native_inventory, synthetic_reaction
from aqueous39b_reference import PR, TR, M, density, dielectric, hkf_state, mp, water_g

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_hkf import AqueousReferenceBasis
from chemworld.physchem.aqueous_hkf_sources import (
    hkf_aqueous_database,
    hkf_reference_basis,
    hkf_source_records,
    hkf_source_species,
    hkf_species_from_record,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import solve_aqueous_reaction_equilibrium
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.water_dielectric import (
    water_dielectric_response,
    water_relative_permittivity,
)
from chemworld.physchem.water_if97 import water_if97_liquid_density
from chemworld.physchem.water_saturation import water_saturation_pressure
from chemworld.runtime.aqueous_phase_equilibrium import equilibrate_aqueous_phase_inventory


def comparison(name, actual, expected, atol=2e-6, rtol=2e-8):
    return {
        "quantity": name,
        "actual": float(actual),
        "expected": float(expected),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "absolute_error": abs(float(actual) - float(expected)),
        "passed": math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
    }


def dielectric_case(T, molar_density, printed):
    rho = molar_density * 1000 * 0.018015268
    native = asdict(water_dielectric_response(T, rho))
    t, d = M(str(T)), M(str(rho))
    refs = (
        dielectric(t, d),
        mp.diff(lambda x: dielectric(x, d), t),
        mp.diff(lambda x: dielectric(t, x), d),
        mp.diff(lambda x: dielectric(x, d), t, 2),
        mp.diff(lambda x: mp.diff(lambda y: dielectric(x, y), d), t),
        mp.diff(lambda x: dielectric(t, x), d, 2),
    )
    return [
        comparison("printed epsilon", native["relative_permittivity"], printed, 2e-5, 0),
        *(comparison(k, v, ref, 2e-11) for (k, v), ref in zip(native.items(), refs, strict=True)),
    ]


def species_case(key, T, P, reference_authority="strict"):
    th = hkf_source_species(key, reference_authority=reference_authority)
    native = th.evaluate_tp(T, P)
    basis = hkf_reference_basis()
    expected = hkf_state(
        hkf_source_records()[key],
        T,
        P,
        {**basis.element_entropy_J_mol_K, "charge": basis.charge_entropy_J_mol_K},
        reference_authority=reference_authority,
    )
    actual = {
        "G": native.gibbs_J_mol,
        "H": native.enthalpy_J_mol,
        "S": native.entropy_J_mol_K,
        "Cp": native.cp_J_mol_K,
        "V": native.molar_volume_m3_mol,
        "density": water_if97_liquid_density(T, P),
        "epsilon": water_relative_permittivity(T, water_if97_liquid_density(T, P)),
        "omega": th.omega(T, P),
    }
    tolerances = {
        "G": 2e-6,
        "H": 2e-6,
        "S": 2e-7,
        "Cp": 2e-7,
        "V": 2e-12,
        "density": 2e-7,
        "epsilon": 2e-8,
        "omega": 2e-6,
    }
    return [comparison(k, v, expected[k], tolerances[k]) for k, v in actual.items()]


def reference_case(index):
    if index == 0:
        s = hkf_source_species("Na+").evaluate_tp(298.15, 1e5)
        return [
            comparison("H datum", s.enthalpy_J_mol, -57433 * 4.184),
            comparison("S datum", s.entropy_J_mol_K, 13.96 * 4.184, 2e-7),
            comparison("G datum", s.gibbs_J_mol, (-57433 - 298.15 * 13.96) * 4.184),
        ]
    if index == 1:
        s = hkf_source_species("Cl-").evaluate_tp(298.15, 1e5)
        se = 111.5395 + 65.34
        gf, entropy = -31379 * 4.184, 13.56 * 4.184
        return [
            comparison("formation G", s.gibbs_J_mol + 298.15 * se, gf),
            comparison("native G", s.gibbs_J_mol, gf - 298.15 * se),
            comparison("converted H", s.enthalpy_J_mol, gf + 298.15 * (entropy - se)),
        ]
    if index == 2:
        s = hkf_source_species("H+").evaluate_tp(573.15, 2e7)
        return [
            comparison(k, getattr(s, k), 0, 2e-8)
            for k in ("gibbs_J_mol", "enthalpy_J_mol", "entropy_J_mol_K")
        ]
    db = synthetic_reaction().database
    pair = [s.standard_state.thermo for s in db.species if s.species_id in ("Cl-", "Cl-test")]
    before = [s.evaluate_tp(473.15, 1e7) for s in pair]
    shifted = []
    for th in pair:
        basis = th.reference_basis
        newbasis = replace(
            basis,
            basis_id="explicit-test-shift",
            element_entropy_J_mol_K={
                **basis.element_entropy_J_mol_K,
                "Cl": basis.element_entropy_J_mol_K["Cl"] + 5,
            },
            charge_entropy_J_mol_K=basis.charge_entropy_J_mol_K + 7,
        )
        delta = newbasis.entropy_shift(th.composition, th.charge) - basis.entropy_shift(
            th.composition, th.charge
        )
        other = replace(
            th,
            reference_basis=newbasis,
            entropy_ref_J_mol_K=th.entropy_ref_J_mol_K + delta,
            formation_gibbs_ref_J_mol=None,
            formation_enthalpy_ref_J_mol=th.enthalpy_ref_J_mol,
        )
        shifted.append(other.evaluate_tp(473.15, 1e7))
    return [
        comparison(
            k,
            getattr(shifted[1], k) - getattr(shifted[0], k),
            getattr(before[1], k) - getattr(before[0], k),
            2e-7 if "entropy" in k else 2e-6,
        )
        for k in ("gibbs_J_mol", "enthalpy_J_mol", "entropy_J_mol_K")
    ]


def integration_case(index):
    T, P = 473.15, 1e7
    if index == 0:
        db = hkf_aqueous_database()
        amounts = inventory(db)
        state = aqueous_phase_properties(db, amounts, temperature_K=T, pressure_Pa=P)
        mass = molecular_weight({"H": 2, "O": 1}) / 1000
        basis = hkf_reference_basis()
        elemental = {**basis.element_entropy_J_mol_K, "charge": basis.charge_entropy_J_mol_K}
        reference = {
            k: hkf_state(hkf_source_records()[k], T, P, elemental) for k in ("Na+", "Cl-", "H+")
        }
        mm, t, p = M(".018015268"), M(str(T)), M(str(P))
        sref = -mp.diff(lambda x: mm * water_g(x, PR), TR)
        href = mm * water_g(TR, PR) + TR * sref
        g = mm * water_g(t, p) + M(-285830) - href - t * (M("69.95") - sref)
        s = -mp.diff(lambda x: mm * water_g(x, p), t) + M("69.95") - sref
        reference["H2O"] = {"G": float(g), "H": float(g + t * s), "V": float(mm / density(t, p))}
        totalg = sum(n * reference[k]["G"] for k, n in amounts.items())
        totalg += (
            R
            * T
            * sum(
                n * (math.log(n / (amounts["H2O"] * mass)) - 1)
                for k, n in amounts.items()
                if k != "H2O" and n > 0
            )
        )
        totalh = sum(n * reference[k]["H"] for k, n in amounts.items())
        totalv = sum(n * reference[k]["V"] for k, n in amounts.items())
        return [
            comparison("G", state.gibbs_energy_J, totalg, 2e-5),
            comparison("H", state.enthalpy_J, totalh, 2e-5),
            comparison("S", state.entropy_J_K, (totalh - totalg) / T, 2e-5),
            comparison("V", state.volume_m3, totalv, 2e-11),
            comparison("U", state.internal_energy_J, totalh - P * totalv, 2e-5),
            comparison("positive V", int(state.volume_m3 > 0), 1, 0, 0),
        ]
    if index == 1:
        spec = synthetic_reaction()
        result = solve_aqueous_reaction_equilibrium(
            spec, inventory(spec.database), temperature_K=T, pressure_Pa=P
        )
        amounts = result.amounts_mol or {}
        K = math.exp(-(2000 - 3 * T) / (R * T))
        return [
            comparison("success", int(result.success), 1, 0, 0),
            comparison(
                "analytic extent", amounts.get("Cl-test", math.nan), 0.1 * K / (1 + K), 2e-9
            ),
            comparison("Na conserved", amounts.get("Na+", math.nan), 0.1, 2e-9),
            comparison("water conserved", amounts.get("H2O", math.nan), 55.5, 2e-9),
            comparison("charge", result.charge_residual_mol, 0, 2e-9),
            comparison("affinity", result.reaction_residuals.get("synthetic-pair"), 0, 2e-8),
        ]
    if index == 2:
        spec, ledger, vessel = native_inventory(T, P)
        before = ledger.to_dict()
        result = equilibrate_aqueous_phase_inventory(
            ledger, spec, "aqueous", vessel=vessel, temperature_K=T, pressure_Pa=P
        )
        extent = 0.1 / (1 + math.exp((2000 - 3 * T) / (R * T)))
        return [
            comparison("success", int(result.success), 1, 0, 0),
            comparison("heat", result.heat_duty_J, 2000 * extent, 2e-5),
            comparison("work", result.boundary_work_J, 0, 2e-5),
            comparison(
                "delta U",
                result.final_properties.internal_energy_J
                - result.initial_properties.internal_energy_J,
                2000 * extent,
                2e-5,
            ),
            comparison("immutable input", int(ledger.to_dict() == before), 1, 0, 0),
            comparison(
                "other owner",
                int(result.ledger.phases["other"].to_dict() == before["other"]),
                1,
                0,
                0,
            ),
        ]
    db = synthetic_reaction().database
    payload = json.loads(json.dumps(db.to_dict(), allow_nan=False))
    restored = AqueousDatabase.from_dict(payload)
    a, b = db.standard_states(T, P), restored.standard_states(T, P)
    return [
        comparison("JSON contract", int(payload == restored.to_dict()), 1, 0, 0),
        *(comparison(key + " G", b[key].gibbs_J_mol, a[key].gibbs_J_mol) for key in a),
    ]


def rejection_case(index):
    record = hkf_source_records()["Na+"]
    args = {
        "reference_basis": hkf_reference_basis(),
        "source_id": "test",
        "units": "cal-mol-bar-K",
        "consistency_tolerance_J_mol": 10.719448,
    }
    try:
        if index == 0:
            record.pop("a")
            hkf_species_from_record("Na+", record, **args)
        elif index == 1:
            args["reference_basis"] = AqueousReferenceBasis("missing", {"H": 65.34}, -65.34, "test")
            hkf_species_from_record("Na+", record, **args)
        elif index == 2:
            args["units"] = "J-mol-Pa-K"
            hkf_species_from_record("Na+", record, **args)
        elif index == 3:
            record["Gf"] = 0
            hkf_species_from_record("Na+", record, **args)
        else:
            T, P = ((650, 3e7), (300, 1.01e8), (500, 1e5), (300, water_saturation_pressure(300)))[
                index - 4
            ]
            hkf_source_species("Na+").evaluate_tp(T, P)
    except ValueError as exc:
        return [{**comparison("explicit rejection", 1, 1, 0, 0), "diagnostic": str(exc)}]
    return [comparison("explicit rejection", 0, 1, 0, 0)]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    definitions = [
        *({"kind": "dielectric", "args": list(row), "comparisons": 7} for row in DIELECTRIC),
        *(
            {"kind": "species", "args": [key, *tp], "comparisons": 8}
            for key in ("Na+", "Cl-", "H+", "OH-")
            for tp in TP
        ),
        *({"kind": "reference", "args": [i], "comparisons": 3} for i in range(4)),
        *({"kind": "integration", "args": [i], "comparisons": 6} for i in range(4)),
        *({"kind": "rejection", "args": [i], "comparisons": 1} for i in range(8)),
    ]
    functions = {
        "dielectric": dielectric_case,
        "species": species_case,
        "reference": reference_case,
        "integration": integration_case,
        "rejection": rejection_case,
    }
    (output / "units.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    started = time.monotonic()
    units, comparisons = [], []
    print("start units=0/48 planned_comparisons=292", flush=True)
    for i, definition in enumerate(definitions, 1):
        record = dict(unit=i, **definition)
        try:
            rows = functions[definition["kind"]](*definition["args"])
            if len(rows) != definition["comparisons"]:
                raise RuntimeError("comparison denominator changed")
            comparisons.extend(dict(unit=i, **row) for row in rows)
            record.update(passed=all(row["passed"] for row in rows), executed=len(rows))
        except Exception as exc:
            record.update(passed=False, executed=0, exception=f"{type(exc).__name__}: {exc}")
        units.append(record)
        elapsed = time.monotonic() - started
        print(
            f"progress units={i}/48 rate={i / max(elapsed, 1e-9):.2f}/s "
            f"eta={(48 - i) * elapsed / i:.1f}s status={'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )
    failures = [row for row in units if not row["passed"]]
    raw = hkf_source_records()["OH-"]
    se = hkf_reference_basis().entropy_shift(raw["composition"], raw["charge"])
    data_residual = raw["Hf"] * 4.184 - 298.15 * (raw["S"] * 4.184 - se) - raw["Gf"] * 4.184
    summary = {
        "mode": "development",
        "planned_units": 48,
        "passed_units": 48 - len(failures),
        "failed_units": len(failures),
        "planned_comparisons": 292,
        "executed_comparisons": len(comparisons),
        "passed_comparisons": sum(row["passed"] for row in comparisons),
        "failed_comparisons": sum(not row["passed"] for row in comparisons),
        "unexecuted_comparisons": 292 - len(comparisons),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "data_precision": {
            "species": "OH-",
            "source_record": raw,
            "reference_residual_J_mol": data_residual,
            "allowed_printing_residual_J_mol": 10.719448,
            "disposition": "full source triple rejected; no authoritative pair",
        },
        "reference": {
            "algorithm": "independent mpmath 50-digit public equations",
            "water": "IF97 region1 + IAPWS97; different from upstream water EOS",
            "upstream_equivalence": "not claimed or numerically qualified",
        },
    }
    for name, rows in (("comparisons", comparisons), ("unit_results", units)):
        (output / (name + ".jsonl")).write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
