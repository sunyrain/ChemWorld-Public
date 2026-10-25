"""Fixed 14-unit/104-comparison parent-53 closure block with exact replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from copy import deepcopy
from math import log, prod
from pathlib import Path

import numpy as np
from catalytic_table53_cases import BOUNDARY_CASES, FIT_CASES, case
from catalytic_table53_reference import weighted_reference

from chemworld.data.catalysis_inputs import assemble_catalytic_mechanism
from chemworld.examples.catalytic_energy_tables import energy_calibration, mechanism_declaration
from chemworld.physchem.catalysis_network import evaluate_catalytic_rates, pack_catalytic_state
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


class Unit:
    def __init__(self, name):
        self.name, self.checks, self.details = name, [], {}

    def check(self, name, actual, expected, *, rtol=0.0, atol=0.0, exact=False):
        if exact:
            passed = actual == expected
            error = None
        else:
            actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            passed = actual.shape == expected.shape and bool(
                np.allclose(actual, expected, rtol=rtol, atol=atol)
            )
            error = float(np.max(np.abs(actual - expected)))
            actual, expected = actual.tolist(), expected.tolist()
        self.checks.append(
            {
                "name": name,
                "actual": actual,
                "expected": expected,
                "passed": bool(passed),
                "rtol": rtol,
                "atol": atol,
                "exact": exact,
                "maximum_absolute_error": error,
            }
        )


def fit_unit(unit, path):
    calibration, options = case(unit.name)
    result = calibration.fit(**options)
    reference = weighted_reference(calibration, options.get("fixed_parameters", {}))
    write(path / "fit.json", result)
    unit.details["resources"] = result["resources"]
    names = [p.name for p in calibration.parameters]
    unit.check("converged", result["status"], "fitted", exact=True)
    # Physical energy and dimensionless tolerances are separately respected.
    errors = [
        abs(result["estimate"][name] - reference["estimate"][name])
        / (2e-5 if name == "intercept_J_mol" else 2e-8)
        for name in names
    ]
    unit.details["parameter_errors_in_fixed_tolerance"] = dict(zip(names, errors, strict=True))
    unit.details["estimate"] = result["estimate"]
    unit.details["reference_estimate"] = reference["estimate"]
    unit.check(
        "parameters_within_fixed_tolerance",
        bool(np.all(np.asarray(errors) <= 1.0)),
        True,
        exact=True,
    )
    uncertainty = result["parameter_uncertainty"]
    unit.check(
        "covariance", uncertainty["covariance"], reference["covariance"], rtol=2e-7, atol=2e-10
    )
    unit.check(
        "standard_errors",
        list(uncertainty["standard_errors"].values()),
        reference["standard_errors"],
        rtol=2e-7,
        atol=2e-10,
    )
    unit.check("weighted_SSE", result["objective"], reference["objective"], rtol=2e-7, atol=2e-5)
    predicted = [
        row["predicted"] * (1000 if row["unit"] == "kJ/mol" else 1)
        for row in result["validation"]["measurements"]
    ]
    unit.check(
        "validation_predictions_J_mol",
        predicted,
        reference["validation_predictions_J_mol"],
        rtol=2e-7,
        atol=2e-5,
    )
    free = [p for p in calibration.parameters if p.name in reference["free_names"]]
    scales = [p.decode_derivative(p.encode(result["estimate"][p.name])) for p in free]
    physical = np.asarray(result["identifiability"]["weighted_jacobian"]) / scales
    unit.check(
        "physical_weighted_J",
        physical,
        reference["physical_weighted_jacobian"],
        rtol=2e-7,
        atol=2e-10,
    )
    unit.check(
        "source_identity",
        result["sources"],
        [source.to_dict() for source in calibration.table.sources],
        exact=True,
    )


def boundary_unit(unit, path):
    calibration, options = case(unit.name)
    result = calibration.fit(**options)
    write(path / "fit.json", result)
    expected = {
        "rank": "fitted",
        "bound": "fitted",
        "optimizer_budget": "optimizer_not_converged",
        "forward_budget": "forward_failed",
    }
    unit.check("fit_status", result["status"], expected[unit.name], exact=True)
    status = {"rank": "rank_deficient", "bound": "active_parameter_bound"}.get(
        unit.name, "fit_not_converged"
    )
    unit.check(
        "uncertainty",
        [result["parameter_uncertainty"]["status"], result["parameter_uncertainty"]["covariance"]],
        [status, None],
        exact=True,
    )
    unit.check("model_export", "fitted_model" in result, unit.name in {"rank", "bound"}, exact=True)
    unit.check(
        "retained_computation",
        bool(result["runs"]) and result["resources"]["forward_groups"] == len(result["runs"]),
        True,
        exact=True,
    )
    unit.details["resources"] = result["resources"]


def state_values(spec, state):
    slot = state.state_slots.slots[0]
    ns = {**state.species_amounts, **{name: slot.values[name] for name in spec.adsorbates}}
    atoms = {}
    for name, amount in ns.items():
        for element, count in spec.species_by_id[name].species.composition.items():
            atoms[element] = atoms.get(element, 0.0) + count * amount
    surface = np.asarray([ns[name] for name in spec.adsorbates])
    excess = float(
        surface
        @ np.asarray(spec.interaction_matrix_J_mol)
        @ surface
        / (2 * sum(slot.values["capacity:" + site] for site in spec.site_types))
    )
    energy = (
        sum(
            ns[name] * entry.internal_energy(state.temperature_K)
            for name, entry in spec.species_by_id.items()
        )
        + excess
        + 5 * (state.temperature_K - 298.15)
    )
    return atoms, energy


def world_unit(unit, path):
    topology, mode = unit.name.split("/")
    declaration = mechanism_declaration(topology)
    before = deepcopy(declaration)
    fits = {name: energy_calibration(name).fit(max_nfev=60) for name in ("bound_A", "bound_B")}
    fits["bep"] = energy_calibration(bep=True).fit(max_nfev=60)
    spec, assembly = assemble_catalytic_mechanism(
        declaration,
        fits,
        species_fits={name: name for name in ("bound_A", "bound_B")},
        reaction_fits={r["reaction_id"]: "bep" for r in declaration["reactions"]},
        descriptors={"binding": 1000.0, "strain": -500.0},
    )
    write(path / "assembly.json", assembly)
    composition = descriptor_catalysis_world(
        spec,
        {spec.gas_species[0]: 0.002},
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_adsorbates_mol={"bound_A": 2e-5, "bound_B": 1e-5},
        volume_L=1.0,
        temperature_K=350.0,
        support_heat_capacity_J_K=5.0,
        stop_temperature_K=440.0,
        maximum_pressure_Pa=1e6,
        isothermal=mode == "isothermal",
    )
    runtime, restored = ChemWorldRuntime(composition=composition), None
    try:
        runtime.reset(seed=53)
        initial_atoms, initial_energy = state_values(spec, runtime._state)
        replies = [runtime.execute({"operation": "wait", "duration_s": 2.0})]
        checkpoint = json.loads(json.dumps(runtime.snapshot().to_dict()))
        write(path / "checkpoint.json", checkpoint)
        replies.append(runtime.execute({"operation": "wait", "duration_s": 3.0}))
        final = runtime._state
        expected = json.loads(json.dumps(runtime.snapshot().to_dict()))
        write(path / "final.json", expected)
        unit.check(
            "committed",
            [r.info["transaction_status"] for r in replies],
            ["committed", "committed"],
            exact=True,
        )
        unit.check(
            "actual_conversion",
            abs(final.species_amounts[spec.gas_species[0]] - 0.002) > 1e-12,
            True,
            exact=True,
        )
        final_atoms, final_energy = state_values(spec, final)
        unit.check(
            "element_balance_mol",
            list(final_atoms.values()),
            list(initial_atoms.values()),
            atol=2e-10,
        )
        heat = final.ledger.energy_jacket_J - final.ledger.heat_loss_J
        unit.check("original_U_heat_balance_J", final_energy - initial_energy, heat, atol=2e-5)
        unit.check(
            "fit_bindings",
            spec.to_dict() == assembly["mechanism"]
            and len(assembly["bindings"]) == 4
            and all(
                s.adsorbate.electronic_energy.to_dict()
                == fits[s.species.species_id]["fitted_model"]
                for s in spec.species
                if s.adsorbate is not None
            ),
            True,
            exact=True,
        )
        slot = final.state_slots.slots[0]
        packed = pack_catalytic_state(
            spec, (slot,), final.species_amounts, temperature_K=final.temperature_K, volume_L=1.0
        )
        rates = evaluate_catalytic_rates(packed, slot, spec)
        activities = {
            name: final.species_amounts[name] * R * final.temperature_K / 0.001 / 1e5
            for name in spec.gas_species
        }
        activities.update(
            {
                name: slot.values[name] / slot.values["capacity:" + spec.adsorbate_sites[name]]
                for name in spec.adsorbates
            }
        )
        activities.update(
            {
                "vacancy:" + site: 1
                - sum(
                    activities[name]
                    for name in spec.adsorbates
                    if spec.adsorbate_sites[name] == site
                )
                for site in spec.site_types
            }
        )
        balance = []
        for reaction in spec.reactions:
            forward_activity = prod(activities[k] ** n for k, n in reaction.reactants.items())
            reverse_activity = prod(activities[k] ** n for k, n in reaction.products.items())
            ratio = (rates.forward_rates_mol_s[reaction.reaction_id] / forward_activity) / (
                rates.reverse_rates_mol_s[reaction.reaction_id] / reverse_activity
            )
            balance.append(
                log(ratio) + rates.delta_g_J_mol[reaction.reaction_id] / (R * final.temperature_K)
            )
        unit.check("detailed_balance", balance, [0.0] * len(balance), atol=2e-10)
        restored = ChemWorldRuntime.from_snapshot(checkpoint)
        restored.execute({"operation": "wait", "duration_s": 3.0})
        actual = json.loads(json.dumps(restored.snapshot().to_dict()))
        write(path / "restored.json", actual)
        unit.check("complete_JSON_continuation", actual == expected, True, exact=True)
        fresh_path = path / "fresh.json"
        child = subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(Path(__file__).resolve()),
                "--resume",
                str(path / "checkpoint.json"),
                "--output",
                str(fresh_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (path / "fresh.log").write_text(child.stdout + child.stderr, encoding="utf-8")
        fresh = (
            json.loads(fresh_path.read_text(encoding="utf-8")) if child.returncode == 0 else None
        )
        unit.check("complete_fresh_process_continuation", fresh == expected, True, exact=True)
        unit.check("authored_declaration_unchanged", declaration, before, exact=True)
        numerical, analytic = [], []
        for entry in spec.species:
            dt, temperature = 0.001, final.temperature_K
            numerical.extend(
                [
                    (
                        entry.internal_energy(temperature + dt)
                        - entry.internal_energy(temperature - dt)
                    )
                    / (2 * dt),
                    -(
                        entry.evaluate(temperature + dt).gibbs_J_mol
                        - entry.evaluate(temperature - dt).gibbs_J_mol
                    )
                    / (2 * dt),
                ]
            )
            analytic.extend([entry.cv(temperature), entry.evaluate(temperature).entropy_J_mol_K])
        unit.check("thermodynamic_derivatives", numerical, analytic, rtol=3e-5, atol=2e-5)
        unit.details.update(
            initial_energy_J=initial_energy,
            final_energy_J=final_energy,
            heat_J=heat,
            actual_duration_s=final.ledger.time_s,
            final_gas=dict(final.species_amounts),
            final_temperature_K=final.temperature_K,
            fit_resources={name: fit["resources"] for name, fit in fits.items()},
            actual_world_waits=4,
            fresh_processes=1,
        )
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            reply = runtime.execute({"operation": "wait", "duration_s": 3.0})
            if reply.info["transaction_status"] != "committed":
                raise RuntimeError(str(reply.info))
            write(args.output, runtime.snapshot().to_dict())
        finally:
            runtime.close()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cases = (
        [(name, 8, fit_unit) for name in FIT_CASES]
        + [(name, 4, boundary_unit) for name in BOUNDARY_CASES]
        + [
            (f"{topology}/{mode}", 10, world_unit)
            for topology in ("exchange", "dissociation")
            for mode in ("isothermal", "adiabatic")
        ]
    )
    start, units = time.monotonic(), []
    for index, (name, planned, callback) in enumerate(cases, 1):
        path = args.output / name.replace("/", "-")
        path.mkdir()
        unit, failure = Unit(name), None
        print(f"53 close {index}/14 starting {name}", flush=True)
        try:
            callback(unit, path)
        except Exception:
            failure = traceback.format_exc()
        passed = sum(check["passed"] for check in unit.checks)
        units.append(
            {
                "name": name,
                "planned": planned,
                "executed": len(unit.checks),
                "passed": passed,
                "unevaluated": planned - len(unit.checks),
                "failure": failure,
                "checks": unit.checks,
                "details": unit.details,
                "success": failure is None and passed == planned and len(unit.checks) == planned,
            }
        )
        elapsed = time.monotonic() - start
        write(path / "unit.json", units[-1])
        rate = f"{index / elapsed:.2f} units/s" if elapsed > 0 else "clock tick pending"
        print(
            f"53 close {index}/14 completed, {passed}/{planned}; "
            f"{rate} ETA {(14 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    summary = {
        "scope": "development_parent53_table_fit_and_original_consumers",
        "units": units,
        "units_planned": 14,
        "units_passed": sum(unit["success"] for unit in units),
        "comparisons_planned": 104,
        "comparisons_executed": sum(u["executed"] for u in units),
        "comparisons_passed": sum(u["passed"] for u in units),
        "comparisons_unevaluated": sum(u["unevaluated"] for u in units),
        "elapsed_s": time.monotonic() - start,
    }
    write(args.output / "summary.json", summary)
    print({key: value for key, value in summary.items() if key != "units"}, flush=True)
    raise SystemExit(0 if summary["units_passed"] == 14 else 1)


if __name__ == "__main__":
    main()
