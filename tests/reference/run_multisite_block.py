"""Fixed 57-unit reversible multisite development comparison; retain attempts."""

import json
import sys
import time
from dataclasses import replace
from math import log
from pathlib import Path

import numpy as np
from multisite_catmap import coverages_from_logits, reference_rates, solve_reference_steady
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from chemworld.physchem.inventory_interactions import InventoryInteraction
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.surface_rates import (
    MULTISITE_REACTIONS,
    adsorbate_key,
    evaluate_surface_rates,
    gas_amounts_from_pressure,
    standard_surface_gibbs,
    surface_state,
)
from chemworld.physchem.surface_specs import (
    MULTISITE_ADSORBATES,
    MultisiteSurfaceSpec,
    make_surface_slot,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.coupled_thermal import local_thermodynamics
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.surface_world import surface_world_spec


def model(strength):
    matrix = np.eye(4) * strength
    matrix[0, 2] = matrix[2, 0] = matrix[1, 3] = matrix[3, 1] = strength / 2
    return MultisiteSurfaceSpec(interaction_matrix_J_mol=tuple(tuple(row) for row in matrix))


def fixture(spec, temperature, coverages):
    slot = make_surface_slot(
        "carrier",
        "reactor",
        spec=spec,
        initial_coverages=coverages[:2],
        second_coverages=coverages[2:],
    )
    gas = gas_amounts_from_pressure(
        {"CO": 0.2, "O2": 0.1, "CO2": 0.05, "N2": 0.65},
        pressure_Pa=100000,
        temperature_K=temperature,
        volume_L=0.01,
    )
    state = surface_state(gas, (slot,), temperature_K=temperature, volume_L=0.01)
    return slot, state


def rates_unit(spec, temperature, coverages):
    slot, state = fixture(spec, temperature, coverages)
    native = evaluate_surface_rates(state, slot, spec)
    reference = reference_rates(temperature, coverages, spec.interaction_matrix_J_mol)
    term = InventoryInteraction.from_slot(slot, slot.spec.energy_interactions[0])
    _, gradient = term.energy_and_gradient(state.amounts_mol)
    gradient_error = max(
        abs(gradient[k] - v)
        for k, v in zip(term.keys, reference["excess_potentials_J_mol"], strict=True)
    )
    derivative_errors = []
    for key in term.keys:
        plus, minus = dict(state.amounts_mol), dict(state.amounts_mol)
        plus[key] += 1e-11
        minus[key] -= 1e-11
        fd = (term.energy_and_gradient(plus)[0] - term.energy_and_gradient(minus)[0]) / 2e-11
        derivative_errors.append(fd - gradient[key])
    progress = np.array(native.progress_rates_mol_s)
    rate_errors = progress - reference["progress_mol_s"]
    potentials = standard_surface_gibbs(spec, temperature)
    potentials.update(
        {k: potentials[k] + gradient[adsorbate_key(slot, k)] for k in MULTISITE_ADSORBATES}
    )
    activity = {k: v / 100000 for k, v in native.partial_pressures_Pa.items()}
    activity.update(native.coverages)
    residuals = []
    for i, (left, right, _) in enumerate(MULTISITE_REACTIONS):
        dg = sum(v * (potentials[k] + R * temperature * log(activity[k])) for k, v in right.items())
        dg -= sum(v * (potentials[k] + R * temperature * log(activity[k])) for k, v in left.items())
        residuals.append(log(progress[2 * i] / progress[2 * i + 1]) + dg / (R * temperature))
    checks = {
        "potential": gradient_error <= 1e-8,
        "finite_difference": max(map(abs, derivative_errors)) <= 1e-3,
        "rates": bool(np.all(abs(rate_errors) <= 1e-8 * abs(reference["progress_mol_s"]) + 1e-16)),
        "detailed_balance": max(map(abs, residuals)) <= 1e-8,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "potential_error_J_mol": float(gradient_error),
        "finite_difference_errors_J_mol": derivative_errors,
        "native_progress_mol_s": progress.tolist(),
        "reference_progress_mol_s": reference["progress_mol_s"].tolist(),
        "detailed_balance_log_residuals": residuals,
    }


def steady_unit(spec, temperature, initial):
    slot, state = fixture(spec, temperature, initial)

    def coverage_rates(coverages):
        candidate = replace(
            state,
            amounts_mol={
                **state.amounts_mol,
                **{
                    adsorbate_key(slot, k): float(v) * slot.values["sites_mol"]
                    for k, v in zip(MULTISITE_ADSORBATES, coverages, strict=True)
                },
            },
        )
        rates = evaluate_surface_rates(candidate, slot, spec)
        return np.array(
            [
                rates.amount_rates_mol_s[adsorbate_key(slot, k)] / slot.values["sites_mol"]
                for k in MULTISITE_ADSORBATES
            ]
        )

    def residual(logits):
        return coverage_rates(coverages_from_logits(logits))

    transient = solve_ivp(
        lambda t, y: coverage_rates(y), (0, 1000), initial, method="BDF", rtol=1e-9, atol=1e-12
    )
    if not transient.success:
        raise RuntimeError("native steady transient preconditioning failed")
    endpoint = transient.y[:, -1]
    start = [
        log(v / (1 - sum(endpoint[(i // 2) * 2 : (i // 2 + 1) * 2])))
        for i, v in enumerate(endpoint)
    ]
    solved = least_squares(residual, start, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2000)
    native = coverages_from_logits(solved.x)
    reference = solve_reference_steady(temperature, initial, spec.interaction_matrix_J_mol)
    rate_residual = float(max(abs(residual(solved.x))))
    diff = float(max(abs(native - reference)))
    return {
        "passed": bool(solved.success and rate_residual <= 1e-8 and diff <= 1e-6),
        "native_coverages": native.tolist(),
        "reference_coverages": reference.tolist(),
        "max_coverage_difference": diff,
        "max_steady_rate_s_inv": rate_residual,
        "native_solver_nfev": solved.nfev,
        "transient_preconditioning_s": 1000,
        "native_transient_nfev": transient.nfev,
    }


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    return result


def snapshot(runtime):
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    payload["operation_record"]["constitution_checks"].sort(
        key=lambda c: json.dumps(c, sort_keys=True)
    )
    return payload


def workflow(spec):
    world = surface_world_spec(surface_spec=spec)

    def new_session():
        runtime = ChemWorldRuntime(composition=world)
        runtime.reset(seed=38)
        execute(runtime, "add_component", component="CO-feed", amount_mol=2e-5)
        execute(runtime, "add_component", component="O2-feed", amount_mol=1e-5)
        return runtime

    runtime = new_session()
    initial = runtime._state
    execute(runtime, "heat", target_temperature_K=400, duration_s=60)
    execute(runtime, "measure", instrument="gas_composition")
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.from_dict(snapshot(runtime)))
    for s in (runtime, restored):
        execute(s, "wait", duration_s=60)
        execute(s, "measure", instrument="gas_composition")
    final = runtime._state
    refined = new_session()
    execute(refined, "heat", target_temperature_K=400, duration_s=30)
    execute(refined, "wait", duration_s=30)
    execute(refined, "measure", instrument="gas_composition")
    execute(refined, "wait", duration_s=30)
    execute(refined, "wait", duration_s=30)
    execute(refined, "measure", instrument="gas_composition")
    temps = final.temperature_K - refined._state.temperature_K
    amounts = [final.state_slots.slots[0].values[k] for k in MULTISITE_ADSORBATES]
    reference = [refined._state.state_slots.slots[0].values[k] for k in MULTISITE_ADSORBATES]
    energies = []
    for state in (initial, final):
        thermo, packed = local_thermodynamics(state, world.physics.models_of('thermal')["reactor"])
        energies.append(thermo.internal_energy_j(packed))
    qin = final.ledger.energy_jacket_J - initial.ledger.energy_jacket_J
    qout = final.ledger.heat_loss_J - initial.ledger.heat_loss_J
    energy_error = energies[1] - energies[0] - qin + qout

    def elements(state):
        totals = {}
        species = {s.species_id: s for s in world.physics.network.species}
        for local in (state, *state.inactive_vessels.values()):
            stocks = list(local.species_amounts.items())
            stocks.extend(
                (f.species_id, s.values[f.name])
                for s in local.state_slots.slots
                for f in s.spec.fields
                if f.species_id
            )
            for key, n in stocks:
                for element, count in species[key].composition.items():
                    totals[element] = totals.get(element, 0) + count * n
        return totals

    before, after = elements(initial), elements(final)
    residuals = {k: after[k] - v for k, v in before.items()}
    sites_ok = all(
        final.state_slots.slots[0].values[k] == initial.state_slots.slots[0].values[k]
        for k in ("sites_mol", "sites_B_mol")
    )
    slot = final.state_slots.slots[0]
    occupied_ok = all(
        sum(slot.values[k] * n for k, n in c.occupied_fields)
        <= slot.values[c.capacity_field] * (1 + 1e-12)
        for c in slot.spec.capacities
    )
    checks = {
        "elements": all(abs(v) <= 1e-8 * before[k] + 1e-12 for k, v in residuals.items()),
        "site_stock": sites_ok and occupied_ok,
        "energy": abs(energy_error) <= 1e-6 * max(abs(energies[0]), abs(qin), abs(qout)) + 1e-7,
        "refinement": abs(temps) <= 1e-4 + 1e-6 * final.temperature_K
        and all(
            abs(a - b) <= 1e-5 * abs(b) + 1e-12 for a, b in zip(amounts, reference, strict=True)
        ),
        "snapshot": snapshot(runtime) == snapshot(restored),
        "time_counted_once": final.ledger.time_s - initial.ledger.time_s == 130,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "energy_error_J": energy_error,
        "element_residuals_mol": residuals,
        "final_temperature_K": final.temperature_K,
        "refinement_temperature_error_K": temps,
        "final_adsorbates_mol": amounts,
        "refined_adsorbates_mol": reference,
        "actual_time_s": final.ledger.time_s,
        "heat_in_J": qin,
        "heat_out_J": qout,
    }


def main(path):
    previous = []
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        previous = [*old.pop("previous_attempts", []), old]
    report = {
        "mode": "development",
        "expected_units": 57,
        "cases": [],
        "previous_attempts": previous,
        "reference": ("CatMAP ed04f9146a787b0e63b6febd96d8fe3b314c251b "
                      "local kernels + independent steady solve"),
    }
    units = []
    for strength in (-5000, 0, 5000):
        for temperature in (325, 350, 400):
            for coverage in ((0.05, 0.05, 0.10, 0.05), (0.6, 0.1, 0.1, 0.6), (0.1, 0.6, 0.6, 0.1)):
                for kind, fn in (("rates", rates_unit), ("steady", steady_unit)):
                    units.append(
                        (
                            {
                                "strength_J_mol": strength,
                                "temperature_K": temperature,
                                "initial_coverages": coverage,
                                "kind": kind,
                            },
                            lambda s=strength, t=temperature, c=coverage, f=fn: f(model(s), t, c),
                        )
                    )
    units.extend(
        ({"kind": "native_workflow", "strength_J_mol": s}, lambda s=s: workflow(model(s)))
        for s in (-5000, 0, 5000)
    )
    started = time.monotonic()
    for metadata, fn in units:
        try:
            row = fn()
        except Exception as exc:
            row = {"passed": False, "error": str(exc), "error_type": type(exc).__name__}
        report["cases"].append({**metadata, **row})
        n, elapsed = len(report["cases"]), time.monotonic() - started
        report.update(
            completed_units=n,
            elapsed_s=elapsed,
            passed_units=sum(r["passed"] for r in report["cases"]),
            failed_units=sum(not r["passed"] for r in report["cases"]),
            skipped_units=0,
        )
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            f"{n}/57 {metadata['kind']} passed={row['passed']} rate={n / elapsed:.2f}/s "
            f"ETA={(57 - n) * elapsed / n:.1f}s",
            flush=True,
        )
    if report["failed_units"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
