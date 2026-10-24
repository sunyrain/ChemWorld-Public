"""Fixed independent descriptor/thermo/rate/dynamic acceptance with all results."""

import argparse
import json
import time
from dataclasses import replace
from math import isclose, log
from pathlib import Path

from catalysis53_fixtures import SOURCE, harmonic, mechanism, state_and_slot
from catalysis53_reference import R, finite_adsorption_amount, interaction, rates, thermo

from chemworld.physchem.catalysis_descriptors import FreeEnergyBEP
from chemworld.physchem.catalysis_network import catalytic_system, evaluate_catalytic_rates
from chemworld.physchem.inventory_interactions import InventoryInteraction
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world


class Record:
    def __init__(self, name):
        self.name, self.comparisons = name, []

    def check(self, label, actual, expected, *, relative=2e-8, absolute=2e-7, exact=False):
        passed = (
            actual == expected
            if exact
            else isclose(actual, expected, rel_tol=relative, abs_tol=absolute)
        )
        self.comparisons.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "passed": passed,
                "relative_tolerance": None if exact else relative,
                "absolute_tolerance": None if exact else absolute,
            }
        )


def descriptor(record, point):
    model = harmonic(descriptors=point)
    expected = -5000 + 0.5 * point["binding"] + 0.25 * point["strain"]
    record.check("affine_energy", model.electronic_energy.evaluate(point), expected, exact=True)


def harmonic_case(record, temperature, modes):
    entry = replace(mechanism().species[1], adsorbate=harmonic(modes=modes))
    expected = thermo(entry, temperature)
    native = entry.evaluate(temperature)
    for field, value in (
        ("u", entry.internal_energy(temperature)),
        ("g", native.gibbs_J_mol),
        ("s", native.entropy_J_mol_K),
        ("cv", entry.cv(temperature)),
    ):
        record.check(field, value, expected[field])
    dt = 1e-3
    derivative_u = (
        entry.internal_energy(temperature + dt) - entry.internal_energy(temperature - dt)
    ) / (2 * dt)
    derivative_g = (
        entry.evaluate(temperature + dt).gibbs_J_mol - entry.evaluate(temperature - dt).gibbs_J_mol
    ) / (2 * dt)
    record.check("dU_dT", derivative_u, expected["cv"], relative=2e-6)
    record.check("minus_dG_dT", -derivative_g, expected["s"], relative=2e-6)


def gas_case(record, temperature):
    entry = mechanism().species[0]
    expected = thermo(entry, temperature)
    native = entry.evaluate(temperature)
    for field, value in (
        ("u", entry.internal_energy(temperature)),
        ("g", native.gibbs_J_mol),
        ("s", native.entropy_J_mol_K),
        ("cv", entry.cv(temperature)),
    ):
        record.check(field, value, expected[field])


def rate_case(record, temperature, dissociative, point):
    spec, slot, state = state_and_slot(
        mechanism(descriptors=point, dissociative=dissociative), temperature=temperature
    )
    expected = rates(spec, slot, state)
    native = evaluate_catalytic_rates(state, slot, spec)
    for key, values in expected.items():
        record.check(key + "/delta_g", native.delta_g_J_mol[key], values["delta"])
        record.check(key + "/barrier_f", native.barriers_J_mol[key][0], values["forward_barrier"])
        record.check(key + "/barrier_r", native.barriers_J_mol[key][1], values["reverse_barrier"])
        record.check(
            key + "/forward_rate", native.forward_rates_mol_s[key], values["f"], absolute=1e-14
        )
        record.check(
            key + "/reverse_rate", native.reverse_rates_mol_s[key], values["r"], absolute=1e-14
        )
        bf, br = native.barriers_J_mol[key]
        record.check(
            key + "/detailed_balance",
            (br - bf) / (R * temperature),
            log(values["kf"] / values["kr"]),
        )


def interaction_case(record, amounts):
    spec, slot, state = state_and_slot(amounts=amounts)
    expected_u, expected_mu = interaction(spec, slot, state.amounts_mol)
    native_u, native_mu = InventoryInteraction.from_slot(
        slot, slot.spec.energy_interactions[0]
    ).energy_and_gradient(state.amounts_mol)
    record.check("excess_U", native_u, expected_u)
    for key, value in expected_mu.items():
        record.check("mu/" + key, native_mu[f"{slot.spec.slot_id}:{key}"], value)


def trajectory_case(record, occupied):
    spec = mechanism(interacting=False)
    spec = replace(spec, reactions=(spec.reactions[0],))
    spec, slot, state = state_and_slot(spec, amounts=(occupied * 1e-4, 1e-5))
    system = catalytic_system(spec, (slot,), support_heat_capacity_J_K=5, isothermal=True)
    result = system.advance(state, 0.5, evaluation_times_s=[0, 0.1, 0.3, 0.5])
    record.check("trajectory_points", len(result.trajectory), 4, exact=True)
    for final in result.trajectory:
        expected = finite_adsorption_amount(spec, slot, state, final.time_s)
        record.check(
            f"{final.time_s}/adsorbed",
            final.amounts_mol[f"{slot.spec.slot_id}:bound_A"],
            expected,
            relative=2e-6,
            absolute=2e-11,
        )
        expected_gas = 0.002 + occupied * 1e-4 - expected
        record.check(
            f"{final.time_s}/finite_gas",
            final.amounts_mol["He"],
            expected_gas,
            relative=2e-6,
            absolute=2e-11,
        )


def energy_case(record, isothermal):
    spec, slot, state = state_and_slot()
    system = catalytic_system(spec, (slot,), support_heat_capacity_J_K=5, isothermal=isothermal)
    result = system.advance(state, 0.5)
    final = result.final_state

    def reference_u(value):
        terms = []
        for entry in spec.species:
            key = entry.species.species_id
            packed = key if entry.species.phase == "gas" else f"{slot.spec.slot_id}:{key}"
            terms.append(value.amounts_mol[packed] * thermo(entry, value.temperature_K)["u"])
        return (
            sum(terms)
            + 5 * (value.temperature_K - 298.15)
            + interaction(spec, slot, value.amounts_mol)[0]
        )

    record.check(
        "energy_balance",
        reference_u(final) - reference_u(state),
        final.boundary_heat_J,
        absolute=2e-6,
    )
    record.check(
        "nuclear_balance",
        sum(final.amounts_mol.values()),
        sum(state.amounts_mol.values()),
        absolute=1e-11,
    )
    record.check("temperature_behavior", final.temperature_K == 350, isothermal, exact=True)
    for key, site in spec.adsorbate_sites.items():
        record.check(
            "finite_site/" + site,
            0 <= final.amounts_mol[f"{slot.spec.slot_id}:{key}"] <= slot.values["capacity:" + site],
            True,
            exact=True,
        )


def world_case(record, output):
    spec = mechanism()
    world = descriptor_catalysis_world(
        spec,
        {"He": 0.002},
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_adsorbates_mol={"bound_A": 2e-5, "bound_B": 1e-5},
        volume_L=1,
        temperature_K=350,
        support_heat_capacity_J_K=5,
        stop_temperature_K=440,
        maximum_pressure_Pa=1e6,
        isothermal=False,
    )
    (output / "declared-world.json").write_text(
        json.dumps(world.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )
    original, resumed = ChemWorldRuntime(composition=world), None
    try:
        original.reset(seed=53)
        first = original.execute({"operation": "wait", "duration_s": 2})
        record.check("first_commit", first.info["transaction_status"], "committed", exact=True)
        snapshot = json.loads(json.dumps(original.snapshot().to_dict()))
        (output / "snapshot.json").write_text(
            json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8"
        )
        resumed = ChemWorldRuntime.from_snapshot(snapshot)
        for runtime, label in ((original, "original"), (resumed, "restored")):
            outcome = runtime.execute({"operation": "wait", "duration_s": 3})
            record.check(
                label + "/commit", outcome.info["transaction_status"], "committed", exact=True
            )
            record.check(label + "/time", runtime._state.ledger.time_s, 5, exact=True)
            slot = runtime._state.state_slots.slots[0]
            total = (
                runtime._state.species_amounts["He"]
                + slot.values["bound_A"]
                + slot.values["bound_B"]
            )
            record.check(label + "/nuclei", total, 0.00203, absolute=1e-11)
        record.check(
            "exact_continuation",
            original._state.to_dict() == resumed._state.to_dict(),
            True,
            exact=True,
        )
    finally:
        original.close()
        if resumed is not None:
            resumed.close()


def boundary_case(record, kind):
    error = None
    try:
        if kind == "descriptor":
            harmonic(descriptors={"binding": 10001, "strain": 0})
        elif kind == "barrier":
            FreeEnergyBEP(1, 0.5, -100, 100, SOURCE).barriers(10)
        elif kind == "site":
            spec = mechanism()
            replace(
                spec, reactions=(replace(spec.reactions[0], reactants={"He": 1, "vacancy:B": 1}),)
            )
        else:
            spec = mechanism()
            replace(
                spec,
                species=(replace(spec.species[0], reference_frame="unrelated"), *spec.species[1:]),
            )
    except ValueError as exc:
        error = str(exc)
    record.check("explicit_rejection", error is not None, True, exact=True)
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for i, point in enumerate(
        (
            {"binding": 0, "strain": 0},
            {"binding": -10000, "strain": -5000},
            {"binding": 10000, "strain": 5000},
            {"binding": 2000, "strain": -1000},
        )
    ):
        cases.append((f"descriptor-{i}", lambda r, p=point: descriptor(r, p)))
    for temperature in (300, 350, 450):
        for modes in ((2000,), (1000, 2000, 3000)):
            cases.append(
                (
                    f"harmonic-{temperature}-{len(modes)}",
                    lambda r, t=temperature, m=modes: harmonic_case(r, t, m),
                )
            )
        cases.append((f"gas-{temperature}", lambda r, t=temperature: gas_case(r, t)))
    for i, (temperature, dissociative, point) in enumerate(
        ((300, False, None), (350, True, None), (450, False, {"binding": 2000, "strain": -1000}))
    ):
        cases.append(
            (f"rates-{i}", lambda r, t=temperature, d=dissociative, p=point: rate_case(r, t, d, p))
        )
    for i, amounts in enumerate(((0, 0), (2e-5, 1e-5), (8e-5, 1.5e-4))):
        cases.append((f"interaction-{i}", lambda r, a=amounts: interaction_case(r, a)))
    for occupied in (0.2, 0.5):
        cases.append((f"finite-adsorption-{occupied}", lambda r, o=occupied: trajectory_case(r, o)))
    for isothermal in (True, False):
        cases.append(
            (f"energy-isothermal-{isothermal}", lambda r, iso=isothermal: energy_case(r, iso))
        )
    cases.append(("world-snapshot", lambda r: world_case(r, args.output)))
    for kind in ("descriptor", "barrier", "site", "reference"):
        cases.append((f"rejection-{kind}", lambda r, k=kind: boundary_case(r, k)))
    start, records = time.perf_counter(), []
    for i, (name, execute) in enumerate(cases, 1):
        record, error, detail = Record(name), None, None
        try:
            detail = execute(record)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        records.append(
            {
                "unit": name,
                "comparisons": record.comparisons,
                "error": error,
                "detail": detail,
                "passed": error is None and all(c["passed"] for c in record.comparisons),
            }
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        print(
            f"Catalysis53 {i}/{len(cases)} {name} passed={records[-1]['passed']} "
            f"throughput={i / elapsed:.1f} units/s ETA={(len(cases) - i) * elapsed / i:.1f}s",
            flush=True,
        )
    comparisons = [c for row in records for c in row["comparisons"]]
    summary = {
        "status": "passed" if all(r["passed"] for r in records) else "failed",
        "evidence_mode": "development",
        "units_total": len(records),
        "units_passed": sum(r["passed"] for r in records),
        "comparisons_total": len(comparisons),
        "comparisons_passed": sum(c["passed"] for c in comparisons),
        "failures": [row for row in records if not row["passed"]],
        "elapsed_s": time.perf_counter() - start,
        "reference": (
            "Independent partition functions, affine/BEP/rate formulas, finite-gas "
            "Riccati trajectory and original world; not same-version CatMAP qualification"
        ),
    }
    for name, value in (("records.json", records), ("summary.json", summary)):
        (args.output / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
