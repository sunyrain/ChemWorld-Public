"""Fixed contact grid; preserve every development attempt and failure."""

import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from chemworld.examples.native_sorbent_research import (
    synthetic_sorbent_spec,
    synthetic_sorbent_thermodynamics,
)
from chemworld.physchem.sorbent_kinetics import (
    advance_sorbent_contact,
    make_sorbent_slot,
    sorbent_key,
)

GRID = tuple(product((0.1, 1.0, 3.0), (298.15, 318.15), (False, True)))


def heartbeat(label):
    last = [time.monotonic()]

    def progress(t, evaluations):
        now = time.monotonic()
        if now - last[0] >= 30:
            print(f"sorbent {label}: RHS={evaluations}, trial_time={t:.6f}s", flush=True)
            last[0] = now

    return progress


def fixture(concentration, temperature, preloaded):
    spec = synthetic_sorbent_spec()
    slot = make_sorbent_slot(
        spec,
        owner_id="contact-carrier",
        location_id="contact",
        mass_kg=0.01,
        initial_loading_mol_per_kg=({"A": 0.3, "B": 0.2} if preloaded else None),
        bound_species_ids={"A": "A_ads", "B": "B_ads"},
    )
    authored = synthetic_sorbent_thermodynamics(spec)
    thermo = replace(
        authored,
        species={
            **{k: authored.species[k] for k in spec.component_ids},
            **{sorbent_key(slot, k): authored.species[f"{k}_ads"] for k in spec.component_ids},
        },
    )
    fluid = {"A": 0.1 * concentration, "B": 0.05 * concentration}
    return spec, slot, thermo, fluid


def independent_reference(concentration, temperature, preloaded):
    # This does not call the native isotherm/finite-equilibrium helpers.
    total = np.array([0.1 * concentration, 0.05 * concentration]) + 0.01 * np.array(
        [0.3, 0.2] if preloaded else [0.0, 0.0]
    )
    affinity = np.array([1.0, 3.0]) * np.exp(
        -np.array([-10000.0, -20000.0]) / 8.31446261815324 * (1 / temperature - 1 / 298.15)
    )

    def residual(log_concentrations):
        values = np.exp(log_concentrations)
        loading = affinity * values / (1 + np.dot(affinity, values))
        return (0.1 * values + 0.01 * loading - total) / total

    result = least_squares(residual, np.log(total / 0.1), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if not result.success or max(abs(residual(result.x))) > 1e-10:
        raise RuntimeError("independent finite-bath equilibrium did not converge")
    values = np.exp(result.x)
    return {
        "loading_mol_kg": dict(
            zip(("A", "B"), affinity * values / (1 + np.dot(affinity, values)), strict=True)
        ),
        "liquid_concentration_mol_L": dict(zip(("A", "B"), values, strict=True)),
        "maximum_normalized_equation_residual": float(max(abs(residual(result.x)))),
        "nfev": result.nfev,
    }


def state_record(state):
    return {
        "amounts_mol": dict(state.amounts_mol),
        "temperature_K": state.temperature_K,
        "time_s": state.time_s,
        "volume_L": state.volume_L,
        "boundary_heat_J": state.boundary_heat_J,
        "heat_loss_J": state.heat_loss_J,
        "chemical_energy_change_J": state.chemical_energy_change_J,
    }


def independent_energy(state, slot):
    rise = state.temperature_K - 298.15
    value = (20 + slot.values["heat_capacity_J_K"]) * rise
    for key, cv, slope, energy in (("A", 50, 0.02, -10000), ("B", 70, 0.03, -20000)):
        bound = state.amounts_mol[sorbent_key(slot, key)]
        total = state.amounts_mol[key] + bound
        value += total * (cv * rise + 0.5 * slope * rise**2) + bound * energy
    return value


def trajectory_row(concentration, temperature, preloaded, *, isothermal, reference):
    spec, slot, thermo, fluid = fixture(concentration, temperature, preloaded)
    grid = (0, 10, 30, 100, 300, 1000, 3000) if isothermal else (0, 10, 30, 100, 300)
    result = advance_sorbent_contact(
        spec,
        slot,
        fluid,
        temperature_K=temperature,
        volume_L=0.1,
        duration_s=grid[-1],
        thermodynamics=thermo,
        isothermal=isothermal,
        evaluation_times_s=grid,
        progress=heartbeat((concentration, temperature, preloaded, isothermal)),
    )
    initial, final = result.initial_state, result.final_state
    errors = dict.fromkeys(spec.component_ids, 0.0)
    largest_energy = 0.0
    for state in result.trajectory:
        assert min(state.amounts_mol.values()) >= 0
        assert 280 <= state.temperature_K <= 430
        assert sum(state.amounts_mol[sorbent_key(slot, k)] for k in spec.component_ids) <= (
            slot.values["sites_mol"] + 1e-10
        )
        for key in spec.component_ids:
            birth = initial.amounts_mol[key] + initial.amounts_mol[sorbent_key(slot, key)]
            error = (
                abs(state.amounts_mol[key] + state.amounts_mol[sorbent_key(slot, key)] - birth)
                / birth
            )
            errors[key] = max(errors[key], error)
        energy = independent_energy(state, slot) - independent_energy(initial, slot)
        external = state.boundary_heat_J - state.heat_loss_J
        largest_energy = max(
            largest_energy, abs(energy - external) / max(abs(energy), abs(external), 1.0)
        )
    assert max(errors.values()) <= 1e-8, errors
    assert largest_energy <= 1e-6, largest_energy
    comparisons = {}
    if isothermal:
        for key in spec.component_ids:
            value = final.amounts_mol[sorbent_key(slot, key)] / 0.01
            expected = reference["loading_mol_kg"][key]
            allowance = max(1e-10, abs(expected) * 1e-7)
            comparisons[key] = {
                "native": value,
                "independent": expected,
                "allowance": allowance,
                "passed": bool(abs(value - expected) <= allowance),
            }
        assert all(x["passed"] for x in comparisons.values()), comparisons
        assert final.temperature_K == temperature
    else:
        assert final.boundary_heat_J == 0
    return {
        "passed": True,
        "isothermal": isothermal,
        "normalized_component_errors": errors,
        "maximum_independent_normalized_energy_error": largest_energy,
        "equilibrium_comparisons": comparisons,
        "solver_diagnostic": dict(result.solver_diagnostic),
        "trajectory": [state_record(s) for s in result.trajectory],
    }


def native_cycles(destination):
    from chemworld.physchem.sorbent_kinetics import loading_field
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.condensed_material_routing import background_source_energy
    from chemworld.runtime.thermal_material_routing import _energy
    from chemworld.world.sorbent_world import sorbent_world_spec

    report = json.loads(destination.read_text(encoding="utf-8"))
    attempt = {
        "started_utc": datetime.now(UTC).isoformat(),
        "cycle_denominator": 3,
        "wash_boundary": "finite_internal_0.03_L_stock",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "cycles": [],
        "actions": [],
    }
    report["native_cycle_attempts"].append(attempt)

    def save():
        destination.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    spec = synthetic_sorbent_spec()
    world = sorbent_world_spec(
        spec, synthetic_sorbent_thermodynamics(spec), {"A": "C2H6O", "B": "C3H8O"}
    )
    session, replay = ChemWorldRuntime(composition=world), None
    start = time.monotonic()
    session.reset(seed=22)

    def locals_of(state):
        return {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}

    def quantities(state):
        locals_ = locals_of(state)
        n = {
            k: sum(
                local.species_amounts[k]
                + sum(slot.values[loading_field(k)] for slot in local.state_slots.slots)
                for local in locals_.values()
            )
            for k in spec.component_ids
        }
        energy = sum(
            _energy(local, world.physics.models_of('thermal')[name])
            if name in world.physics.models_of('thermal')
            else background_source_energy(local, world.physics)
            for name, local in locals_.items()
        )
        sites = sum(
            slot.values["sites_mol"]
            for local in locals_.values()
            for slot in local.state_slots.slots
        )
        mass = sum(
            slot.values["mass_kg"] for local in locals_.values() for slot in local.state_slots.slots
        )
        return n, energy, sites, mass

    birth, initial_energy, initial_sites, initial_mass = quantities(session._state)

    def check_state():
        current, energy, sites, mass = quantities(session._state)
        component_error = {k: abs(current[k] - birth[k]) / birth[k] for k in birth}
        inlet_energy = sum(
            r["inlet_energy_J"]
            for local in locals_of(session._state).values()
            for r in local.metadata.get("carrier_packet_history", ())
            if r.get("boundary_kind") != "internal"
        )
        utilities = session._state.ledger.energy_jacket_J - session._state.ledger.heat_loss_J
        energy_error = abs(energy - initial_energy - utilities - inlet_energy) / max(
            abs(energy - initial_energy), abs(utilities), abs(inlet_energy), 1.0
        )
        assert max(component_error.values()) <= 1e-8, component_error
        assert energy_error <= 1e-6, energy_error
        assert abs(sites - initial_sites) <= 1e-12
        assert abs(mass - initial_mass) <= 1e-12
        assert session._state.state_slots.slots[0].spec.owner_id == "reactor:bed"
        total_carrier = sum(local.volume_L for local in locals_of(session._state).values())
        assert abs(total_carrier - 0.11) <= 1e-12
        assert session._state.solvent_accounting.added.volume_L == 0
        return {
            "normalized_component_errors": component_error,
            "normalized_energy_error": energy_error,
            "total_internal_energy_J": energy,
            "total_sites_mol": sites,
            "total_carrier_mass_kg": mass,
            "total_liquid_carrier_L": total_carrier,
            "remaining_wash_supply_L": locals_of(session._state)["wash_supply"].volume_L,
        }

    def execute(**action):
        for current in (session, replay):
            if current is None:
                continue
            result = current.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(
                    str(
                        {
                            "action": action,
                            "error": result.info.get("error_message"),
                            "diagnostics": result.info.get("outcome_diagnostics"),
                        }
                    )
                )
            if current is session:
                attempt["actions"].append(
                    {
                        "action": action,
                        "outcome": result.info["operation_outcome"],
                        "actual_duration_s": result.info["actual_duration_s"],
                        "time_s": current._state.ledger.time_s,
                    }
                )
                save()

    save()
    try:
        for cycle in range(1, 4):
            print(f"Native finite regeneration cycle {cycle}/3", flush=True)
            execute(operation="dry")
            execute(operation="measure", instrument="phase_state")
            loaded = sum(
                session._state.state_slots.slots[0].values[loading_field(k)]
                for k in spec.component_ids
            )
            assert loaded > 0
            execute(
                operation="route_material",
                connection="reactor-to-waste",
                transfer_fraction=1.0,
                mixing=1,
            )
            execute(operation="add_solvent", solvent=0, volume_L=0.01)
            initial_heat = session._state.ledger.energy_jacket_J
            execute(
                operation="heat", target_temperature_K=350, duration_s=300, stirring_speed_rpm=600
            )
            warm = sum(
                session._state.state_slots.slots[0].values[loading_field(k)]
                for k in spec.component_ids
            )
            heat = session._state.ledger.energy_jacket_J - initial_heat
            assert -1e-10 <= heat <= 1500 + 1e-6
            execute(operation="pause_control")
            execute(operation="wait", duration_s=180, stirring_speed_rpm=600)
            cooled = sum(
                session._state.state_slots.slots[0].values[loading_field(k)]
                for k in spec.component_ids
            )
            execute(
                operation="route_material",
                connection="reactor-to-waste",
                transfer_fraction=1.0,
                mixing=1,
            )
            if cycle < 3:
                execute(operation="select_vessel", vessel="feed")
                fraction = 0.02 / session._state.volume_L
                execute(
                    operation="route_material",
                    connection="feed-to-reactor",
                    transfer_fraction=fraction,
                    mixing=1,
                )
                execute(operation="select_vessel", vessel="reactor")
            row = {
                "cycle": cycle,
                "passed": True,
                "loaded_amount_mol": loaded,
                "warm_wash_remaining_fraction": warm / loaded,
                "cooled_wash_remaining_fraction": cooled / loaded,
                "warm_regeneration_goal_met": bool(warm / loaded <= 0.5),
                "heater_energy_J": heat,
                **check_state(),
            }
            attempt["cycles"].append(row)
            if cycle == 1:
                checkpoint = json.loads(json.dumps(session.snapshot().to_dict()))
                replay = ChemWorldRuntime.from_snapshot(checkpoint)
                attempt["snapshot_checkpoint_after_cycle"] = 1
            save()
        assert session._state.to_dict() == replay._state.to_dict()
        locals_ = locals_of(session._state)
        assert abs(session._state.ledger.time_s - 2343) <= 1e-7
        assert abs(locals_["waste"].volume_L - 0.09) <= 1e-12
        assert abs(locals_["feed"].volume_L - 0.02) <= 1e-12
        assert session._state.volume_L == 0
        assert locals_["wash_supply"].volume_L == 0
        attempt.update(
            passed=True,
            exact_snapshot_continuation=True,
            unmet_warm_regeneration_goals=sum(
                not row["warm_regeneration_goal_met"] for row in attempt["cycles"]
            ),
            final_time_s=session._state.ledger.time_s,
            final_states={name: state.to_dict() for name, state in locals_.items()},
        )
    except Exception as exc:
        attempt.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        session.close()
        if replay is not None:
            replay.close()
    attempt.update(elapsed_s=time.monotonic() - start, finished_utc=datetime.now(UTC).isoformat())
    save()
    print(
        {
            k: attempt.get(k)
            for k in (
                "passed",
                "error",
                "elapsed_s",
                "final_time_s",
                "unmet_warm_regeneration_goals",
            )
        },
        flush=True,
    )
    if not attempt["passed"]:
        raise SystemExit(1)


def main():
    destination = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "cycles":
        return native_cycles(destination)
    report = (
        json.loads(destination.read_text(encoding="utf-8"))
        if destination.exists()
        else {
            "classification": "development evidence",
            "contact_attempts": [],
            "native_cycle_attempts": [],
        }
    )
    attempt = {
        "started_utc": datetime.now(UTC).isoformat(),
        "condition_denominator": 12,
        "trajectory_denominator": 24,
        "equilibrium_reference_denominator": 12,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "conditions": [],
    }
    report["contact_attempts"].append(attempt)

    def save():
        destination.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    start = time.monotonic()
    save()
    for index, (concentration, temperature, preloaded) in enumerate(GRID, 1):
        print(
            f"Condition {index}/12: C_A={concentration}, T={temperature}, loaded={preloaded}",
            flush=True,
        )
        condition = {
            "C_A_mol_L": concentration,
            "initial_temperature_K": temperature,
            "preloaded": preloaded,
            "trajectories": [],
        }
        try:
            reference = independent_reference(concentration, temperature, preloaded)
            condition["reference"] = {"passed": True, **reference}
        except Exception as exc:
            reference = None
            condition["reference"] = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        for isothermal in (True, False):
            try:
                row = trajectory_row(
                    concentration,
                    temperature,
                    preloaded,
                    isothermal=isothermal,
                    reference=reference,
                )
            except Exception as exc:
                row = {
                    "isothermal": isothermal,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            condition["trajectories"].append(row)
        condition["passed"] = condition["reference"]["passed"] and all(
            row["passed"] for row in condition["trajectories"]
        )
        attempt["conditions"].append(condition)
        save()
        elapsed = time.monotonic() - start
        print(
            f"Completed {index}/12 conditions, passed={condition['passed']}, "
            f"elapsed={elapsed:.1f}s, ETA={(12 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    rows = [row for c in attempt["conditions"] for row in c["trajectories"]]
    attempt.update(
        passed_conditions=sum(c["passed"] for c in attempt["conditions"]),
        passed_trajectories=sum(row["passed"] for row in rows),
        failed_trajectories=sum(not row["passed"] for row in rows),
        elapsed_s=time.monotonic() - start,
        finished_utc=datetime.now(UTC).isoformat(),
    )
    save()
    print(
        {
            k: attempt[k]
            for k in (
                "passed_conditions",
                "passed_trajectories",
                "failed_trajectories",
                "elapsed_s",
            )
        },
        flush=True,
    )
    if attempt["passed_conditions"] != 12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
