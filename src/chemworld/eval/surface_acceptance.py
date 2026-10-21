"""Fixed development comparisons for the authored native surface apparatus.

Coverage and thresholds are defined before execution in
workstreams/world_foundation/SURFACE_ACCEPTANCE_EXPERIMENT.md. No parameter fit.
"""

from dataclasses import replace
from itertools import product

import numpy as np

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.surface_rates import (
    adsorbate_key,
    evaluate_surface_rates,
    gas_amounts_from_pressure,
    make_surface_system,
    surface_state,
)
from chemworld.physchem.surface_reference import CanteraSurfaceReference
from chemworld.physchem.surface_specs import GAS_SPECIES, make_surface_slot
from chemworld.runtime.coupled_thermal import assemble_local_system, local_thermodynamics
from chemworld.world.surface_world import surface_world_spec

TIMES_S = (0, 1, 3, 10, 30, 100, 300, 600)
TEMPERATURES_K = (300, 350, 430)
MIXTURES = ((0.1, 0.2, 0.7), (0.2, 0.1, 0.7), (0.45, 0.05, 0.5))


def acceptance_cases():
    rows = []
    for temperature, pressure, coverage in product(
        TEMPERATURES_K, (50000, 100000, 200000), ((0, 0), (0.5, 0), (0, 0.5), (0.3, 0.2))
    ):
        rows.append(
            {
                "block": "rate",
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "coverage": coverage,
                "mixture": (0.2, 0.1, 0.7),
            }
        )
    for temperature, mixture, coverage in product(
        TEMPERATURES_K, MIXTURES, ((0, 0), (0.8, 0), (0, 0.8))
    ):
        rows.append(
            {
                "block": "isothermal",
                "temperature_K": temperature,
                "pressure_Pa": 100000,
                "coverage": coverage,
                "mixture": mixture,
            }
        )
    for heater, bulk, coverage in product((None, 380, 420), (False, True), ((0, 0), (0.8, 0))):
        rows.append(
            {
                "block": "thermal",
                "heater_K": heater,
                "include_bulk": bulk,
                "temperature_K": 350,
                "pressure_Pa": 100000,
                "coverage": coverage,
                "mixture": (0.2, 0.1, 0.7),
            }
        )
    counters = dict.fromkeys(("rate", "isothermal", "thermal"), 0)
    for row in rows:
        counters[row["block"]] += 1
        row["case_id"] = f"{row['block']}-{counters[row['block']]:02}"
    return tuple(rows)


def initial_state(case):
    slot = make_surface_slot("catalyst", "reactor", initial_coverages=tuple(case["coverage"]))
    gas = gas_amounts_from_pressure(
        dict(zip(("CO", "O2", "N2"), case["mixture"], strict=True)),
        pressure_Pa=case["pressure_Pa"],
        temperature_K=case["temperature_K"],
        volume_L=0.01,
    )
    state = surface_state(gas, (slot,), temperature_K=case["temperature_K"], volume_L=0.01)
    return slot, state


def _scales(state, slot):
    gas_scale = sum(state.amounts_mol[k] for k in GAS_SPECIES)
    return {
        key: gas_scale if key in GAS_SPECIES else slot.values["sites_mol"]
        for key in state.amounts_mol
    }


def balance_report(system, result, slot):
    initial = result.initial_state
    initial_u = system.thermodynamics.internal_energy_j(initial)
    rows = []
    for state in result.trajectory:
        amounts = state.amounts_mol
        elements = {}
        for name, weights in system.conservation_laws.items():
            start = sum(initial.amounts_mol[key] * weight for key, weight in weights.items())
            final = sum(amounts[key] * weight for key, weight in weights.items())
            elements[name] = {"residual_mol": final - start, "limit_mol": 1e-12 + 1e-8 * abs(start)}
        coverage = [
            amounts[adsorbate_key(slot, k)] / slot.values["sites_mol"] for k in ("CO_ads", "O_ads")
        ]
        site_error = max(0, -min(coverage), sum(coverage) - 1)
        delta_u = system.thermodynamics.internal_energy_j(state) - initial_u
        residual = delta_u - (state.boundary_heat_J - state.heat_loss_J - state.work_J)
        limit = 1e-8 + 1e-6 * max(
            1, abs(delta_u), abs(state.boundary_heat_J) + abs(state.heat_loss_J) + abs(state.work_J)
        )
        rows.append(
            {
                "time_s": state.time_s,
                "elements": elements,
                "site_fraction_violation": site_error,
                "energy_residual_J": residual,
                "energy_limit_J": limit,
                "passed": all(abs(e["residual_mol"]) <= e["limit_mol"] for e in elements.values())
                and site_error <= 1e-10
                and abs(residual) <= limit,
            }
        )
    return rows


def compare_rates(case):
    slot, state = initial_state(case)
    reference = CanteraSurfaceReference(state, slot)
    native = np.asarray(evaluate_surface_rates(state, slot).per_site_rates_s_inv)
    expected = np.asarray(reference.per_site_rates_s_inv())
    error = np.abs(native - expected)
    return {
        "passed": bool(np.all(error <= 1e-12 + 1e-9 * np.abs(expected))),
        "reference_version": reference.version,
        "native_rates_s_inv": native.tolist(),
        "reference_rates_s_inv": expected.tolist(),
        "absolute_errors_s_inv": error.tolist(),
    }


def compare_isothermal(case):
    slot, state = initial_state(case)
    system = make_surface_system((slot,), isothermal=True)
    native = system.advance(state, 600, evaluation_times_s=TIMES_S)
    reference = CanteraSurfaceReference(state, slot)
    expected = reference.trajectory(TIMES_S)
    scales = _scales(state, slot)
    rows = []
    for actual, target in zip(native.trajectory, expected, strict=True):
        errors = {
            key: abs(actual.amounts_mol[key] - target["amounts_mol"][key]) / scales[key]
            for key in scales
        }
        passed = all(
            error <= 1e-8 + 1e-5 * abs(target["amounts_mol"][key] / scales[key])
            for key, error in errors.items()
        )
        rows.append(
            {
                "time_s": actual.time_s,
                "native_amounts_mol": dict(actual.amounts_mol),
                "reference_amounts_mol": target["amounts_mol"],
                "normalized_errors": errors,
                "passed": passed,
            }
        )
    balances = balance_report(system, native, slot)
    return {
        "passed": all(r["passed"] for r in (*rows, *balances)),
        "reference_version": reference.version,
        "samples": rows,
        "balances": balances,
        "actual_duration_s": native.actual_duration_s,
        "outcome": native.outcome,
        "stopping_event": native.stopping_event,
    }


def thermal_spec(case):
    spec = surface_world_spec(include_bulk=case["include_bulk"])
    _, initial = initial_state(case)
    state = spec.physics.initial_states["reactor"]
    old_slot = state.state_slots.slots[0]
    slot = make_surface_slot(
        old_slot.spec.owner_id, "reactor", initial_coverages=tuple(case["coverage"])
    )
    amounts = {key: initial.amounts_mol[key] for key in GAS_SPECIES}
    devices = dict(state.equipment.equipment)
    heater = devices["reactor:heater"]
    devices[heater.equipment_id] = replace(
        heater,
        settings={
            "thermal_device": ThermalDeviceSpec(
                setpoint_K=case["heater_K"] or 350,
                enabled=case["heater_K"] is not None,
            ).to_dict(),
        },
    )
    state = state.replace(
        species_amounts=amounts,
        species=None,
        state_slots=StateSlots((slot,)),
        equipment=EquipmentLedger(devices),
        phases=PhaseLedger(
            {
                name: replace(phase, species_amounts_mol=amounts)
                for name, phase in state.phases.phases.items()
            }
        ),
    )
    return replace(
        spec,
        physics=replace(
            spec.physics,
            initial_states={
                **spec.physics.initial_states,
                "reactor": state,
            },
        ),
    )


def compare_thermal(case):
    spec = thermal_spec(case)
    env = ChemWorldEnv(composition=spec)
    env.reset(seed=0)
    try:
        initial_world = env._state
        slot = initial_world.state_slots.slots[0]
        system, state = assemble_local_system(
            initial_world, spec.physics, spec.modules, spec.components
        )
        native = system.advance(state, 600, evaluation_times_s=TIMES_S)
        tighter = replace(
            system,
            policy=replace(
                system.policy,
                rtol=system.policy.rtol / 10,
                atol=system.policy.atol / 10,
            ),
        ).advance(state, 600, evaluation_times_s=TIMES_S)
        scales = _scales(state, slot)
        differences = []
        for a, b in zip(native.trajectory, tighter.trajectory, strict=True):
            differences.extend(abs(a.amounts_mol[k] - b.amounts_mol[k]) / scales[k] for k in scales)
            differences.append(abs(a.temperature_K - b.temperature_K) / 350)
        _, _, _, _, info = env.step(
            {"operation": "wait", "duration_s": 600, "stirring_speed_rpm": 600}
        )
        _, actual = local_thermodynamics(env._state, spec.physics.models_of("thermal")["reactor"])
        endpoint_error = max(
            *(
                abs(actual.amounts_mol[k] - native.final_state.amounts_mol[k]) / scales[k]
                for k in scales
            ),
            abs(actual.temperature_K - native.final_state.temperature_K) / 350,
        )
        balances = balance_report(system, native, slot)
        return {
            "passed": all(row["passed"] for row in balances)
            and max(differences) <= 1e-6
            and endpoint_error <= 1e-8
            and native.outcome == tighter.outcome
            and info["transaction_status"] == "committed"
            and info["operation_outcome"] == native.outcome
            and abs(info["actual_duration_s"] - native.actual_duration_s) <= 1e-6,
            "balances": balances,
            "max_normalized_convergence_error": max(differences),
            "max_runtime_endpoint_error": endpoint_error,
            "actual_duration_s": native.actual_duration_s,
            "outcome": native.outcome,
            "stopping_event": native.stopping_event,
            "samples": [
                {
                    "time_s": s.time_s,
                    "temperature_K": s.temperature_K,
                    "amounts_mol": dict(s.amounts_mol),
                    "heater_J": s.boundary_heat_J,
                    "loss_J": s.heat_loss_J,
                }
                for s in native.trajectory
            ],
        }
    finally:
        env.close()


def evaluate_case(case):
    evaluators = {
        "rate": compare_rates,
        "isothermal": compare_isothermal,
        "thermal": compare_thermal,
    }
    return {"case": case, **evaluators[case["block"]](case)}


def summarize_cases(cases, rows):
    """Unavailable references remain uncompleted units, never successful skips."""
    result = {}
    for block in ("rate", "isothermal", "thermal"):
        selected = [r for r in rows if r["case"]["block"] == block]
        unavailable = sum(r.get("status") == "unavailable" for r in selected)
        result[block] = {
            "expected": sum(c["block"] == block for c in cases),
            "attempted": len(selected),
            "completed": len(selected) - unavailable,
            "passed": sum(r["passed"] for r in selected),
            "failed": sum(not r["passed"] for r in selected) - unavailable,
            "unavailable": unavailable,
        }
    result.update(skipped=0, all_passed=len(rows) == len(cases) and all(r["passed"] for r in rows))
    return result


def compact_report(report):
    """One readable machine summary; detailed time series remain in the raw output."""
    rows = []
    for item in report["cases"]:
        row = {
            key: item[key]
            for key in (
                "case",
                "passed",
                "status",
                "exception",
                "reason",
                "reference_version",
                "actual_duration_s",
                "outcome",
                "stopping_event",
                "max_normalized_convergence_error",
                "max_runtime_endpoint_error",
            )
            if key in item
        }
        if "absolute_errors_s_inv" in item:
            row["max_absolute_rate_error_s_inv"] = max(item["absolute_errors_s_inv"])
        if item["case"]["block"] == "isothermal":
            row["sample_count"] = len(item.get("samples", ()))
            row["max_normalized_reference_error"] = max(
                (max(s["normalized_errors"].values()) for s in item.get("samples", ())),
                default=None,
            )
        balances = item.get("balances", ())
        if balances:
            row["max_absolute_element_residual_mol"] = max(
                abs(e["residual_mol"]) for b in balances for e in b["elements"].values()
            )
            row["max_absolute_energy_residual_J"] = max(
                abs(b["energy_residual_J"]) for b in balances
            )
            row["max_site_fraction_violation"] = max(b["site_fraction_violation"] for b in balances)
        rows.append(row)
    result = {key: value for key, value in report.items() if key not in {"cases", "summary"}}
    result.update(
        cases=rows,
        summary=summarize_cases(
            [
                case
                for case in acceptance_cases()
                if report["summary"].get(case["block"], {}).get("expected", 0)
            ],
            rows,
        ),
        experiment_note="workstreams/world_foundation/SURFACE_ACCEPTANCE_EXPERIMENT.md",
    )
    result["summary"]["elapsed_s"] = report["summary"]["elapsed_s"]
    return result
