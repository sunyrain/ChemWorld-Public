"""Fixed 34D development qualification; all units and failures remain visible."""

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import cantera as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chemworld.physchem.coupled_system import CoupledState, CoupledSystem, HeatFlux
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.molecular_reaction import (
    mechanism_caloric_model,
    molecular_reaction_process,
    molecular_reaction_rates,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.nonideal_mechanism_world import nonideal_mechanism_world
from reference import nonideal34_cases as cases
from reference import nonideal34_reference as reference

POLICY = ODESolverPolicy("nonideal34-reference", method="BDF", rtol=1e-11, atol=1e-13)
INITIAL = {"A": 1.0, "B": 0.3, "C": 0.2}


def check(row, name, actual, expected, *, atol=1e-9, rtol=2e-8):
    a, b = np.array(actual), np.array(expected)
    tolerance = atol + rtol * np.abs(b)
    error = np.abs(a - b)
    row["comparisons"].append(
        {
            "name": name,
            "actual": a.tolist(),
            "expected": b.tolist(),
            "atol": atol,
            "rtol": rtol,
            "maximum_error": float(error.max()),
            "passed": bool(np.all(error <= tolerance)),
        }
    )


def exact(row, name, passed):
    check(row, name, int(passed), 1, atol=0, rtol=0)


def native_advance(network, thermo, *, power=0.0, initial=None):
    initial = initial or CoupledState(INITIAL, 600.0, 1.0)
    return CoupledSystem(
        thermo,
        processes=(molecular_reaction_process(network, thermo),),
        heat_flux=lambda _: HeatFlux(power),
        policy=POLICY,
        conservation_laws={"H": {"A": 2.0, "B": 2.0, "C": 4.0}},
    ).advance(initial, 0.3)


def phase_unit(row, family, index):
    temperature, rho, fractions, interaction = (
        (600.0, 1500.0, (2 / 3, 0.2, 2 / 15), 0.0),
        (900.0, 4000.0, (0.25, 0.5, 0.25), 0.025),
        (700.0, 800.0, (1.0, 0.0, 0.0), 0.0),
    )[index]
    amounts = dict(zip(("A", "B", "C"), (x * rho / 1000 for x in fractions), strict=True))
    network, thermo = cases.inputs(family, interaction=interaction)
    initial = CoupledState(amounts, temperature, 1.0)
    state = thermo.molar_state(initial)
    expected = reference.properties(
        reference.phase(family, temperature, amounts, interaction=interaction)
    )
    activity = (
        np.array(list(amounts.values()))
        * state.compressibility_factor
        * np.exp(state.log_fugacity_coefficients)
    )
    actual = {
        "P": state.pressure_Pa,
        "U": state.internal_energy_J_mol,
        "H": state.enthalpy_J_mol,
        "S": state.entropy_J_mol_K,
        "Cv": state.cv_J_mol_K,
        "Cp": state.cp_J_mol_K,
        "activity": activity,
        "rates": list(molecular_reaction_rates(network, thermo, initial).values()),
    }
    for key in actual:
        check(
            row,
            key,
            actual[key],
            expected[key],
            atol={
                "P": 1e-6,
                "U": 1e-7,
                "H": 1e-7,
                "S": 1e-9,
                "Cv": 1e-9,
                "Cp": 1e-9,
                "activity": 1e-10,
                "rates": 1e-10,
            }[key],
        )


def trajectory_unit(row, family):
    network, thermo = cases.inputs(family, multiple=family == "constant-cp")
    result = native_advance(network, thermo)
    expected = reference.trajectory(family)
    final = result.final_state
    check(row, "T", final.temperature_K, expected["T"], atol=2e-6, rtol=0)
    check(row, "n", list(final.amounts_mol.values()), expected["n"], atol=2e-8, rtol=0)
    check(row, "P", thermo.pressure_pa(final), expected["P"], rtol=2e-7)
    check(row, "U_closure", result.energy_balance_residual_J, 0.0, atol=2e-6, rtol=0)
    row["resources"] = {"native": dict(result.solver_diagnostic), "cantera": expected["resources"]}


def raw_unit(row):
    network, thermo = cases.inputs()
    molar = thermo.molar_state(CoupledState(INITIAL, 600.0, 1.0))
    raw = reference.raw_pr(600.0, list(INITIAL.values()))
    check(row, "P", molar.pressure_Pa, raw["P"], atol=1e-6)
    check(row, "U", molar.internal_energy_J_mol, raw["U"], atol=1e-7)
    expected = reference.extent_trajectory()
    actual = native_advance(network, thermo, power=20.0)
    # Scale the temperature component so the common absolute tolerance is 2e-8 mol / 2e-6 K.
    check(
        row,
        "heated_extent_n_T",
        [*actual.final_state.amounts_mol.values(), actual.final_state.temperature_K / 100],
        [*expected["n"], expected["T"] / 100],
        atol=2e-8,
        rtol=0,
    )
    row["resources"] = {
        "native": dict(actual.solver_diagnostic),
        "reference_nfev": expected["nfev"],
    }


def dilute_unit(row):
    network, thermo = cases.inputs()
    initial = CoupledState({k: v / 1.5 * 1e-7 for k, v in INITIAL.items()}, 600.0, 1.0)
    check(row, "Z", thermo.molar_state(initial).compressibility_factor, 1.0, atol=1e-7, rtol=0)
    check(
        row,
        "ideal_rates",
        list(molecular_reaction_rates(network, thermo, initial).values()),
        list(
            network.reaction_rates(initial.amounts_mol, volume_L=1.0, temperature_K=600.0).values()
        ),
        rtol=2e-7,
        atol=1e-15,
    )


def equilibrium_unit(row):
    network, thermo = cases.inputs(interaction=0.025)
    gas = reference.phase("constant-cp", 600.0, INITIAL, interaction=0.025)
    gas.equilibrate("TV", rtol=1e-12)
    state = CoupledState(dict(zip(gas.species_names, gas.concentrations, strict=True)), gas.T, 1.0)
    check(
        row,
        "zero_rates",
        list(molecular_reaction_rates(network, thermo, state).values()),
        [0.0, 0.0],
        atol=1e-9,
        rtol=0,
    )
    check(row, "P", thermo.pressure_pa(state), gas.P)


def shift_unit(row):
    network, thermo = cases.inputs()
    source = cases.source()
    for index, species in enumerate(source["species"]):
        species["thermo"]["h0"] = (
            f"{-10000 - 2000 * index + 7000 * species['composition']['H']} J/mol"
        )
    shifted = convert_mechanism(source)
    shifted_thermo = replace(thermo, model=mechanism_caloric_model(shifted, cases.residual()))
    a, b = native_advance(network, thermo), native_advance(shifted, shifted_thermo)
    check(
        row,
        "n",
        list(a.final_state.amounts_mol.values()),
        list(b.final_state.amounts_mol.values()),
        atol=2e-9,
        rtol=0,
    )
    check(row, "T", a.final_state.temperature_K, b.final_state.temperature_K, atol=2e-6, rtol=0)
    initial = CoupledState(INITIAL, 600.0, 1.0)
    check(
        row,
        "extensive_H_offset",
        shifted_thermo.internal_energy_j(initial) - thermo.internal_energy_j(initial),
        7000 * (2 * 1.0 + 2 * 0.3 + 4 * 0.2),
        atol=2e-6,
        rtol=0,
    )


def world_unit(row, shared):
    network, thermo = cases.inputs()
    world = nonideal_mechanism_world(network, cases.residual(), INITIAL, shared_clock=shared)
    runtime = ChemWorldRuntime(composition=world)
    restored = None
    try:
        runtime.reset(seed=34)
        before = CoupledState(INITIAL, runtime._state.temperature_K, 1.0)
        first = runtime.execute(
            {
                "operation": "queue_lab_heat" if shared else "heat",
                "target_temperature_K": 650.0,
                "duration_s": 1.0 if shared else 0.3,
            }
        )
        second = runtime.execute({"operation": "wait", "duration_s": 0.3}) if shared else first
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        action = {"operation": "wait", "duration_s": 0.3}
        result = runtime.execute(action)
        repeated = restored.execute(action)
        exact(
            row,
            "all_actions_committed",
            all(
                r.info["transaction_status"] == "committed"
                for r in (first, second, result, repeated)
            ),
        )
        exact(row, "full_JSON_continuation", runtime._state.to_dict() == restored._state.to_dict())
        state = runtime._state
        exact(
            row,
            "phase_inventory",
            next(iter(state.phases.phases.values())).species_amounts_mol == state.species_amounts,
        )
        exact(
            row,
            "global_local_clock",
            state.ledger.time_s == 0.6 and state.vessel_elapsed_s["reactor"] == 0.6,
        )
        final = CoupledState(state.species_amounts, state.temperature_K, state.volume_L)
        check(
            row,
            "U_heat_receipts",
            thermo.internal_energy_j(final) - thermo.internal_energy_j(before),
            state.ledger.energy_jacket_J - state.ledger.heat_loss_J,
            atol=2e-6,
            rtol=0,
        )
        row["resources"] = {
            "time_s": state.ledger.time_s,
            "jacket_J": state.ledger.energy_jacket_J,
            "last_outcome": result.info["outcome_diagnostics"],
        }
    finally:
        runtime.close()
        if restored:
            restored.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    units = [
        (f"phase:{family}:{i}", 8, phase_unit, (family, i))
        for family in cases.THERMOS
        for i in range(3)
    ]
    units += [(f"reactor:{family}", 4, trajectory_unit, (family,)) for family in cases.THERMOS]
    units += [
        ("raw_PR_extent_U", 3, raw_unit, ()),
        ("dilute", 2, dilute_unit, ()),
        ("equilibrium", 2, equilibrium_unit, ()),
        ("reference_shift", 3, shift_unit, ()),
        ("world:direct", 5, world_unit, (False,)),
        ("world:shared", 5, world_unit, (True,)),
    ]
    rows = []
    for identity, expected_count, function, parameters in units:
        row = {"id": identity, "expected_comparisons": expected_count, "comparisons": []}
        before = time.perf_counter()
        try:
            function(row, *parameters)
        except Exception:
            row["exception"] = traceback.format_exc()
        row["wall_s"] = time.perf_counter() - before
        row["passed"] = (
            len(row["comparisons"]) == expected_count
            and all(c["passed"] for c in row["comparisons"])
            and "exception" not in row
        )
        rows.append(row)
        elapsed = time.perf_counter() - start
        print(
            f"34D {len(rows)}/{len(units)} {identity}: {'PASS' if row['passed'] else 'FAIL'}; "
            f"{len(rows) / elapsed:.2f} units/s; "
            f"ETA {(len(units) - len(rows)) * elapsed / len(rows):.1f}s",
            flush=True,
        )
    checked = [c for row in rows for c in row["comparisons"]]
    summary = {
        "mode": "development",
        "cantera_version": ct.__version__,
        "units_total": 26,
        "units_passed": sum(r["passed"] for r in rows),
        "comparisons_total": 160,
        "comparisons_evaluated": len(checked),
        "comparisons_passed": sum(c["passed"] for c in checked),
        "wall_s": time.perf_counter() - start,
        "all_units": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "all_units"}), flush=True)
    return 0 if summary["units_passed"] == 26 and summary["comparisons_passed"] == 160 else 1


if __name__ == "__main__":
    raise SystemExit(main())
