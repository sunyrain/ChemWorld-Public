"""Fixed 30-unit energy-grain development qualification; all failures retained."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from math import exp, isclose
from pathlib import Path

import energy_grain38_cases as cases
import energy_grain38_reference as oracle
import numpy as np

from chemworld.physchem.energy_grain_binding import compile_energy_grain_network, energy_grain_world
from chemworld.physchem.energy_grain_master_equation import (
    boltzmann_grain_fractions,
    collision_transition_probabilities,
    energy_grain_generator,
    grain_population_report,
    well_collision_generator,
)
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, kind="exact"):
        if hasattr(actual, "item"):
            actual = actual.item()
        if hasattr(expected, "item"):
            expected = expected.item()
        tolerance = {
            "rate": (2e-11, 2e-12),
            "amount": (3e-6, 2e-10),
            "heat": (5e-6, 2e-6),
            "balance": (0, 2e-10),
            "equilibrium": (3e-6, 2e-8),
        }
        passed = (
            actual == expected
            if kind == "exact"
            else isclose(actual, expected, rel_tol=tolerance[kind][0], abs_tol=tolerance[kind][1])
        )
        self.checks.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "kind": kind,
                "passed": bool(passed),
            }
        )


def model_and_reactor(card, temperature):
    model = EnergyGrainMasterEquationSpec.from_dict(card)
    network = compile_energy_grain_network(model)
    reactor = GasReactorNetworkSpec((GasReactorNodeSpec("cell", 1.0),), (), temperature, "cell")
    return model, network, reactor


def compare_generator(record, card, model, temperature, amounts):
    baths = {b["species_id"]: amounts[b["species_id"]] for b in card["bath_gases"]}
    expected_names, expected = oracle.generator(card, temperature, baths)
    actual = energy_grain_generator(
        model, temperature_K=temperature, bath_concentrations_mol_L=baths
    )
    record.check("molecular-state-order", list(actual["species_ids"]), expected_names)
    for i, name in enumerate(expected_names):
        for j, source in enumerate(expected_names):
            record.check(
                f"generator:{source}->{name}",
                actual["generator_s_inverse"][i][j],
                expected[i, j],
                "rate",
            )
        record.check(
            f"generator-column-balance:{name}",
            sum(row[i] for row in actual["generator_s_inverse"]),
            0,
            "rate",
        )
    for well in model.wells:
        energies = np.array([g.energy_J_mol for g in well.grains])
        weights = np.array([g.statistical_weight for g in well.grains])
        equilibrium = weights * np.exp(-(energies - min(energies)) / (cases.R * temperature))
        equilibrium /= sum(equilibrium)
        native_pi = boltzmann_grain_fractions(well, temperature)
        for i, grain in enumerate(well.grains):
            record.check(
                f"equilibrium:{grain.species_id}",
                native_pi[grain.species_id],
                equilibrium[i],
                "rate",
            )
        for bath in well.baths:
            probability = collision_transition_probabilities(well, bath, temperature)
            prefix = f"{well.well_id}:{bath.bath_species_id}"
            record.check(
                f"probability-nonnegative:{prefix}",
                min(x for row in probability for x in row) >= 0,
                True,
            )
            for i in range(len(well.grains)):
                record.check(
                    f"probability-column:{prefix}:{i}",
                    sum(row[i] for row in probability),
                    1,
                    "rate",
                )
                for j in range(i):
                    record.check(
                        f"detailed-balance:{prefix}:{i}:{j}",
                        equilibrium[i] * probability[j][i],
                        equilibrium[j] * probability[i][j],
                        "rate",
                    )


def compare_trajectory(record, card, temperature=600.0, pressure=1e4, times=cases.TIMES):
    model, network, reactor = model_and_reactor(card, temperature)
    amounts = cases.initial_amounts(card, temperature, pressure)
    compare_generator(record, card, model, temperature, amounts)
    reference = oracle.trajectory(card, temperature, amounts, times)
    state = pack_gas_reactor_network_state(reactor, network, {"cell": amounts})
    h = oracle.enthalpies(card, temperature)
    original_u = sum(n * (h[name] - cases.R * temperature) for name, n in amounts.items())
    record.evidence.update(
        {
            "card": card,
            "temperature_K": temperature,
            "bath_pressure_Pa": pressure,
            "initial_amounts_mol": amounts,
            "samples": [],
        }
    )
    final_amounts = None
    for t, expected in zip(times, reference, strict=True):
        result, diagnostic = advance_gas_reactor_network(reactor, network, state, t - state.time_s)
        state = result.final_state
        actual = diagnostic["nodes"]["cell"]["amounts_mol"]
        for name in network.species_ids:
            record.check(f"amount:{name}:t{t}", actual[name], expected[name], "amount")
        independent_u = sum(n * (h[name] - cases.R * temperature) for name, n in expected.items())
        record.check(
            f"cumulative-Q:t{t}", state.boundary_heat_J, independent_u - original_u, "heat"
        )
        record.check(
            f"local-energy-residual:t{t}",
            diagnostic["nodes"]["cell"]["energy_residual_J"],
            0,
            "heat",
        )
        record.check(f"group-energy-residual:t{t}", diagnostic["energy_residual_J"], 0, "heat")
        record.check(
            f"molecular-balance:t{t}",
            grain_population_report(model, actual)["molecular_amount_mol"],
            0.01,
            "balance",
        )
        for name, residual in result.conservation_residuals.items():
            record.check(f"element-balance:{name}:t{t}", residual, 0, "balance")
        record.check(f"actual-clock:t{t}", state.time_s, t, "rate")
        record.evidence["samples"].append(
            {
                "time_s": t,
                "amounts_mol": actual,
                "bath_heat_J": state.boundary_heat_J,
                "solver": dict(result.solver_diagnostic),
            }
        )
        final_amounts = actual
    return model, network, reactor, amounts, state, final_amounts


def two_level(record, temperature):
    card = cases.declaration(topology="two-level")
    _, network, reactor, initial, _, _ = compare_trajectory(record, card, temperature)
    _, matrix = oracle.generator(card, temperature, {k: initial[k] for k in ("He", "Ar")})
    up, down = matrix[1, 0], matrix[0, 1]
    relaxation = up + down
    times = (*cases.TIMES, 50 / relaxation)
    state = pack_gas_reactor_network_state(reactor, network, {"cell": initial})
    for t in times:
        result, diagnostic = advance_gas_reactor_network(reactor, network, state, t - state.time_s)
        state = result.final_state
        high = 0.01 * (up / relaxation + (1 - up / relaxation) * exp(-relaxation * t))
        actual = diagnostic["nodes"]["cell"]["amounts_mol"]
        record.check(f"two-level-analytic-high:{t}", actual["A1"], high, "amount")
        record.check(f"two-level-analytic-low:{t}", actual["A0"], 0.01 - high, "amount")
    record.check("long-time-Boltzmann", actual["A1"] / 0.01, up / relaxation, "equilibrium")
    record.evidence["long_time_s"] = times[-1]


def mixed_bath(record):
    card = cases.declaration()
    model, _, _, amounts, _, _ = compare_trajectory(record, card)
    well = model.wells[0]
    together = well_collision_generator(well, temperature_K=600, concentrations_mol_L=amounts)
    separate = [
        well_collision_generator(well, temperature_K=600, concentrations_mol_L={k: amounts[k]})
        for k in ("He", "Ar")
    ]
    for i in range(3):
        for j in range(3):
            record.check(
                f"bath-additivity:{i}:{j}", together[i][j], sum(m[i][j] for m in separate), "rate"
            )


def exact_three_state(record):
    card = cases.declaration(topology="chain")
    compare_trajectory(record, card, pressure=0)
    _, network, reactor = model_and_reactor(card, 600)
    amounts = cases.initial_amounts(card, 600, 0)
    initial = pack_gas_reactor_network_state(reactor, network, {"cell": amounts})
    for t in cases.TIMES:
        _, diagnostic = advance_gas_reactor_network(reactor, network, initial, t)
        actual = diagnostic["nodes"]["cell"]["amounts_mol"]
        surviving = 0.01 * exp(-2.7 * t)
        for name, expected in {
            "A2": surviving,
            "P": (0.01 - surviving) * 0.7 / 2.7,
            "Q": (0.01 - surviving) * 2 / 2.7,
        }.items():
            record.check(f"exact-branch:{name}:t{t}", actual[name], expected, "amount")


def flow_setup():
    card, amounts, temperature, flow, duration = cases.flow_inputs()
    model = EnergyGrainMasterEquationSpec.from_dict(card)
    reactor = GasReactorNetworkSpec(
        (GasReactorNodeSpec("feed", 1, chemistry_enabled=False), GasReactorNodeSpec("reactor", 1)),
        (GasFlowConnectionSpec("bath-feed", "feed", "reactor", coefficient=flow),),
        temperature,
        "reactor",
    )
    return card, model, reactor, amounts, flow, duration


def flowing_bath(record):
    card, model, reactor, amounts, flow, duration = flow_setup()
    network = compile_energy_grain_network(model)
    state = pack_gas_reactor_network_state(reactor, network, amounts)
    result, diagnostic = advance_gas_reactor_network(reactor, network, state, duration)
    expected = oracle.finite_flow(card, amounts, reactor.temperature_K, flow, duration)
    for name, local in diagnostic["nodes"].items():
        for species in network.species_ids:
            record.check(
                f"flow-amount:{name}:{species}",
                local["amounts_mol"][species],
                expected["amounts"][name][species],
                "amount",
            )
        record.check(
            f"node-heat:{name}", local["bath_heat_J"], expected["bath_heat_J"][name], "heat"
        )
        record.check(f"node-energy-residual:{name}", local["energy_residual_J"], 0, "heat")
    record.check(
        "internal-transferred-H",
        diagnostic["connections"]["bath-feed"]["enthalpy_J"],
        expected["transferred_enthalpy_J"],
        "heat",
    )
    record.check(
        "global-Q",
        result.final_state.boundary_heat_J,
        sum(expected["bath_heat_J"].values()),
        "heat",
    )
    for name in ("He", "Ar"):
        record.check(
            f"finite-bath-balance:{name}",
            sum(n["amounts_mol"][name] for n in diagnostic["nodes"].values()),
            sum(a[name] for a in amounts.values()),
            "balance",
        )
    record.check(
        "actual-pressure-increased",
        diagnostic["nodes"]["reactor"]["pressure_Pa"]
        > sum(amounts["reactor"].values()) * cases.R * 600 * 1000,
        True,
    )
    record.evidence.update({"native": diagnostic, "independent": expected, "initial": amounts})


def runtime_replay(record, directory, fresh_process=False):
    card, model, reactor, amounts, flow, _ = flow_setup()
    runtime = ChemWorldRuntime(composition=energy_grain_world(model, reactor, amounts))
    restored = None
    try:
        runtime.reset(seed=38)
        first = runtime.execute({"operation": "wait", "duration_s": 0.2})
        record.check("first-shared-wait", first.info["transaction_status"], "committed")
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        second = runtime.execute({"operation": "wait", "duration_s": 0.3})
        record.check("second-shared-wait", second.info["transaction_status"], "committed")
        if fresh_process:
            snapshot_path, result_path = (
                directory / "snapshot.json",
                directory / "resumed-state.json",
            )
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            code = (
                "import json,sys; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                "payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "r=ChemWorldRuntime.from_snapshot(payload); "
                "outcome=r.execute({'operation':'wait','duration_s':.3}); "
                "assert outcome.info['transaction_status']=='committed',outcome.info; "
                "Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict()),encoding='utf-8'); "
                "r.close()"
            )
            process = subprocess.run(
                [sys.executable, "-c", code, str(snapshot_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=45,
            )
            record.check("fresh-process-exit", process.returncode, 0)
            record.evidence["fresh_process_stderr"] = process.stderr
            replayed = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            restored = ChemWorldRuntime.from_snapshot(snapshot)
            result = restored.execute({"operation": "wait", "duration_s": 0.3})
            record.check("restored-shared-wait", result.info["transaction_status"], "committed")
            replayed = json.loads(json.dumps(restored._state.to_dict()))
        actual = json.loads(json.dumps(runtime._state.to_dict()))
        record.check("complete-raw-state-exact", replayed, actual)
        expected = oracle.finite_flow(card, amounts, 600, flow, 0.5)
        for name, state in local_states(runtime._state).items():
            for species, amount in state.species_amounts.items():
                record.check(
                    f"world-amount:{name}:{species}",
                    amount,
                    expected["amounts"][name][species],
                    "amount",
                )
            record.check(
                f"original-phase-owner:{name}",
                state.phases.phases["gas"].species_amounts_mol,
                state.species_amounts,
            )
            record.check(
                f"world-Q:{name}",
                state.thermal.vessels[name].energy_jacket_J,
                expected["bath_heat_J"][name],
                "heat",
            )
        record.check("shared-clock", runtime._state.ledger.time_s, 0.5)
        record.check(
            "all-vessel-clocks", runtime._state.vessel_elapsed_s, {"feed": 0.5, "reactor": 0.5}
        )
        record.check(
            "world-group-Q",
            runtime._state.ledger.energy_jacket_J,
            sum(expected["bath_heat_J"].values()),
            "heat",
        )
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def pressure_branching(record):
    card = cases.declaration(alpha=6000)
    yields = []
    for pressure in cases.PRESSURES:
        _, _, _, _, _, final = compare_trajectory(record, card, pressure=pressure)
        yields.append(final["P"])
    record.check("nontrivial-pressure-branching", max(yields) - min(yields) > 1e-4, True)
    record.evidence["pressure_Pa"] = list(cases.PRESSURES)
    record.evidence["product_P_yield_mol"] = yields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("retain each prior attempt; choose new output and summary names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / args.output.stem
    work.mkdir(exist_ok=False)
    units = []
    for temperature in cases.TEMPERATURES:
        for pressure in cases.PRESSURES:
            for alpha in cases.ALPHAS:
                units.append(
                    (
                        f"grid-T{temperature}-P{pressure}-alpha{alpha}",
                        lambda r, t=temperature, p=pressure, a=alpha: compare_trajectory(
                            r, cases.declaration(a), t, p
                        ),
                    )
                )
    for temperature in cases.TEMPERATURES:
        units.append(
            (f"analytic-two-level-T{temperature}", lambda r, t=temperature: two_level(r, t))
        )
    units.extend(
        [
            ("zero-bath", lambda r: compare_trajectory(r, cases.declaration(), pressure=0)),
            ("mixed-bath-additivity", mixed_bath),
            ("two-well", lambda r: compare_trajectory(r, cases.declaration(topology="two-well"))),
            ("six-grain", lambda r: compare_trajectory(r, cases.declaration(topology="six-grain"))),
            ("exact-three-state-branches", exact_three_state),
            ("finite-bath-flow", flowing_bath),
            ("world-json-replay", lambda r: runtime_replay(r, work)),
            ("fresh-process-replay", lambda r: runtime_replay(r, work, True)),
            ("pressure-dependent-branching", pressure_branching),
        ]
    )
    assert len(units) == 30
    started, reports = time.monotonic(), []
    for index, (name, function) in enumerate(units, 1):
        record, error = Record(name), None
        unit_start = time.monotonic()
        try:
            function(record)
        except Exception:
            error = traceback.format_exc()
        reports.append(
            {
                "unit": name,
                "comparisons": record.checks,
                "evidence": record.evidence,
                "error": error,
                "elapsed_s": time.monotonic() - unit_start,
                "passed": error is None and all(c["passed"] for c in record.checks),
            }
        )
        elapsed = time.monotonic() - started
        print(
            f"38B {index}/30 {name} {'PASS' if reports[-1]['passed'] else 'FAIL'} "
            f"checks={len(record.checks)} throughput={index / elapsed:.2f} units/s "
            f"ETA={(30 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    report = {
        "mode": "development",
        "planned_units": 30,
        "completed_units": len(reports),
        "passed_units": sum(r["passed"] for r in reports),
        "executed_comparisons": sum(len(r["comparisons"]) for r in reports),
        "passed_comparisons": sum(c["passed"] for r in reports for c in r["comparisons"]),
        "elapsed_s": time.monotonic() - started,
        "units": reports,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    compact = {key: value for key, value in report.items() if key != "units"}
    compact["full_evidence"] = str(args.output)
    compact["units"] = []
    for result in reports:
        checks = result["comparisons"]
        discrepancies = {}
        for kind in ("rate", "amount", "heat", "balance", "equilibrium"):
            values = [abs(c["actual"] - c["expected"]) for c in checks if c["kind"] == kind]
            if values:
                discrepancies[kind] = max(values)
        compact["units"].append(
            {
                "unit": result["unit"],
                "passed": result["passed"],
                "comparison_count": len(checks),
                "passed_comparisons": sum(c["passed"] for c in checks),
                "failures": [c for c in checks if not c["passed"]],
                "error": result["error"],
                "elapsed_s": result["elapsed_s"],
                "maximum_absolute_discrepancies": discrepancies,
            }
        )
    compact["pressure_branching"] = reports[-1]["evidence"].get("product_P_yield_mol")
    args.summary.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in compact.items() if k != "units"}), flush=True)
    return 0 if report["passed_units"] == 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
