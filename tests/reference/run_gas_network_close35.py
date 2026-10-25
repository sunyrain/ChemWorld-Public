"""Fixed 23-unit original gas-network development block; preserve every attempt."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import cantera as ct
import gas_network_close35_cases as cases
import gas_network_close35_reference as reference
import numpy as np
import yaml
from mechanism34_cases import THERMOS, payload

from chemworld.physchem.gas_network_energy import gas_loss_key
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_network_sensitivity import gas_network_trajectory_sensitivities
from chemworld.physchem.gas_network_steady import gas_network_steady_state
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    advance_gas_reactor_network,
    gas_inventory_key,
    gas_node_view,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_thermochemistry_parameters import (
    GasThermochemistryParameter,
    species_enthalpy_parameter,
    with_species_enthalpy,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.residual_system import RootPolicy


def snapshot(state):
    from collections.abc import Mapping
    from dataclasses import fields

    return {
        f.name: dict(v) if isinstance(v, Mapping) else v
        for f in fields(state)
        for v in (getattr(state, f.name),)
    }


class Unit:
    def __init__(self, name):
        self.name, self.checks, self.artifacts = name, [], {}

    def check(self, name, actual, expected, atol=0.0, rtol=0.0):
        a, b = np.asarray(actual), np.asarray(expected)
        passed = a.shape == b.shape and bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))
        self.checks.append(
            {
                "name": name,
                "passed": passed,
                "actual": a.tolist(),
                "expected": b.tolist(),
                "atol": atol,
                "rtol": rtol,
            }
        )

    def yes(self, name, value):
        self.check(name, int(bool(value)), 1)


def source(unit, index):
    model = tuple(THERMOS)[index // 2]
    delta = 25.0 if index % 2 == 0 else -25.0
    raw = payload(thermo=model)
    network = convert_mechanism(raw)
    before = deepcopy(network.to_dict())
    parameter = GasThermochemistryParameter("A", 10)
    varied = with_species_enthalpy(
        network, parameter, species_enthalpy_parameter(network, parameter) + delta
    )
    temps = (350.0, 700.0, 1200.0)

    def evaluate(n):
        rows = []
        for t in temps:
            q = n.imported_gas_thermochemistry["A"].evaluate(t)
            rows.append([q.enthalpy_J_mol, q.gibbs_J_mol, q.cp_J_mol_K, q.entropy_J_mol_K])
        return np.asarray(rows)

    native, original = evaluate(varied), evaluate(network)
    independent = reference.thermochemistry(reference.shift(raw, "A", delta), temps)
    for i, name in enumerate(("H", "G", "Cp", "S")):
        unit.check(
            "source_" + name,
            native[:, i] - original[:, i],
            np.full(3, delta if i < 2 else 0),
            1e-6 if i < 2 else 1e-8,
            1e-9,
        )
        unit.check(
            "Cantera_" + name, native[:, i], independent[:, i], 1e-6 if i < 2 else 1e-8, 1e-9
        )
    assert network.to_dict() == before
    unit.artifacts.update(model=model, shift_J_mol=delta, immutable=True)


def derivative(unit, index):
    raw, spec, network, amounts = cases.closed(bool(index))
    state = pack_gas_reactor_network_state(spec, network, amounts)
    before = deepcopy(snapshot(state))
    parameters = tuple(GasThermochemistryParameter(s, 10) for s in ("A", "B"))
    result = gas_network_trajectory_sensitivities(spec, network, state, parameters, cases.TIMES)
    unit.artifacts["native"] = result
    native, expected = (
        {k: [] for k in ("amount", "temperature", "delta_U", "bath", "pressure")},
        {k: [] for k in ("amount", "temperature", "delta_U", "bath", "pressure")},
    )
    for p, response in zip(parameters, result["responses"], strict=True):
        plus = reference.trajectory(
            reference.shift(raw, p.species_id, 1), spec.to_dict(), amounts, cases.TIMES
        )
        minus = reference.trajectory(
            reference.shift(raw, p.species_id, -1), spec.to_dict(), amounts, cases.TIMES
        )
        for row, a, b in zip(response["records"], plus, minus, strict=True):
            for field, refkey in (
                ("amount", "amounts"),
                ("temperature", "temperatures"),
                ("delta_U", "delta_U"),
                ("bath", "bath"),
                ("pressure", "pressures"),
            ):
                if field == "amount":
                    for s in ("A", "B"):
                        native[field].append(row["coefficients"][f"node:r:amount:{s}"]["dy_dp"])
                        expected[field].append((a[refkey]["r"][s] - b[refkey]["r"][s]) / 2)
                else:
                    native[field].append(row["coefficients"][f"node:r:{field}"]["dy_dp"])
                    expected[field].append((a[refkey]["r"] - b[refkey]["r"]) / 2)
    for field in native:
        unit.check(
            field,
            native[field],
            expected[field],
            2e-9 if field == "amount" else 2e-3 if field == "pressure" else 2e-5,
            2e-3,
        )
    unit.yes("complete", all(c["status"] == "complete" for c in result["trajectories"]))
    unit.yes("immutable", snapshot(state) == before)
    unit.yes("work", result["work"]["reported_nfev"] > 0 and result["work"]["failed_advances"] == 0)


def steady_inputs(index):
    slots = ()
    held = ()
    if index < 3:
        raw, spec, network, amounts = cases.closed(index > 0, index == 2)
        expect, temps = reference.closed_steady(index > 0, index == 2)
    elif index < 6:
        roots = reference.stirred_roots()
        if len(roots) != 3:
            raise RuntimeError(f"expected three independent branches, found {len(roots)}")
        expect_r, t = roots[index - 3]
        raw, spec, network, amounts = cases.stirred(t + 5)
        expect = {"r": expect_r}
        temps = {"r": t}
        held = ("feed", "sink")
    else:
        raw, spec, network, amounts, slots = cases.surface()
        independent = reference.trajectory(raw, spec.to_dict(), amounts, (100.0,), surface=True)[0]
        expect = independent["amounts"]
        temps = independent["temperatures"]
        expect["surface"] = independent["surface"]
    state = pack_gas_reactor_network_state(spec, network, amounts, surface_slots=slots)
    scales = {k: 0.01 for k in state.amounts_mol if not any(f'"{n}"' in k for n in held)}
    return raw, spec, network, state, slots, held, scales, expect, temps


def steady(unit, index):
    _, spec, network, state, slots, held, scales, expect, temps = steady_inputs(index)
    before = deepcopy(snapshot(state))
    result = gas_network_steady_state(
        spec,
        network,
        state,
        amount_scales_mol=scales,
        held_vessel_ids=held,
        surface_slots=slots,
        root_policy=RootPolicy(maximum_iterations=120),
    )
    unit.artifacts["native"] = result
    unit.yes("success", result["status"] == "steady_state_found")
    if result["status"] != "steady_state_found":
        raise RuntimeError(result["failure"])
    actual, expected = [], []
    for name, row in expect.items():
        for s, v in row.items():
            if name == "surface":
                from chemworld.physchem.catalysis_network import catalytic_inventory_key

                key = catalytic_inventory_key(slots[0], s)
            else:
                key = gas_inventory_key(name, s)
            actual.append(result["amounts_mol"][key])
            expected.append(v)
    unit.check("amounts", actual, expected, 2e-8, 2e-5)
    unit.check(
        "temperature",
        [
            result["temperature_K"]
            if n == spec.anchor_vessel_id
            else result["coordinates"][json.dumps(["node_temperature_K", n], separators=(",", ":"))]
            for n in temps
        ],
        list(temps.values()),
        0.02,
        2e-5,
    )
    unit.yes(
        "full_material_residual",
        result["material_residual_mol_s"] <= 1e-9
        and (
            not held
            or any(
                abs(v) > 1e-8
                for k, v in result["material_rates_mol_s"].items()
                if any(f'"{n}"' in k for n in held)
            )
        ),
    )
    unit.yes(
        "original_energy",
        result["temperature_residual_K_s"] <= 1e-7
        and all(abs(v) <= 1e-5 for v in result["energy_invariant_errors_J"]),
    )
    unit.yes("conservation", all(abs(v) * 0.01 <= 1e-9 for v in result["conservation_error"]))
    unit.yes("immutable", snapshot(state) == before)
    unit.yes("work", 0 < result["original_rate_evaluations"] <= 20000)


def ignition_inputs(index):
    if index < 2:
        raw, spec, network, amounts = cases.stirred(
            300 if index == 0 else 900,
            environment=900 if index == 0 else 300,
            loss=0.03 if index == 0 else 0.3,
        )
        return raw, spec, network, amounts, (15.0, 30.0, 60.0)
    path = next(
        Path(p) / "h2o2.yaml" for p in ct.get_data_directories() if (Path(p) / "h2o2.yaml").exists()
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    network = convert_mechanism(
        raw, phase_name="ohmech", source="Cantera installed h2o2.yaml (complete)"
    )
    gas = ct.Solution(str(path), "ohmech")
    gas.TPX = 1100, 101325, {"H2": 2, "O2": 1, "AR": 7}
    amounts = {"r": dict(zip(gas.species_names, gas.concentrations, strict=True))}
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("r", 1),),
        (),
        1100,
        "r",
        (
            GasNodeEnergySpec(
                "r",
                1100,
                0,
                environment_temperature_K=300,
                loss_conductance_W_K=0 if index == 2 else 5000,
                minimum_temperature_K=300,
                maximum_temperature_K=3400,
            ),
        ),
    )
    return raw, spec, network, amounts, (0.0001, 0.0005, 0.002)


def ignition(unit, index):
    raw, spec, network, amounts, times = ignition_inputs(index)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    original = state
    expected = reference.trajectory(raw, spec.to_dict(), amounts, times)
    actual = []
    unit.artifacts.update(
        native=actual,
        reference=expected,
        initial=snapshot(original),
        mechanism_species=len(network.species),
        mechanism_reactions=len(network.reactions),
    )
    from chemworld.physchem.gas_reactor_network import gas_reactor_network_system

    system = gas_reactor_network_system(spec, network)
    initial_u = system.thermodynamics.internal_energy_j(state)
    for t in times:
        start = time.perf_counter()
        result, _diagnostics = advance_gas_reactor_network(spec, network, state, t - state.time_s)
        state = result.final_state
        row = {
            "T": [],
            "n": [],
            "P": [],
            "U": system.thermodynamics.internal_energy_j(state) - initial_u,
            "heat": sum(state.coordinates[gas_loss_key(n.vessel_id)] for n in spec.nodes),
            "time": state.time_s,
            "work": dict(result.solver_diagnostic),
            "stopping_event": result.stopping_event,
            "wall_time_s": time.perf_counter() - start,
        }
        for node in spec.nodes:
            view = gas_node_view(state, node, network)
            row["T"].append(view.temperature_K)
            row["n"].extend(view.amounts_mol[s] for s in network.species_ids)
            row["P"].append(
                sum(view.amounts_mol.values()) * cases.R * view.temperature_K * 1000 / node.volume_L
            )
        actual.append(row)
        print(
            json.dumps(
                {
                    "unit": unit.name,
                    "stage": "native trajectory",
                    "completed": len(actual),
                    "total": 3,
                    "physical_time_s": t,
                    "nfev": row["work"].get("nfev"),
                }
            ),
            flush=True,
        )
    unit.artifacts.update(
        native=actual,
        reference=expected,
        initial=snapshot(original),
        mechanism_species=len(network.species),
        mechanism_reactions=len(network.reactions),
    )
    unit.check(
        "T",
        [a["T"] for a in actual],
        [[e["temperatures"][n.vessel_id] for n in spec.nodes] for e in expected],
        0.1,
        3e-4,
    )
    unit.check(
        "n",
        [a["n"] for a in actual],
        [
            [e["amounts"][n.vessel_id][s] for n in spec.nodes for s in network.species_ids]
            for e in expected
        ],
        2e-7,
        3e-4,
    )
    unit.check(
        "P",
        [a["P"] for a in actual],
        [[e["pressures"][n.vessel_id] for n in spec.nodes] for e in expected],
        10,
        3e-4,
    )
    unit.check(
        "U", [a["U"] for a in actual], [sum(e["delta_U"].values()) for e in expected], 0.1, 3e-4
    )
    unit.check(
        "heat",
        [a["heat"] for a in actual],
        [-sum(e["delta_U"].values()) for e in expected],
        0.1,
        3e-4,
    )
    final = state.temperature_K
    unit.yes(
        "transition",
        final > 700
        if index == 0
        else final < 450
        if index == 1
        else final > 1400
        if index == 2
        else final < 600,
    )
    unit.check("clock", [a["time"] for a in actual], times, 1e-12)
    unit.yes(
        "work", all(a["stopping_event"] is None and a["work"].get("nfev", 0) > 0 for a in actual)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--units", default="")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing attempts")
    names = [
        *(f"S{i}" for i in range(1, 11)),
        "D1",
        "D2",
        *(f"R{i}" for i in range(1, 8)),
        *(f"I{i}" for i in range(1, 5)),
    ]
    selected = args.units.split(",") if args.units else names
    if set(selected) - set(names):
        raise ValueError("unknown fixed unit")
    report = {
        "mode": "canary" if args.units else "fixed-development-block",
        "planned_units": len(selected),
        "planned_comparison_groups": 8 * len(selected),
        "cantera_version": ct.__version__,
        "units": [],
    }
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in selected:
        unit = Unit(name)
        begin = time.perf_counter()
        error = None
        try:
            {"S": source, "D": derivative, "R": steady, "I": ignition}[name[0]](
                unit, int(name[1:]) - 1
            )
        except Exception:
            import traceback

            error = traceback.format_exc()
        missing = 8 - len(unit.checks)
        unit.checks.extend(
            {"name": f"unexecuted-{i}", "passed": False, "not_executed": True}
            for i in range(missing)
        )
        entry = {
            "id": name,
            "passed": error is None
            and len(unit.checks) == 8
            and all(c["passed"] for c in unit.checks),
            "failure": error,
            "checks": unit.checks,
            "artifacts": unit.artifacts,
            "wall_time_s": time.perf_counter() - begin,
        }
        report["units"].append(entry)
        report.update(
            passed_units=sum(u["passed"] for u in report["units"]),
            passed_comparison_groups=sum(c["passed"] for u in report["units"] for c in u["checks"]),
            wall_time_s=time.perf_counter() - started,
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "unit": name,
                    "passed": entry["passed"],
                    "completed": len(report["units"]),
                    "total": len(selected),
                    "wall_time_s": report["wall_time_s"],
                    "units_per_s": len(report["units"]) / report["wall_time_s"],
                    "eta_s": report["wall_time_s"]
                    / len(report["units"])
                    * (len(selected) - len(report["units"])),
                }
            ),
            flush=True,
        )
    if report["passed_units"] != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
