"""Fixed 16-unit nonisothermal spatial development acceptance; never replace attempts."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import gas_spatial36_cases as sealed
import gas_spatial_energy36_cases as cases
import gas_spatial_energy36_reference as reference
import gas_spatial_flow36_cases as flowing
import numpy as np

from chemworld.examples.gas_thermal_column import example_world
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime

COUNTS = (6, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 4, 2, 4, 4, 2)


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class Unit:
    def __init__(self, number):
        self.data = {
            "unit": number,
            "expected_checks": COUNTS[number - 1],
            "checks": [],
            "exceptions": [],
            "evidence": {},
        }

    def check(self, name, actual, expected, atol=0, rtol=0):
        if atol or rtol:
            a, b = np.asarray(actual), np.asarray(expected)
            error = np.abs(a - b)
            tolerance = atol + rtol * np.abs(b)
            passed = bool(np.all(error <= tolerance))
            detail = {"maximum_error": float(np.max(error)), "atol": atol, "rtol": rtol}
        else:
            passed = actual == expected
            detail = {}
        self.data["checks"].append(
            {
                "name": name,
                "passed": passed,
                "actual": plain(actual),
                "expected": plain(expected),
                **detail,
            }
        )


def advance(card, duration=None, device=None):
    spec, network, amounts = cases.native(card)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    result, detail = advance_gas_reactor_network(
        spec,
        network,
        state,
        card["duration"] if duration is None else duration,
        thermal_device=device,
    )
    return spec, network, state, result, detail


def numerical(unit):
    card = cases.case(unit.data["unit"])
    spec, _, _, result, detail = advance(card)
    oracle = reference.solve(card)
    names = list(spec.spatial_transport.cell_vessel_ids)
    if card["flow"]:
        names = ["feed", *names, "receiver"]
    nodes = [detail["nodes"][name] for name in names]
    for label, actual, expected, atol in (
        ("inventories", [list(n["amounts_mol"].values()) for n in nodes], oracle["amounts"], 2e-10),
        ("temperatures", [n["temperature_K"] for n in nodes], oracle["temperatures"], 3e-5),
        ("pressures", [n["pressure_Pa"] for n in nodes], oracle["pressures"], 0.02),
        (
            "node-energy-and-loss",
            [[n["internal_energy_change_J"], n["heat_loss_J"]] for n in nodes],
            np.column_stack((oracle["energy_changes"], oracle["loss"])),
            2e-6,
        ),
        (
            "face-material-enthalpy-fourier",
            [
                [*n["amounts_mol"].values(), n["enthalpy_J"], n["fourier_heat_J"]]
                for n in detail["connections"].values()
            ],
            oracle["faces"],
            [2e-10] * len(card["species"]) + [2e-6, 2e-6],
        ),
    ):
        unit.check(label, actual, expected, atol, 3e-6)
    unit.check(
        "common-pressure-and-first-laws",
        [
            max(abs(v) for v in detail["spatial"]["common_pressure_residual_Pa"].values()),
            max(abs(n["energy_residual_J"]) for n in nodes),
            result.energy_balance_residual_J,
            max(abs(v) for v in result.conservation_residuals.values()),
            result.final_state.work_J,
        ],
        [0] * 5,
        2e-6,
    )
    unit.data["evidence"] = {
        "card": card,
        "spec": spec.to_dict(),
        "result": plain(result),
        "detail": detail,
        "reference": plain(oracle),
    }


def fourier(unit):
    for cells in (4, 8, 16):
        card = cases.pure(cells)
        _, _, _, result, detail = advance(card)
        initial = np.array(card["temperatures"])
        shape = np.cos(np.pi * (np.arange(cells) + 0.5) / cells)
        decay = 4 * np.sin(np.pi / (2 * cells)) ** 2 / (card["length"] / cells) ** 2
        conductivity = reference.conductivity(card, 600, [1])
        alpha = conductivity / (card["pressure"] / (cases.R * 600) * 20.8)
        expected = 0.01 * np.exp(-alpha * decay * card["duration"])
        actual = (
            np.array([n["temperature_K"] for n in detail["nodes"].values()])
            @ shape
            / (shape @ shape)
        )
        unit.check(f"Cp-mode-{cells}-cells", actual, expected, 2e-7)
        unit.data["evidence"][str(cells)] = {
            "Cp_prediction": expected,
            "actual": actual,
            "incorrect_Cv_prediction": 0.01
            * np.exp(-alpha * 20.8 / (20.8 - cases.R) * decay * card["duration"]),
            "pressure_change": detail["spatial"]["pressure_Pa"] - card["pressure"],
            "initial": initial.tolist(),
            "work_J": result.final_state.work_J,
        }


def heating(unit):
    card = cases.pure(1, 0)
    device = ThermalDeviceSpec(maximum_power_W=0.1, setpoint_K=1000, maximum_setpoint_K=1400)
    _, _, state, result, detail = advance(card, 0.05, device)
    n = sum(state.amounts_mol.values())
    expected = 600 + 0.005 / (n * (20.8 - cases.R))
    unit.check("homogeneous-Cv", result.final_state.temperature_K, expected, 1e-8)
    unit.check("pressure-rise", detail["spatial"]["pressure_Pa"], 100000 * expected / 600, 2e-6)
    unit.check(
        "heater-once-zero-work",
        [
            result.final_state.boundary_heat_J,
            result.final_state.work_J,
            result.energy_balance_residual_J,
        ],
        [0.005, 0, 0],
        2e-9,
    )


def cantera_homogeneous(unit):
    import cantera as ct

    kind = "equimolar" if unit.data["unit"] == 9 else "non-equimolar"
    card = sealed.case(kind, cells=4)
    card.update(flow=False, temperatures=[600] * 4, support=0, loss=0, link=0, duration=0.1)
    card["fractions"] = [card["fractions"][0]] * 4
    _, _, _, result, detail = advance(card)
    gas = reference.cantera_gas(card)
    gas.TPX = 600, 100000, card["fractions"][0]
    reactor = ct.IdealGasReactor(gas, volume=card["area"] * card["length"] / 4, clone=True)
    solver = ct.ReactorNet([reactor])
    solver.rtol, solver.atol = 1e-11, 1e-18
    solver.advance(0.1)
    expected = reactor.phase.concentrations * 1000 * reactor.volume
    unit.check(
        "Cantera-amounts",
        list(detail["nodes"]["cell-000"]["amounts_mol"].values()),
        expected,
        2e-10,
        2e-6,
    )
    unit.check(
        "Cantera-T-P",
        [result.final_state.temperature_K, detail["spatial"]["pressure_Pa"]],
        [reactor.T, reactor.phase.P],
        2e-5,
        2e-7,
    )
    unit.check(
        "uniform-no-bulk", [r["bulk_mol"] for r in detail["connections"].values()], [0] * 3, 1e-12
    )
    unit.data["evidence"] = {"cantera_version": ct.__version__, "solver": dict(solver.solver_stats)}


def cantera_flow(unit):
    import cantera as ct

    card = flowing.case(4)
    card.update(
        flow=True,
        support=0,
        loss=0,
        link=0,
        k=0,
        temperatures=[650, 600, 500],
        feed_x=1,
        receiver_x=1,
        fractions=[{"A": 1, "B": 0}],
        duration=0.05,
    )
    spec, _, _, _, detail = advance(card)
    species = ct.Species("A", {"H": 2})
    species.thermo = ct.ConstantCp(200, 2000, 101325, [298.15, 0, 30000, 30000])
    reactors = []
    ordered = [
        next(n for n in spec.nodes if n.vessel_id == name)
        for name in ("feed", "cell-000", "receiver")
    ]
    for node, t, p in zip(ordered, card["temperatures"], (160000, 100000, 40000), strict=True):
        gas = ct.Solution(thermo="ideal-gas", species=[species])
        gas.TP = t, p
        reactors.append(ct.IdealGasReactor(gas, volume=node.volume_L / 1000, clone=True))
    controllers = [
        ct.MassFlowController(reactors[i], reactors[i + 1], mdot=0.0004 * 0.002016)
        for i in range(2)
    ]
    solver = ct.ReactorNet(reactors)
    solver.rtol, solver.atol = 1e-11, 1e-18
    solver.advance(0.05)
    rows = [detail["nodes"][n.vessel_id] for n in ordered]
    unit.check(
        "Cantera-finite-T", [r["temperature_K"] for r in rows], [r.T for r in reactors], 3e-6
    )
    unit.check(
        "Cantera-finite-P", [r["pressure_Pa"] for r in rows], [r.phase.P for r in reactors], 0.01
    )
    unit.check(
        "finite-stocks",
        [sum(r["amounts_mol"].values()) for r in rows],
        [r.mass / 0.002016 for r in reactors],
        2e-10,
    )
    unit.data["evidence"] = {
        "controller_count": len(controllers),
        "solver": dict(solver.solver_stats),
    }


def limits(unit):
    card = cases.pure(1, 0)
    spec, network, amounts = cases.native(card)
    n = sum(amounts["cell-000"].values())
    device = ThermalDeviceSpec(maximum_power_W=0.1, setpoint_K=1000, maximum_setpoint_K=1400)
    for kind, target in (("P", 100010), ("T", 600.1)):
        changed = (
            replace(spec, nodes=(replace(spec.nodes[0], maximum_pressure_Pa=target),))
            if kind == "P"
            else replace(
                spec, energy_nodes=(replace(spec.energy_nodes[0], maximum_temperature_K=target),)
            )
        )
        actual_device = (
            replace(device, maximum_setpoint_K=target, setpoint_K=target) if kind == "T" else device
        )
        # Setpoint above node ceiling is rejected by design; use maximum cooling
        # boundary? For high-T event author an environment hotter than the node.
        if kind == "T":
            changed = replace(
                changed,
                energy_nodes=(
                    replace(
                        changed.energy_nodes[0],
                        loss_conductance_W_K=0.1,
                        environment_temperature_K=1000,
                    ),
                ),
            )
            actual_device = None
        state = pack_gas_reactor_network_state(changed, network, amounts)
        result, _ = advance_gas_reactor_network(
            changed, network, state, 1, thermal_device=actual_device
        )
        expected_time = (
            (target / 100000 * 600 - 600) * n * (20.8 - cases.R) / 0.1
            if kind == "P"
            else -n * (20.8 - cases.R) / 0.1 * np.log((1000 - target) / (1000 - 600))
        )
        unit.check(
            f"{kind}-event",
            result.stopping_event,
            "spatial:maximum_pressure" if kind == "P" else "gas:cell-000:maximum_temperature",
        )
        unit.check(f"{kind}-time", result.actual_duration_s, expected_time, 3e-7)


def inventory_mach(unit):
    card = cases.case(5)
    card.update(k=0, support=0.1, loss=0, link=0)
    spec, network, amounts = cases.native(card)
    floor = sum(amounts["feed"].values()) - card["Fin"] * 0.01
    spec = replace(
        spec,
        nodes=tuple(
            replace(n, minimum_amount_mol=floor) if n.vessel_id == "feed" else n for n in spec.nodes
        ),
    )
    state = pack_gas_reactor_network_state(spec, network, amounts)
    result, _ = advance_gas_reactor_network(spec, network, state, 0.1)
    unit.check(
        "finite-feed-stop",
        [result.stopping_event, round(result.actual_duration_s, 7)],
        ["spatial:minimum_amount", 0.01],
    )
    bad = replace(spec, spatial_transport=replace(spec.spatial_transport, maximum_mach=1e-12))
    try:
        gas_reactor_network_system(bad, network).validate_state(state)
    except ValueError as error:
        unit.check("Mach-refusal", "Mach" in str(error), True)
    else:
        unit.check("Mach-refusal", False, True)


def world(unit, directory):
    runtime = ChemWorldRuntime(composition=example_world())
    restored = None
    try:
        runtime.reset(seed=36)
        statuses = []
        for command in (
            {"operation": "queue_lab_heat", "target_temperature_K": 800, "duration_s": 1},
            {"operation": "select_vessel", "vessel": "receiver"},
            {"operation": "wait", "duration_s": 0.02},
        ):
            statuses.append(runtime.execute(command).info["transaction_status"])
        unit.check("original-world-queue-nonanchor", statuses, ["committed"] * 3)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        source = directory / "snapshot.json"
        source.write_text(json.dumps(snapshot), encoding="utf-8")
        for item in (runtime, restored):
            result = item.execute({"operation": "wait", "duration_s": 0.03})
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(result.info["error_message"])
        unit.check(
            "same-process-complete-state", runtime._state.to_dict(), restored._state.to_dict()
        )
        script = """import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()))
o=r.execute({'operation':'wait','duration_s':.03})
assert o.info['transaction_status']=='committed'
Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict()))
r.close()
"""
        target = directory / "continued.json"
        completed = subprocess.run(
            [sys.executable, "-c", script, str(source), str(target)],
            capture_output=True,
            text=True,
            check=True,
        )
        unit.check(
            "new-process-complete-state",
            json.loads(json.dumps(runtime._state.to_dict())),
            json.loads(target.read_text()),
        )
        states = [runtime._state, *runtime._state.inactive_vessels.values()]
        unit.check(
            "one-clock-and-no-K-accounting",
            [
                all(abs(s.ledger.time_s - 0.05) < 1e-12 for s in states),
                all(
                    f.unit != "K"
                    for s in states
                    for slot in s.state_slots.slots
                    if slot.spec.state_type == "gas_network_accounting_v1"
                    for f in slot.spec.fields
                ),
            ],
            [True, True],
        )
        unit.data["evidence"] = {"state": runtime._state.to_dict(), "subprocess": completed.stdout}
    finally:
        runtime.close()
        if restored:
            restored.close()


def declarations(unit):
    spec, _, _ = cases.native(cases.case(1))
    unit.check(
        "thermal-roundtrip",
        GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))),
        spec,
    )
    zero = GasNodeEnergySpec("gas", 600, 0)
    unit.check("zero-support", zero.support_heat_capacity_J_K, 0)
    refused = []
    for value in (-1, float("nan"), float("inf"), True):
        try:
            replace(zero, support_heat_capacity_J_K=value)
        except ValueError:
            refused.append(True)
    unit.check("invalid-support", len(refused), 4)
    payload = spec.to_dict()
    payload["spatial_transport"]["model"] = "sealed-isothermal-low-mach-v1"
    try:
        GasReactorNetworkSpec.from_dict(payload)
    except ValueError:
        unit.check("mixed-codec-refusal", True, True)
    else:
        unit.check("mixed-codec-refusal", False, True)


def compatibility(unit):
    baseline = json.loads(
        (Path(os.environ["TEMP"]) / "chemworld-36D-original/objects.json").read_text()
    )
    for name, inputs in (
        ("sealed", sealed.native(sealed.case(cells=2))),
        ("finite-flow", flowing.native(flowing.case(1))),
    ):
        spec, network, amounts = inputs
        state = pack_gas_reactor_network_state(spec, network, amounts)
        result, detail = advance_gas_reactor_network(spec, network, state, 0.01)
        actual = {
            "spec": spec.to_dict(),
            "initial": plain(state),
            "result": plain(result),
            "diagnostic": plain(detail),
        }
        expected = baseline[name]
        unit.check(f"complete-old-{name}-objects", actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows = []
    for number in range(1, 17):
        unit = Unit(number)
        started = time.perf_counter()
        try:
            if number <= 6:
                numerical(unit)
            elif number in (9, 10):
                cantera_homogeneous(unit)
            elif number == 14:
                world(unit, args.output)
            else:
                {
                    7: fourier,
                    8: heating,
                    11: cantera_flow,
                    12: limits,
                    13: inventory_mach,
                    15: declarations,
                    16: compatibility,
                }[number](unit)
        except Exception:
            unit.data["exceptions"].append(traceback.format_exc())
        unit.data["elapsed_s"] = time.perf_counter() - started
        unit.data["passed"] = (
            not unit.data["exceptions"]
            and len(unit.data["checks"]) == COUNTS[number - 1]
            and all(c["passed"] for c in unit.data["checks"])
        )
        rows.append(unit.data)
        (args.output / f"unit-{number:02d}.json").write_text(
            json.dumps(plain(unit.data), indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "completed": number,
                    "total": 16,
                    "passed": unit.data["passed"],
                    "units_per_s": number / elapsed,
                    "eta_s": (16 - number) * elapsed / number,
                }
            ),
            flush=True,
        )
    result = {
        "evidence_class": "development",
        "units": 16,
        "checks": sum(COUNTS),
        "passed_units": sum(r["passed"] for r in rows),
        "evaluated_checks": sum(len(r["checks"]) for r in rows),
        "passed_checks": sum(c["passed"] for r in rows for c in r["checks"]),
        "elapsed_s": time.perf_counter() - start,
        "results": rows,
    }
    (args.output / "RESULTS.json").write_text(
        json.dumps(plain(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    if result["passed_units"] != 16:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
