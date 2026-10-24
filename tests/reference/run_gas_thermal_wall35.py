"""Fixed 18 units / 180 comparisons for the finite thermal wall consumer."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gas_thermal_wall35_cases as cases
import gas_thermal_wall35_reference as oracle
import numpy as np

from chemworld.examples.gas_thermal_wall import example_inputs, wall_temperature
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.gas_network_pressure import GasPressureControlSpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_thermal_wall import GAS_WALL_TYPE, GasWallContact, wall_key
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import GAS_ACCOUNTING_TYPE, unpack_gas_network
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, atol=None, rtol=0.0):
        passed = (
            actual == expected
            if atol is None
            else np.allclose(actual, expected, atol=atol, rtol=rtol)
        )
        self.checks.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "atol": atol,
                "rtol": rtol if atol is not None else None,
                "passed": bool(passed),
            }
        )


def physical(record, name):
    spec, network, amounts, segments = cases.native(name)
    card = cases.card(name)
    initial = pack_gas_reactor_network_state(spec, network, amounts)
    state = initial
    advances = []
    for duration, device in segments:
        result, diagnostic = advance_gas_reactor_network(
            spec, network, state, duration, thermal_device=device
        )
        advances.append({"diagnostic": diagnostic, "solver": result.solver_diagnostic})
        state = result.final_state
        if result.stopping_event:
            break
    expected = oracle.solve(card)
    names = card["nodes"]
    native_amounts = {n: diagnostic["nodes"][n]["amounts_mol"] for n in names}
    native_temperatures = {n: diagnostic["nodes"][n]["temperature_K"] for n in names}
    wall = spec.thermal_wall
    tw = state.coordinates[wall_key(wall.owner_id, "temperature_K")]
    contacts = {
        n: state.coordinates.get(wall_key(wall.owner_id, "contact_J", n), 0.0) for n in names
    }
    losses = [
        state.coordinates[wall_key(wall.owner_id, k)]
        for k in ("environment_conduction_J", "environment_radiation_J")
    ]
    record.check(
        "all-species",
        [native_amounts[n][s] for n in names for s in ("A", "B")],
        [expected["amounts"][n][s] for n in names for s in ("A", "B")],
        2e-11,
        2e-7,
    )
    record.check(
        "gas-temperatures",
        list(native_temperatures.values()),
        list(expected["temperatures"].values()),
        2e-6,
        2e-7,
    )
    record.check("wall-temperature", tw, expected["wall_temperature"], 2e-6, 2e-7)
    record.check(
        "contact-heat", list(contacts.values()), list(expected["contacts"].values()), 2e-6, 5e-7
    )
    record.check(
        "enclosure-heat", losses, [expected["conduction"], expected["radiation"]], 2e-6, 5e-7
    )
    delta_u = oracle.energy(native_amounts, native_temperatures, tw, card) - oracle.energy(
        amounts, card["temperatures"], wall.initial_temperature_K, card
    )
    record.check("global-first-law", delta_u - state.boundary_heat_J + sum(losses), 0.0, 5e-5)
    record.check(
        "wall-first-law",
        wall.heat_capacity_J_K * (tw - wall.initial_temperature_K)
        - sum(contacts.values())
        + sum(losses),
        0.0,
        5e-5,
    )
    record.check(
        "elements",
        2 * sum(sum(a.values()) for a in native_amounts.values()),
        2 * sum(sum(a.values()) for a in amounts.values()),
        2e-11,
    )
    record.check("actual-time", state.time_s, expected["duration"], 2e-7, 2e-6)
    record.check(
        "external-heat-and-no-work",
        [state.boundary_heat_J, state.work_J],
        [expected["heater"], 0.0],
        2e-6,
        5e-7,
    )
    record.evidence = {
        "input": card,
        "native_state": {
            k: dict(v) if isinstance(v, Mapping) else v for k, v in vars(state).items()
        },
        "reference": expected,
        "advances": advances,
    }


def execute(runtime, **action):
    outcome = runtime.execute(action)
    if outcome.info["transaction_status"] != "committed":
        raise RuntimeError(
            {
                k: outcome.info.get(k)
                for k in (
                    "outcome_diagnostics",
                    "error_message",
                    "failed_preconditions",
                    "operation_outcome",
                )
            }
        )
    return outcome


def world(record, kind, directory):
    spec, network, amounts = example_inputs(connected=True, reactive=False)
    device = ThermalDeviceSpec(
        conductance_W_K=10.0,
        maximum_power_W=20.0,
        setpoint_K=1100.0,
        enabled=False,
        minimum_setpoint_K=250.0,
        maximum_setpoint_K=1200.0,
    )
    runtime = ChemWorldRuntime(
        composition=gas_reactor_network_world(spec, network, amounts, anchor_thermal_device=device)
    )
    restored = None
    try:
        runtime.reset(seed=35)
        execute(runtime, operation="select_vessel", vessel="reactor")
        execute(runtime, operation="queue_lab_heat", target_temperature_K=1100.0, duration_s=1.0)
        execute(runtime, operation="select_vessel", vessel="receiver")
        execute(runtime, operation="wait", duration_s=0.1)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        replay = True
        if kind == "rollback":
            before = json.loads(json.dumps(runtime._state.to_dict()))
            with patch(
                "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
                side_effect=RuntimeError("35F injected failure"),
            ):
                failure = runtime.execute({"operation": "wait", "duration_s": 0.3})
            after = json.loads(json.dumps(runtime._state.to_dict()))
            record.evidence.update(
                rollback_before=before, rollback_after=after, rollback_info=dict(failure.info)
            )
            for key in ("ledger", "process"):
                for field in ("cost", "risk"):
                    before[key].pop(field)
                    after[key].pop(field)
            replay = (
                before == after
                and failure.info["actual_duration_s"] == 0
                and failure.info["operation_outcome"] == "numerical_failure"
            )
        if kind == "json":
            restored = ChemWorldRuntime.from_snapshot(snapshot)
        continuation = [
            {"operation": "wait", "duration_s": 1.1},
            {"operation": "wait", "duration_s": 0.2},
        ]
        for action in continuation:
            execute(runtime, **action)
            if restored is not None:
                execute(restored, **action)
                replay = replay and restored._state.to_dict() == runtime._state.to_dict()
        if kind == "process":
            source, target = directory / "snapshot.json", directory / "continued.json"
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            code = """import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
for a in json.loads(sys.argv[3]):
    z=r.execute(a)
    assert z.info['transaction_status']=='committed', z.info.get('outcome_diagnostics')
Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict()),encoding='utf-8')
r.close()
"""
            subprocess.run(
                [sys.executable, "-c", code, str(source), str(target), json.dumps(continuation)],
                check=True,
            )
            replay = json.loads(target.read_text(encoding="utf-8")) == json.loads(
                json.dumps(runtime._state.to_dict())
            )
        states = local_states(runtime._state)
        slots = [
            s
            for local in states.values()
            for s in local.state_slots.slots
            if s.spec.state_type == GAS_WALL_TYPE
        ]
        account = next(
            s
            for s in states["reactor"].state_slots.slots
            if s.spec.state_type == GAS_ACCOUNTING_TYPE
        )
        qloss = sum(
            account.values[wall_key(spec.thermal_wall.owner_id, k)]
            for k in ("environment_conduction_J", "environment_radiation_J")
        )
        tw = wall_temperature(runtime)
        temperatures = {n: s.temperature_K for n, s in states.items()}
        final_amounts = {n: s.species_amounts for n, s in states.items()}
        card = {
            "nodes": list(states),
            "supports": dict.fromkeys(states, 0.5),
            "wall": spec.thermal_wall.to_dict(),
        }
        delta_u = oracle.energy(final_amounts, temperatures, tw, card) - oracle.energy(
            amounts, spec.initial_temperatures_K, spec.thermal_wall.initial_temperature_K, card
        )
        record.check(
            "shared-clock", list(runtime._state.vessel_elapsed_s.values()), [1.2] * 3, 1e-12
        )
        record.check("selected-member", runtime._state.vessel_id, "receiver")
        record.check(
            "one-owned-wall",
            [(s.spec.owner_id, s.location_id, s.spec.heat_capacity_field) for s in slots],
            [("wall", "reactor", None)],
        )
        record.check(
            "original-phase-stock",
            all(
                s.phases.phases["gas"].species_amounts_mol == s.species_amounts
                for s in states.values()
            ),
            True,
        )
        record.check(
            "independent-persistent-temperature",
            tw != 800.0
            and tw != temperatures["reactor"]
            and states["reactor"].temperature_K > 450.0,
            True,
        )
        record.check("loss-once", runtime._state.ledger.heat_loss_J, qloss, 2e-6)
        record.check("heater-once", runtime._state.ledger.energy_jacket_J, 20.0, 2e-6)
        heater = states["reactor"].equipment.equipment["reactor:heater"]
        record.check(
            "queue-shutoff",
            [
                len(runtime._state.lab_clock.running),
                len(runtime._state.lab_clock.completed),
                heater.settings["thermal_device"]["enabled"],
            ],
            [0, 1, False],
        )
        record.check("global-energy", delta_u + qloss - 20.0, 0.0, 5e-5)
        record.check("exact-continuation-or-rollback", replay, True)
        record.evidence.update(snapshot=snapshot, final_state=runtime._state.to_dict())
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def refusals(record):
    spec, network, amounts = example_inputs()
    wall = spec.thermal_wall

    def codec():
        payload = spec.to_dict()
        payload["model"] = "finite-thermal-gas-network-v1"
        GasReactorNetworkSpec.from_dict(payload)

    def bad_slot(moved):
        state = gas_reactor_network_world(spec, network, amounts).physics.initial_states["reactor"]
        if moved:
            records = dict(state.equipment.equipment)
            records[wall.owner_id] = replace(records[wall.owner_id], attached_vessel_id="elsewhere")
            state = state.replace(equipment=EquipmentLedger(records))
        else:
            state = state.replace(
                state_slots=StateSlots(
                    tuple(s for s in state.state_slots.slots if s.spec.state_type != GAS_WALL_TYPE)
                )
            )
        unpack_gas_network(state, network)

    checks = [
        ("zero-capacity", lambda: replace(wall, heat_capacity_J_K=0)),
        ("emissivity", lambda: replace(wall, emissivity=1.1)),
        ("missing-radiating-area", lambda: replace(wall, radiating_area_m2=0)),
        ("duplicate-contact", lambda: replace(wall, contacts=wall.contacts * 2)),
        (
            "unknown-vessel",
            lambda: replace(
                spec, thermal_wall=replace(wall, contacts=(GasWallContact("unknown", 1),))
            ),
        ),
        ("isothermal", lambda: replace(spec, energy_nodes=())),
        (
            "piston",
            lambda: replace(
                spec, pressure_control=GasPressureControlSpec("reactor", 100000, 0.1, 2.0)
            ),
        ),
        ("codec", codec),
        ("missing-slot", lambda: bad_slot(False)),
        ("moved-owner", lambda: bad_slot(True)),
    ]
    for name, action in checks:
        try:
            action()
            error = None
        except (ValueError, TypeError) as exc:
            error = str(exc)
        record.check(name, error is not None, True)
        record.evidence[name] = error


PROBE = r"""
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
import chemworld.physchem.gas_reactor_network as gas
from chemworld.examples.gas_network_sensitivity import example_inputs as simple
from chemworld.examples.gas_piston import example_inputs as piston
from chemworld.examples.gas_flow_column import example_inputs as spatial
from chemworld.examples.rrkm_gas_network import example_inputs as rrkm
from chemworld.physchem.rrkm_grain_binding import rrkm_grain_network
from gas_network35_energy_cases import case,native
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world
rows=[]
for kind in ('35A','35B','35E','36C','38C'):
    if kind=='35A':s,n,a=simple()
    elif kind=='35B':
        card=case('loop');s,n,_=native(card);a=card['amounts']
    elif kind=='35E':s,n,a=piston(thermal=True,connected=True)
    elif kind=='36C':s,n,a=spatial(non_equimolar=True)
    else:
        model,s,a=rrkm(flowing=True);n=rrkm_grain_network(model)
    w=gas_reactor_network_world(s,n,a)
    r=ChemWorldRuntime(composition=w);r.reset(seed=35)
    out=r.execute({'operation':'wait','duration_s':.2})
    assert out.info['transaction_status']=='committed',out.info.get('outcome_diagnostics')
    rows.append({'kind':kind,'codec':[s.to_dict(),w.to_dict()],
        'output':[r._state.to_dict(),out.info['outcome_diagnostics']]})
    r.close()
Path(sys.argv[1]).write_text(json.dumps({'module_path':gas.__file__,'rows':rows},allow_nan=False),encoding='utf-8')
"""


def compatibility(record, directory, baseline):
    root = Path(__file__).resolve().parents[2]
    reports = []
    for name, source in (("old", baseline), ("current", root / "src")):
        target = directory / (name + ".json")
        subprocess.run(
            [sys.executable, "-c", PROBE, str(target), str(Path(__file__).parent.resolve())],
            cwd=directory,
            env={**os.environ, "PYTHONPATH": str(source)},
            check=True,
        )
        report = json.loads(target.read_text(encoding="utf-8"))
        if not Path(report["module_path"]).is_relative_to(source):
            raise ValueError("wrong source imported")
        reports.append(report)
    for a, b in zip(reports[0]["rows"], reports[1]["rows"], strict=True):
        for field in ("codec", "output"):
            record.check(a["kind"] + ":" + field, a[field], b[field])
    record.evidence = {"baseline_source": str(baseline), "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-src", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("retain all attempts; choose fresh paths")
    names = [
        *cases.PHYSICAL,
        *["world-" + s for s in ("ownership", "json", "process", "rollback")],
        "refusals",
        "compatibility",
    ]
    records, start, stage = [], time.perf_counter(), ["starting"]
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            elapsed = time.perf_counter() - start
            print(
                f"35F stage={stage[0]} complete={len(records)}/18 elapsed={elapsed:.1f}s "
                f"throughput={len(records) / elapsed:.3f} units/s",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    for i, name in enumerate(names, 1):
        stage[0] = name
        record = Record(name)
        directory = args.output.parent / (args.output.stem + "-" + name)
        try:
            if name in cases.PHYSICAL:
                physical(record, name)
            elif name == "refusals":
                refusals(record)
            else:
                directory.mkdir(exist_ok=False)
                if name == "compatibility":
                    compatibility(record, directory, args.baseline_src.resolve())
                else:
                    world(record, name[6:], directory)
        except Exception:
            record.evidence["exception"] = traceback.format_exc()
        passed = (
            len(record.checks) == 10
            and all(c["passed"] for c in record.checks)
            and "exception" not in record.evidence
        )
        records.append(
            {"unit": name, "passed": passed, "checks": record.checks, "evidence": record.evidence}
        )
        elapsed = time.perf_counter() - start
        print(
            f"35F {i}/18 {name}: {'PASS' if passed else 'FAIL'}; "
            f"{i / elapsed:.3f} units/s; ETA {(18 - i) * elapsed / i:.1f}s",
            flush=True,
        )
        args.output.write_text(
            json.dumps({"records": records}, indent=2, allow_nan=False), encoding="utf-8"
        )
    done.set()
    summary = {
        "scope": "35F development independent finite wall",
        "planned_units": 18,
        "planned_comparisons": 180,
        "passed_units": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["checks"]) for r in records),
        "passed_comparisons": sum(c["passed"] for r in records for c in r["checks"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": [
            {
                "unit": r["unit"],
                "checks": [c for c in r["checks"] if not c["passed"]],
                "exception": r["evidence"].get("exception"),
            }
            for r in records
            if not r["passed"]
        ],
        "units": [
            {"unit": r["unit"], "passed": r["passed"], "checks": len(r["checks"])} for r in records
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"failures", "units"}}), flush=True
    )
    return 0 if summary["passed_units"] == 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
