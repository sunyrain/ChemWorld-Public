"""Fixed 23-unit / 276-comparison development qualification for finite pistons."""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import gas_piston35_cases as cases
import gas_piston35_reference as oracle
import numpy as np

from chemworld.examples.gas_piston import example_inputs
from chemworld.physchem.gas_network_pressure import GasPressureControlSpec, gas_work_key
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import unpack_gas_network
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, atol=None, rtol=2e-6):
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
    spec, network, amounts, device, duration = cases.native(name)
    card = cases.card(name)
    event = name in cases.EVENTS
    reference = oracle.ode(card, events=event)
    times = [reference["time"]] if event else [0.2, 0.5, duration]
    reference = oracle.ode(card, times, events=event)
    upstream = oracle.cantera(card, times)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    heat = np.zeros(len(spec.nodes))
    work = heat.copy()
    edges = np.zeros((len(spec.connections), 3))
    actual = []
    diagnostics = []
    resources = []
    residuals = []
    for t in times:
        result, d = advance_gas_reactor_network(
            spec, network, state, duration if event else t - state.time_s, thermal_device=device
        )
        state = result.final_state
        rows = [d["nodes"][n.vessel_id] for n in spec.nodes]
        heat += np.array(
            [
                r["bath_heat_J"] + r.get("conducted_heat_J", 0.0) - r.get("heat_loss_J", 0.0)
                for r in rows
            ]
        )
        work += np.array([r["boundary_work_on_J"] for r in rows])
        if spec.connections:
            edges += np.array(
                [
                    [
                        *[d["connections"][e.connection_id]["amounts_mol"][s] for s in ("A", "B")],
                        d["connections"][e.connection_id]["enthalpy_J"],
                    ]
                    for e in spec.connections
                ]
            )
        actual.append(
            {
                "time": state.time_s,
                "amounts": [[r["amounts_mol"][s] for s in ("A", "B")] for r in rows],
                "temperatures": [r["temperature_K"] for r in rows],
                "volumes": [r["volume_L"] for r in rows],
                "pressures": [r["pressure_Pa"] for r in rows],
                "heat": heat.tolist(),
                "work": work.tolist(),
                "edges": edges.tolist(),
            }
        )
        diagnostics.append(d)
        resources.append(dict(result.solver_diagnostic))
        residuals.append(result.energy_balance_residual_J)
    expected = reference["samples"]
    for field, atol in (
        ("amounts", 3e-10),
        ("temperatures", 2e-5),
        ("volumes", 2e-6),
        ("pressures", 0.05),
        ("heat", 3e-5),
        ("work", 3e-5),
        ("edges", 3e-5),
    ):
        record.check("ode-" + field, [a[field] for a in actual], [a[field] for a in expected], atol)
    ct_errors = []
    for a, b in zip(actual, upstream["samples"], strict=True):
        for f, atol in (
            ("amounts", 3e-10),
            ("temperatures", 2e-5),
            ("volumes", 2e-6),
            ("pressures", 0.05),
        ):
            ct_errors.append(bool(np.allclose(a[f], b[f], rtol=2e-6, atol=atol)))
    record.check("actual-cantera-n-T-V-P", ct_errors, [True] * len(ct_errors))
    residuals.extend(r["energy_residual_J"] for d in diagnostics for r in d["nodes"].values())
    residuals.extend(v for d in diagnostics for v in d["element_residuals_mol"].values())
    record.check("original-first-law-and-elements", residuals, [0.0] * len(residuals), 5e-5, 0.0)
    i = next(i for i, n in enumerate(spec.nodes) if n.vessel_id == spec.pressure_control.vessel_id)
    record.check(
        "signed-PV-work",
        [a["work"][i] for a in actual],
        [
            -spec.pressure_control.pressure_Pa * (a["volumes"][i] - spec.nodes[i].volume_L) / 1000
            for a in actual
        ],
        3e-5,
    )
    record.check("earliest-shared-time", state.time_s, reference["time"], 2e-5)
    record.check("earliest-stop-identity", result.stopping_event, reference["event"])
    record.evidence = {
        "card": card,
        "native": actual,
        "diagnostics": diagnostics,
        "resources": resources,
        "independent_ode": reference,
        "cantera": upstream,
    }


def execute(runtime, **action):
    result = runtime.execute(action)
    if result.info["transaction_status"] != "committed":
        raise AssertionError(
            {
                k: result.info.get(k)
                for k in ("transaction_status", "outcome_diagnostics", "error_message")
            }
        )
    return result


def world(record, kind, directory):
    thermal = kind != "isothermal"
    spec, network, amounts = example_inputs(thermal=thermal, connected=True)
    device = (
        ThermalDeviceSpec(
            conductance_W_K=1.0,
            maximum_power_W=8.0,
            setpoint_K=1100.0,
            enabled=False,
            minimum_setpoint_K=300.0,
            maximum_setpoint_K=1200.0,
        )
        if thermal
        else None
    )
    composition = gas_reactor_network_world(spec, network, amounts, anchor_thermal_device=device)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=35)
        execute(runtime, operation="select_vessel", vessel="piston")
        if thermal:
            execute(
                runtime, operation="queue_lab_heat", target_temperature_K=1100.0, duration_s=1.0
            )
        execute(runtime, operation="select_vessel", vessel="receiver")
        execute(runtime, operation="wait", duration_s=0.2)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        record.evidence["snapshot"] = snapshot
        replay = True
        if kind == "rollback":
            before = json.loads(json.dumps(runtime._state.to_dict()))
            with patch(
                "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
                side_effect=RuntimeError("35E injected failure"),
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
            {"operation": "wait", "duration_s": 0.8},
            {"operation": "select_vessel", "vessel": "feed"},
            {"operation": "wait", "duration_s": 0.2},
        ]
        for action in continuation:
            execute(runtime, **action)
            if restored:
                execute(restored, **action)
                replay = replay and restored._state.to_dict() == runtime._state.to_dict()
        if kind == "process":
            source, target = directory / "snapshot.json", directory / "continued.json"
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            code = (
                "import json;from pathlib import Path;"
                "from chemworld.runtime import ChemWorldRuntime;"
                f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text(encoding='utf-8')));"
                f"z=[r.execute(a) for a in {continuation!r}];"
                "assert all(x.info['transaction_status']=='committed' for x in z);"
                f"Path({str(target)!r}).write_text(json.dumps(r._state.to_dict()),encoding='utf-8');r.close()"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            replay = json.loads(target.read_text(encoding="utf-8")) == json.loads(
                json.dumps(runtime._state.to_dict())
            )
        states = local_states(runtime._state)
        piston = states["piston"]
        account = next(
            s for s in piston.state_slots.slots if s.spec.state_type == "gas_network_accounting_v1"
        )
        work = account.values[gas_work_key("piston")]
        expected_by = 100000.0 * (piston.volume_L - 1.0) / 1000
        record.check(
            "original-shared-clock",
            list(runtime._state.vessel_elapsed_s.values()),
            [1.2] * len(states),
            1e-12,
        )
        record.check("selected-original-member", runtime._state.vessel_id, "feed")
        record.check(
            "actual-EOS-volume",
            piston.volume_L,
            sum(piston.species_amounts.values()) * oracle.R * piston.temperature_K * 0.01,
            1e-11,
        )
        record.check("original-phase-volume", piston.phases.phases["gas"].volume_L, piston.volume_L)
        record.check(
            "original-phase-stock",
            piston.phases.phases["gas"].species_amounts_mol,
            piston.species_amounts,
        )
        record.check("original-capacity", piston.vessels.vessels["piston"].max_volume_L, 2.0)
        record.check("constant-external-pressure", piston.pressure_Pa, 100000.0)
        record.check("work-on-accounting", work, -expected_by, 3e-5)
        record.check(
            "work-by-original-ledger", runtime._state.ledger.boundary_work_J, expected_by, 3e-5
        )
        record.check(
            "heater-energy-once",
            runtime._state.ledger.energy_jacket_J,
            8.0 if thermal else runtime._state.ledger.energy_jacket_J,
            3e-5,
        )
        record.check(
            "queue-complete-or-no-queue",
            [
                len(runtime._state.lab_clock.running),
                len(runtime._state.lab_clock.queued),
                len(runtime._state.lab_clock.completed),
            ],
            [0, 0, 1 if thermal else 0],
        )
        record.check("exact-continuation-or-rollback", replay, True)
        record.evidence["final_state"] = runtime._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def refusals(record):
    spec, network, amounts = example_inputs(thermal=True, connected=True)

    def malformed_world(kind):
        w = gas_reactor_network_world(spec, network, amounts)
        states = dict(w.physics.initial_states)
        piston = states.pop("piston")
        if kind == "volume":
            piston = piston.replace(volume_L=1.1)
        if kind == "pressure":
            piston = piston.replace(pressure_Pa=90000.0)
        if kind == "phase":
            piston = piston.replace(species_amounts={"A": 0.01, "B": 0.01})
        unpack_gas_network(piston.replace(inactive_vessels=states), network)

    def wrong_codec():
        value = spec.to_dict()
        value["model"] = "finite-thermal-gas-network-v1"
        GasReactorNetworkSpec.from_dict(value)

    checks = [
        ("zero-pressure", lambda: GasPressureControlSpec("piston", 0, 0.4, 2)),
        ("bad-stroke", lambda: GasPressureControlSpec("piston", 1, 2, 1)),
        ("infinite-stroke", lambda: GasPressureControlSpec("piston", 1, 0.4, float("inf"))),
        ("boolean-pressure", lambda: GasPressureControlSpec("piston", True, 0.4, 2)),
        (
            "unknown-node",
            lambda: replace(
                spec, pressure_control=replace(spec.pressure_control, vessel_id="missing")
            ),
        ),
        (
            "outside-reference",
            lambda: replace(
                spec, pressure_control=replace(spec.pressure_control, minimum_volume_L=1.1)
            ),
        ),
        (
            "pressure-ceiling",
            lambda: replace(spec, pressure_control=replace(spec.pressure_control, pressure_Pa=1e9)),
        ),
        (
            "inconsistent-initial-stock",
            lambda: gas_reactor_network_world(
                spec, network, {**amounts, "piston": {"A": 0.02, "B": 0.02}}
            ),
        ),
        ("wrong-codec", wrong_codec),
        *[
            ("restore-" + kind, lambda kind=kind: malformed_world(kind))
            for kind in ("volume", "pressure", "phase")
        ],
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
from chemworld.examples.gas_surface_network import example_inputs as surface
from chemworld.examples.gas_flow_column import example_inputs as spatial
from gas_network35_energy_cases import case,native
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world
rows=[]
for kind in ('35A','35B','35D','36C'):
    extra={}
    if kind=='35A':s,n,a=simple()
    elif kind=='35B':
        card=case('loop');s,n,_=native(card);a=card['amounts']
    elif kind=='35D':
        _,s,n,a,slots=surface(thermal=True);extra={'surface_amounts_by_owner':slots}
    else:s,n,a=spatial(non_equimolar=True)
    w=gas_reactor_network_world(s,n,a,**extra)
    r=ChemWorldRuntime(composition=w);r.reset(seed=35)
    out=r.execute({'operation':'wait','duration_s':.2})
    assert out.info['transaction_status']=='committed',out.info
    rows.append({'kind':kind,'codec':[s.to_dict(),w.to_dict()],
        'state':r._state.to_dict(),'diagnostics':out.info['outcome_diagnostics']})
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
        for field in ("codec", "state", "diagnostics"):
            record.check(a["kind"] + ":" + field, a[field], b[field])
    record.evidence = {"baseline_commit": "68adce7b", "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--baseline-src", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("preserve prior attempts")
    names = [
        *cases.PHYSICAL,
        *cases.EVENTS,
        *["world-" + k for k in ("isothermal", "queue", "json", "process", "rollback")],
        "refusals",
        "compatibility",
    ]
    records = []
    start = time.perf_counter()
    for i, name in enumerate(names, 1):
        record = Record(name)
        directory = args.output.parent / (args.output.stem + "-" + name)
        try:
            if name in cases.PHYSICAL or name in cases.EVENTS:
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
        passed = sum(c["passed"] for c in record.checks)
        records.append(
            {
                "unit": name,
                "passed": passed == 12
                and len(record.checks) == 12
                and "exception" not in record.evidence,
                "checks": record.checks,
                "evidence": record.evidence,
            }
        )
        elapsed = time.perf_counter() - start
        print(
            f"35E {i}/23 {name}: {'PASS' if records[-1]['passed'] else 'FAIL'}; "
            f"{i / elapsed:.2f} units/s; ETA {(23 - i) * elapsed / i:.1f}s",
            flush=True,
        )
        args.output.write_text(
            json.dumps({"records": records}, indent=2, allow_nan=False), encoding="utf-8"
        )
    summary = {
        "scope": "35E development finite-stroke constant-load piston",
        "planned_units": 23,
        "planned_comparisons": 276,
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
        json.dumps({k: v for k, v in summary.items() if k not in {"units", "failures"}}), flush=True
    )
    return 0 if summary["passed_units"] == 23 else 1


if __name__ == "__main__":
    raise SystemExit(main())
