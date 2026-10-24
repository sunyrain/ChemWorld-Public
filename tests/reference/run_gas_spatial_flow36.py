"""Fixed 20-unit / 240-slot development block; preserve every attempt."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import gas_spatial_flow36_cases as cases
import gas_spatial_flow36_reference as reference
import numpy as np

from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_spatial_reactor import spatial_rates
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import GasReactorNetworkRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world

GROUPS = ("n", "P", "Q", "material", "H", "bulk", "diffusion", "diffusion_H", "instant")
FLOORS = {
    "n": 2e-10,
    "P": 0.03,
    "Q": 3e-6,
    "material": 2e-10,
    "H": 3e-6,
    "bulk": 2e-10,
    "diffusion": 2e-10,
    "diffusion_H": 3e-6,
    "instant": 2e-9,
}


class Unit:
    def __init__(self, number):
        self.data = {"unit": number, "checks": [], "exceptions": []}

    def check(self, label, passed, **values):
        self.data["checks"].append(
            {"label": label, "evaluated": True, "passed": bool(passed), **values}
        )

    def compare(self, label, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        error = np.abs(a - b)
        tolerance = FLOORS[label] + 3e-5 * np.abs(b)
        self.check(
            label,
            bool(np.all(error <= tolerance)),
            scalar_count=a.size,
            maximum_absolute_error=float(np.max(error)),
            maximum_error_fraction=float(np.max(error / tolerance)),
            failed_values=[
                {"actual": float(x), "expected": float(y), "tolerance": float(t)}
                for x, y, e, t in zip(
                    a.ravel(), b.ravel(), error.ravel(), tolerance.ravel(), strict=True
                )
                if e > t
            ],
        )

    def refuses(self, label, call):
        try:
            call()
        except (ValueError, RuntimeError) as error:
            self.check(label, True, message=str(error))
        else:
            self.check(label, False)


def observe(spec, network, state, detail):
    names = ("feed", *spec.spatial_transport.cell_vessel_ids, "receiver")
    faces = ("inlet", *(f"spatial-face:{i}" for i in range(1, len(names) - 2)), "outlet")
    if detail is None:
        n = {
            name: {
                s: state.amounts_mol[json.dumps(["inventory", name, s], separators=(",", ":"))]
                for s in network.species_ids
            }
            for name in names
        }
        nodes = {node.vessel_id: node for node in spec.nodes}
        _, packet = spatial_rates(spec, network, state)
        output = {
            "time": state.time_s,
            "n": [n[name][s] for name in names for s in network.species_ids],
            "P": [
                sum(n[name].values()) * cases.R * spec.temperature_K * 1000 / nodes[name].volume_L
                for name in names
            ],
            "Q": [0.0] * len(names),
        }
        for label, size in (
            ("material", 2),
            ("H", 1),
            ("bulk", 1),
            ("diffusion", 2),
            ("diffusion_H", 1),
        ):
            output[label] = [0.0] * (len(faces) * size)
    else:
        packet = detail["spatial"]
        output = {
            "time": state.time_s,
            "n": [
                detail["nodes"][name]["amounts_mol"][s]
                for name in names
                for s in network.species_ids
            ],
            "P": [detail["nodes"][name]["pressure_Pa"] for name in names],
            "Q": [detail["nodes"][name]["bath_heat_J"] for name in names],
        }
        for label, key, vector in (
            ("material", "amounts_mol", True),
            ("H", "enthalpy_J", False),
            ("bulk", "bulk_mol", False),
            ("diffusion", "diffusion_mol", True),
            ("diffusion_H", "diffusion_enthalpy_J", False),
        ):
            output[label] = (
                [detail["connections"][f][key][s] for f in faces for s in network.species_ids]
                if vector
                else [detail["connections"][f][key] for f in faces]
            )
    output["instant"] = [
        packet["faces"][f]["amount_rates_mol_s"][s] for f in faces for s in network.species_ids
    ]
    return output


def numerical(unit):
    number = unit.data["unit"]
    card = cases.case(number)
    spec, network, amounts = cases.native(card)
    initial = pack_gas_reactor_network_state(spec, network, amounts)
    original = json.dumps({"spec": spec.to_dict(), "amounts": amounts}, sort_keys=True)
    state = initial
    samples = [observe(spec, network, state, None)]
    work = []
    details = []
    residuals = []
    event = number in (13, 14, 15, 16)
    cumulative = {
        label: np.zeros(len(samples[0][label]))
        for label in ("Q", "material", "H", "bulk", "diffusion", "diffusion_H")
    }
    for endpoint in card["times"][1:]:
        result, detail = advance_gas_reactor_network(spec, network, state, endpoint - state.time_s)
        state = result.final_state
        row = observe(spec, network, state, detail)
        for label in cumulative:
            cumulative[label] += row[label]
            row[label] = cumulative[label].tolist()
        samples.append(row)
        details.append(detail)
        work.append(dict(result.solver_diagnostic))
        residuals.append(
            {
                "atoms": dict(result.conservation_residuals),
                "energy": result.energy_balance_residual_J,
                "node_energy": [n["energy_residual_J"] for n in detail["nodes"].values()],
            }
        )
    expected = reference.solve(card, matrix=number in (1, 2, 3, 4, 9, 10, 11, 12), event=event)
    if event:
        samples = samples[-1:]
    unit.data.update(
        card=card,
        native=samples,
        reference=expected,
        details=details,
        work=work,
        residuals=residuals,
    )
    for label in GROUPS[:8]:
        unit.compare(label, [r[label] for r in samples], [r[label] for r in expected["samples"]])
    if number in (9, 10, 11, 12):
        actual = np.array(samples[-1]["n"]).reshape(-1, 2)[1:-1]
        fraction = actual[:, 0] / actual.sum(axis=1)
        analytic = reference.continuous_profile(card)
        error = float(max(abs(fraction - analytic)))
        previous = numerical.grid_error
        ratio = None if previous is None else previous / error
        unit.check(
            "continuous_profile",
            error <= 0.20 / card["cells"] and (ratio is None or ratio >= 1.5),
            error=error,
            limit=0.20 / card["cells"],
            previous_ratio=ratio,
        )
        numerical.grid_error = error
        unit.data["continuous_profile"] = analytic.tolist()
    elif event:
        unit.check(
            "actual_event_time",
            abs(samples[-1]["time"] - expected["samples"][-1]["time"]) <= 2e-6,
            actual=samples[-1]["time"],
            expected=expected["samples"][-1]["time"],
        )
    else:
        unit.compare(
            "instant", [r["instant"] for r in samples], [r["instant"] for r in expected["samples"]]
        )
    closed = all(
        max(map(abs, r["atoms"].values()), default=0) <= 1e-10
        and abs(r["energy"]) <= 2e-6
        and max(map(abs, r["node_energy"])) <= 2e-6
        for r in residuals
    )
    if event:
        expected_stop = (
            "spatial:minimum_amount"
            if number == 13
            else "spatial:maximum_mach"
            if number == 16
            else "spatial:maximum_pressure"
        )
        closed = (
            closed
            and details[-1]["stopping_event"] == expected_stop
            and 0 < state.time_s < card["times"][-1]
        )
    unit.check("conservation_and_endpoint", closed)
    unit.check(
        "actual_work",
        len(work) == len(card["times"]) - 1
        and all(w["nfev"] > 0 and w["solver_calls"] > 0 for w in work),
    )
    unit.check(
        "unchanged_input",
        initial.time_s == 0
        and original == json.dumps({"spec": spec.to_dict(), "amounts": amounts}, sort_keys=True),
    )


numerical.grid_error = None


def world(unit, temp):
    spec, network, amounts = cases.native(cases.case(unit.data["unit"]))
    runtime = ChemWorldRuntime(composition=gas_reactor_network_world(spec, network, amounts))
    restored = None
    try:
        runtime.reset(seed=36)
        unit.check(
            "original_species", set(runtime._state.species_amounts) == set(network.species_ids)
        )
        before = json.loads(json.dumps(runtime._state.to_dict()))
        first = runtime.execute({"operation": "wait", "duration_s": 0.1})
        unit.check("first_wait", first.info["transaction_status"] == "committed")
        local = {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}
        unit.check(
            "finite_feed",
            sum(local["feed"].species_amounts.values()) < sum(amounts["feed"].values()),
        )
        p = [local[n].pressure_Pa for n in spec.spatial_transport.cell_vessel_ids]
        unit.check("grid_pressure", max(p) - min(p) < 1e-6)
        unit.check(
            "distinct_tanks", local["feed"].pressure_Pa > max(p) > local["receiver"].pressure_Pa
        )
        selected = runtime.execute({"operation": "select_vessel", "vessel": "receiver"})
        unit.check(
            "actual_selection",
            selected.info["transaction_status"] == "committed"
            and runtime._state.vessel_id == "receiver",
        )
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        unit.check("strict_json_snapshot", isinstance(snapshot, dict))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        results = [r.execute({"operation": "wait", "duration_s": 0.2}) for r in (runtime, restored)]
        final = json.loads(json.dumps(runtime._state.to_dict()))
        unit.check("second_wait", all(r.info["transaction_status"] == "committed" for r in results))
        unit.check("same_process", final == json.loads(json.dumps(restored._state.to_dict())))
        source = temp / f"world-{unit.data['unit']}-source.json"
        target = temp / f"world-{unit.data['unit']}-target.json"
        source.write_text(json.dumps(snapshot), encoding="utf-8")
        code = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()));"
            "o=r.execute({'operation':'wait','duration_s':.2});"
            "assert o.info['transaction_status']=='committed';"
            "Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict(),allow_nan=False));r.close()"
        )
        subprocess.run([sys.executable, "-c", code, str(source), str(target)], check=True)
        fresh = json.loads(target.read_text())
        unit.check("new_process", fresh == final)
        unit.check("one_clock", abs(runtime._state.ledger.time_s - 0.3) < 1e-14)
        unit.check(
            "unique_stock",
            all(
                set(s.phases.phases) == {"gas"}
                and s.phases.phases["gas"].species_amounts_mol == s.species_amounts
                for s in {
                    runtime._state.vessel_id: runtime._state,
                    **runtime._state.inactive_vessels,
                }.values()
            ),
        )
        unit.data.update(
            initial=before,
            snapshot=snapshot,
            final=final,
            fresh_process_final=fresh,
            first_info=dict(first.info),
        )
    finally:
        runtime.close()
        if restored:
            restored.close()


def refusals(unit):
    spec, network, amounts = cases.native(cases.case(1))
    unit.refuses(
        "reactive_tank",
        lambda: replace(
            spec,
            nodes=tuple(
                replace(n, chemistry_enabled=True) if n.vessel_id == "feed" else n
                for n in spec.nodes
            ),
        ),
    )
    unit.refuses(
        "wrong_inlet",
        lambda: replace(
            spec,
            connections=(
                replace(spec.connections[0], target_vessel_id="cell-001"),
                spec.connections[1],
            ),
        ),
    )
    unit.refuses(
        "same_tank",
        lambda: replace(
            spec,
            connections=(
                spec.connections[0],
                replace(spec.connections[1], target_vessel_id="feed"),
            ),
        ),
    )
    unit.refuses(
        "extra_connection",
        lambda: replace(
            spec,
            connections=(*spec.connections, GasFlowConnectionSpec("extra", "feed", "receiver")),
        ),
    )
    unit.refuses(
        "nonmolar",
        lambda: replace(
            spec,
            connections=(replace(spec.connections[0], mode="pressure-valve"), spec.connections[1]),
        ),
    )
    unit.refuses(
        "face_collision",
        lambda: replace(
            spec,
            connections=(
                replace(spec.connections[0], connection_id="spatial-face:1"),
                spec.connections[1],
            ),
        ),
    )
    unit.refuses(
        "sealed_model",
        lambda: GasReactorNetworkSpec.from_dict(
            {**spec.to_dict(), "model": "sealed-spatial-gas-network-v1"}
        ),
    )
    unit.refuses(
        "empty_cell",
        lambda: gas_reactor_network_system(spec, network).validate_state(
            pack_gas_reactor_network_state(
                spec, network, {**amounts, "cell-000": {"A": 0.0, "B": 0.0}}
            )
        ),
    )
    unit.refuses(
        "unequal_cell_pressure",
        lambda: gas_reactor_network_system(spec, network).validate_state(
            pack_gas_reactor_network_state(
                spec, network, {**amounts, "cell-000": {"A": 0.1, "B": 0.1}}
            )
        ),
    )
    unit.refuses("negative_flow", lambda: replace(spec.connections[0], coefficient=-1.0))
    unit.refuses(
        "extra_tank",
        lambda: replace(spec, nodes=(*spec.nodes, replace(spec.nodes[-1], vessel_id="extra"))),
    )
    composition = gas_reactor_network_world(spec, network, amounts)
    states = composition.physics.initial_states
    anchor = states[spec.anchor_vessel_id].replace(
        inactive_vessels={
            k: v for k, v in states.items() if k not in (spec.anchor_vessel_id, "feed")
        }
    )
    before = anchor.to_dict()
    outcome = GasReactorNetworkRuntime(composition).advance(anchor, {"duration_s": 0.1})
    unit.check(
        "atomic_rollback",
        outcome.state is None and outcome.actual_duration_s == 0 and anchor.to_dict() == before,
    )
    unit.data["rollback"] = dict(outcome.diagnostics)


COMPAT = r"""
import json,sys
from dataclasses import fields
from collections.abc import Mapping
from pathlib import Path
sys.path.insert(0,sys.argv[2])
import gas_spatial36_cases as cases
from chemworld.examples.gas_network_sensitivity import example_inputs as old
from chemworld.examples.gas_surface_network import example_inputs as surface
import chemworld.physchem.gas_reactor_network as gas
from chemworld.world.gas_reactor_network import gas_reactor_network_world
rows=[]
for name in ('sealed','finite','surface'):
 slots=()
 if name=='sealed':spec,network,amounts=cases.native(cases.case())
 elif name=='finite':spec,network,amounts=old()
 else:
  _,spec,network,amounts,initial=surface(size=2,both=True)
  slots=tuple(a.make_slot(initial[a.owner_id]) for a in spec.surfaces)
 state=gas.pack_gas_reactor_network_state(spec,network,amounts,surface_slots=slots)
 result,diagnostic=gas.advance_gas_reactor_network(spec,network,state,.1,surface_slots=slots)
 final={f.name:(dict(v) if isinstance(v,Mapping) else v) for f in fields(result.final_state)
        for v in [getattr(result.final_state,f.name)]}
 world=gas_reactor_network_world(spec,network,amounts,
       **({'surface_amounts_by_owner':initial} if name=='surface' else {}))
 rows.append({'scenario':name,'spec':spec.to_dict(),'world':world.to_dict(),'endpoint':final,'diagnostic':diagnostic})
Path(sys.argv[1]).write_text(json.dumps({'module':gas.__file__,'rows':rows},allow_nan=False))
"""


def compatibility(unit, temp, baseline):
    raw = []
    root = Path(__file__).resolve().parents[2]
    for name, src in (("baseline", baseline), ("current", root / "src")):
        output = temp / f"compat-{name}.json"
        subprocess.run(
            [sys.executable, "-c", COMPAT, str(output), str(Path(__file__).parent.resolve())],
            env={**os.environ, "PYTHONPATH": str(src.resolve())},
            cwd=temp,
            check=True,
        )
        data = json.loads(output.read_text())
        if not Path(data["module"]).is_relative_to(src.resolve()):
            raise ValueError("wrong source loaded")
        raw.append(data)
    for old, new in zip(raw[0]["rows"], raw[1]["rows"], strict=True):
        for key in ("spec", "world", "endpoint", "diagnostic"):
            unit.check(old["scenario"] + ":" + key, old[key] == new[key])
    unit.data.update(baseline_commit="975e2af6", raw=raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--baseline-src", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError("preserve attempts; choose fresh paths")
    payload = {
        "schema": "chemworld-36C-development-1",
        "planned_units": 20,
        "planned_comparisons": 240,
        "units": [],
    }
    start = time.monotonic()
    progress = {"active_unit": 0, "completed": 0}
    stopped = threading.Event()

    def heartbeat():
        ticks = 0
        while not stopped.wait(30):
            ticks += 1
            print(
                json.dumps(
                    {
                        "stage": "live",
                        **progress,
                        "total": 20,
                        "liveness_counter": ticks,
                        "elapsed_s": time.monotonic() - start,
                    }
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    with tempfile.TemporaryDirectory(prefix="chemworld-36C-") as directory:
        for number in range(1, 21):
            progress["active_unit"] = number
            print(json.dumps({"stage": "start", "unit": number, "total": 20}), flush=True)
            unit = Unit(number)
            try:
                if number <= 16:
                    numerical(unit)
                elif number <= 18:
                    world(unit, Path(directory))
                elif number == 19:
                    refusals(unit)
                else:
                    compatibility(unit, Path(directory), args.baseline_src)
            except Exception as error:
                unit.data["exceptions"].append(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            while len(unit.data["checks"]) < 12:
                unit.data["checks"].append(
                    {
                        "label": f"unexecuted-{len(unit.data['checks']) + 1}",
                        "evaluated": False,
                        "passed": False,
                    }
                )
            payload["units"].append(unit.data)
            progress["completed"] = number
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "stage": "complete",
                        "completed": number,
                        "total": 20,
                        "passed": sum(c["passed"] for c in unit.data["checks"]),
                        "elapsed_s": elapsed,
                        "units_per_s": number / elapsed,
                        "eta_s": elapsed * (20 - number) / number,
                    }
                ),
                flush=True,
            )
    stopped.set()
    checks = [c for u in payload["units"] for c in u["checks"]]
    payload.update(
        elapsed_s=time.monotonic() - start,
        evaluated_comparisons=sum(c["evaluated"] for c in checks),
        passed_comparisons=sum(c["passed"] for c in checks),
        failed_comparisons=sum(c["evaluated"] and not c["passed"] for c in checks),
        unexecuted_comparisons=sum(not c["evaluated"] for c in checks),
    )
    payload["failures"] = [
        {"unit": u["unit"], **c} for u in payload["units"] for c in u["checks"] if not c["passed"]
    ]
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        **{k: v for k, v in payload.items() if k != "units"},
        "full_evidence": str(args.output),
        "units": [{k: u[k] for k in ("unit", "checks", "exceptions")} for u in payload["units"]],
    }
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in payload.items() if k not in ("units", "failures")}), flush=True
    )
    if payload["passed_comparisons"] != 240:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
