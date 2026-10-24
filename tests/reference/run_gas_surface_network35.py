"""Fixed 12-unit/144-comparison finite gas/surface development qualification."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

import gas_surface_network35_cases as cases
import gas_surface_network35_reference as reference

from chemworld.examples.gas_surface_network import example_world
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.catalysis_network import catalytic_inventory_key
from chemworld.physchem.gas_network_sensitivity import (
    GasSensitivityParameter,
    gas_network_trajectory_sensitivities,
)
from chemworld.physchem.gas_network_surface import GasSurfaceContext
from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    gas_bath_key,
    gas_enthalpy_key,
    gas_node_view,
    gas_transfer_key,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import GasReactorNetworkRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world

ATOL = {
    "gas": 2e-9,
    "surface": 2e-11,
    "coverage": 2e-5,
    "temperature": 0.002,
    "pressure": 0.2,
    "energy": 0.003,
    "bath": 0.003,
    "transport": 2e-9,
    "enthalpy": 0.003,
}


class Unit:
    def __init__(self, index):
        self.index = index
        labels = (
            cases.NUMERIC
            if index <= 8
            else cases.WORLD
            if index <= 10
            else cases.EVENT
            if index == 11
            else cases.REJECT
        )
        self.record = {
            "unit": index,
            "checks": [{"label": k, "status": "not_executed", "passed": False} for k in labels],
            "exceptions": [],
        }

    def check(self, label, actual, expected=True):
        row = next(r for r in self.record["checks"] if r["label"] == label)
        row.update(
            status="evaluated", actual=actual, expected=expected, passed=bool(actual == expected)
        )

    def compare(self, label, actual, expected):
        values = []
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            if set(a) != set(e):
                raise ValueError("numerical output identities differ")
            for key in a:
                error = abs(a[key] - e[key])
                tolerance = ATOL[label] + 3e-4 * abs(e[key])
                values.append(
                    {
                        "sample": i,
                        "key": key,
                        "actual": float(a[key]),
                        "expected": float(e[key]),
                        "absolute_error": float(error),
                        "tolerance": float(tolerance),
                        "passed": bool(error <= tolerance),
                    }
                )
        self.check(label, all(v["passed"] for v in values))
        next(r for r in self.record["checks"] if r["label"] == label)["values"] = values

    def refuses(self, label, operation):
        try:
            operation()
        except ValueError as error:
            self.check(label, True)
            next(r for r in self.record["checks"] if r["label"] == label)["refusal"] = str(error)
        else:
            self.check(label, False)


def observe(spec, network, context, state, initial):
    row = {key: {} for key in ATOL}
    row.update(elements={}, sites={})
    for node in spec.nodes:
        name = node.vessel_id
        gas = gas_node_view(state, node, network)
        row["gas"].update({name + ":" + s: n for s, n in gas.amounts_mol.items()})
        row["temperature"][name] = gas.temperature_K
        row["pressure"][name] = (
            sum(gas.amounts_mol.values()) * reference.R * gas.temperature_K * 1000 / node.volume_L
        )
        model = context.models[name]
        row["energy"][name] = model.internal_energy_j(
            context.view(state, node)
        ) - model.internal_energy_j(context.view(initial, node))
        row["bath"][name] = (
            state.coordinates[gas_bath_key(name)] - initial.coordinates[gas_bath_key(name)]
        )
        if name in context.slots:
            slot, attachment = context.slots[name], context.attachments[name]
            for species in attachment.mechanism.adsorbates:
                n = state.amounts_mol[catalytic_inventory_key(slot, species)]
                row["surface"][name + ":" + species] = n
                row["coverage"][name + ":" + species] = (
                    n * attachment.mechanism.site_occupancies[species] / 1e-6
                )
    for name, weights in context.laws.items():
        value = sum(state.amounts_mol[k] * v for k, v in weights.items())
        if name.startswith("sites:"):
            row["sites"][name] = value
        else:
            row["elements"][name] = value
    row["transport"] = {
        "feed:" + s: state.coordinates[gas_transfer_key("feed", s)]
        - initial.coordinates[gas_transfer_key("feed", s)]
        for s in network.species_ids
    }
    row["enthalpy"] = {
        "feed": state.coordinates[gas_enthalpy_key("feed")]
        - initial.coordinates[gas_enthalpy_key("feed")]
    }
    return row


def numerical(unit):
    card, spec, network, slots, start, context = cases.case(unit.index)
    before = (
        spec.to_dict(),
        network.to_dict(),
        [s.to_dict() for s in slots],
        dict(start.amounts_mol),
        dict(start.coordinates),
    )
    native = [observe(spec, network, context, start, start)]
    state, details, work = start, [], []
    unit.record.update(card=card, native=native, native_details=details, native_work=work)
    for target in cases.TIMES[1:]:
        result, diagnostic = advance_gas_reactor_network(
            spec, network, state, target - state.time_s, surface_slots=slots
        )
        state = result.final_state
        native.append(observe(spec, network, context, state, start))
        details.append(diagnostic)
        work.append(dict(result.solver_diagnostic))
    expected = (
        reference.equation_history(card, cases.TIMES)
        if card["thermal"]
        else reference.reactor_history(card, cases.TIMES)
    )
    unit.record["reference"] = expected
    for group in ATOL:
        unit.compare(group, [r[group] for r in native], [r[group] for r in expected["rows"]])
    conserved = all(
        abs(row[group][k] - native[0][group][k]) <= tolerance
        for row in native
        for group, tolerance in (("elements", 2e-9), ("sites", 2e-11))
        for k in row[group]
    ) and all(
        abs(d["energy_residual_J"]) <= 3e-6
        and all(abs(n["energy_residual_J"]) <= 3e-6 for n in d["nodes"].values())
        for d in details
    )
    unit.check("conservation", conserved)
    unit.check("work", len(work) == 4 and state.time_s == 0.5 and all(w["nfev"] > 0 for w in work))
    unit.check(
        "immutable",
        before
        == (
            spec.to_dict(),
            network.to_dict(),
            [s.to_dict() for s in slots],
            dict(start.amounts_mol),
            dict(start.coordinates),
        ),
    )


def world(unit, directory):
    thermal = unit.index == 10
    composition = example_world(size=2, thermal=thermal, both=True)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=35)
        before = {name: local.to_dict() for name, local in local_states(runtime._state).items()}
        unit.check(
            "identity",
            {s.phase for s in composition.physics.network.species} == {"gas", "surface"},
        )
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.1})
        unit.check("first_wait", outcome.info["transaction_status"] == "committed")
        unit.record["first_info"] = {
            k: outcome.info.get(k)
            for k in ("transaction_status", "actual_duration_s", "outcome_diagnostics")
        }
        after = local_states(runtime._state)
        unit.check(
            "slot_evolution",
            all(
                next(
                    s for s in local.state_slots.slots if s.spec.owner_id == name + ":catalyst"
                ).values["HA"]
                != 1e-7
                for name, local in after.items()
            ),
        )
        unit.check(
            "gas_only", all(set(local.species_amounts) == {"A", "B"} for local in after.values())
        )
        selected = runtime.execute({"operation": "select_vessel", "vessel": "source"})
        unit.check(
            "selection",
            selected.info["transaction_status"] == "committed"
            and runtime._state.vessel_id == "source",
        )
        source = directory / f"world{unit.index}.json"
        output = directory / f"world{unit.index}-continued.json"
        runtime.snapshot().save(source)
        snapshot = json.loads(source.read_text(encoding="utf-8"))
        json.dumps(snapshot, allow_nan=False)
        unit.check("json", True)
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        action = {"operation": "wait", "duration_s": 0.2}
        outcomes = [consumer.execute(action) for consumer in (runtime, restored)]
        unit.check(
            "second_wait", all(o.info["transaction_status"] == "committed" for o in outcomes)
        )
        unit.check("exact_restore", runtime._state.to_dict() == restored._state.to_dict())
        code = (
            "import json;from pathlib import Path;from chemworld.runtime import ChemWorldRuntime;"
            f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text(encoding='utf-8')));"
            f"o=r.execute({action!r});assert o.info['transaction_status']=='committed';"
            f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict(),allow_nan=False));r.close()"
        )
        subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
        fresh_state = json.loads(output.read_text())
        unit.record["new_process_state"] = fresh_state
        unit.check("new_process", fresh_state == json.loads(json.dumps(runtime._state.to_dict())))
        stock, initial = True, True
        for name, local in local_states(runtime._state).items():
            slot = next(s for s in local.state_slots.slots if s.spec.owner_id == name + ":catalyst")
            stock &= (
                abs(slot.values["Pt"] + 2 * sum(slot.values[s] for s in ("HA", "HB", "HAA")) - 1e-6)
                <= 2e-11
            )
            original = next(
                s
                for s in before[name]["state_slots"].values()
                if s["spec"]["owner_id"] == name + ":catalyst"
            )
            initial &= dict(slot.initial_values) == original["initial_values"]
        unit.check("site_stock", bool(stock))
        unit.check("initial_stock", bool(initial))
        unit.check(
            "resource_once",
            abs(runtime._state.ledger.time_s - 0.3) < 1e-15
            and runtime._state.ledger.cost == 0
            and runtime._state.ledger.risk == 0
            and all(abs(v - 0.3) < 1e-15 for v in runtime._state.vessel_elapsed_s.values()),
        )
        unit.record.update(initial=before, final=runtime._state.to_dict(), snapshot=snapshot)
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def events(unit):
    card, spec, network, slots, start, _ = cases.case(1)
    spec = replace(
        spec,
        nodes=tuple(
            replace(n, minimum_amount_mol=0.0295) if n.vessel_id == "source" else n
            for n in spec.nodes
        ),
    )
    context = GasSurfaceContext(spec, network, slots)
    result, detail = advance_gas_reactor_network(spec, network, start, 1, surface_slots=slots)
    expected = reference.equation_history(card, (0.0, 1.0), event=True)
    native = observe(spec, network, context, result.final_state, start)
    unit.record.update(
        card=card,
        native=native,
        reference=expected,
        native_details=detail,
        native_work=[dict(result.solver_diagnostic)],
    )
    unit.check("stop_reason", result.stopping_event, "gas:source:minimum_amount")
    unit.check("stop_time", abs(result.actual_duration_s - expected["rows"][-1]["time_s"]) <= 2e-6)
    for group in ATOL:
        unit.compare(group, [native[group]], [expected["rows"][-1][group]])
    unit.check(
        "finite_work", 0 < result.actual_duration_s < 1 and result.solver_diagnostic["nfev"] > 0
    )


def refusals(unit):
    _, spec, network, slots, start, _ = cases.case(1)
    slot = slots[0]
    unit.refuses("missing_slot", lambda: GasSurfaceContext(spec, network, ()))
    unit.refuses(
        "moved_slot",
        lambda: GasSurfaceContext(spec, network, (replace(slot, location_id="receiver"),)),
    )
    unit.refuses(
        "capacity",
        lambda: GasSurfaceContext(
            spec, network, (replace(slot, values={**slot.values, "heat_capacity_J_K": 999}),)
        ),
    )
    unit.refuses("area", lambda: replace(spec.surfaces[0], area_m2=0))
    unit.refuses("duplicate_node", lambda: replace(spec, surfaces=spec.surfaces * 2))
    unit.refuses(
        "unknown_node",
        lambda: replace(spec, surfaces=(replace(spec.surfaces[0], vessel_id="unknown"),)),
    )
    unit.refuses(
        "identity",
        lambda: GasSurfaceContext(
            spec,
            replace(
                network, species=(replace(network.species[0], formula="H"), network.species[1])
            ),
            slots,
        ),
    )
    unit.refuses(
        "sensitivity",
        lambda: gas_network_trajectory_sensitivities(
            spec,
            network,
            start,
            (GasSensitivityParameter("flow", "feed", "coefficient"),),
            (0, 0.1),
        ),
    )
    amounts = {"source": {"A": 0.02, "B": 0.01}, "receiver": {"A": 0.003, "B": 0.002}}
    unit.refuses("missing_initial", lambda: gas_reactor_network_world(spec, network, amounts))
    unit.refuses(
        "shared_species",
        lambda: GasSurfaceContext(
            spec,
            replace(
                network,
                species=(
                    *network.species,
                    replace(
                        spec.surfaces[0].mechanism.species_by_id["Pt"].species, species_id="unused"
                    ),
                ),
            ),
            slots,
        ),
    )
    unit.refuses(
        "no_surface_slots",
        lambda: advance_gas_reactor_network(
            replace(spec, surfaces=()), network, start, 0.1, surface_slots=slots
        ),
    )
    composition = example_world()
    states = composition.physics.initial_states
    anchor = states["source"].replace(inactive_vessels={"receiver": states["receiver"]})
    broken = anchor.replace(
        state_slots=StateSlots(
            tuple(s for s in anchor.state_slots.slots if s.spec.owner_id != "source:catalyst")
        )
    )
    before = broken.to_dict()
    outcome = GasReactorNetworkRuntime(composition).advance(broken, {"duration_s": 0.1})
    unit.check(
        "rollback",
        outcome.outcome == "numerical_failure"
        and outcome.state is None
        and outcome.actual_duration_s == 0
        and broken.to_dict() == before,
    )
    unit.record["rollback_diagnostics"] = dict(outcome.diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError("preserve attempts; choose new paths")
    payload = {
        "schema": "chemworld-35D-fixed-development-1",
        "planned_units": 12,
        "planned_comparisons": 144,
        "units": [],
    }
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chemworld-35D-") as temp:
        for index in range(1, 13):
            unit = Unit(index)
            print(json.dumps({"stage": "unit_start", "unit": index, "total": 12}), flush=True)
            try:
                if index <= 8:
                    numerical(unit)
                elif index <= 10:
                    world(unit, Path(temp))
                elif index == 11:
                    events(unit)
                else:
                    refusals(unit)
            except Exception as error:
                unit.record["exceptions"].append(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            payload["units"].append(unit.record)
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "stage": "unit_complete",
                        "completed": index,
                        "total": 12,
                        "passed": sum(c["passed"] for c in unit.record["checks"]),
                        "comparisons": 12,
                        "elapsed_s": elapsed,
                        "units_per_s": index / elapsed,
                        "eta_s": elapsed * (12 - index) / index,
                    }
                ),
                flush=True,
            )
    checks = [c for u in payload["units"] for c in u["checks"]]
    payload.update(
        elapsed_s=time.monotonic() - start,
        attempted_units=12,
        passed_units=sum(all(c["passed"] for c in u["checks"]) for u in payload["units"]),
        evaluated_comparisons=sum(c["status"] == "evaluated" for c in checks),
        passed_comparisons=sum(c["passed"] for c in checks),
        failed_comparisons=sum(c["status"] == "evaluated" and not c["passed"] for c in checks),
        unexecuted_comparisons=sum(c["status"] == "not_executed" for c in checks),
    )
    payload["failures"] = [
        {"unit": u["unit"], **c} for u in payload["units"] for c in u["checks"] if not c["passed"]
    ]
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = {k: v for k, v in payload.items() if k != "units"}
    summary["full_evidence"] = str(args.output)
    summary["units"] = []
    for unit in payload["units"]:
        row = {k: unit[k] for k in ("unit", "checks", "exceptions")}
        row = json.loads(json.dumps(row))
        for check in row["checks"]:
            if "values" in check:
                values = check.pop("values")
                check.update(
                    scalar_count=len(values),
                    maximum_absolute_error=max(v["absolute_error"] for v in values),
                    maximum_error_fraction=max(
                        v["absolute_error"] / v["tolerance"] for v in values
                    ),
                    failed_values=[v for v in values if not v["passed"]],
                )
        work = unit.get("native_work", [])
        row["native_work"] = {
            "reported_advances": len(work),
            **{k: sum(w.get(k, 0) for w in work) for k in ("nfev", "njev", "nlu")},
        }
        summary["units"].append(row)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in payload.items() if k not in {"units", "failures"}}), flush=True
    )
    if payload["passed_comparisons"] != 144:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
