"""Existing-world workloads for the single CPU baseline entry point."""

import json
import time
from collections.abc import Mapping
from dataclasses import fields
from math import exp


def block_b():
    return (
        [
            {"id": "world-" + name, "kind": "world", "workflow": name}
            for name in (
                "heater",
                "membrane",
                "mineral",
                "shared-clock",
                "distillation",
                "crystal",
                "solid",
                "heterogeneous",
                "open-closed",
                "device-lifecycle",
            )
        ]
        + [{"id": "world-sbml-events", "kind": "sbml"}]
        + [
            {"id": f"dfn-{nx}-{nr}", "kind": "dfn", "nx": nx, "nr": nr}
            for nx, nr in ((6, 16), (12, 32), (24, 64))
        ]
        + [{"id": "same-physics-task-binding", "kind": "task_binding"}]
    )


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False, default=lambda other: other.tolist()))


def coupled_state(state):
    return {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state)
        for value in (getattr(state, field.name),)
    }


def make_workload(case, output, write):
    if case["kind"] == "dfn":
        return make_dfn(case, output, write)
    return make_history(case, output, write)


def make_dfn(case, output, write):
    from chemworld.physchem.battery_dfn import BatteryDFNSpec
    from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
    from chemworld.physchem.battery_models import ParticleSpec, advance_battery
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = BatteryDFNSpec(
        negative=ParticleSpec(cells=case["nr"]),
        positive=ParticleSpec(cells=case["nr"]),
        electrolyte=BatteryElectrolyteSpec(cells=(case["nx"],) * 3),
    )
    initial = spec.initial_state()
    program = ScalarControlProgram(((0, 0.2),), "A")
    case["physical_inputs"] = {
        "model": spec.to_dict(),
        "initial": coupled_state(initial),
        "current_A": 0.2,
        "duration_s": 10,
        "inventory_size": len(initial.amounts_mol),
        "scope": "cost and conservation only; coarse-grid accuracy is not requalified",
    }
    number = 0

    def run(_):
        return advance_battery(spec, initial, program, 10)

    def validate(result):
        nonlocal number
        path = output / f"dfn-result-{number}.json"
        number += 1
        checks = {
            "success": result.outcome == "success",
            "charge": abs(result.final_state.coordinates["discharged_charge_C"] - 2) <= 1e-8,
            "conservation": all(
                abs(value) <= (1e-8 if key.endswith("_C") else 1e-12)
                for key, value in result.conservation_residuals.items()
            ),
        }
        details = {
            "final_state": coupled_state(result.final_state),
            "outcome": result.outcome,
            "actual_duration_s": result.actual_duration_s,
            "checks": checks,
            "conservation_residuals": dict(result.conservation_residuals),
            "solver_diagnostic": dict(result.solver_diagnostic),
        }
        write(path, details)
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "raw_result": str(path),
            "conservation_residuals": details["conservation_residuals"],
            "solver_diagnostic": details["solver_diagnostic"],
        }

    return lambda: None, run, validate


def make_history(case, output, write):
    from chemworld.examples.native_domains import workflow, workflow_initializations
    from chemworld.runtime import ChemWorldRuntime

    if case["kind"] == "sbml":
        from tests.test_sbml import model

        from chemworld.physchem.sbml_system import SBMLObservable
        from chemworld.world.sbml_world import sbml_world

        spec = sbml_world(
            model(),
            maximum_step_s=0.02,
            readout_relative_sd=0.01,
            observables=(
                SBMLObservable("amount", "A"),
                SBMLObservable("concentration", "B"),
                SBMLObservable("volume", "c1"),
            ),
        )
        actions = [
            {"operation": "wait", "duration_s": 2.25},
            {"operation": "measure", "instrument": "formal_quantity_meter"},
            {"operation": "wait", "duration_s": 0.75},
        ]
        boundaries = {}
    else:
        spec, actions = workflow(case.get("workflow", "heater"))
        boundaries = workflow_initializations(case.get("workflow", "heater"))
    specs = [spec]
    if case["kind"] == "task_binding":
        from chemworld.world.composition import compile_world_composition

        compiled = compile_world_composition(spec)
        task = {
            **spec.task.to_dict(),
            "objective": "independent observation task on the same apparatus",
        }
        rebound = compiled.bind_task(task)
        assert rebound.spec.physics.to_dict() == compiled.spec.physics.to_dict()
        specs = [compiled, rebound]
        actions = actions[:10]
    case["physical_inputs"] = {
        "compositions": [(s.spec if hasattr(s, "spec") else s).to_dict() for s in specs],
        "actions": actions,
        "initializations_after_action": boundaries,
        "seed": 7,
        "observation_seed": 12,
        "noise_mode": "sequential",
        "operation_scope": "whole history including state export and cleanup; subcalls separate",
    }
    number = 0
    reference = None

    def prepare():
        nonlocal number
        sessions = [
            ChemWorldRuntime(
                composition=s, observation_noise_mode="sequential", observation_seed_override=12
            )
            for s in specs
        ]
        for session in sessions:
            session.reset(seed=7)
        current = number
        number += 1
        return sessions, current

    def run(context):
        sessions, index = context
        histories = []
        try:
            for session in sessions:
                trace, timings, mid_snapshot = [], {}, None
                for i, action in enumerate(actions):
                    if i in boundaries:
                        start = time.perf_counter()
                        session.reset(options=boundaries[i])
                        timings[f"material_reset_before_{i}"] = time.perf_counter() - start
                    started = time.perf_counter()
                    result = session.execute(action)
                    elapsed = time.perf_counter() - started
                    trace.append(
                        {
                            "index": i,
                            "action": action,
                            "execute_s": elapsed,
                            "info": result.info,
                            "observation": result.observation.to_dict(),
                        }
                    )
                    print(f"{case['id']} sample {index} action {i + 1}/{len(actions)}", flush=True)
                    if i + 1 == max(1, len(actions) // 2):
                        started = time.perf_counter()
                        saved = session.snapshot().to_dict()
                        timings["mid_snapshot_s"] = time.perf_counter() - started
                        started = time.perf_counter()
                        mid_snapshot = canonical(saved)
                        timings["mid_json_roundtrip_s"] = time.perf_counter() - started
                    if result.info["transaction_status"] != "committed":
                        break
                before_observe = session._state.to_dict()
                started = time.perf_counter()
                observation = session.observe()
                timings["observe_s"] = time.perf_counter() - started
                observed_unchanged = before_observe == session._state.to_dict()
                started = time.perf_counter()
                final_snapshot = session.snapshot().to_dict()
                timings["final_snapshot_s"] = time.perf_counter() - started
                started = time.perf_counter()
                serialized = json.dumps(
                    final_snapshot, allow_nan=False, default=lambda value: value.tolist()
                )
                timings["final_json_encode_s"] = time.perf_counter() - started
                histories.append(
                    {
                        "trace": trace,
                        "final_state": canonical(session._state.to_dict()),
                        "mid_snapshot": mid_snapshot,
                        "component_timings_s": timings,
                        "observe_unchanged": observed_unchanged,
                        "observation": observation.to_dict(),
                        "serialized_bytes": len(serialized.encode("utf-8")),
                    }
                )
            return histories, index
        finally:
            for session in sessions:
                session.close()

    def validate(result):
        nonlocal reference
        histories, index = result
        checks = {
            "complete_actions": all(len(h["trace"]) == len(actions) for h in histories),
            "committed": all(
                r["info"]["transaction_status"] == "committed"
                for h in histories
                for r in h["trace"]
            ),
            "valid_measurement": all(
                r["info"].get("measurement_status") == "valid"
                for h in histories
                for r in h["trace"]
                if r["action"]["operation"] == "measure"
            ),
            "observe_unchanged": all(h["observe_unchanged"] for h in histories),
        }
        final_states = [h["final_state"] for h in histories]
        if case["kind"] == "task_binding":
            checks["same_physics_different_task"] = final_states[0] == final_states[1]
        if case["kind"] == "sbml":
            from chemworld.foundation import WorldState
            from chemworld.runtime.sbml_models import model_record

            final = WorldState.from_dict(final_states[0])
            coordinates = model_record(final).settings["coordinates"]
            middle = WorldState.from_dict(histories[0]["mid_snapshot"]["world_state"])
            pending = model_record(middle).settings["continuation"]["events"]["pending"]
            checks["formal_pending_event"] = len(pending) == 1
            expected = {
                "N:A": 0.02 * exp(-0.15),
                "N:B": 0.01 * (1 - exp(-0.75)) + 0.02 * (1 - exp(-0.15)),
                "P:p": 5.05,
            }
            checks["formal_analytic_values"] = all(
                abs(coordinates[k] - v) <= 1e-8 for k, v in expected.items()
            )
            checks["formal_clock"] = final.ledger.time_s == 3
            histories[0]["formal_reference"] = {
                "expected": expected,
                "actual": {key: coordinates[key] for key in expected},
                "absolute_errors": {
                    key: abs(coordinates[key] - value) for key, value in expected.items()
                },
            }
        if reference is None and all(checks.values()):
            checks["mid_snapshot_exact_resume"] = True
            for history in histories:
                resumed = ChemWorldRuntime.from_snapshot(history["mid_snapshot"])
                try:
                    for i in range(max(1, len(actions) // 2), len(actions)):
                        if i in boundaries:
                            resumed.reset(options=boundaries[i])
                        resumed.execute(actions[i])
                    checks["mid_snapshot_exact_resume"] &= (
                        canonical(resumed._state.to_dict()) == history["final_state"]
                    )
                finally:
                    resumed.close()
            reference = final_states
        elif reference is not None:
            checks["exact_repeated_state"] = final_states == reference
        path = output / f"history-{index}.json"
        write(path, {"histories": histories, "checks": checks})
        passed = all(checks.values())
        return {
            "passed": passed,
            "checks": checks,
            "raw_result": str(path),
            "component_timings_s": [h["component_timings_s"] for h in histories],
            "execute_times_s": [[r["execute_s"] for r in h["trace"]] for h in histories],
            "serialized_bytes": [h["serialized_bytes"] for h in histories],
            **({"failure_details": histories} if not passed else {}),
        }

    return prepare, run, validate
