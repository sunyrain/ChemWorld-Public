"""Fixed 26-unit / 300-check thermal feedback development block."""

import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

from run_thermal_feedback33_canary import execute, memory, reactor
from run_thermal_feedback33_canary import unit as world_unit
from thermal_feedback33_cases import case, dynamic_cases
from thermal_feedback33_reference import Reference

from chemworld.examples.thermal_feedback import thermal_feedback_world
from chemworld.foundation.state_ownership import StateSlot, StateSlots
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thermal_feedback import feedback_slot_spec

CHECKS = (
    "T",
    "I",
    "Tf",
    "Q",
    "loss",
    "control",
    "event",
    "reference_closure",
    "native_closure",
    "stock",
    "no_fake_work",
    "memory_pause",
)
LIFECYCLE = (
    "handoff_owner",
    "handoff_memory",
    "material_memory",
    "resources_preserved",
    "replacement_memory",
    "missing_memory_rejected",
    "snapshot_exact",
    "fresh_process_exact",
    "one_shared_clock",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def composition(card):
    spec = thermal_feedback_world()
    physics = spec.physics
    state = physics.initial_states["reactor"]
    temperature, integral, filtered = card["initial"][:3]
    records = dict(state.equipment.equipment)
    record = records["feedback-heater"]
    records[record.equipment_id] = replace(
        record,
        settings={
            "thermal_device": card["device"],
            "thermal_feedback": card["controller"],
        },
    )
    slot = memory(state)
    state = state.replace(
        temperature_K=temperature,
        pressure_Pa=0.01 * 8.31446261815324 * temperature * 1000,
        equipment=replace(state.equipment, equipment=records),
        state_slots=StateSlots(
            (
                replace(
                    slot,
                    values={"integral_K": integral, "filtered_temperature_K": filtered},
                    initial_values={"integral_K": integral, "filtered_temperature_K": filtered},
                ),
            )
        ),
    )
    p = card["plant"]
    model = replace(
        physics.models_of("thermal")["reactor"],
        environment_temperature_K=p["ambient_K"],
        loss_conductance_W_K=p["loss_W_K"],
        minimum_temperature_K=p["minimum_temperature_K"],
        stop_temperature_K=p["stop_temperature_K"],
        maximum_pressure_Pa=p["maximum_pressure_Pa"],
    )
    return replace(
        spec,
        components=tuple(
            replace(
                component,
                parameters={
                    **component.parameters,
                    "temperature_range_K": [
                        model.minimum_temperature_K,
                        model.maximum_temperature_K,
                    ],
                },
            )
            if component.kind == "thermal"
            else component
            for component in spec.components
        ),
        physics=replace(
            physics, initial_states={"reactor": state}, models={"thermal": {"reactor": model}}
        ),
    )


def native(card):
    runtime = ChemWorldRuntime(composition=composition(card))
    samples, reports, paused = [], [], []
    stopping = None
    try:
        runtime.reset(seed=33)
        for segment in card["segments"]:
            before_memory = dict(memory(runtime._state).values)
            if not segment["enabled"]:
                execute(runtime, "pause_control")
            local = 0.0
            for index, target in enumerate(segment["sample_times_s"] or [segment["duration_s"]]):
                before = runtime._state.ledger.time_s
                result = execute(
                    runtime,
                    "heat" if segment["enabled"] and index == 0 else "wait",
                    duration_s=target - local,
                    **(
                        {"target_temperature_K": segment["target_K"]}
                        if segment["enabled"] and index == 0
                        else {}
                    ),
                )
                state = runtime._state
                local += state.ledger.time_s - before
                diag = result.info["outcome_diagnostics"]
                stopping = diag.get("stopping_event")
                m, ledger = memory(state), state.ledger
                samples.append(
                    {
                        "time_s": ledger.time_s,
                        "values": [
                            state.temperature_K,
                            m.values["integral_K"],
                            m.values["filtered_temperature_K"],
                            ledger.energy_jacket_J,
                            ledger.heat_loss_J,
                        ],
                        "point": diag["thermal_feedback"],
                        "amounts": dict(state.species_amounts),
                        "work": [
                            ledger.boundary_work_J,
                            ledger.flow_shaft_work_J,
                            ledger.heat_reaction_J,
                        ],
                        "memory_owner": [m.spec.owner_kind, m.spec.owner_id, m.location_id],
                    }
                )
                reports.append(diag)
                if stopping:
                    break
            if not segment["enabled"]:
                paused.append(
                    {"before": before_memory, "after": dict(memory(runtime._state).values)}
                )
            if stopping:
                break
        return {
            "samples": samples,
            "stopping_event": stopping,
            "reports": reports,
            "paused": paused,
        }
    finally:
        runtime.close()


def dynamic_unit(card, row, _):
    row["input"] = card
    row["reference"] = ref = Reference(card, idaes=card.get("idaes", False)).solve()
    row["native"] = actual = native(card)
    pairs = list(zip(actual["samples"], ref["samples"], strict=True))
    errors = {}
    for index, key, tolerance in (
        (0, "T", 2e-4),
        (1, "I", 2e-3),
        (2, "Tf", 2e-4),
        (3, "Q", 0.02),
        (4, "loss", 0.02),
    ):
        error = max(abs(a["values"][index] - b["values"][index]) for a, b in pairs)
        errors[key] = {"maximum_absolute_error": error, "tolerance": tolerance}
        row["checks"][key] = error <= tolerance
    control = max(
        abs(a["point"][key] - b["point"][other])
        for a, b in pairs
        for key, other in (
            ("actual_heater_power_W", "power"),
            ("command_K", "command"),
            ("realized_command_K", "realized"),
        )
    )
    row["checks"]["control"] = control <= 0.002
    errors["control"] = {"maximum_absolute_error": control, "tolerance": 0.002}
    time_error = max(abs(a["time_s"] - b["time_s"]) for a, b in pairs)
    row["checks"]["event"] = (
        actual["stopping_event"] == ref["stopping_event"] == card["expected_event"]
        and time_error <= 2e-4
    )
    errors["event"] = {"maximum_time_error_s": time_error, "tolerance_s": 2e-4}
    closure = max(
        abs(
            card["plant"]["capacity_J_K"] * (r["values"][0] - card["initial"][0])
            - r["values"][3]
            + r["values"][4]
        )
        for r in ref["samples"]
    )
    matrix = max(
        (
            max(abs(a - b) for a, b in zip(r["values"], r["matrix_exponential"], strict=True))
            for r in ref["samples"]
            if "matrix_exponential" in r
        ),
        default=0.0,
    )
    row["checks"]["reference_closure"] = closure <= 2e-7 and matrix <= 2e-7
    errors["reference_closure"] = {
        "energy_error_J": closure,
        "matrix_maximum_error": matrix,
        "tolerance": 2e-7,
    }
    row["checks"]["native_closure"] = all(
        abs(r["energy_balance_residual_J"])
        <= 2e-6 + 5e-6 * (abs(s["values"][3]) + abs(s["values"][4]))
        for r, s in zip(actual["reports"], actual["samples"], strict=True)
    )
    row["checks"]["stock"] = all(abs(s["amounts"]["A"] - 0.01) <= 1e-12 for s in actual["samples"])
    row["checks"]["no_fake_work"] = all(all(v == 0 for v in s["work"]) for s in actual["samples"])
    row["checks"]["memory_pause"] = all(
        s["memory_owner"] == ["equipment", "feedback-heater", "reactor"] for s in actual["samples"]
    ) and all(
        abs(p["before"]["integral_K"] - p["after"]["integral_K"]) <= 1e-10 for p in actual["paused"]
    )
    row["errors"] = errors


def continue_snapshot(runtime, folder, prefix):
    payload = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
    source, target = folder / f"{prefix}-snapshot.json", folder / f"{prefix}-fresh.json"
    save(source, payload)
    replay = ChemWorldRuntime.from_snapshot(payload)
    code = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"p=json.loads(Path({str(source)!r}).read_text(encoding='utf8')); "
        "r=ChemWorldRuntime.from_snapshot(p); "
        "z=r.execute({'operation':'wait','duration_s':.25}); "
        "assert z.info['transaction_status']=='committed',z.info; "
        f"p=Path({str(target)!r}); p.write_text(json.dumps(r._state.to_dict()),encoding='utf8'); "
        "r.close()"
    )
    try:
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if child.returncode:
            raise RuntimeError(child.stderr)
        for session in (runtime, replay):
            execute(session, "wait", duration_s=0.25)
        return {
            "in_memory": runtime._state.to_dict() == replay._state.to_dict(),
            "fresh": json.loads(target.read_text(encoding="utf8"))
            == json.loads(json.dumps(runtime._state.to_dict())),
            "child": {
                "returncode": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
            },
        }
    finally:
        replay.close()


def lifecycle_unit(_, row, folder):
    runtime = local = None
    try:
        runtime = ChemWorldRuntime(composition=thermal_feedback_world("PID", shared=True))
        runtime.reset(seed=33)
        execute(runtime, "queue_lab_heat", target_temperature_K=350.0, duration_s=1.0)
        execute(runtime, "wait", duration_s=1.0)
        previous = dict(memory(runtime._state).values)
        execute(runtime, "select_vessel", vessel="witness")
        execute(runtime, "queue_lab_heat", target_temperature_K=350.0, duration_s=1.0)
        execute(runtime, "wait", duration_s=1e-6)
        state = runtime._state
        moved = next(
            s for s in state.state_slots.slots if s.spec.state_type == "thermal-feedback-memory-v1"
        )
        row["checks"]["handoff_owner"] = (
            moved.location_id == "witness"
            and moved.spec.owner_id == "feedback-heater"
            and not any(
                s.spec.owner_id == "feedback-heater" for s in reactor(state).state_slots.slots
            )
        )
        row["checks"]["handoff_memory"] = all(
            abs(moved.values[k] - v) < 0.001 for k, v in previous.items()
        )
        execute(runtime, "wait", duration_s=1.0)
        before = runtime._state
        row["checks"]["one_shared_clock"] = abs(before.ledger.time_s - 2.0) < 1e-12 and all(
            abs(t - 2.0) < 1e-12 for t in before.vessel_elapsed_s.values()
        )
        runtime.reset(options={"scope": "material", "vessel_ids": ["witness"]})
        row["checks"]["material_memory"] = runtime._state.state_slots == before.state_slots
        row["checks"]["resources_preserved"] = runtime._state.ledger == before.ledger
        row["shared_continuation"] = shared = continue_snapshot(runtime, folder, "lifecycle-shared")
        local = ChemWorldRuntime(composition=thermal_feedback_world("PID"))
        local.reset(seed=33)
        execute(local, "heat", target_temperature_K=350.0, duration_s=1.0)
        record = local._state.equipment.equipment["feedback-heater"]
        replacement = replace(record, equipment_id="replacement-heater")
        before = local.snapshot().to_dict()
        try:
            local.reset(
                options={
                    "scope": "equipment",
                    "equipment_id": "feedback-heater",
                    "replacement": replacement.to_dict(),
                }
            )
        except ValueError:
            row["checks"]["missing_memory_rejected"] = local.snapshot().to_dict() == before
        slot = StateSlot(
            feedback_slot_spec("replacement-heater"),
            "reactor",
            {"integral_K": 7.0, "filtered_temperature_K": 320.0},
        )
        ledger = local._state.ledger
        local.reset(
            options={
                "scope": "equipment",
                "equipment_id": "feedback-heater",
                "replacement": replacement.to_dict(),
                "slots": StateSlots((slot,)).to_dict(),
            }
        )
        row["checks"]["replacement_memory"] = (
            memory(local._state).values == slot.values
            and memory(local._state).spec.owner_id == "replacement-heater"
        )
        row["checks"]["resources_preserved"] &= local._state.ledger == ledger
        row["replacement_continuation"] = replaced = continue_snapshot(
            local, folder, "lifecycle-replacement"
        )
        row["checks"]["snapshot_exact"] = shared["in_memory"] and replaced["in_memory"]
        row["checks"]["fresh_process_exact"] = shared["fresh"] and replaced["fresh"]
        row["final_shared_state"], row["final_replaced_state"] = (
            runtime._state.to_dict(),
            local._state.to_dict(),
        )
    finally:
        if runtime:
            runtime.close()
        if local:
            local.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = [(c["id"], dynamic_unit, c, CHECKS) for c in dynamic_cases()]
    for kind in ("P", "PI"):
        card = case(kind)
        card.update(id="actual-IDAES-" + kind, idaes=True)
        units.append((card["id"], dynamic_unit, card, CHECKS))
    from run_thermal_feedback33_canary import CHECKS as WORLD_CHECKS

    for identity in ("local-PI", "local-PID-pause", "shared-PID"):
        units.append((identity, None, {}, WORLD_CHECKS))
    units.append(("device-lifecycle", lifecycle_unit, {}, LIFECYCLE))
    rows, done, start = [], Event(), time.monotonic()

    def progress():
        elapsed = time.monotonic() - start
        rate = len(rows) / elapsed
        passed = sum(sum(r["checks"].values()) for r in rows)
        print(
            f"33D units={len(rows)}/26 checks={passed}/300 elapsed={elapsed:.1f}s "
            f"rate={rate:.3f}/s ETA={(26 - len(rows)) / rate if rate else -1:.1f}s",
            flush=True,
        )

    def heartbeat():
        while not done.wait(30):
            progress()

    Thread(target=heartbeat, daemon=True).start()
    for identity, run, card, checks in units:
        row = {"id": identity, "checks": dict.fromkeys(checks, False), "failures": []}
        begin = time.monotonic()
        print("33D start", identity, flush=True)
        try:
            if run is None:
                row = world_unit(identity, args.output)
            else:
                run(copy.deepcopy(card), row, args.output)
        except Exception as exc:
            row["failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        row["elapsed_s"] = time.monotonic() - begin
        row["passed"] = all(row["checks"].values()) and not row["failures"]
        rows.append(row)
        save(args.output / (identity + ".json"), row)
        save(
            args.output / "results.json",
            {
                "development_only": True,
                "units_planned": 26,
                "checks_planned": 300,
                "units_completed": len(rows),
                "units_passed": sum(r["passed"] for r in rows),
                "checks_passed": sum(sum(r["checks"].values()) for r in rows),
                "elapsed_s": time.monotonic() - start,
                "rows": rows,
            },
        )
        progress()
    done.set()
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
