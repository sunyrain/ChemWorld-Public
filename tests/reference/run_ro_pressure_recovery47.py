"""Fixed 18-unit / 172-comparison pressure-recovery qualification block."""

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from chemworld.physchem.distributed_ro import (
    DistributedROSpec,
    distributed_ro_profile,
)
from chemworld.physchem.pressure_recovery import (
    IsothermalLiquidPressureRecoverySpec as Recovery,
)
from chemworld.physchem.pressure_recovery import (
    evaluate_isothermal_liquid_pressure_recovery as evaluate,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.distributed_ro import (
    distributed_ro_record,
    unpack_distributed_ro,
)
from chemworld.world.distributed_ro_world import distributed_ro_world_spec
from reference.ro47_fixtures import feed, spec
from reference.ro_pressure_recovery47_reference import axial, finite, recovery

DEVICE = (
    "flow_identity_and_volume",
    "G_availability",
    "H_rate_change",
    "shaft_output",
    "irreversibility",
    "device_heat",
    "entropy_production",
    "energy_entropy_identities",
    "ratings_and_limit_status",
    "codec_and_no_fictitious_phase",
)
PROFILE = (
    "pump_flow",
    "retentate_rates",
    "permeate_rates",
    "outlet_pressure",
    "reference_refinement",
    "G_availability",
    "shaft_output",
    "irreversibility",
    "device_heat",
    "gross_pump_limits",
    "original_material_and_membrane_path",
    "codec_and_T_composition",
)
WORLD = (
    "outcome_reason",
    "actual_time",
    "species_endpoints",
    "H_endpoints",
    "material_charge",
    "original_T_P_V",
    "exported_shaft_work",
    "gross_pump_cost_no_electricity",
    "global_bath",
    "original_energy_balance",
    "event_capacity_and_clocks",
    "exact_snapshot_and_ownership",
)
REJECT = (
    "explicit_rejection",
    "reason_or_physical_preservation",
    "input_or_resources_unchanged",
    "declaration_unchanged",
)


def rows(names):
    return {
        n: {"assessed": False, "passed": False, "detail": "calculation unavailable"} for n in names
    }


def check(data, key, passed, detail=None):
    data["checks"][key] = {"assessed": True, "passed": bool(passed), "detail": detail}


def close(data, key, actual, expected, absolute, relative=0.0):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    error = float(np.max(np.abs(a - b)))
    check(
        data,
        key,
        np.all(np.abs(a - b) <= absolute + relative * np.abs(b)),
        {"maximum_error": error, "absolute_tolerance": absolute, "relative_tolerance": relative},
    )


def codec(model):
    return (
        DistributedROSpec.from_dict(json.loads(json.dumps(model.to_dict()))).to_dict()
        == model.to_dict()
    )


def protected(names, body):
    data = {"checks": rows(names)}
    try:
        body(data)
    except Exception as exc:
        data.update(failure_reason=str(exc), traceback=traceback.format_exc())
    return data


def device_unit(index):
    def run(data):
        family = "ideal" if index <= 4 else "Davies"
        eta, maximum = ({1: 0.0, 2: 0.75, 3: 1.0}.get(index, 0.75), 0.02 if index == 4 else 1.0)
        model = spec(family)
        device = Recovery(eta, maximum, 1.5e6, 1e-6)
        n = feed(model, water_mol=0.0 if index == 6 else 0.01)
        declaration = device.to_dict()
        a = evaluate(
            device,
            model.properties,
            n,
            temperature_K=310.0,
            inlet_pressure_Pa=6e5,
            outlet_pressure_Pa=1e5,
        )
        b = recovery(family, n, 310.0, 6e5, 1e5, eta, maximum)
        data.update(native=a, reference=b, declaration=declaration)
        check(
            data,
            DEVICE[0],
            a["component_rates_mol_s"] == n
            and abs(a["inlet_volume_flow_m3_s"] - b["volume"]) <= 1e-14 + 2e-7 * abs(b["volume"]),
        )
        keys = (
            ("pressure_availability_W", "availability"),
            ("enthalpy_rate_change_W", "enthalpy_change"),
            ("recovered_shaft_power_W", "work"),
            ("irreversibility_W", "dissipation"),
            ("device_heat_into_liquid_W", "heat"),
            ("entropy_production_W_K", "entropy"),
        )
        for name, (actual, expected) in zip(DEVICE[1:7], keys, strict=True):
            close(data, name, a[actual], b[expected], 1e-9 if expected == "entropy" else 1e-7)
        balance = (
            a["enthalpy_rate_change_W"]
            - a["device_heat_into_liquid_W"]
            + a["recovered_shaft_power_W"]
        )
        entropy = (
            0.0 if a["no_material"] else a["outlet"]["entropy_J_K"] - a["inlet"]["entropy_J_K"]
        )
        close(
            data,
            DEVICE[7],
            [balance, 310 * entropy - a["device_heat_into_liquid_W"] - a["irreversibility_W"]],
            [0, 0],
            1e-9,
        )
        check(
            data,
            DEVICE[8],
            a["power_limited"] == b["power_limited"]
            and 0 <= a["recovered_shaft_power_W"] <= min(a["pressure_availability_W"], maximum)
            and a["inlet_volume_flow_m3_s"] <= 1e-6,
        )
        check(
            data,
            DEVICE[9],
            device.to_dict() == declaration
            and Recovery.from_dict(json.loads(json.dumps(declaration))).to_dict() == declaration
            and (index != 6 or a["inlet"] is a["outlet"] is None),
        )

    return protected(DEVICE, run)


def profile_unit(family, temperature):
    def run(data):
        model = replace(
            spec(family, temperature=temperature),
            energy_recovery=Recovery(0.8, 0.02 if family == "ideal" else 1.0, 1.5e6, 1e-6),
        )
        n = feed(model)
        a = distributed_ro_profile(model, n, 10 / 6e7)
        old = distributed_ro_profile(replace(model, energy_recovery=None), n, 10 / 6e7)
        b, coarse = axial(model, n), axial(model, n, cells=64)
        data.update(native=a, reference=b, coarse=coarse, declaration=model.to_dict())
        close(data, PROFILE[0], a["actual_flow_m3_s"], b["flow"], 1e-14, 2e-5)
        for name, key in zip(PROFILE[1:3], ("retentate_mol_s", "permeate_mol_s"), strict=True):
            close(data, name, list(a[key].values()), [b[key][k] for k in a[key]], 1e-12, 2e-5)
        close(data, PROFILE[3], a["stations"][-1]["pressure_Pa"], b["pressure_Pa"], 1e-4, 2e-5)
        check(
            data,
            PROFILE[4],
            abs(coarse["pressure_Pa"] - b["pressure_Pa"]) <= 1e-4 + 1e-5 * abs(b["pressure_Pa"])
            and all(
                abs(coarse["retentate_mol_s"][k] - v) <= 1e-12 + 1e-5 * abs(v)
                for k, v in b["retentate_mol_s"].items()
            ),
        )
        for name, key, ref in zip(
            PROFILE[5:9],
            (
                "pressure_availability_W",
                "recovered_shaft_power_W",
                "irreversibility_W",
                "device_heat_into_liquid_W",
            ),
            ("availability", "work", "dissipation", "heat"),
            strict=True,
        ):
            close(data, name, a["pressure_recovery"][key], b["recovery"][ref], 1e-7)
        check(
            data,
            PROFILE[9],
            abs(a["shaft_power_W"] - b["shaft"]) <= 1e-7
            and a["shaft_power_W"] <= model.pump.maximum_shaft_power_W
            and a["pressure_rise_Pa"] <= model.pump.maximum_pressure_rise_Pa
            and a["shaft_power_W"] == old["shaft_power_W"],
        )
        check(
            data,
            PROFILE[10],
            all(
                abs(a["inlet_mol_s"][k] - a["retentate_mol_s"][k] - a["permeate_mol_s"][k]) <= 1e-12
                for k in n
            )
            and a["stations"] == old["stations"],
        )
        r = a["pressure_recovery"]
        check(
            data,
            PROFILE[11],
            codec(model)
            and r["component_rates_mol_s"] == a["retentate_mol_s"]
            and r["inlet"]["temperature_K"] == r["outlet"]["temperature_K"] == temperature,
        )

    return protected(PROFILE, run)


def all_tanks(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def physical(runtime):
    return {
        n: {
            "amounts": dict(s.species_amounts),
            "T": s.temperature_K,
            "P": s.pressure_Pa,
            "V": s.volume_L,
            "phases": s.phases.to_dict(),
            "H": s.state_slots.to_dict(),
        }
        for n, s in all_tanks(runtime).items()
    }


def make_world(kind):
    model = replace(spec(), energy_recovery=Recovery(0.8, 1.0, 1.5e6, 1e-6))
    n = feed(model)
    reference = axial(model, n)
    if kind == "source":
        target = model.pickup_volume_L + 1000 * reference["flow"] * 2
        scale = target / (model.properties.volume(n, 310.0, 1e5) * 1000)
        n = {k: v * scale for k, v in n.items()}
    elif kind == "receiver":
        model = replace(model, retentate_capacity_L=1000 * reference["recovery"]["volume"] * 1.5)
    elif kind == "blocked":
        model = replace(model, pump=replace(model.pump, maximum_pressure_rise_Pa=1e5))
    elif kind in {"pressure", "flow"}:
        device = replace(
            model.energy_recovery,
            **(
                {"maximum_pressure_drop_Pa": 1e4}
                if kind == "pressure"
                else {"maximum_inlet_volume_flow_m3_s": 1e-12}
            ),
        )
        model = replace(model, energy_recovery=device)
    runtime = ChemWorldRuntime(
        composition=distributed_ro_world_spec(model, n, shared_clock=kind in {"source", "receiver"})
    )
    runtime.reset(seed=47)
    configured = runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 10})
    if configured.info["transaction_status"] != "committed":
        raise RuntimeError("original flow configuration rejected")
    return model, runtime


def world_unit(kind, output):
    def run(data):
        model, runtime = make_world(kind)
        try:
            declaration = model.to_dict()
            initial = unpack_distributed_ro(runtime._state)[2]
            initial_cost = runtime._state.ledger.cost
            snapshot = None
            first_committed = True
            if kind == "receiver":
                first = runtime.execute({"operation": "wait", "duration_s": 0.5})
                data["first_info"] = first.info
                data["first_time_s"] = runtime._state.ledger.time_s
                first_committed = (
                    first.info["transaction_status"] == "committed"
                    and abs(runtime._state.ledger.time_s - 0.5) <= 2e-7
                )
                snapshot = output / "unit13-before-resume.json"
                snapshot.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
            requested = {"normal": 2.0, "source": 3.0, "receiver": 2.0, "blocked": 1.0}[kind]
            outcome = runtime.execute(
                {
                    "operation": "wait" if kind in {"source", "receiver"} else "run_flow",
                    "duration_s": requested,
                }
            )
            data.update(info=outcome.info, declaration=declaration)
            actual = unpack_distributed_ro(runtime._state)[2]
            expected_time = {"normal": 2.0, "source": 2.0, "receiver": 1.5, "blocked": 0.0}[kind]
            event = {
                "normal": None,
                "source": "feed_pickup_exhausted",
                "receiver": "retentate_full",
                "blocked": "no_forward_pump_drive",
            }[kind]
            check(
                data,
                WORLD[0],
                outcome.info["transaction_status"] == "committed"
                and outcome.info["operation_outcome"] == ("success" if event is None else "stopped")
                and (event is None or event in str(outcome.info["outcome_diagnostics"])),
            )
            ref = finite(model, initial.amounts_mol, expected_time)
            data["reference"] = ref
            names = (runtime._state.vessel_id, model.permeate_vessel_id, model.retentate_vessel_id)
            ts = all_tanks(runtime)
            ledger = runtime._state.ledger
            close(data, WORLD[1], actual.time_s, expected_time, 2e-7)

            def ordered(ns):
                return [[n[k] for k in model.properties.component_ids] for n in ns]

            close(data, WORLD[2], ordered(actual.amounts_mol), ordered(ref["amounts"]), 1e-10)
            close(
                data, WORLD[3], actual.enthalpies_J, [s["enthalpy_J"] for s in ref["final"]], 1e-6
            )
            errors = np.sum(ordered(actual.amounts_mol), axis=0) - np.sum(
                ordered(initial.amounts_mol), axis=0
            )
            close(
                data,
                WORLD[4],
                [*errors, *(n["Na+"] - n["Cl-"] for n in actual.amounts_mol)],
                np.zeros(6),
                1e-10,
            )
            check(
                data,
                WORLD[5],
                all(
                    abs(ts[n].temperature_K - 310.0) <= 1e-8
                    and abs(ts[n].pressure_Pa - 1e5) <= 1e-5
                    and abs(ts[n].volume_L - 1000 * p["volume_m3"])
                    <= 1e-10 + 2e-5 * abs(1000 * p["volume_m3"])
                    for n, p in zip(names, ref["final"], strict=True)
                ),
            )
            close(
                data,
                WORLD[6],
                -ledger.flow_shaft_work_J,
                ref["recovery"]["work"] * expected_time,
                1e-6,
            )
            check(
                data,
                WORLD[7],
                abs(ledger.flow_pump_work_J - ref["shaft"] * expected_time) <= 1e-6
                and ledger.flow_electrical_work_J == 0
                and abs(ledger.cost - initial_cost - ledger.flow_pump_work_J / 250000) <= 1e-12,
            )
            motor = (ref["shaft"] - ref["hydraulic"]) * expected_time
            bath = ledger.energy_jacket_J - ledger.heat_loss_J + motor
            close(data, WORLD[8], bath, ref["bath"], 1e-6)
            close(
                data,
                WORLD[9],
                sum(actual.enthalpies_J) - sum(initial.enthalpies_J),
                ledger.flow_pump_work_J + ledger.flow_shaft_work_J - motor + bath,
                1e-6,
            )
            check(
                data,
                WORLD[10],
                all(abs(s.ledger.time_s - expected_time) <= 2e-7 for s in ts.values())
                and ts[model.retentate_vessel_id].volume_L <= model.retentate_capacity_L + 1e-10
                and ts[model.permeate_vessel_id].volume_L <= model.permeate_capacity_L + 1e-10
                and (event is None or event in str(outcome.info["outcome_diagnostics"])),
            )
            consistent = first_committed and (
                declaration == distributed_ro_record(runtime._state).settings["physical_model"]
            )
            if snapshot is not None:
                target = output / "unit13-resumed.json"
                process = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "python",
                        str(Path(__file__).with_name("ro_pressure_recovery47_resume.py")),
                        str(snapshot),
                        str(target),
                        "2",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                data["resume_stderr"] = process.stderr
                consistent = (
                    consistent
                    and process.returncode == 0
                    and json.loads(target.read_text(encoding="utf-8"))["snapshot"]
                    == json.loads(json.dumps(runtime.snapshot().to_dict()))
                )
            else:
                restored = ChemWorldRuntime.from_snapshot(
                    json.loads(json.dumps(runtime.snapshot().to_dict()))
                )
                consistent = consistent and physical(restored) == physical(runtime)
                restored.close()
            check(data, WORLD[11], consistent)
        finally:
            runtime.close()

    return protected(WORLD, run)


def rejection_unit(kind):
    def run(data):
        if kind in {"pressure", "flow"}:
            model, runtime = make_world(kind)
            try:
                before = physical(runtime)
                declaration = model.to_dict()
                outcome = runtime.execute({"operation": "wait", "duration_s": 1.0})
                data["info"] = outcome.info
                reason = "pressure-drop" if kind == "pressure" else "inlet-volume-flow"
                check(
                    data,
                    REJECT[0],
                    outcome.info["operation_outcome"] == "numerical_failure"
                    and reason in str(outcome.info["outcome_diagnostics"]),
                )
                check(data, REJECT[1], before == physical(runtime))
                check(
                    data,
                    REJECT[2],
                    all(s.ledger.time_s == 0 for s in all_tanks(runtime).values())
                    and all(
                        getattr(runtime._state.ledger, k) == 0
                        for k in (
                            "energy_jacket_J",
                            "heat_loss_J",
                            "flow_pump_work_J",
                            "flow_shaft_work_J",
                            "flow_electrical_work_J",
                        )
                    ),
                )
                check(
                    data,
                    REJECT[3],
                    distributed_ro_record(runtime._state).settings["physical_model"] == declaration,
                )
            finally:
                runtime.close()
        else:
            model, device = spec("ideal"), Recovery(0.75, 1.0, 1.5e6, 1e-6)
            n = feed(model, water_mol=0.01)
            original, declaration = dict(n), device.to_dict()
            failure = None
            try:
                evaluate(
                    device,
                    model.properties,
                    n,
                    temperature_K=350.0 if kind == "temperature" else 310.0,
                    inlet_pressure_Pa=1e5 if kind == "reverse" else 6e5,
                    outlet_pressure_Pa=6e5 if kind == "reverse" else 1e5,
                )
            except ValueError as exc:
                failure = str(exc)
            check(data, REJECT[0], failure is not None, failure)
            check(
                data,
                REJECT[1],
                failure is not None
                and (
                    "forward" in failure
                    if kind == "reverse"
                    else any(w in failure for w in ("range", "domain", "outside"))
                ),
                failure,
            )
            check(data, REJECT[2], n == original)
            check(
                data,
                REJECT[3],
                device.to_dict() == declaration
                and Recovery.from_dict(declaration).to_dict() == declaration,
            )

    return protected(REJECT, run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = [(f"device-{i}", lambda i=i: device_unit(i), 10) for i in range(1, 7)]
    units += [
        (f"profile-{f}-{t}", lambda f=f, t=t: profile_unit(f, t), 12)
        for f in ("ideal", "Davies")
        for t in (290.0, 330.0)
    ]
    units += [
        ("world-" + k, lambda k=k: world_unit(k, args.output), 12)
        for k in ("normal", "source", "receiver", "blocked")
    ]
    units += [
        ("reject-" + k, lambda k=k: rejection_unit(k), 4)
        for k in ("pressure", "flow", "reverse", "temperature")
    ]
    start, completed, stop = monotonic(), [], Event()
    live = {"stage": "start", "completed": 0}

    def progress():
        elapsed, n = monotonic() - start, live["completed"]
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = f"{(18 - n) / rate:.1f}s" if n and rate else "pending first unit"
        print(
            f"47B stage={live['stage']} completed={n}/18 elapsed={elapsed:.1f}s "
            f"rate={rate:.3f}/s ETA={eta}",
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for i, (name, execute, count) in enumerate(units, 1):
            live["stage"] = name
            progress()
            tick = monotonic()
            data = execute()
            if len(data["checks"]) != count:
                raise RuntimeError("fixed comparison denominator changed")
            data.update(id=name, elapsed_s=monotonic() - tick, comparisons_total=count)
            data["comparisons_assessed"] = sum(c["assessed"] for c in data["checks"].values())
            data["comparisons_passed"] = sum(c["passed"] for c in data["checks"].values())
            data["passed"] = data["comparisons_passed"] == count
            (args.output / f"unit{i:02d}.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            completed.append(data)
            live["completed"] = i
            progress()
    finally:
        stop.set()
        thread.join()
    assessed = sum(d["comparisons_assessed"] for d in completed)
    passed = sum(d["comparisons_passed"] for d in completed)
    summary = {
        "task": "47B",
        "mode": "development",
        "units_total": 18,
        "units_attempted": len(completed),
        "units_passed": sum(d["passed"] for d in completed),
        "comparisons_total": 172,
        "comparisons_assessed": assessed,
        "comparisons_passed": passed,
        "comparisons_measured_failed": assessed - passed,
        "comparisons_unexecuted_unavailable": 172 - assessed,
        "elapsed_s": monotonic() - start,
        "rows": [
            {
                k: d[k]
                for k in (
                    "id",
                    "passed",
                    "comparisons_total",
                    "comparisons_assessed",
                    "comparisons_passed",
                )
            }
            | {
                "failed_comparisons": [k for k, v in d["checks"].items() if not v["passed"]],
                "failure_reason": d.get("failure_reason", ""),
            }
            for d in completed
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
