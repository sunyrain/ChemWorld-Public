"""Fixed 40C 24-unit independent qualification, with all outcomes retained."""

import argparse
import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic


def main():
    # Standalone entry and pytest use the same reference package, without a
    # second fixture copy or importing native physics into the oracle.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chemworld.physchem.aqueous_mineral_kinetics import advance_aqueous_minerals
    from reference.aqueous40_kinetics_cases import cases, inputs, observation_times
    from reference.aqueous40_kinetics_reference import reference
    from reference.aqueous40_mineral_reference import invariants

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    if output.exists():
        raise ValueError("completed results are immutable; choose a new output")
    started, completed, stage = monotonic(), 0, "initializing"
    stop = Event()

    def heartbeat():
        while not stop.wait(30):
            elapsed = monotonic() - started
            rate = completed / elapsed
            print(
                f"40C {completed}/24 {stage}; elapsed={elapsed:.1f}s rate={rate:.3f}/s "
                f"ETA={((24 - completed) / rate if rate else 0):.1f}s",
                flush=True,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    keys = ("species", "areas", "rates", "events", "conservation", "heat", "work", "first_law")
    tolerances = dict(zip(keys, (2e-6, 2e-6, 2e-9, 0.5, 2e-8, 0.002, 2e-7, 2e-7), strict=True))
    rows = []
    try:
        for case in cases():
            for strategy in ("BDF", "Radau"):
                checks = dict.fromkeys(keys, False)
                row = {"condition": case["id"], "strategy": strategy, "checks": checks}
                try:
                    stage = case["id"] + ":independent"
                    times = observation_times(case)
                    ref = reference(case, times)
                    row["reference"] = ref
                    stage = case["id"] + ":" + strategy
                    spec, amounts, geometry = inputs(case, strategy)
                    native = advance_aqueous_minerals(
                        spec,
                        amounts,
                        geometry,
                        case["duration_s"],
                        temperature_K=case["temperature_K"],
                        pressure_Pa=case["pressure_Pa"],
                        evaluation_times_s=times,
                    )
                    row["native"] = native.to_dict()
                    errors = dict.fromkeys(keys, 0.0)
                    original = invariants(case, ref["initial"])
                    for point, expected in zip(native.trajectory, ref["trajectory"], strict=True):
                        for k, n in expected["amounts_mol"].items():
                            errors["species"] = max(
                                errors["species"],
                                abs(point.amounts_mol[k] - n) / max(ref["initial"][k], 0.01),
                            )
                        for mineral, area in expected["areas"].items():
                            errors["areas"] = max(
                                errors["areas"], abs(point.surface_areas_m2[mineral] - area)
                            )
                            errors["rates"] = max(
                                errors["rates"],
                                abs(
                                    point.dissolution_rates_mol_s[mineral]
                                    - expected["rates"][mineral]
                                ),
                            )
                        conserved = invariants(case, point.amounts_mol)
                        charge = (
                            point.amounts_mol["Na+"]
                            + point.amounts_mol.get("K+", 0)
                            - point.amounts_mol["Cl-"]
                        )
                        errors["conservation"] = max(
                            errors["conservation"],
                            abs(charge),
                            *(abs(conserved[k] - v) for k, v in original.items()),
                        )
                        errors["heat"] = max(
                            errors["heat"], abs(point.heat_duty_J - expected["heat_J"])
                        )
                        errors["work"] = max(
                            errors["work"], abs(point.boundary_work_J - expected["work_J"])
                        )
                        errors["first_law"] = max(
                            errors["first_law"],
                            abs(
                                point.internal_energy_change_J
                                - point.heat_duty_J
                                + point.boundary_work_J
                            ),
                        )
                    actual_events = {e["mineral_id"]: e["time_s"] for e in native.events}
                    expected_events = {e["mineral_id"]: e["time_s"] for e in ref["events"]}
                    if set(actual_events) != set(expected_events) or len(native.events) != len(
                        ref["events"]
                    ):
                        errors["events"] = 1e100
                    else:
                        errors["events"] = max(
                            (abs(actual_events[k] - v) for k, v in expected_events.items()),
                            default=0.0,
                        )
                    checks.update({k: bool(errors[k] <= tolerances[k]) for k in keys})
                    row["errors"] = errors
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                completed += 1
                elapsed = monotonic() - started
                print(
                    f"40C {completed}/24 {case['id']} {strategy}: {sum(checks.values())}/8 "
                    f"rate={completed / elapsed:.3f}/s "
                    f"ETA={(24 - completed) * elapsed / completed:.1f}s",
                    flush=True,
                )
        summary = {
            "units": 24,
            "native_successful": sum("native" in row for row in rows),
            "state_observations": 144,
            "comparisons": 192,
            "passed": sum(sum(r["checks"].values()) for r in rows),
            "failures": [
                {
                    "condition": r["condition"],
                    "strategy": r["strategy"],
                    "checks": [k for k, v in r["checks"].items() if not v],
                    "error": r.get("error"),
                }
                for r in rows
                if not all(r["checks"].values())
            ],
            "elapsed_s": monotonic() - started,
        }
        payload = {
            "mode": "development",
            "fixed_cases": cases(),
            "tolerances": tolerances,
            "summary": summary,
            "runs": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
