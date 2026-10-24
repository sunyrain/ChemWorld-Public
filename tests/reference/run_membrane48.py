"""Fixed 30-unit independent DCMD development experiment."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from membrane48_fixtures import amounts, devices, spec
from membrane48_reference import finite, liquid, point

from chemworld.physchem.membrane_distillation import (
    advance_membrane_distillation,
    membrane_distillation_inventory,
    membrane_distillation_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.membrane_distillation import read_membrane_distillation
from chemworld.world.membrane_distillation_world import membrane_distillation_world_spec


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    units, records, outputs = [], [], {}
    started = time.monotonic()

    def compare(name, actual, expected, *, relative=2e-7, absolute=1e-10):
        passed = (
            actual == expected
            if isinstance(expected, (bool, str))
            else math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute)
        )
        records.append(
            {
                "unit": units[-1]["name"],
                "quantity": name,
                "actual": actual,
                "expected": expected,
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "passed": bool(passed),
            }
        )

    def unit(name, function):
        before = len(records)
        row = {"name": name, "error": None}
        units.append(row)
        try:
            outputs[name] = function()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        selected = records[before:]
        row.update(
            comparisons_total=len(selected),
            comparisons_passed=sum(r["passed"] for r in selected),
            failures=[r for r in selected if not r["passed"]],
        )
        row["passed"] = row["error"] is None and not row["failures"]
        elapsed = time.monotonic() - started
        print(
            f"units={len(units)}/30 elapsed={elapsed:.2f}s "
            f"throughput={len(units) / max(elapsed, 0.001):.2f}/s "
            f"ETA={elapsed / len(units) * (30 - len(units)):.2f}s {name} passed={row['passed']}",
            flush=True,
        )

    def property_case(family, t):
        model = spec(family)
        n = amounts(model)
        expected = liquid(family, n, t, 1e5)
        actual = model.properties.liquid(n, t, 1e5)
        p = membrane_distillation_point(model, (n, n), (t, t))
        compare("H", actual.enthalpy_J, expected["enthalpy_J"], absolute=2e-6)
        compare("V", actual.volume_m3, expected["volume_m3"])
        compare("p_vapor", p["vapor_pressures_Pa"][0], expected["vapor_pressure_Pa"])
        compare("PH", model.at_ph(0, n, model.enthalpy(0, n, t)), t, absolute=1e-8)
        return p

    for family in ("ideal", "Davies"):
        for t in (290, 310, 330):
            unit(f"properties:{family}:{t}", lambda f=family, t=t: property_case(f, t))

    def local_case(family, temperatures):
        model = spec(family)
        ns = (amounts(model), amounts(model))
        actual, expected = (
            membrane_distillation_point(model, ns, temperatures),
            point(model, ns, temperatures),
        )
        for key in expected:
            compare(key, actual[key], expected[key])
        return actual

    for family in ("ideal", "Davies"):
        for temperatures in ((320, 300), (300, 320), (310, 310)):
            unit(
                f"local:{family}:{temperatures}", lambda f=family, t=temperatures: local_case(f, t)
            )

    def geometry(field):
        original = spec()
        model = replace(original, **{field: 2 * getattr(original, field)})
        ns = (amounts(model), amounts(model, concentration=0))
        actual, expected = (
            membrane_distillation_point(model, ns, (320, 300)),
            point(model, ns, (320, 300)),
        )
        for key in expected:
            compare(key, actual[key], expected[key])
        return actual

    for field in ("membrane_area_m2", "pore_radius_m", "thickness_m"):
        unit("geometry:" + field, lambda field=field: geometry(field))

    def trajectory(family, mode):
        model = spec(family)
        ns = (amounts(model), amounts(model, concentration=0))
        temperatures = (300, 320) if mode == "reverse" else (320, 300)
        heater, cooler = devices(mode == "heated")
        initial = membrane_distillation_inventory(model, *ns, *temperatures)
        result, receipt = advance_membrane_distillation(
            model, initial, heater=heater, cooler=cooler, duration_s=30
        )
        expected = finite(model, ns, temperatures, heater, cooler, 30)
        final = result.final_state
        for i in range(2):
            compare(
                f"water:{i}",
                final.amounts_mol[i]["H2O"],
                expected["amounts_mol"][i]["H2O"],
                relative=2e-5,
                absolute=2e-9,
            )
            compare(
                f"T:{i}",
                receipt["final"]["temperatures_K"][i],
                expected["temperatures_K"][i],
                relative=0,
                absolute=2e-5,
            )
            compare(
                f"H:{i}",
                final.enthalpies_J[i],
                expected["enthalpies_J"][i],
                relative=2e-5,
                absolute=2e-5,
            )
        compare(
            "heater",
            receipt["heater_energy_into_hot_J"],
            expected["heater_J"],
            relative=2e-5,
            absolute=2e-5,
        )
        compare(
            "cooler",
            receipt["cooler_energy_out_of_cold_J"],
            expected["cooler_J"],
            relative=2e-5,
            absolute=2e-5,
        )
        compare(
            "species_balance",
            max(map(abs, result.component_residuals_mol.values())),
            0,
            absolute=2e-9,
        )
        compare("energy_balance", result.enthalpy_residual_J, 0, absolute=2e-5)
        return {"receipt": receipt, "reference": expected, "solver": result.solver_diagnostic}

    for family in ("ideal", "Davies"):
        for mode in ("passive", "heated", "reverse"):
            unit(f"finite:{family}:{mode}", lambda f=family, m=mode: trajectory(f, m))

    def world(reverse):
        model = spec()
        heater, cooler = devices()
        ns = (amounts(model), amounts(model, concentration=0))
        temperatures = (300, 320) if reverse else (320, 300)
        runtime = ChemWorldRuntime(
            composition=membrane_distillation_world_spec(
                model,
                *ns,
                hot_temperature_K=temperatures[0],
                cold_temperature_K=temperatures[1],
                heater=heater,
                cooler=cooler,
            )
        )
        restored = None
        try:
            runtime.reset(seed=48)
            actions = [
                {"operation": "heat", "duration_s": 2, "target_temperature_K": 325},
                {"operation": "pause_control"},
            ]
            statuses = [runtime.execute(action).info["transaction_status"] for action in actions]
            snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
            (output / f"world-{reverse}-snapshot.json").write_text(
                json.dumps(snapshot, indent=2), encoding="utf-8"
            )
            restored = ChemWorldRuntime.from_snapshot(snapshot)
            for session in (runtime, restored):
                statuses.append(
                    session.execute({"operation": "wait", "duration_s": 3}).info[
                        "transaction_status"
                    ]
                )
            compare("committed", all(s == "committed" for s in statuses), True)
            compare("snapshot_exact", runtime._state.to_dict() == restored._state.to_dict(), True)
            expected = finite(model, ns, temperatures, heater, cooler, 2)
            expected = finite(
                model,
                expected["amounts_mol"],
                expected["temperatures_K"],
                replace(heater, enabled=False),
                cooler,
                3,
            )
            state = runtime._state
            locals_ = (state, state.inactive_vessels["cold"])
            for i, tank in enumerate(locals_):
                compare(
                    f"T:{i}",
                    tank.temperature_K,
                    expected["temperatures_K"][i],
                    relative=0,
                    absolute=2e-5,
                )
                compare(
                    f"water:{i}",
                    tank.species_amounts["H2O"],
                    expected["amounts_mol"][i]["H2O"],
                    relative=2e-5,
                    absolute=2e-9,
                )
            compare(
                "cold_thermal_owner",
                state.inactive_vessels["cold"].thermal.vessels["cold"].heat_loss_J,
                state.ledger.heat_loss_J,
                absolute=2e-8,
            )
            compare("hot_thermal_owner", state.thermal.vessels["reactor"].heat_loss_J, 0.0)
            reading = read_membrane_distillation(state)
            compare("query_not_instrument", reading["instrument_id"] is None, True)
            return {"state": state.to_dict(), "query": reading, "statuses": statuses}
        finally:
            runtime.close()
            if restored:
                restored.close()

    for reverse in (False, True):
        unit("world:" + str(reverse), lambda reverse=reverse: world(reverse))

    def rejection(kind):
        model = spec()
        ns, temperatures = (amounts(model), amounts(model)), (320, 300)
        caught = None
        try:
            if kind == "reference":
                replace(
                    model,
                    vapor=replace(
                        model.vapor,
                        standard_state=replace(
                            model.vapor.standard_state, reference_basis_id="different-basis"
                        ),
                    ),
                )
            else:
                if kind == "entry":
                    model = replace(model, liquid_entry_pressure_Pa=50000)
                if kind == "knudsen":
                    model = replace(model, pore_radius_m=1e-5)
                if kind == "temperature":
                    temperatures = (335, 285)
                if kind == "salinity":
                    ns = (amounts(model, concentration=250), amounts(model))
                membrane_distillation_point(model, ns, temperatures)
        except ValueError as exc:
            caught = str(exc)
        compare("declared_rejection", caught is not None, True)
        return {"expected_failure": caught}

    for kind in ("reference", "entry", "knudsen", "temperature", "salinity"):
        unit("boundary:" + kind, lambda kind=kind: rejection(kind))

    def stopping(kind):
        model = spec()
        ns = (amounts(model), amounts(model, concentration=0))
        if kind == "capacity":
            model = replace(
                model,
                capacities_L=(0.04, liquid("ideal", ns[1], 300, 1e5)["volume_m3"] * 1000 + 1e-7),
            )
        else:
            model = replace(model, minimum_water_mol=1 - 1e-5)
        initial = membrane_distillation_inventory(model, *ns, 320, 300)
        heater, cooler = devices(False)
        result, receipt = advance_membrane_distillation(
            model, initial, heater=heater, cooler=cooler, duration_s=30
        )
        event = "capacity_1" if kind == "capacity" else "water_pickup_0"
        compare("event", result.stopping_event, event)
        compare("actual_time", 0 < result.actual_duration_s < 30, True)
        if kind == "capacity":
            compare(
                "finite_capacity",
                receipt["final"]["volumes_L"][1],
                model.capacities_L[1],
                absolute=1e-10,
            )
        else:
            compare(
                "finite_pickup",
                result.final_state.amounts_mol[0]["H2O"],
                model.minimum_water_mol,
                absolute=1e-10,
            )
        return {"receipt": receipt, "solver": result.solver_diagnostic}

    for kind in ("capacity", "pickup"):
        unit("stop:" + kind, lambda kind=kind: stopping(kind))

    assert len(units) == 30
    summary = {
        "scope": "development; synthetic DCMD equations and native consumers; no upstream parity",
        "units_total": len(units),
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_total": len(records),
        "comparisons_passed": sum(r["passed"] for r in records),
        "elapsed_s": time.monotonic() - started,
        "units": units,
        "failures": [u for u in units if not u["passed"]],
        "expected_rejections": {k: v for k, v in outputs.items() if k.startswith("boundary:")},
    }
    for name, data in (("summary", summary), ("records", records), ("outputs", outputs)):
        (output / (name + ".json")).write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"units", "expected_rejections"}},
            indent=2,
        ),
        flush=True,
    )
    return not summary["failures"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
