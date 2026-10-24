"""Prospective fixed 30-unit independent mathematical/native-consumer block."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from ro47_fixtures import feed, spec
from ro47_reference import coefficient, local, profile, properties

from chemworld.physchem.distributed_ro import (
    advance_distributed_ro,
    distributed_ro_inventory,
    distributed_ro_profile,
    local_ro_flux,
)
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.distributed_ro_world import distributed_ro_world_spec


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    records, units = [], []
    started = time.monotonic()

    def compare(name, actual, expected, *, relative=2e-7, absolute=1e-10):
        ok = (
            actual == expected
            if isinstance(expected, bool)
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
                "passed": bool(ok),
            }
        )

    def unit(name, function):
        before = len(records)
        row = {"name": name, "error": None}
        units.append(row)
        try:
            function()
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
        model, p = spec(family, temperature=t), 5e5
        n = feed(model, 40)
        expected = properties(family, n, t, p)
        result = model.properties.liquid(n, t, p)
        activity = model.properties.database.chemical_potentials(n, temperature_K=t, pressure_Pa=p)
        compare("H", result.enthalpy_J, expected["enthalpy_J"], absolute=1e-6)
        compare("V", result.volume_m3, expected["volume_m3"])
        compare("ln_aw", activity.log_activities["H2O"], expected["log_water_activity"])

    for family in ("ideal", "Davies"):
        for t in (290, 310, 330):
            unit(f"properties:{family}:{t}", lambda f=family, t=t: property_case(f, t))

    def local_case(family, p):
        model = spec(family)
        actual, expected = local_ro_flux(model, 30, p), local(model, 30, p)
        for name in expected:
            compare(name, actual[name], expected[name])

    for family in ("ideal", "Davies"):
        for p in (3e5, 5e5, 8e5):
            unit(f"local:{family}:{p}", lambda f=family, p=p: local_case(f, p))

    def pure(t):
        model = spec(temperature=t)
        row = local_ro_flux(model, 0, 5e5)
        compare(
            "pure_water_flux",
            row["water_flux_m_s"],
            coefficient(model.water_permeability, t, 5e5, 0) * 4e5,
        )
        compare("no_salt_flux", row["salt_flux_mol_m2_s"], 0.0)

    for t in (290, 310, 330):
        unit(f"pure:{t}", lambda t=t: pure(t))

    def axial(family, length):
        model = spec(family, length=length)
        n = feed(model)
        actual = distributed_ro_profile(model, n, 10 / 6e7)
        coarse, fine = profile(model, n, 10 / 6e7, 64), profile(model, n, 10 / 6e7, 128)
        for key in ("water_mol_s", "salt_mol_s", "pressure_Pa"):
            compare(key, actual["stations"][-1][key], fine[key], relative=2e-5)
            compare("refinement:" + key, coarse[key], fine[key], relative=1e-5)
        for key in n:
            compare(
                "water_ions:" + key,
                actual["permeate_mol_s"][key] + actual["retentate_mol_s"][key],
                actual["inlet_mol_s"][key],
                absolute=1e-9,
            )

    for family, length in (("ideal", 0.5), ("ideal", 2.0), ("Davies", 1.0)):
        unit(f"axial:{family}:{length}", lambda f=family, length=length: axial(f, length))

    def finite(prefilled):
        model = spec()
        n = feed(model)
        initial = distributed_ro_inventory(
            model,
            n,
            permeate=feed(model, 70, 0.2) if prefilled else None,
            retentate=feed(model, 10, 0.3) if prefilled else None,
        )
        result, d = advance_distributed_ro(model, initial, 10 / 6e7, 60)
        ref = profile(model, n, 10 / 6e7)
        flows = (
            {k: -v for k, v in ref["inlet_mol_s"].items()},
            ref["permeate_mol_s"],
            ref["retentate_mol_s"],
        )
        for i, row in enumerate(result.final_state.amounts_mol):
            for key in row:
                compare(
                    f"tank:{i}:{key}",
                    row[key],
                    initial.amounts_mol[i][key] + 60 * flows[i][key],
                    absolute=1e-9,
                )
            compare(
                f"tank_H:{i}",
                result.final_state.enthalpies_J[i],
                properties("Davies", row, 310, 1e5)["enthalpy_J"],
                absolute=2e-5,
            )
        for key in n:
            compare(
                "closed:" + key,
                sum(r[key] for r in result.final_state.amounts_mol),
                sum(r[key] for r in initial.amounts_mol),
                absolute=1e-9,
            )
        compare("shaft", d["shaft_work_J"], (10 / 6e7) * 4e5 / 0.75 * 60, absolute=2e-5)
        compare(
            "energy",
            d["shaft_work_J"] - d["motor_loss_J"] + d["bath_heat_into_liquid_J"],
            d["liquid_enthalpy_change_J"],
            absolute=2e-5,
        )

    for prefilled in (False, True):
        unit(f"finite:{prefilled}", lambda p=prefilled: finite(p))

    def world(t):
        model = spec(temperature=t)
        composition = distributed_ro_world_spec(model, feed(model))
        original = ChemWorldRuntime(composition=composition)
        resumed = None
        try:
            original.reset(seed=47)
            for action in (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
                {"operation": "wait", "duration_s": 2},
            ):
                response = original.execute(action)
                compare("committed", response.info["transaction_status"] == "committed", True)
            snapshot = json.loads(json.dumps(original.snapshot().to_dict()))
            (output / f"world-{t}.json").write_text(json.dumps(composition.to_dict(), indent=2))
            (output / f"snapshot-{t}.json").write_text(json.dumps(snapshot, indent=2))
            resumed = ChemWorldRuntime.from_snapshot(snapshot)
            for runtime in (original, resumed):
                response = runtime.execute({"operation": "run_flow", "duration_s": 3})
                compare("continued", response.info["transaction_status"] == "committed", True)
                compare("time", runtime._state.ledger.time_s, 5.0)
            compare("exact_snapshot", original._state.to_dict() == resumed._state.to_dict(), True)
            states = (original._state, *original._state.inactive_vessels.values())
            for key, amount in feed(model).items():
                compare(
                    "actual_stock:" + key,
                    sum(s.species_amounts[key] for s in states),
                    amount,
                    absolute=1e-9,
                )
        finally:
            original.close()
            if resumed is not None:
                resumed.close()

    for t in (290, 310, 330):
        unit(f"world:{t}", lambda t=t: world(t))

    def stop():
        model = replace(spec(), retentate_capacity_L=0.001)
        r, _ = advance_distributed_ro(
            model, distributed_ro_inventory(model, feed(model)), 10 / 6e7, 60
        )
        compare("stop", r.stopping_event == "retentate_full", True)
        compare("actual_time", 0 < r.actual_duration_s < 60, True)
        compare(
            "volume",
            properties("Davies", r.final_state.amounts_mol[2], 310, 1e5)["volume_m3"],
            1e-6,
        )

    unit("finite_receiver_stop", stop)

    def blocked():
        model = replace(spec(), pump=PumpSpec(100, 1, 0.75))
        r, d = advance_distributed_ro(
            model, distributed_ro_inventory(model, feed(model)), 10 / 6e7, 60
        )
        compare("blocked", r.stopping_event == "no_forward_pump_drive", True)
        compare("time", r.actual_duration_s, 0.0)
        compare("shaft", d["shaft_work_J"], 0.0)

    unit("blocked_pump", blocked)

    model = spec()
    unbalanced = feed(model)
    unbalanced["Cl-"] *= 1.1
    for name, call in (
        ("ions", lambda: model.properties.validate_amounts(unbalanced)),
        ("temperature", lambda: replace(model, temperature_K=350)),
        ("interface", lambda: local_ro_flux(model, 201, 5e5)),
        ("reverse_pressure", lambda: local_ro_flux(model, 30, 9e4)),
        (
            "source",
            lambda: replace(
                model, water_permeability=replace(model.water_permeability, source_id="absent")
            ),
        ),
    ):

        def rejected(call=call):
            try:
                call()
            except ValueError:
                compare("reject", True, True)
            else:
                compare("reject", False, True)

        unit("reject:" + name, rejected)

    summary = {
        "status": "passed" if all(u["passed"] for u in units) and len(units) == 30 else "failed",
        "units_total": len(units),
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_total": len(records),
        "comparisons_passed": sum(r["passed"] for r in records),
        "failures": [u for u in units if not u["passed"]],
        "units": units,
        "elapsed_seconds": time.monotonic() - started,
        "evidence_status": "development; synthetic data; independent mathematical reference",
    }
    (output / "records.json").write_text(json.dumps(records, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("units", "failures")}), flush=True
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
