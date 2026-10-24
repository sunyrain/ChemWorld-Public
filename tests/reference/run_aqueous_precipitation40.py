"""Fixed development block: eight finite mixing routes, eight comparisons each."""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

from chemworld.examples.aqueous_precipitation import route, synthetic_world
from chemworld.physchem.aqueous_mineral_equilibrium import aqueous_mineral_properties
from chemworld.runtime import ChemWorldRuntime

ROUTE = (
    ("feed-a", "feed-a-reactor", 0.5),
    ("feed-b", "feed-b-reactor", 0.5),
    ("diluent", "dilute-reactor", 0.25),
    ("diluent", "dilute-reactor", 0.75),
)
CHECKS = (
    "initial_precursors_no_solid",
    "single_precursor_no_solid",
    "mixture_precipitates",
    "partial_dilution_redissolves",
    "final_dilution_exhausts",
    "elements_and_charge",
    "total_heat_work_energy",
    "exact_json_continuation",
)


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )


def members(state):
    return {state.vessel_id: state, **state.inactive_vessels}


def conserved(spec, state):
    elements = {}
    charges = []
    for local in members(state).values():
        charge = 0.0
        for species in spec.database.species:
            n = local.species_amounts[species.species_id]
            charge += n * species.charge
            for key, count in species.composition.items():
                elements[key] = elements.get(key, 0.0) + n * count
        charges.append(charge)
    return elements, charges


def properties(spec, state):
    values = [
        aqueous_mineral_properties(
            spec,
            local.species_amounts,
            temperature_K=local.temperature_K,
            pressure_Pa=local.pressure_Pa,
        )
        for local in members(state).values()
    ]
    return {
        key: sum(getattr(v, key) for v in values)
        for key in ("enthalpy_J", "internal_energy_J", "volume_m3")
    }


def unit(family, association, output):
    spec, composition = synthetic_world(family, association=association)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=40)
    replay = None
    raw = {"family": family, "association": association, "composition": composition.to_dict()}
    try:
        initial = session._state
        raw["initial"] = initial.to_dict()
        initial_elements, _ = conserved(spec, initial)
        initial_properties = properties(spec, initial)
        solids, stages, material_errors, charge_errors, heat_errors, work_errors, energy_errors = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for index, (source, edge, fraction) in enumerate(ROUTE):
            result = route(session, source, edge, fraction)
            state = session._state
            solids.append(members(state)["reactor"].species_amounts["NaCl(s)"])
            stages.append({"state": state.to_dict(), "info": result.info})
            elements, charges = conserved(spec, state)
            material_errors.append(max(abs(elements[k] - v) for k, v in initial_elements.items()))
            charge_errors.append(max(map(abs, charges)))
            props = properties(spec, state)
            heat = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
            work = state.ledger.boundary_work_J
            heat_errors.append(abs(props["enthalpy_J"] - initial_properties["enthalpy_J"] - heat))
            work_errors.append(
                abs(1e5 * (props["volume_m3"] - initial_properties["volume_m3"]) - work)
            )
            energy_errors.append(
                abs(
                    props["internal_energy_J"]
                    - initial_properties["internal_energy_J"]
                    - heat
                    + work
                )
            )
            if index == 1:
                snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
                raw["snapshot"] = snapshot
                replay = ChemWorldRuntime.from_snapshot(snapshot)
        assert replay is not None
        for source, edge, fraction in ROUTE[2:]:
            route(replay, source, edge, fraction)
        final = json.loads(json.dumps(session._state.to_dict()))
        restored = json.loads(json.dumps(replay._state.to_dict()))
        checks = (
            all(members(initial)[k].species_amounts["NaCl(s)"] == 0 for k in ("feed-a", "feed-b")),
            solids[0] == 0,
            solids[1] > 1e-6,
            1e-6 < solids[2] < solids[1] - 1e-6,
            solids[3] <= 1e-10,
            max(material_errors + charge_errors) <= 2e-10,
            max(heat_errors + energy_errors) <= 2e-7 and max(work_errors) <= 2e-8,
            final == restored,
        )
        raw.update(
            stages=stages,
            solids_mol=solids,
            material_errors_mol=material_errors,
            charge_errors_mol=charge_errors,
            heat_errors_J=heat_errors,
            work_errors_J=work_errors,
            energy_errors_J=energy_errors,
            replay_final=restored,
            final=final,
        )
        write(output / "raw.json", raw)
        return [
            {"check": key, "passed": bool(value)} for key, value in zip(CHECKS, checks, strict=True)
        ]
    except Exception:
        raw["failure"] = traceback.format_exc()
        write(output / "raw.json", raw)
        raise
    finally:
        session.close()
        if replay is not None:
            replay.close()


def resume(snapshot, output):
    session = ChemWorldRuntime.from_snapshot(json.loads(snapshot.read_text(encoding="utf8")))
    try:
        for source, edge, fraction in ROUTE[2:]:
            route(session, source, edge, fraction)
        write(output, session._state.to_dict())
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        resume(args.resume, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started, rows, units = time.monotonic(), [], []
    for family in ("ideal", "Davies", "SIT", "Pitzer"):
        for association in (False, True):
            name = family + ("-association" if association else "-salt")
            folder = args.output / name
            folder.mkdir()
            try:
                checks = unit(family, association, folder)
            except Exception:
                failure = traceback.format_exc()
                checks = [
                    {"check": key, "passed": False, "not_completed": failure} for key in CHECKS
                ]
            rows.extend({"unit": name, **r} for r in checks)
            units.append({"unit": name, "passed": all(r["passed"] for r in checks)})
            elapsed = time.monotonic() - started
            print(
                f"40H {len(units)}/8 routes, {len(units) / elapsed:.2f} routes/s, "
                f"ETA {(8 - len(units)) * elapsed / len(units):.1f}s: {name} "
                f"{sum(r['passed'] for r in checks)}/8",
                flush=True,
            )
    fresh = []
    for name in ("ideal-salt", "Davies-association"):
        folder = args.output / name
        try:
            raw = json.loads((folder / "raw.json").read_text(encoding="utf8"))
            snapshot_path, final_path = folder / "snapshot.json", folder / "fresh-final.json"
            write(snapshot_path, raw["snapshot"])
            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    __file__,
                    "--resume",
                    str(snapshot_path),
                    "--output",
                    str(final_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            passed = (
                completed.returncode == 0
                and json.loads(final_path.read_text(encoding="utf8")) == raw["final"]
            )
            fresh.append(
                {
                    "unit": name,
                    "passed": passed,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        except Exception:
            fresh.append({"unit": name, "passed": False, "failure": traceback.format_exc()})
    summary = {
        "mode": "development",
        "units_total": 8,
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_total": 64,
        "comparisons_passed": sum(r["passed"] for r in rows),
        "comparisons_failed": sum(not r["passed"] for r in rows),
        "units": units,
        "comparisons": rows,
        "fresh_process_functional_checks": fresh,
        "elapsed_s": time.monotonic() - started,
    }
    write(args.output / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"comparisons", "units"}}),
        flush=True,
    )
    if summary["comparisons_failed"] or not all(r["passed"] for r in fresh):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
