"""Fixed development block: eight finite filter/wash routes, ten checks each."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from chemworld.examples.aqueous_filtration import filter_vessel, synthetic_filtration_world
from chemworld.examples.aqueous_precipitation import route
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_precipitation import precipitation_properties

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reference.aqueous_filtration40_reference import reference_history, totals

CHECKS = (
    "first_all_species_distribution",
    "first_source_exact_empty",
    "leaked_solid_retains_phase",
    "both_products_actual_water",
    "finite_wash_species_distribution",
    "second_all_species_distribution",
    "both_sources_exact_empty",
    "all_stage_elements_and_charge",
    "independent_all_stage_H_U_V_and_ledger",
    "exact_json_continuation",
)


def write(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )


def members(state):
    return {state.vessel_id: state, **state.inactive_vessels}


def empty(state):
    return (
        state.volume_L == 0
        and not any(state.species_amounts.values())
        and all(
            p.volume_L == 0 and not any(p.species_amounts_mol.values())
            for p in state.phases.phases.values()
        )
    )


def element_totals(stocks):
    # Explicit analytic stoichiometry for this synthetic six-species system.
    n = {key: sum(row[key] for row in stocks.values()) for key in next(iter(stocks.values()))}
    return {
        "H": 2 * n["H2O"],
        "O": n["H2O"],
        "Na": n["Na+"] + n["NaCl(s)"],
        "Cl": n["Cl-"] + n["NaCl(s)"],
        "K": n["K+"],
        "Br": n["Br-"],
    }


def unit(family, eta, retention, output):
    _, composition = synthetic_filtration_world(
        family, solid_retention=eta, liquid_retention=retention
    )
    expected = reference_history(family, eta, retention)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=41)
    replay = None
    raw = {
        "family": family,
        "solid_retention": eta,
        "liquid_retention": retention,
        "composition": composition.to_dict(),
        "stages": [],
    }
    quantity_errors, balance_errors, property_errors = [], [], []
    initial_props = totals(expected[0], family)
    initial_elements = element_totals(expected[0])

    def record(info=None):
        index = len(raw["stages"])
        state = session._state
        locals_ = members(state)
        stocks = {key: local.species_amounts for key, local in locals_.items()}
        quantity_errors.append(
            max(
                abs(n - expected[index][key][species])
                for key, amounts in stocks.items()
                for species, n in amounts.items()
            )
        )
        elements = element_totals(stocks)
        balance_errors.append(
            max(
                *(abs(elements[k] - n) for k, n in initial_elements.items()),
                *(abs(n["Na+"] + n["K+"] - n["Cl-"] - n["Br-"]) for n in stocks.values()),
            )
        )
        props = [precipitation_properties(local) for local in locals_.values()]
        actual = {
            "H": sum(p.enthalpy_J for p in props),
            "U": sum(p.internal_energy_J for p in props),
            "V": sum(local.volume_L / 1000 for local in locals_.values()),
        }
        independent = totals(expected[index], family)
        heat = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
        work = state.ledger.boundary_work_J
        property_errors.append(
            {
                "H_J": abs(actual["H"] - independent["H"]),
                "U_J": abs(actual["U"] - independent["U"]),
                "V_m3": abs(actual["V"] - independent["V"]),
                "Q_J": abs(heat - independent["H"] + initial_props["H"]),
                "W_J": abs(work - 1e5 * (independent["V"] - initial_props["V"])),
                "first_law_J": abs(independent["U"] - initial_props["U"] - heat + work),
            }
        )
        raw["stages"].append({"state": state.to_dict(), "info": info})

    try:
        record()
        record(route(session, "feed-a", "feed-a-reactor", 0.5).info)
        record(route(session, "feed-b", "feed-b-reactor", 0.5).info)
        first = filter_vessel(session, "reactor")
        record(first.info)
        first_members = members(session._state)
        source_empty = empty(first_members["reactor"])
        filtrate = first_members["filtrate"]
        leaked = filtrate.phases.phases["mineral:NaCl(s)"]
        leaked_correct = (
            leaked.phase_type == "solid"
            and abs(leaked.species_amounts_mol["NaCl(s)"] - expected[3]["filtrate"]["NaCl(s)"])
            <= 2e-9
            and (
                leaked.species_amounts_mol["NaCl(s)"] > 1e-6
                if eta < 1
                else leaked.species_amounts_mol["NaCl(s)"] <= 1e-10
            )
        )
        product_water = all(
            first_members[k].species_amounts["H2O"] > 0
            and abs(first_members[k].species_amounts["H2O"] - expected[3][k]["H2O"]) <= 2e-9
            for k in ("cake", "filtrate")
        )
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        raw["snapshot"] = snapshot
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        record(route(session, "diluent", "wash-cake", 0.025).info)
        second = filter_vessel(session, "cake")
        record(second.info)
        route(replay, "diluent", "wash-cake", 0.025)
        filter_vessel(replay, "cake")
        final = json.loads(json.dumps(session._state.to_dict()))
        continued = json.loads(json.dumps(replay._state.to_dict()))
        final_members = members(session._state)
        calorics = all(
            all(value <= (2e-12 if key == "V_m3" else 2e-7) for key, value in row.items())
            for row in property_errors
        )
        checks = (
            quantity_errors[3] <= 2e-9,
            source_empty,
            leaked_correct,
            product_water,
            quantity_errors[4] <= 2e-9,
            quantity_errors[5] <= 2e-9,
            all(empty(final_members[k]) for k in ("reactor", "cake")),
            max(balance_errors) <= 2e-10,
            calorics,
            final == continued,
        )
        raw.update(
            quantity_errors_mol=quantity_errors,
            balance_errors_mol=balance_errors,
            property_errors=property_errors,
            final=final,
            replay_final=continued,
            first_filter_duration_s=first.info["actual_duration_s"],
            second_filter_duration_s=second.info["actual_duration_s"],
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
        route(session, "diluent", "wash-cake", 0.025)
        filter_vessel(session, "cake")
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
    for family in ("ideal", "Davies"):
        for eta in (0.6, 1.0):
            for retention in (0.05, 0.2):
                name = f"{family}-eta{eta:g}-liquid{retention:g}"
                folder = args.output / name
                folder.mkdir()
                try:
                    checks = unit(family, eta, retention, folder)
                except Exception:
                    failure = traceback.format_exc()
                    checks = [
                        {"check": key, "passed": False, "not_completed": failure} for key in CHECKS
                    ]
                rows.extend({"unit": name, **r} for r in checks)
                units.append({"unit": name, "passed": all(r["passed"] for r in checks)})
                elapsed = time.monotonic() - started
                print(
                    f"40I {len(units)}/8 routes, {len(units) / elapsed:.2f} routes/s, "
                    f"ETA {(8 - len(units)) * elapsed / len(units):.1f}s: {name} "
                    f"{sum(r['passed'] for r in checks)}/10",
                    flush=True,
                )
    fresh = []
    for name in ("ideal-eta0.6-liquid0.05", "Davies-eta1-liquid0.2"):
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
                timeout=55,
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
        print(f"40I fresh consumer {len(fresh)}/2: {name} passed={fresh[-1]['passed']}", flush=True)
    summary = {
        "mode": "development",
        "units_total": 8,
        "units_passed": sum(u["passed"] for u in units),
        "comparisons_total": 80,
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
