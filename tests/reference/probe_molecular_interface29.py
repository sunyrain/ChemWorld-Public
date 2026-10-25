"""Self-contained real-session interface probe; also runs from an installed wheel.

Inputs are serialized original world/phase declarations, never imported test or
reference factories. All query results, states and failures remain in the output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import isclose
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_interface import SquareGradientInfluence
from chemworld.runtime import ChemWorldRuntime, molecular_interface_from_state
from chemworld.runtime.molecular_interface import molecular_interface_from_ledger
from chemworld.runtime.molecular_multiphase_inventory import MolecularMultiphaseInventorySpec
from chemworld.world.composition import WorldCompositionSpec


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def options():
    return {
        "left_phase_id": "vapor", "right_phase_id": "liquid", "area_m2": 2e-4,
        "mesh": DiffusionMesh.planar(12e-9, 512),
        "component_density_bounds": ((.001, 14000.0),) * 2,
        "max_iterations": 500, "residual_tolerance": 1e-7, "first_integral_tolerance": .05,
    }


def query(session, unit):
    spec = MolecularMultiphaseInventorySpec.from_dict(unit["inventory"])
    influence = SquareGradientInfluence.from_dict(unit["influence"])
    return molecular_interface_from_state(session._state, spec, influence, **options())


def advance(session):
    return session.execute({"operation": "wait", "duration_s": 1}).info


def forbidden_modules():
    names = {"tests", "reference", "phasepy", "thermo", "teqp"}
    return sorted(name for name in sys.modules if name.split(".")[0] in names)


def child_run(payload, output):
    unit = payload["unit"]
    session = ChemWorldRuntime.from_snapshot(payload["snapshot"])
    try:
        info = advance(session)
        report = query(session, unit).to_dict()
        write(output, {"state": session._state.to_dict(), "query": report, "info": info,
                       "forbidden_modules": forbidden_modules()})
    finally:
        session.close()


def run_unit(unit, root):
    started = monotonic()
    report = {"id": unit["id"], "checks": [], "queries": {}, "states": {}}
    finished = Event()

    def progress():
        while not finished.wait(30):
            elapsed = monotonic() - started
            completed = len(report["checks"])
            stage = report["checks"][-1]["name"] if completed else "initial query"
            print(f"interface29 {unit['id']}: checks={completed}/18; "
                  f"last={stage}; elapsed={elapsed:.1f}s", flush=True)

    heartbeat = Thread(target=progress, daemon=True)
    heartbeat.start()

    def check(name, passed):
        report["checks"].append({"name": name, "pass": bool(passed)})

    session = replay = None
    try:
        session = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(unit["composition"]))
        session.reset(seed=19)
        initial = session._state.to_dict()
        report["states"]["initial"] = initial
        before = query(session, unit)
        report["queries"]["initial"] = before.to_dict()
        check("initial_success", before.interface.success)
        check("initial_read_only", session._state.to_dict() == initial)
        gamma = before.interface.surface_tension_N_m
        check("declared_area_grand_potential", gamma is not None
              and before.surface_excess_grand_potential_J == 2e-4 * gamma)
        check("first_wait_committed", advance(session).get("transaction_status") == "committed")
        heated = session._state.to_dict()
        report["states"]["heated"] = heated
        after = query(session, unit)
        report["queries"]["heated"] = after.to_dict()
        check("heated_success", after.interface.success)
        check("heated_read_only", session._state.to_dict() == heated)
        check("actual_temperature_increases", heated["temperature_K"] > initial["temperature_K"])
        check("interface_responds_to_live_state", gamma is not None
              and after.interface.surface_tension_N_m is not None
              and not isclose(gamma, after.interface.surface_tension_N_m, rel_tol=1e-8, abs_tol=0))
        spec = MolecularMultiphaseInventorySpec.from_dict(unit["inventory"])
        low = molecular_interface_from_ledger(
            session._state.phases, spec.equilibrium.model.residual_model,
            SquareGradientInfluence.from_dict(unit["influence"]),
            vessel_id=session._state.vessel_id, temperature_K=session._state.temperature_K,
            pressure_Pa=session._state.pressure_Pa, **options(),
        )
        check("same_declared_ledger_core", low.to_dict() == after.to_dict())
        snapshot = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        check("snapshot_query_exact", query(replay, unit).to_dict() == after.to_dict())
        info, replay_info = advance(session), advance(replay)
        check("suffix_committed", all(r.get("transaction_status") == "committed"
                                       for r in (info, replay_info)))
        check("suffix_state_exact", session._state.to_dict() == replay._state.to_dict())
        suffix = query(session, unit).to_dict()
        report["queries"]["suffix"] = suffix
        check("suffix_query_success", suffix["interface"]["success"])
        check("suffix_query_exact", suffix == query(replay, unit).to_dict())
        report["states"]["suffix"] = session._state.to_dict()
        child_input, child_output = root / "child_input.json", root / "child_output.json"
        write(child_input, {"unit": unit, "snapshot": snapshot})
        # -I drops the working tree and PYTHONPATH. Reinsert only the actual
        # imported package root, whether this is source or the installed target.
        import chemworld

        package_root = str(Path(chemworld.__file__).resolve().parent.parent)
        script = str(Path(__file__).resolve())
        code = ("import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); sys.argv.pop(0); "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
        process = subprocess.run(
            [sys.executable, "-I", "-c", code, package_root, script,
             "--child", "--payload", str(child_input), "--output", str(child_output)],
            capture_output=True, text=True, timeout=300, check=False,
        )
        report["child_process"] = {"returncode": process.returncode,
                                   "stdout": process.stdout, "stderr": process.stderr}
        check("child_completed", process.returncode == 0 and child_output.is_file())
        child = read(child_output) if child_output.is_file() else {}
        check("child_state_exact", child.get("state") == session._state.to_dict())
        # JSON arrays encode both Python tuples and lists; compare the complete
        # encoded schema without rounding or changing any numeric value.
        check("child_query_exact", child.get("query") == json.loads(json.dumps(suffix)))
        check("reference_free_imports", not forbidden_modules()
              and child.get("forbidden_modules") == [])
    except Exception as exc:
        report["exception"] = f"{type(exc).__name__}: {exc}"
        check("unit_execution", False)
    finally:
        finished.set()
        heartbeat.join()
        if replay is not None:
            replay.close()
        if session is not None:
            session.close()
    report["elapsed_s"] = monotonic() - started
    report["success"] = len(report["checks"]) == 18 and all(c["pass"] for c in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    payload = read(args.payload)
    if args.child:
        child_run(payload, args.output)
        return 0
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing qualification result")
    started = monotonic()
    report = {"mode": "development", "expected_units": 2, "expected_checks": 36, "units": []}
    for index, unit in enumerate(payload["units"], 1):
        print(f"interface29 consumer {index}/2 {unit['id']} starting", flush=True)
        root = args.output.parent / unit["id"]
        result = run_unit(unit, root)
        report["units"].append(result)
        report["elapsed_s"] = monotonic() - started
        report["completed_units"] = len(report["units"])
        report["passed_units"] = sum(u["success"] for u in report["units"])
        report["completed_checks"] = sum(len(u["checks"]) for u in report["units"])
        report["passed_checks"] = sum(c["pass"] for u in report["units"] for c in u["checks"])
        report["success"] = report["completed_units"] == 2 and report["passed_units"] == 2
        write(args.output, report)
        elapsed = report["elapsed_s"]
        print(f"interface29 consumer {index}/2: passed={report['passed_units']}/{index}; "
              f"elapsed={elapsed:.1f}s; rate={index/max(elapsed,1e-9):.3g}/s; "
              f"ETA={elapsed/index*(2-index):.1f}s", flush=True)
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
