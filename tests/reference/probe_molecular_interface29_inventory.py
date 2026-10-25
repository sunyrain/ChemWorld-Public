"""Self-contained pure and ternary live-inventory qualification for source/wheel."""

import argparse
import json
import sys
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem import molecular_influence_from_dict
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.runtime import molecular_interface_from_ledger, molecular_interface_from_state
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseEquilibriumSpec

CHECKS = (
    "query_success",
    "positive_gamma",
    "all_world_ledgers_unchanged",
    "gamma_area",
    "current_vs_ledger_exact",
    "JSON_state_and_influence_query_exact",
    "wrong_pressure_rejected",
    "missing_phase_rejected",
    "identity_collision_rejected",
    "no_reference_imports",
)


def run_unit(unit):
    started, finished = monotonic(), Event()
    result = {
        "id": unit["id"],
        "checks": [{"name": name, "pass": False, "attempted": False} for name in CHECKS],
    }

    def check(name, passed):
        row = result["checks"][CHECKS.index(name)]
        row.update(attempted=True, **{"pass": bool(passed)})

    def heartbeat():
        while not finished.wait(30):
            completed = sum(row["attempted"] for row in result["checks"])
            print(
                f"interface29 {unit['id']}: {completed}/10 checks; "
                f"elapsed={monotonic() - started:.1f}s",
                flush=True,
            )

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        state = WorldState.from_dict(unit["state"])
        declaration = MolecularPhaseEquilibriumSpec.from_dict(unit["inventory"])
        influence = molecular_influence_from_dict(unit["influence"])
        options = dict(unit["options"])
        if "mesh" in options:
            options["mesh"] = DiffusionMesh.planar(**options["mesh"])

        def query(current, model=declaration, coefficients=influence, right="right"):
            return molecular_interface_from_state(
                current,
                model,
                coefficients,
                left_phase_id="left",
                right_phase_id=right,
                **options,
            )

        before = state.to_dict()
        interface = query(state)
        result["query"] = interface.to_dict()
        check("query_success", interface.interface.success)
        gamma = interface.interface.surface_tension_N_m
        check("positive_gamma", gamma is not None and gamma > 0)
        check("all_world_ledgers_unchanged", state.to_dict() == before)
        check(
            "gamma_area",
            gamma is not None and interface.surface_excess_grand_potential_J == gamma * 2e-4,
        )
        low = molecular_interface_from_ledger(
            state.phases,
            declaration.residual_model,
            influence,
            vessel_id=state.vessel_id,
            left_phase_id="left",
            right_phase_id="right",
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
            **options,
        )
        check("current_vs_ledger_exact", low.to_dict() == interface.to_dict())
        restored = WorldState.from_dict(json.loads(json.dumps(before, allow_nan=False)))
        declared = MolecularPhaseEquilibriumSpec.from_dict(
            json.loads(json.dumps(declaration.to_dict(), allow_nan=False)),
        )
        coefficients = molecular_influence_from_dict(
            json.loads(json.dumps(influence.to_dict(), allow_nan=False)),
        )
        check(
            "JSON_state_and_influence_query_exact",
            query(restored, declared, coefficients).to_dict() == interface.to_dict(),
        )
        rejected = []
        failures = (
            (
                "wrong_pressure_rejected",
                state.replace(pressure_Pa=1.2 * state.pressure_Pa),
                "right",
            ),
            (
                "missing_phase_rejected",
                state.replace(
                    phases=PhaseLedger(
                        {
                            "left": state.phases.phases["left"],
                        }
                    )
                ),
                "right",
            ),
            ("identity_collision_rejected", state, "left"),
        )
        for name, current, right in failures:
            try:
                query(current, right=right)
            except ValueError as exc:
                check(name, True)
                rejected.append({"name": name, "reason": str(exc)})
            else:
                check(name, False)
        result["rejections"] = rejected
        forbidden = {"tests", "reference", "phasepy", "thermo", "teqp"}
        imported = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
        result["forbidden_modules"] = imported
        check("no_reference_imports", not imported)
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        finished.set()
        worker.join()
    result["success"] = all(row["attempted"] and row["pass"] for row in result["checks"])
    result["elapsed_s"] = monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing qualification result")
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    report = {"mode": "development", "expected_units": 3, "expected_checks": 30, "units": []}
    started = monotonic()
    for index, unit in enumerate(payload["units"], 1):
        print(f"interface29 inventory {index}/3 {unit['id']} starting", flush=True)
        report["units"].append(run_unit(unit))
        report["completed_units"] = len(report["units"])
        report["passed_units"] = sum(unit["success"] for unit in report["units"])
        report["attempted_checks"] = sum(
            c["attempted"] for u in report["units"] for c in u["checks"]
        )
        report["passed_checks"] = sum(c["pass"] for u in report["units"] for c in u["checks"])
        report["success"] = report["passed_units"] == 3 and report["completed_units"] == 3
        elapsed = monotonic() - started
        report["elapsed_s"] = elapsed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            f"interface29 inventory {index}/3: passed={report['passed_units']}/{index}; "
            f"elapsed={elapsed:.1f}s; rate={index / max(elapsed, 1e-9):.3g}/s; "
            f"ETA={elapsed / index * (3 - index):.1f}s",
            flush=True,
        )
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
