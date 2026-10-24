"""Fixed prospective finite-headspace canary and preserved machine records."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from aqueous_gas40_reference import history

from chemworld.examples.aqueous_headspace import example_inputs
from chemworld.physchem.aqueous_gas_transfer import advance_aqueous_headspace
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows, units, details = [], [], []
    started = time.monotonic()
    for kind in ("neutral", "acid", "desorption"):
        before, error = len(rows), None

        def check(name, actual, expected, atol=0.0, rtol=0.0, unit_name=kind):
            if isinstance(expected, bool):
                passed, maximum = actual == expected, None
            else:
                a, b = np.asarray(actual), np.asarray(expected)
                passed = bool(
                    a.shape == b.shape
                    and np.all(np.isfinite(a))
                    and np.allclose(a, b, atol=atol, rtol=rtol)
                )
                maximum = float(np.max(abs(a - b))) if a.shape == b.shape else None
            rows.append(
                {
                    "unit": unit_name,
                    "quantity": name,
                    "passed": passed,
                    "actual": actual,
                    "expected": expected,
                    "atol": atol,
                    "rtol": rtol,
                    "maximum_absolute_error": maximum,
                }
            )

        runtime = restored = None
        try:
            spec, n = example_inputs(kind)
            native = advance_aqueous_headspace(spec, n, 5.0, evaluation_times_s=(0.0, 1.0, 5.0))
            reference = history(kind, 0, (0.0, 1.0, 5.0))
            details.append({"unit": kind, "native": native.to_dict(), "reference": reference})
            ids = list(n)
            check(
                "all_species",
                [[p["amounts_mol"][k] for k in ids] for p in native.trajectory],
                [[p["amounts"][k] for k in ids] for p in reference],
                atol=2e-10,
                rtol=3e-6,
            )
            elements = {e for s in spec.equilibrium.database.species for e in s.composition}

            def conserved(stocks, specification=spec, element_ids=elements):
                return [
                    sum(
                        s.composition.get(e, 0) * stocks[s.species_id]
                        for s in specification.equilibrium.database.species
                    )
                    for e in sorted(element_ids)
                ] + [
                    sum(
                        s.charge * stocks[s.species_id]
                        for s in specification.equilibrium.database.species
                    )
                ]

            material = max(
                abs(a - b)
                for p in native.trajectory
                for a, b in zip(conserved(p["amounts_mol"]), conserved(n), strict=True)
            )
            heat = max(
                abs(p["bath_heat_J"] - r["heat"])
                for p, r in zip(native.trajectory, reference, strict=True)
            )
            check(
                "atoms_charge_rigid_heat",
                bool(material <= 2e-10 and heat <= 3e-6),
                True,
            )
            runtime = ChemWorldRuntime(composition=build_aqueous_headspace_world(spec, n))
            runtime.reset(seed=40)
            first = runtime.execute({"operation": "wait", "duration_s": 1.0})
            restored = ChemWorldRuntime.from_snapshot(
                json.loads(json.dumps(runtime.snapshot().to_dict()))
            )
            replies = [
                w.execute({"operation": "wait", "duration_s": 4.0}) for w in (runtime, restored)
            ]
            details[-1]["world"] = {
                "first_info": first.info,
                "next_info": [r.info for r in replies],
                "snapshot": runtime.snapshot().to_dict(),
            }
            expected = reference[-1]["amounts"]
            passed = first.info["transaction_status"] == "committed" and all(
                r.info["transaction_status"] == "committed" for r in replies
            )
            passed &= runtime._state.ledger.time_s == 5.0
            passed &= np.allclose(
                [runtime._state.species_amounts[k] for k in ids],
                [expected[k] for k in ids],
                atol=2e-10,
                rtol=3e-6,
            )
            passed &= runtime.snapshot().to_dict() == restored.snapshot().to_dict()
            check("original_world_and_json", bool(passed), True)
        except Exception:
            error = traceback.format_exc()
        finally:
            if runtime is not None:
                runtime.close()
            if restored is not None:
                restored.close()
        selected = rows[before:]
        unit = {
            "unit": kind,
            "planned_comparisons": 3,
            "executed": len(selected),
            "passed_comparisons": sum(r["passed"] for r in selected),
            "error": error,
            "passed": error is None and len(selected) == 3 and all(r["passed"] for r in selected),
        }
        units.append(unit)
        for name, data in (
            ("comparisons.json", rows),
            ("units.json", units),
            ("details.json", details),
        ):
            (output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(
            f"40F canary {len(units)}/3 {kind} {'PASS' if unit['passed'] else 'FAIL'} "
            f"comparisons={len(rows)}/9 elapsed={time.monotonic() - started:.2f}s {error or ''}",
            flush=True,
        )
    summary = {
        "schema": "chemworld-aqueous40f-canary-v1",
        "planned_units": 3,
        "units_passed": sum(u["passed"] for u in units),
        "planned_comparisons": 9,
        "comparisons_executed": len(rows),
        "comparisons_passed": sum(r["passed"] for r in rows),
        "unexecuted_comparisons": 9 - len(rows),
        "failures": [u for u in units if not u["passed"]],
        "failed_comparisons": [r for r in rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": all(u["passed"] for u in units),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("failures", "failed_comparisons")}
        )
    )
    raise SystemExit(0 if summary["passed"] else 1)
