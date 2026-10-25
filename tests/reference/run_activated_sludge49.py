"""Fixed 49G development block; every failed unit is retained and summarized."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ["TEMP"]) / "ChemWorld-49E-reference"))

from chemworld.physchem.activated_sludge import advance_activated_sludge, flow_profile
from chemworld.physchem.activated_sludge_kinetics import (
    ASM1_COMPONENTS,
    ASM1_REACTIONS,
    asm1_quality,
    asm1_rates,
    asm1_stoichiometry,
)
from reference.activated_sludge49_fixtures import inventory_case, rate_cases, transient_cases
from reference.activated_sludge49_reference import WaterTAPReference, independent


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows, started = [], time.monotonic()
    total = len(rate_cases()) + len(transient_cases())
    refs = {}

    def ref(params):
        key = tuple(params.to_dict().values())
        if key not in refs:
            refs[key] = WaterTAPReference(params)
        return refs[key]

    def record(row):
        rows.append(row)
        (output / f"{len(rows):02d}-{row['name']}.json").write_text(
            json.dumps(row, indent=2), encoding="utf8"
        )
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "stage": row["name"],
                    "passed": row["passed"],
                    "completed": len(rows),
                    "total": total,
                    "units_per_second": len(rows) / elapsed,
                    "ETA_s": elapsed / len(rows) * (total - len(rows)),
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )

    for name, params, c in rate_cases():
        row = {"name": name, "kind": "actual-WaterTAP-expression", "checks": 0}
        try:
            reference = ref(params)
            actual = asm1_rates(params, c)
            np.testing.assert_allclose(
                [actual[r] for r in ASM1_REACTIONS], reference.rates(c), atol=1e-16, rtol=2e-12
            )
            row["checks"] += 8
            native = asm1_stoichiometry(params)
            np.testing.assert_allclose(
                [[native[r].get(k, 0.0) for r in ASM1_REACTIONS] for k in ASM1_COMPONENTS],
                reference.matrix,
                atol=1e-12,
                rtol=2e-12,
            )
            row["checks"] += 104
            quality = asm1_quality(params, c)
            expected = reference.quality(c)
            np.testing.assert_allclose(
                list(quality.values()), list(expected.values()), atol=1e-12, rtol=2e-12
            )
            row["checks"] += 8
            row.update(passed=True, rates=actual, quality=quality)
        except Exception as exc:
            row.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
        record(row)
    for name, spec, options, q, duration in transient_cases():
        row = {"name": name, "kind": "finite-independent-Radau", "checks": 0}
        try:
            initial = inventory_case(spec, **options)
            reference = independent(spec, initial, q, duration, ref(spec.parameters))
            result = advance_activated_sludge(spec, initial, q, duration)
            np.testing.assert_allclose(
                flow_profile(spec, q)["actual_flow_m3_s"], q, atol=1e-20, rtol=1e-9
            )
            row["checks"] += 1
            final = result.final
            comparisons = (
                (
                    [r[k] for r in final.components for k in ASM1_COMPONENTS],
                    reference["components"].ravel(),
                    2e-12,
                ),
                (
                    [r[k] for r in final.extents for k in ASM1_REACTIONS],
                    reference["extents"].ravel(),
                    2e-12,
                ),
                (list(final.utilities.values()), reference["utilities"], 2e-12),
                (
                    np.array(final.water_mol) * spec.water_molar_volume_m3,
                    reference["volumes_m3"],
                    2e-14,
                ),
                (final.work["aeration_electricity_J"], reference["electricity_J"], 2e-9),
                (result.actual_duration_s, reference["duration_s"], 2e-5),
            )
            for actual, expected, absolute in comparisons:
                np.testing.assert_allclose(actual, expected, atol=absolute, rtol=3e-6)
                row["checks"] += 1
            if result.stopping_event != reference["stop"]:
                raise AssertionError((result.stopping_event, reference["stop"]))
            row["checks"] += 1
            residuals = result.diagnostic["source_budget_residuals"]
            if max(map(abs, residuals.values())) > 1e-9:
                raise AssertionError(residuals)
            if abs(result.diagnostic["carrier_energy_residual_J"]) > 1e-6:
                raise AssertionError("carrier energy residual")
            row["checks"] += 2
            row.update(
                passed=True,
                stop=result.stopping_event,
                duration_s=result.actual_duration_s,
                final_components=final.components,
                extents=final.extents,
                utilities=final.utilities,
                work=final.work,
                diagnostic=result.diagnostic,
            )
        except Exception as exc:
            row.update(passed=False, error=repr(exc), traceback=traceback.format_exc())
        record(row)
    summary = {
        "schema": "49G-development-qualification-v1",
        "units": total,
        "passed": sum(r["passed"] for r in rows),
        "failed": sum(not r["passed"] for r in rows),
        "checks_passed": sum(r["checks"] for r in rows),
        "checks_expected": 11 * 120 + 22 * 10,
        "failed_units": [r["name"] for r in rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "reference": "actual WaterTAP 1.8.0 Pyomo expressions; independent finite ODE Radau",
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    return summary["failed"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
