"""17-unit higher-resolution DFN block, separate from the retained coarse block."""

import argparse
import json
import platform
import threading
import time
import traceback
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pybamm
import scipy
from battery_dfn_source import load_dfn_source
from run_battery_dfn_block import compare, limit

from chemworld.physchem.battery_models import battery_spec_from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=int,
                        help="explicit zero-based aggregate attempt index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_dfn_source(args.source, args.source_index)
    originals = {r["unit"]: r for r in source["units"]}
    rows, started, stop, active = [], time.monotonic(), threading.Event(), ["initializing"]

    def progress():
        while not stop.wait(30):
            elapsed = time.monotonic() - started
            print(json.dumps({"completed": len(rows), "total": 17, "active": active[0],
                "elapsed_s": elapsed,
                "eta_s": elapsed * (17 - len(rows)) / len(rows) if rows else None}), flush=True)

    with args.output.open("x") as output:
        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        for index in range(1, 18):
            original = originals[f"D{index if index < 16 else index - 7:02}"]
            base = battery_spec_from_dict(original["parameters"])
            nx, nr = {11: (12, 32), 13: (48, 128), 16: (48, 128), 17: (48, 128)}.get(
                index, (24, 64))
            spec = replace(base, negative=replace(base.negative, cells=nr),
                positive=replace(base.positive, cells=nr),
                electrolyte=replace(base.electrolyte, cells=(nx,) * 3))
            event = {
                "D09": ("battery:minimum-voltage", "event: Minimum voltage [V]"),
                "D10": ("battery:electrolyte:depleted", "event: native:electrolyte:depleted"),
            }.get(original["unit"])
            identity = f"R{index:02}"
            active[0] = identity
            print(f"{identity}: starting ({len(rows)}/17 complete)", flush=True)
            begin = time.monotonic()
            try:
                result = (limit(spec, original["limit"]) if original["limit"] else
                          compare(spec, original["protocol"], event))
            except Exception:
                result = {"passed": False, "exception": traceback.format_exc()}
            rows.append({"unit": identity, "source_unit": original["unit"],
                "parameters": spec.to_dict(), "protocol": original["protocol"],
                "limit": original["limit"], "elapsed_s": time.monotonic() - begin, **result})
            print(f"{identity}: passed={result['passed']} ({len(rows)}/17 complete)", flush=True)
            output.seek(0)
            json.dump({"mode": "development-resolved-qualification", "total": 17,
                "completed": len(rows), "source": args.source.name, "units": rows,
                "source_index": args.source_index,
                "elapsed_s": time.monotonic() - started}, output, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
        grid = rows[10:13]
        converged = all("segments" in r for r in grid) and all(
            b["segments"][0][key] <= a["segments"][0][key] + tolerance
            for a, b in pairwise(grid) for key, tolerance in (
                ("voltage_max_error_V", 1e-6), ("salt_max_error_mol_m3", .01)))
        if not converged:
            for row in grid:
                row.update(passed=False, grid_convergence_failed=True)
        endpoint_checks = []
        for coarse, fine in ((rows[8], rows[15]), (rows[9], rows[16])):
            check = {"pair": [coarse["unit"], fine["unit"]], "passed": False}
            if "segments" in coarse and "segments" in fine:
                a, b = coarse["segments"][0], fine["segments"][0]
                change = abs(a["native_duration_s"] - b["native_duration_s"])
                bound = max(1e-4, .01 * b["reference_duration_s"])
                check.update(change_s=change, bound_s=bound, passed=change <= bound)
            endpoint_checks.append(check)
            if not check["passed"]:
                coarse.update(passed=False, endpoint_refinement_failed=True)
                fine.update(passed=False, endpoint_refinement_failed=True)
        result = {"mode": "development-resolved-qualification", "total": 17,
            "completed": len(rows), "passed": sum(r["passed"] for r in rows),
            "grid_converged": converged, "endpoint_refinement": endpoint_checks,
            "source": args.source.name, "elapsed_s": time.monotonic() - started,
            "source_index": args.source_index,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pybamm": pybamm.__version__, "units": rows}
        output.seek(0)
        json.dump(result, output, indent=2, allow_nan=False)
        output.truncate()
        output.flush()
        stop.set()
        thread.join()
        print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return int(result["passed"] != 17)


if __name__ == "__main__":
    raise SystemExit(main())
