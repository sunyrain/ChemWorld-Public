"""Execute all four preselected D08 resolutions and preserve coarse-grid failures."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
from run_thevenin_frequency_block import capture, convert
from thevenin_diffusion_cases import CORRECTION_URL, case_data
from thevenin_diffusion_resolution import (
    RESOLUTION_GRIDS,
    assess_resolution_block,
    assess_resolution_grid,
    continuous_solution,
    native_resolution,
    reference_resolution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import casadi
    import pybamm
    import scipy

    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("fixed reference versions required")
    data = {"mode": "development", "denominator": 4, "path_denominator": 8, "paths_completed": 0,
        "execution_source": ("development working tree; source_commit identifies base, "
                             "not frozen execution tree"),
        "units": [], "passed": 0, "resolution_qualified": False, "original_D08_status": "unchanged",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": ("workstreams/world_foundation/"
                     "NATIVE_CONV45_THEVENIN_DIFFUSION_RESOLUTION_EXPERIMENT.md"),
        "versions": {"pybamm": pybamm.__version__, "casadi": casadi.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "correction_url": CORRECTION_URL, "grids": list(RESOLUTION_GRIDS), "inputs": case_data(8)}
    start, stop, stage = time.monotonic(), threading.Event(), {"name": "continuous solution"}
    with args.output.open("x", encoding="utf8") as output:
        def checkpoint():
            data["elapsed_s"] = time.monotonic()-start
            output.seek(0)
            output.write(json.dumps(data, default=convert, allow_nan=False, indent=2)+"\n")
            output.truncate()
            output.flush()

        def heartbeat():
            while not stop.wait(20):
                elapsed, done = time.monotonic()-start, data["paths_completed"]
                rate = done/max(elapsed, 1.)
                eta = f"{(8-done)/rate:.1f}s" if rate else "unknown"
                print(f"D08 resolution {done}/8 paths; {stage['name']}; elapsed={elapsed:.1f}s; "
                      f"throughput={rate:.5f}/s; ETA={eta}", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            data["analytic"] = capture(continuous_solution)
            checkpoint()
            for cells in RESOLUTION_GRIDS:
                row = {"cells": cells, "passed": False}
                data["units"].append(row)
                for name, function in (("native", native_resolution),
                                       ("corrected", reference_resolution)):
                    stage["name"] = f"cells={cells} {name}"
                    print(stage["name"], flush=True)
                    started = time.monotonic()
                    row[name] = capture(partial(function, cells))
                    row[name]["elapsed_s"] = time.monotonic()-started
                    data["paths_completed"] += 1
                    checkpoint()
                try:
                    assess_resolution_grid(row, data["analytic"])
                except Exception as exc:
                    row.update(exception=str(exc), traceback=traceback.format_exc())
                data["passed"] = sum(unit["passed"] for unit in data["units"])
                print(f"cells={cells} complete; continuous target={row['passed']}", flush=True)
                checkpoint()
            try:
                assess_resolution_block(data)
            except Exception as exc:
                data.update(exception=str(exc), traceback=traceback.format_exc())
            checkpoint()
        finally:
            stop.set()
            thread.join()
    return int(not data["resolution_qualified"])


if __name__ == "__main__":
    raise SystemExit(main())
