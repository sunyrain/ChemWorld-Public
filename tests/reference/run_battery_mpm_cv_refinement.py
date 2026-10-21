"""One predeclared exponent-two CV grid block, retaining all previous failures."""

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from battery_mpm_cases import case
from battery_mpm_traces import RESPONSE_NAMES, native_trace, reference_trace
from run_battery_mpm_block import guarded
from run_battery_mpm_resolution_block import assess_cv

NATIVE_GRIDS = ((16, 1024), (32, 1024), (64, 1024), (64, 256), (64, 512))
REFERENCE_GRIDS = ((160, 512), (320, 1024))
FINE = (64, 1024)
DIRECTIONS = (((16, 1024), (32, 1024), FINE), ((64, 256), (64, 512), FINE))


def native_spec(size, radial):
    return case(6, size_nodes=size, radial_cells=radial)


def run_unit(record, progress, checkpoint):
    record.update(inputs=case(6).to_dict(), native_mesh_exponent=2.0,
                  native=[], reference=[], response_order=list(RESPONSE_NAMES))
    for size, radial in NATIVE_GRIDS:
        progress(f"CVR01 native size={size}, radial={radial}, exponent=2")
        record["native"].append(guarded(
            lambda size=size, radial=radial: native_trace(native_spec(size, radial), 6, progress),
            size_nodes=size, radial_cells=radial))
        checkpoint()
    for size, radial in REFERENCE_GRIDS:
        progress(f"CVR01 reference size={size}, radial={radial}")
        record["reference"].append(guarded(
            lambda size=size, radial=radial: reference_trace(case(6), 6, size, radial, progress),
            size_nodes=size, radial_cells=radial))
        checkpoint()
    assess_cv(record, fine_grid=FINE, directions=DIRECTIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import pybamm
    import scipy

    record = {"id": "CVR01", "original_case": "G06", "passed": False}
    data = {
        "mode": "development", "denominator": 1, "units": [], "passed": 0,
        "active_unit": record, "does_not_replace_original_blocks": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "pybamm_version": pybamm.__version__, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "protocol": "workstreams/world_foundation/NATIVE_CONV45_MPM_CV_REFINEMENT_EXPERIMENT.md",
    }
    start, stop = time.monotonic(), threading.Event()
    stage = {"name": "starting"}
    with args.output.open("x", encoding="utf8") as output:

        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)

        def checkpoint():
            data["elapsed_s"] = time.monotonic() - start
            payload = json.dumps(data, default=convert, indent=2, allow_nan=False) + "\n"
            output.seek(0)
            output.write(payload)
            output.truncate()
            output.flush()

        def progress(name):
            stage["name"] = name
            print(name, flush=True)

        def heartbeat():
            while not stop.wait(20):
                elapsed, done = time.monotonic() - start, len(data["units"])
                print(f"MPM CV refinement active {done}/1; {stage['name']}; "
                      f"elapsed={elapsed:.1f}s; throughput={done/max(elapsed, 1):.4f}/s; "
                      "ETA=unknown",
                      flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        checkpoint()
        try:
            try:
                run_unit(record, progress, checkpoint)
            except Exception as exc:
                record.update(passed=False, exception=str(exc), traceback=traceback.format_exc())
            record["elapsed_s"] = time.monotonic() - start
            data["units"].append(record)
            data.pop("active_unit")
            data["passed"] = int(record["passed"])
            checkpoint()
            print(f"CVR01 complete 1/1; passed={data['passed']}/1; "
                  f"elapsed={data['elapsed_s']:.1f}s",
                  flush=True)
        finally:
            stop.set()
            thread.join()
    return int(not record["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
