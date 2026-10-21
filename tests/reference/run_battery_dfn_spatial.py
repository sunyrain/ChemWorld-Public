"""24-run endpoint diagnostic; D09/D10 traces use individual endpoint clocks.

Only D08 has aligned pointwise clocks in this retained diagnostic. D09/D10
support endpoint-convergence comparisons, not pointwise error subtraction.
See the recorded sampling deviation in NATIVE_CONV44_EXPERIMENT.md.
"""

import argparse
import json
import threading
import time
import traceback
from dataclasses import replace

import numpy as np
from battery_dfn_pybamm import reference_dfn
from run_battery_dfn_block import advance_battery, dfn_fields, program, units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, started, stop, active = [], time.monotonic(), threading.Event(), ["initializing"]

    def progress():
        while not stop.wait(30):
            elapsed = time.monotonic() - started
            print(json.dumps({"completed": len(rows), "total": 24, "active": active[0],
                "elapsed_s": elapsed,
                "eta_s": elapsed * (24 - len(rows)) / len(rows) if rows else None}), flush=True)

    settings = [{"spatial_cells": 64, "radial_cells": n, "radial_mesh": "uniform"}
                for n in (100, 200, 400)] + [
        {"spatial_cells": 64, "radial_cells": n, "radial_mesh": "exponential-right"}
        for n in (100, 200)] + [
        {"spatial_cells": 128, "radial_cells": 100, "radial_mesh": "exponential-right"}]
    with open(args.output, "x") as output:
        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        for identity, spec, segments, _, _ in units():
            if identity not in {"D08", "D09", "D10"}:
                continue
            duration, value, unit = segments[0]
            for config in [*settings, {"native_x": 8, "native_r": 24},
                           {"native_x": 16, "native_r": 48}]:
                active[0] = f"{identity} {config}"
                print(f"starting {active[0]} ({len(rows)}/24)", flush=True)
                start = time.monotonic()
                row = {"unit": identity, "config": config}
                try:
                    if "native_x" in config:
                        native = replace(spec,
                            negative=replace(spec.negative, cells=config["native_r"]),
                            positive=replace(spec.positive, cells=config["native_r"]),
                            electrolyte=replace(spec.electrolyte, cells=(config["native_x"],) * 3))
                        initial = native.initial_state()
                        result = advance_battery(native, initial, program(value, unit), duration)
                        clock = np.linspace(0, result.actual_duration_s, 31)
                        sampled = advance_battery(native, initial, program(value, unit), clock[-1],
                                                   evaluation_times_s=clock)
                        states = sampled.trajectory[:31]
                        fields = [dfn_fields(native, s, program=program(value, unit))
                                  for s in states]
                        row.update(end_s=result.actual_duration_s, event=result.stopping_event,
                            voltage=[f["voltage_V"] for f in fields],
                            current=[f["current_A"] for f in fields],
                            charge=[s.coordinates["discharged_charge_C"] for s in states],
                            salt_min=[float(min(f["electrolyte_concentration_mol_m3"]))
                                      for f in fields],
                            salt_max=[float(max(f["electrolyte_concentration_mol_m3"]))
                                      for f in fields],
                            conservation=dict(result.conservation_residuals))
                    else:
                        end = reference_dfn(spec, segments, **config)[0][1]
                        clock = np.linspace(0, end, 31)
                        _, end, solution = reference_dfn(spec, segments, evaluation_times=[clock],
                                                        **config)[0]
                        clock = np.minimum(clock, end)
                        salt = np.asarray(solution["Electrolyte concentration [mol.m-3]"](clock))
                        row.update(end_s=end, event=solution.termination,
                            voltage=np.asarray(solution["Voltage [V]"](clock)).reshape(-1).tolist(),
                            current=np.asarray(solution["Current [A]"](clock)).reshape(-1).tolist(),
                            charge=(np.asarray(solution["Discharge capacity [A.h]"](clock))
                                    .reshape(-1) * 3600).tolist(),
                            salt_min=np.min(salt, axis=0).tolist(),
                            salt_max=np.max(salt, axis=0).tolist())
                    row["times_s"] = clock.tolist()
                except Exception:
                    row["exception"] = traceback.format_exc()
                row["elapsed_s"] = time.monotonic() - start
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("unit", "config", "end_s", "elapsed_s")
                                  if k in row} | {"completed": len(rows),
                                                 "exception": row.get("exception")}), flush=True)
            # Flush completed units even if a later native/reference solve is interrupted.
            output.seek(0)
            json.dump({"mode": "development-spatial-diagnostic", "planned": 24,
                       "sampling": "individual endpoint clocks; D09/D10 endpoint-only comparison",
                       "completed": len(rows), "units": rows,
                       "elapsed_s": time.monotonic() - started}, output, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
        stop.set()
        thread.join()
    return int(any("exception" in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
