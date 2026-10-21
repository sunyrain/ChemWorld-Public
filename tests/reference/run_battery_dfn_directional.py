"""Twelve predeclared native endpoint runs; no qualification reclassification."""

import argparse
import json
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from battery_dfn_source import load_dfn_source

from chemworld.physchem.battery_dfn import dfn_fields
from chemworld.physchem.battery_models import advance_battery, battery_spec_from_dict
from chemworld.physchem.reactor_shared import ScalarControlProgram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=int,
                        help="explicit zero-based aggregate attempt index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_dfn_source(args.source, args.source_index)
    rows, started, stop, active = [], time.monotonic(), threading.Event(), ["initializing"]

    def progress():
        while not stop.wait(30):
            elapsed = time.monotonic() - started
            print(json.dumps({"completed": len(rows), "total": 12, "active": active[0],
                "elapsed_s": elapsed,
                "eta_s": elapsed * (12 - len(rows)) / len(rows) if rows else None}), flush=True)

    with args.output.open("x") as output:
        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        for identity in ("D09", "D10"):
            original = next(r for r in source["units"] if r["unit"] == identity)
            base = battery_spec_from_dict(original["parameters"])
            duration, value, unit = original["protocol"][0]
            program = ScalarControlProgram(((0, value),), unit)
            for nx, nr, exponent in ((8, 24, 2), (8, 48, 2), (16, 24, 2),
                                     (16, 48, 2), (8, 24, 3), (16, 48, 3)):
                row = {"unit": identity, "cells_x": nx, "cells_r": nr,
                       "radial_exponent": exponent}
                active[0] = f"{identity}:x{nx}:r{nr}:p{exponent}"
                print(f"starting {active[0]} ({len(rows)}/12)", flush=True)
                begin = time.monotonic()
                try:
                    spec = replace(base,
                        negative=replace(base.negative, cells=nr, mesh_exponent=exponent),
                        positive=replace(base.positive, cells=nr, mesh_exponent=exponent),
                        electrolyte=replace(base.electrolyte, cells=(nx,) * 3))
                    result = advance_battery(spec, spec.initial_state(), program, duration)
                    fields = dfn_fields(spec, result.final_state, program=program)
                    row.update(parameters=spec.to_dict(), end_s=result.actual_duration_s,
                        event=result.stopping_event,
                        conservation=dict(result.conservation_residuals),
                        fields={key: val.tolist() if hasattr(val, "tolist") else val
                                for key, val in fields.items()})
                except Exception:
                    row["exception"] = traceback.format_exc()
                row["elapsed_s"] = time.monotonic() - begin
                rows.append(row)
                print(json.dumps({key: row[key] for key in ("unit", "cells_x", "cells_r",
                    "radial_exponent", "end_s", "event", "elapsed_s", "exception")
                    if key in row}), flush=True)
                output.seek(0)
                json.dump({"mode": "development-directional-diagnostic", "planned": 12,
                    "completed": len(rows), "source": args.source.name,
                    "source_index": args.source_index,
                    "source_inputs": [r for r in source["units"] if r["unit"] in {"D09", "D10"}],
                    "units": rows, "elapsed_s": time.monotonic() - started},
                    output, indent=2, allow_nan=False)
                output.truncate()
                output.flush()
        stop.set()
        thread.join()
    return int(any("exception" in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
