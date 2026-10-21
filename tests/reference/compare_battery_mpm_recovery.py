"""Compare retained native records; this is not a new qualification or replay."""

import argparse
import json
from itertools import product
from math import isfinite
from pathlib import Path

GRIDS = ((8, 64), (16, 64), (32, 64), (32, 16), (32, 32))
MOMENTS = set(product(("number", "area", "volume"), (8, 16, 32), (-3, -1, 0, 1, 2, 3)))


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, float) and not isfinite(value):
        raise ValueError("nonfinite numerical record cannot establish equality")


def indexed(rows, keys, expected):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"duplicate record: {key}")
        result[key] = row
    if result.keys() != expected:
        raise ValueError("records do not cover the complete original fixed matrix")
    return result


def complete_block(block):
    require_finite(block)
    if block["denominator"] != 9 or "active_unit" in block:
        raise ValueError("only complete original nine-unit blocks may be compared")
    units = indexed(block["units"], ("id",), {(f"G{i:02}",) for i in range(1, 10)})
    if block["passed"] != sum(u["passed"] for u in units.values()):
        raise ValueError("qualification pass count does not match retained units")
    return units


def compare_blocks(before, after):
    old, new = complete_block(before), complete_block(after)
    results = []
    for identity, unit in new.items():
        previous = old[identity]
        fields = ["inputs", "passed"]
        row = {"id": identity[0], "native_records": 0, "moment_records": 0, "differences": []}
        if identity == ("G08",):
            keys = ("basis", "nodes", "moment_order")
            a = indexed(previous["comparisons"], keys, MOMENTS)
            b = indexed(unit["comparisons"], keys, MOMENTS)
            row["moment_records"] = len(b)
            for key, record in b.items():
                if record != a[key]:
                    row["differences"].append({"moment": list(key)})
        else:
            grids = {(1, radial) for radial in (16, 32, 64)} if identity == ("G01",) else set(GRIDS)
            a = indexed(previous["native"], ("size_nodes", "radial_cells"), grids)
            b = indexed(unit["native"], ("size_nodes", "radial_cells"), grids)
            row["native_records"] = len(b)
            if identity == ("G09",):
                fields.append("frequencies_Hz")
                responses = ("real", "imag", "success")
                for record in (*a.values(), *b.values()):
                    if len(record["real"]) != 7 or len(record["imag"]) != 7:
                        raise ValueError("all seven frequency responses are required")
            else:
                fields.append("response_order")
                responses = ("spec", "segments")
                if any(not record["segments"] for record in (*a.values(), *b.values())):
                    raise ValueError("missing native trace segments")
            for key, record in b.items():
                changed = [name for name in responses if record[name] != a[key][name]]
                if changed:
                    row["differences"].append({"grid": list(key), "fields": changed})
        changed = [name for name in fields if unit[name] != previous[name]]
        if changed:
            row["differences"].append({"unit_fields": changed})
        row["equal"] = not row["differences"]
        results.append(row)
    return {
        "scope": "original nine units: native retained values and distribution moments",
        "comparison": "exact JSON value equality, not cross-platform bitwise equivalence",
        "excludes": "independent reference traces, wall times and solver diagnostics",
        "does_not_replace_scientific_qualification": True,
        "before_source_commit": before["source_commit"],
        "after_source_commit": after["source_commit"],
        "denominator": 9,
        "equal_units": sum(row["equal"] for row in results),
        "native_records": sum(row["native_records"] for row in results),
        "moment_records": sum(row["moment_records"] for row in results),
        "units": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_blocks(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    result.update(before=str(args.before), after=str(args.after))
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Native value comparison: {result['equal_units']}/9 units equal; "
          f"{result['native_records']} grid/spectrum and {result['moment_records']} moment records")
    return int(result["equal_units"] != 9)


if __name__ == "__main__":
    raise SystemExit(main())
