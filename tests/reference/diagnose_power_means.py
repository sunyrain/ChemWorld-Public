"""Prospectively fixed 26-call independent diagnosis; upstream errors remain failures."""

import argparse
import json
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals.utils import mixing_power
from run_scientific_utils import compare
from scientific_utils_catalog import ADAPTERS, cases


def run(path):
    mp.mp.dps = 60
    selected = [row for row in cases() if row["name"] == "mixing_power"]
    assert len(selected) == 26
    rows = []
    for row in selected:
        xs, ps, r = row["args"]
        fractions = [mp.mpf(str(v)) for v in xs]
        properties = [mp.mpf(str(v)) for v in ps]
        power = mp.mpf(str(r))
        expected = sum(x * p**power for x, p in zip(fractions, properties, strict=True)) ** (
            1 / power
        )
        native = ADAPTERS["mixing_power"](xs, ps, r)
        reference = mixing_power(list(xs), list(ps), r)
        rows.append(
            {
                "input": row,
                "formula_60_digits": str(expected),
                "native": compare(native, expected),
                "chemicals": compare(reference, expected),
            }
        )
    result = {
        "schema": "chemworld-power-mean-diagnostic-1",
        "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "expected": 26,
        "completed": len(rows),
        "cases": rows,
        "native_passed": sum(row["native"][0]["passed"] for row in rows),
        "reference_passed": sum(row["chemicals"][0]["passed"] for row in rows),
        "boundary": "Independent published formula; does not relabel the original upstream "
        "agreement failures or modify the reference package.",
    }
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("completed", "native_passed", "reference_passed")}))
    return result["native_passed"] == 26


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
