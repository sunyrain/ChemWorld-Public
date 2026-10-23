"""Fixed alternative-parameter coil and 80-digit circle inverse checks."""

import argparse
import json
from itertools import product
from pathlib import Path

import mpmath as mp
from fluids.geometry import HelicalCoil
from run_equipment_geometry_block import compare, compare_metrics

from chemworld.physchem.equipment_geometry import helical_coil_geometry
from chemworld.physchem.geometric_primitives import circular_segment_area, circular_segment_height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve attempts")
    records = []
    for (dt, n), diameter in product(((0.01, 2), (0.03, 10)), (0.2, 1)):
        native = helical_coil_geometry(
            tube_diameter_m=dt,
            total_diameter_m=diameter + dt,
            total_height_m=n * dt + dt,
            turns=n,
            inner_diameter_m=0.8 * dt,
        )
        source = HelicalCoil(Dt=dt, Do=diameter, pitch=dt, N=n, Di=0.8 * dt)
        records.append(
            {
                "input": {"family": "coil", "Dt": dt, "N": n, "Do": diameter},
                "comparisons": compare_metrics(native, source, native.metrics),
            }
        )
    mp.mp.dps = 80
    for diameter, fraction in product((0.01, 1), (0.9, 1, 1.1)):
        area = circular_segment_area(diameter, diameter * fraction)
        native = circular_segment_height(diameter, area)
        if fraction >= 1:
            expected = mp.mpf(str(diameter))
        else:
            d = mp.mpf(str(diameter))
            r = d / 2
            target = mp.mpf(area)
            lower, upper = mp.mpf(0), d
            for _ in range(280):
                h = (lower + upper) / 2
                value = r * r * mp.acos((r - h) / r) - (r - h) * mp.sqrt(h * (d - h))
                if value > target:
                    upper = h
                else:
                    lower = h
            expected = (lower + upper) / 2
        records.append(
            {
                "input": {"family": "circle_inverse", "D": diameter, "fraction": fraction},
                "reference80": str(expected),
                "comparisons": [compare("height80", native, float(expected))],
            }
        )
    comparisons = [c for r in records for c in r["comparisons"]]
    result = {
        "inputs": len(records),
        "comparisons": len(comparisons),
        "passed": sum(c["passed"] for c in comparisons),
        "failed": sum(not c["passed"] for c in comparisons),
        "failures": [r for r in records if any(not c["passed"] for c in r["comparisons"])],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
