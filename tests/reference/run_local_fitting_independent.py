"""Single independent Crane friction root and all fixed bend recompositions."""

import argparse
import json
from pathlib import Path

import mpmath as mp
from fluids.friction import ft_Crane

from chemworld.physchem.straight_friction import crane_fully_turbulent_friction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve all attempts")
    mp.mp.dps = 80
    d = mp.mpf(".1")
    re = mp.mpf("7.5e6") * d
    rough = mp.mpf("3.4126825352925e-5") * d ** mp.mpf("-1.0112")

    def residual(y):
        return y + 2 * mp.log10(rough / mp.mpf("3.7") + mp.mpf("2.51") * y / re)

    y = mp.findroot(residual, (5, 10))
    root = float(1 / y**2)
    native = crane_fully_turbulent_friction(0.1)
    records = [
        {
            "input": {"D": 0.1},
            "root80": str(1 / y**2),
            "comparisons": [
                {
                    "quantity": "friction80",
                    "native": native,
                    "reference": root,
                    "rtol": 1e-11,
                    "atol": 1e-12,
                    "passed": abs(native - root) <= 1e-12 + 1e-11 * root,
                },
                {
                    "quantity": "Colebrook_residual80",
                    "native": float(residual(1 / mp.sqrt(native))),
                    "reference": 0,
                    "rtol": 1e-11,
                    "atol": 1e-12,
                    "passed": abs(residual(1 / mp.sqrt(native))) <= 1e-12,
                },
            ],
        }
    ]
    source = json.loads(args.source.read_text())
    cases = [
        r
        for r in source["records"]
        if r["input"]["family"] == "bend" and r["input"]["method"] in {"Crane", "Crane standard"}
    ]
    assert len(cases) == 272
    for r in cases:
        c = r["comparisons"][0]
        expected = c["reference"] / ft_Crane(0.1) * root
        records.append(
            {
                "input": r["input"],
                "comparisons": [
                    {
                        "quantity": "root80_recomposition",
                        "native": c["native"],
                        "reference": expected,
                        "rtol": 3e-8,
                        "atol": 1e-10,
                        "passed": abs(c["native"] - expected) <= 1e-10 + 3e-8 * abs(expected),
                    }
                ],
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
