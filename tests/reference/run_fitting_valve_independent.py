"""Prospectively fixed 80-digit Colebrook and valve recomposition check."""

import argparse
import json
from pathlib import Path

import mpmath as mp
from fluids import fittings as reference
from run_fitting_valve_block import compare, inputs

from chemworld.physchem.straight_friction import crane_fully_turbulent_friction
from chemworld.physchem.valve_fittings import CRANE_VALVE_TYPES, crane_valve_coefficient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve all attempts")
    mp.mp.dps = 80
    records = []
    roots = {}
    for diameter in (0.05, 0.2286, 0.381, 0.5):
        d = mp.mpf(str(diameter))
        re = mp.mpf("7.5e6") * d
        rough = mp.mpf("3.4126825352925e-5") * d ** mp.mpf("-1.0112")
        # Solve directly in inverse square-root friction, a separate equation/solver.
        def residual(y, rough=rough, re=re):
            return y + 2 * mp.log10(rough / mp.mpf("3.7") + mp.mpf("2.51") * y / re)
        y = mp.findroot(residual, (mp.mpf(5), mp.mpf(10)))
        roots[diameter] = float(1 / y**2)
        native = crane_fully_turbulent_friction(diameter)
        native_residual = float(residual(1 / mp.sqrt(native)))
        records.append(
            {
                "diameter_m": diameter,
                "root80": str(1 / y**2),
                "comparisons": [
                    {
                        "quantity": "friction80",
                        "native": native,
                        "reference": roots[diameter],
                        "rtol": 1e-11,
                        "atol": 1e-12,
                        "passed": abs(native - roots[diameter]) <= 1e-12 + 1e-11 * roots[diameter],
                    },
                    {
                        "quantity": "Colebrook_residual80",
                        "native": native_residual,
                        "reference": 0,
                        "rtol": 1e-11,
                        "atol": 1e-12,
                        "passed": abs(native_residual) <= 1e-12,
                    },
                ],
            }
        )
    for case in inputs():
        if case["family"] != "crane" or case["fd"] is not None:
            continue
        d, beta, angle = case["D"], case["beta"], case["angle"]
        native = crane_valve_coefficient(
            CRANE_VALVE_TYPES[case["variant"]],
            pipe_diameter_m=d,
            throat_diameter_m=d * beta,
            angle_degrees=angle,
        )
        oracle = reference.Crane_loss_coefficient(
            d * beta, d, angle, reference.CRANE_VALVES[case["variant"]], roots[d]
        )
        records.append(
            {"input": case, "comparisons": [compare("root80_recomposition", native, oracle)]}
        )
    assert len(records) == 1124
    comparisons = [c for r in records for c in r["comparisons"]]
    summary = {
        "inputs": 1124,
        "comparisons": len(comparisons),
        "passed": sum(c["passed"] for c in comparisons),
        "failed": sum(not c["passed"] for c in comparisons),
        "failures": [r for r in records if any(not c["passed"] for c in r["comparisons"])],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
