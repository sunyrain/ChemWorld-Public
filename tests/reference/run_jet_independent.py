"""Six prospectively fixed alternate-root / scalar-reduction diagnostics."""

import argparse
import json
import time
from math import isclose
from pathlib import Path

import mpmath as mp
from run_jet_block import JET_VARIABLE_UNITS, base

from chemworld.physchem.jet_pumps import solve_liquid_jet_pump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mp.mp.dps = 80
    m = mp.mpf
    designs = [
        ((0.8, 0.45, 1.25, True), ("nozzle_diameter_m", "mixing_diameter_m")),
        ((0.8, 0.55, 0.75, True), ("nozzle_diameter_m", "mixing_diameter_m")),
        ((1.2, 0.45, 1.25, False), ("nozzle_diameter_m", "mixing_diameter_m")),
        ((1.2, 0.55, 0.75, False), ("nozzle_diameter_m", "mixing_diameter_m")),
        ((1.2, 0.55, 1.25, False), ("nozzle_diameter_m", "motive_flow_m3_s")),
        ((0.8, 0.45, 1.25, True), ("suction_flow_m3_s", "suction_pressure_Pa")),
    ]
    started = time.perf_counter()
    records = []
    for parameters, missing in designs:
        model, values = base(*parameters)
        report = solve_liquid_jet_pump(
            model,
            {k: v for k, v in values.items() if k not in missing},
            initial={k: values[k] * f for k, f in zip(missing, (0.97, 1.03), strict=True)},
            bounds={k: (values[k] * 0.75, values[k] * 1.25) for k in missing},
            diffuser_diameter_m=0.1,
        )

        def residual(x, y, missing=missing, values=values, parameters=parameters):
            v = {k: m(value) for k, value in values.items()}
            v.update(zip(missing, (x, y), strict=True))
            dn, dm, qp, qs, p1, p2, p5 = (v[k] for k in JET_VARIABLE_UNITS)
            rho1, rho2 = m(1000), m(1000 * parameters[0])
            kp, ks, km, kd = map(m, (0.04, 0.11, 0.186, 0.12))
            an, annulus = mp.pi * dn**2 / 4, mp.pi * (dm**2 - dn**2) / 4
            pressure_difference = (
                rho1 * (qp / an) ** 2 * (1 + kp) - rho2 * (qs / annulus) ** 2 * (1 + ks)
            ) / 2
            r, alpha, ratio, c = dn**2 / dm**2, dm**2 / m(0.1) ** 2, qs / qp, rho2 / rho1
            numerator = 2 * r + 2 * c * ratio**2 * r**2 / (1 - r)
            numerator -= r**2 * (1 + c * ratio) * (1 + ratio) * (1 + km + kd + alpha**2)
            numerator -= c * ratio**2 * r**2 * (1 + ks) / (1 - r) ** 2
            denominator = 1 + kp - 2 * r - 2 * c * ratio**2 * r**2 / (1 - r)
            denominator += r**2 * (1 + c * ratio) * (1 + ratio) * (1 + km + kd + alpha**2)
            if parameters[3]:
                denominator += c * ratio**2 * r**2 * (1 - ks) / (1 - r) ** 2
            return (p1 - p2 - pressure_difference) / p1, (p5 - p2) / (
                p1 - p5
            ) * denominator - numerator

        native = [report[k] for k in missing]
        solved = mp.findroot(residual, tuple(map(m, native)), tol=m("1e-65"), maxsteps=80)
        errors = [float(value) for value in residual(*map(m, native))]
        checks = [
            isclose(a, float(b), rel_tol=2e-7, abs_tol=1e-10)
            for a, b in zip(native, solved, strict=True)
        ]
        checks.extend(abs(error) <= 2e-10 for error in errors)
        records.append(
            {
                "parameters": parameters,
                "missing": missing,
                "native": native,
                "independent80_root": list(map(str, solved)),
                "scaled_residuals": errors,
                "checks": checks,
                "solver": report["solver"],
            }
        )
    result = {
        "inputs": 6,
        "comparisons": 24,
        "passed": sum(sum(r["checks"]) for r in records),
        "records": records,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    return 0 if result["passed"] == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
