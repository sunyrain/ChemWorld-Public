"""Fixed 8-input/14-comparison diagnosis, preserving the primary block failures."""

import argparse
import json
from math import isclose
from pathlib import Path

import fluids.drag as reference
import mpmath as mp
from scipy.integrate import solve_ivp

from chemworld.physchem.particle_drag import (
    ParticleDragSpec,
    ParticleMotionState,
    advance_particle_motion,
    particle_terminal_velocity,
)
from chemworld.physchem.transport import FluidState


def compare(quantity, native, expected, rtol):
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "rtol": rtol,
        "atol": 1e-12,
        "passed": isclose(native, expected, rel_tol=rtol, abs_tol=1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior attempts")
    records = []
    for diameter in (1e-4, 1e-3):
        spec = ParticleDragSpec(diameter, 500, FluidState(1000, 0.001))
        with mp.workdps(80):
            d = mp.mpf(str(diameter))
            target = mp.mpf(4) / 3 * mp.mpf("9.80665") * d**3 * 1000 * 500 / mp.mpf(".001") ** 2

            def residual(reynolds, target=target):
                cd = (
                    mp.mpf("5.4856e9") * mp.tanh(mp.mpf("4.3774e-9") / reynolds)
                    + mp.mpf(".0709") * mp.tanh(mp.mpf("700.6574") / reynolds)
                    + mp.mpf(".3894") * mp.tanh(mp.mpf("74.1539") / reynolds)
                    - mp.mpf(".1198") * mp.tanh(mp.mpf("7429.0843") / reynolds)
                    + mp.mpf("1.7174") * mp.tanh(mp.mpf("9.9851") / (reynolds + mp.mpf("2.3384")))
                    + mp.mpf(".4744")
                )
                return cd * reynolds**2 - target

            reynolds = mp.findroot(residual, (mp.mpf(".1"), mp.mpf(100)))
            expected = -float(reynolds * mp.mpf(".001") / (1000 * d))
        records.append(
            {
                "family": "rising_force_balance",
                "D": diameter,
                "comparisons": [
                    compare(
                        "velocity", particle_terminal_velocity(spec).velocity_m_s, expected, 2e-10
                    )
                ],
            }
        )
    for diameter, duration, initial in (
        (1e-6, 0.001, 0.1),
        (1e-6, 0.1, 0.1),
        (1e-6, 1, 0.1),
        (1e-4, 0.1, 0.1),
        (1e-4, 1, 0),
        (1e-4, 1, 0.1),
    ):
        spec = ParticleDragSpec(diameter, 3000, FluidState(1000, 0.001))

        def rhs(time, values, diameter=diameter):
            velocity = values[0]
            reynolds = abs(velocity) * 1000 * diameter / 0.001
            drag = (
                0
                if velocity == 0
                else 3
                * 1000
                * reference.drag_sphere(reynolds)
                * velocity
                * abs(velocity)
                / (4 * diameter * 3000)
            )
            return [9.80665 * (1 - 1000 / 3000) - drag, velocity]

        solved = solve_ivp(rhs, (0, duration), (initial, 0), method="Radau", rtol=1e-11, atol=1e-14)
        assert solved.success
        native = advance_particle_motion(spec, ParticleMotionState(velocity_m_s=initial), duration)
        records.append(
            {
                "family": "independent_Radau",
                "D": diameter,
                "t": duration,
                "V0": initial,
                "function_evaluations": solved.nfev,
                "comparisons": [
                    compare("velocity", native.state.velocity_m_s, solved.y[0, -1], 2e-9),
                    compare("displacement", native.displacement_m, solved.y[1, -1], 2e-9),
                ],
            }
        )
        print(f"drag independent {len(records)}/8", flush=True)
    comparisons = [item for record in records for item in record["comparisons"]]
    assert len(records) == 8 and len(comparisons) == 14
    summary = {
        "inputs": 8,
        "comparisons": 14,
        "passed": sum(item["passed"] for item in comparisons),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))


if __name__ == "__main__":
    main()
