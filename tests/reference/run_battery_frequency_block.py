"""Fixed ten-unit SPMe/DFN frequency qualification; development evidence only."""

import argparse
import json
import platform
import subprocess
import threading
import time
import traceback
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy
from battery_frequency_pybamm import reference_frequency

from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import REGIONS, BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec


def cases():
    result = []
    for family, factory in (("SPMe", BatterySPMSpec), ("DFN", BatteryDFNSpec)):
        base = factory(electrolyte=BatteryElectrolyteSpec(), numerical_form="residual")
        grids = [
            replace(
                base,
                negative=replace(base.negative, cells=nr),
                positive=replace(base.positive, cells=nr),
                electrolyte=replace(base.electrolyte, cells=(nx,) * 3),
            )
            for nx, nr in ((12, 32), (24, 64), (48, 128))
        ]
        asymmetric = replace(
            grids[1],
            negative=ParticleSpec(
                cells=64, radius_m=8e-6, diffusivity_m2_s=2e-14, exchange_current_scale_A_m2=0.8
            ),
            positive=ParticleSpec(
                cells=64, radius_m=12e-6, diffusivity_m2_s=8e-15, exchange_current_scale_A_m2=1.3
            ),
            electrolyte=replace(
                grids[1].electrolyte,
                area_m2=0.03,
                lengths_m=(80e-6, 30e-6, 120e-6),
                porosities=(0.35, 0.55, 0.45),
                diffusivity_m2_s=3e-10,
                conductivity_S_m=0.8,
                solid_conductivity_S_m=(70.0, 120.0),
            ),
        )
        stressed = replace(
            grids[1],
            temperature_K=315.0,
            initial_negative_fraction=0.45,
            initial_positive_fraction=0.55,
            electrolyte=replace(
                grids[1].electrolyte,
                diffusivity_m2_s=2e-11,
                conductivity_S_m=0.15,
                cation_transference=0.25,
                thermodynamic_factor=1.4,
            ),
        )
        result.extend((family, spec) for spec in (*grids, asymmetric, stressed))
    return result


def native_frequency(spec, frequencies):
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    spectrum = battery_impedance(spec, frequencies, state=state)
    indices = {v.name: i for i, v in enumerate(spectrum.response.variables)}
    electrolyte = spec.electrolyte
    salt_indices = np.array([indices[k] for k in electrolyte.keys])
    pore_volumes = electrolyte.diffusion.storage_volumes_m3
    points = []
    for point, z in zip(spectrum.response.points, spectrum.impedance_ohm, strict=True):
        row = {
            "frequency_Hz": point.frequency_Hz,
            "success": point.success,
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason,
        }
        if point.success:
            amounts = point.response[salt_indices, 0]
            concentrations = [
                sum(amounts[electrolyte.region_slices[r]])
                / sum(pore_volumes[electrolyte.region_slices[r]])
                for r in REGIONS
            ]
            values = np.array([z, *concentrations])
            row.update(response_real=values.real.tolist(), response_imag=values.imag.tolist())
        points.append(row)
    unchanged = before == (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    return {
        "success": spectrum.success and unchanged,
        "points": points,
        "state_unchanged": unchanged,
        "diagnostics": spectrum.response.diagnostics,
    }


def values(record):
    return np.array([p["response_real"] for p in record["points"]]) + 1j * np.array(
        [p["response_imag"] for p in record["points"]]
    )


def compare(spec, family, frequencies, index, progress, *, spme_radial_mesh="exponential-right"):
    result = {"passed": False}
    try:
        for name, solve in (
            ("native", lambda: native_frequency(spec, frequencies)),
            (
                "reference",
                lambda: reference_frequency(
                    spec, family, frequencies, spme_radial_mesh=spme_radial_mesh
                ),
            ),
            (
                "refined_reference",
                lambda: reference_frequency(
                    spec, family, frequencies, refined=True, spme_radial_mesh=spme_radial_mesh
                ),
            ),
        ):
            progress(name)
            result[name] = solve()
        if not all(result[k]["success"] for k in ("native", "reference", "refined_reference")):
            result["failure"] = "frequency_or_stationary_failure"
            return result
        native, reference, refined = [
            values(result[k]) for k in ("native", "reference", "refined_reference")
        ]
        coarse = index in (1, 6)
        bound = np.array([0.01 if coarse else 0.002, 1.0, 1.0, 1.0]) + np.array(
            [0.02 if coarse else 0.005, 0.01, 0.01, 0.01]
        ) * np.abs(reference)
        reference_bound = np.array([0.0004, 0.2, 0.2, 0.2]) + np.array(
            [0.001, 0.002, 0.002, 0.002]
        ) * np.abs(refined)
        errors, refinement = np.abs(native - reference), np.abs(reference - refined)
        result.update(
            passed=bool(np.all(errors <= bound) and np.all(refinement <= reference_bound)),
            errors=errors.tolist(),
            bounds=bound.tolist(),
            refinement_errors=refinement.tolist(),
            refinement_bounds=reference_bound.tolist(),
            maximum_impedance_error_ohm=float(np.max(errors[:, 0])),
            maximum_salt_error_mol_m3_A=float(np.max(errors[:, 1:])),
        )
    except Exception:
        result["exception"] = traceback.format_exc()
    return result


def grid_checks(units):
    checks = []
    for family, rows in (("SPMe", units[:3]), ("DFN", units[5:8])):
        for metric in ("maximum_impedance_error_ohm", "maximum_salt_error_mol_m3_A"):
            errors = [row.get(metric) for row in rows]
            passed = (
                all(v is not None for v in errors)
                and all(b <= a + 1e-8 for a, b in pairwise(errors))
                and errors[-1] < errors[0]
            )
            checks.append(
                {"family": family, "metric": metric, "errors": errors, "passed": bool(passed)}
            )
            if not passed:
                for row in rows:
                    row["passed"] = False
                    row.setdefault("grid_failures", []).append(metric)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spme-radial-mesh",
        choices=("uniform", "exponential-right"),
        default="exponential-right",
        help="uniform reproduces the retained failed block",
    )
    args = parser.parse_args()
    frequencies = np.logspace(-5, -1, 13)
    data = {
        "mode": "development",
        "block": "SPMe/DFN independent frequency qualification",
        "reference_design": f"SPMe {args.spme_radial_mesh} radial grid; original bounds retained",
        "total": 10,
        "completed": 0,
        "units": [],
        "frequencies_Hz": frequencies.tolist(),
        "frequency_count_per_case": 13,
        "native_conv45_complete": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    start = time.monotonic()
    status, stopped = {"stage": "initializing"}, threading.Event()

    def heartbeat():
        while not stopped.wait(20):
            elapsed, count = time.monotonic() - start, data["completed"]
            eta = f"{elapsed / count * (10 - count):.1f}s" if count else "unknown"
            print(
                f"{status['stage']}: {count}/10 complete, elapsed={elapsed:.1f}s, "
                f"rate={count * 60 / elapsed:.2f}/min, ETA={eta}",
                flush=True,
            )

    def progress(stage):
        status["stage"] = f"F{data['completed'] + 1:02} {stage}"
        print(status["stage"], flush=True)

    with args.output.open("x") as output:
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()

        def save():
            data["elapsed_s"] = time.monotonic() - start
            serialized = json.dumps(data, indent=2, allow_nan=False)
            output.seek(0)
            output.write(serialized)
            output.truncate()
            output.flush()

        try:
            save()
            for index, (family, spec) in enumerate(cases(), 1):
                begin = time.monotonic()
                try:
                    result = compare(
                        spec,
                        family,
                        frequencies,
                        index,
                        progress,
                        spme_radial_mesh=args.spme_radial_mesh,
                    )
                except Exception:
                    result = {"passed": False, "exception": traceback.format_exc()}
                data["units"].append(
                    {
                        "unit": f"F{index:02}",
                        "family": family,
                        "parameters": spec.to_dict(),
                        **result,
                        "elapsed_s": time.monotonic() - begin,
                    }
                )
                data["completed"] = index
                save()
                print(
                    f"F{index:02}: passed={result['passed']} ({index}/10), "
                    f"elapsed={data['elapsed_s']:.2f}s",
                    flush=True,
                )
            data["grid_checks"] = grid_checks(data["units"])
            data["passed"] = sum(row["passed"] for row in data["units"])
            data["block_passed"] = data["passed"] == data["total"]
            save()
        finally:
            stopped.set()
            worker.join()
    print(f"Completed: {data['passed']}/10, grid included", flush=True)
    return 0 if data["block_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
