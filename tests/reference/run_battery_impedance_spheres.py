"""Fixed six-unit continuous-sphere impedance block; no upstream runtime needed."""

import argparse
import json
import platform
import subprocess
import time
import traceback
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy

from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec


def analytic(spec, frequencies):
    # Independent continuous PDE solution: no native FV, flux or derivative calls.
    gas_constant, faraday = 8.314462618, 96485.33212
    result = np.full(len(frequencies), spec.series_resistance_ohm, dtype=complex)
    for name in ("negative", "positive"):
        p = getattr(spec, name)
        theta = getattr(spec, f"initial_{name}_fraction")
        area = 3 * p.active_volume_m3 / p.radius_m
        j0 = p.exchange_current_scale_A_m2 * np.sqrt(theta * (1 - theta))
        result += gas_constant * spec.temperature_K / (faraday * area * j0)
        z = np.sqrt(2j * np.pi * frequencies * p.radius_m**2 / p.diffusivity_m2_s)
        g = p.radius_m / (
            faraday
            * area
            * p.diffusivity_m2_s
            * p.maximum_concentration_mol_m3
            * (z / np.tanh(z) - 1)
        )
        result -= getattr(spec, f"{name}_ocv_slope_V") * g
    return result


def cases():
    base = BatterySPMSpec()
    result = [
        replace(
            base, negative=replace(base.negative, cells=n), positive=replace(base.positive, cells=n)
        )
        for n in (16, 32, 64, 128)
    ]
    second = replace(
        base,
        negative=ParticleSpec(
            cells=64,
            radius_m=8e-6,
            diffusivity_m2_s=2e-14,
            maximum_concentration_mol_m3=27000,
            active_volume_m3=8e-7,
            exchange_current_scale_A_m2=0.8,
        ),
        positive=ParticleSpec(
            cells=64,
            radius_m=12e-6,
            diffusivity_m2_s=8e-15,
            maximum_concentration_mol_m3=33000,
            active_volume_m3=1.2e-6,
            exchange_current_scale_A_m2=1.3,
        ),
        initial_negative_fraction=0.6,
        initial_positive_fraction=0.4,
        negative_ocv_slope_V=-0.15,
        positive_ocv_slope_V=-0.65,
        temperature_K=310.15,
        series_resistance_ohm=0.03,
    )
    return [*result, second, replace(second, negative_ocv_slope_V=0, positive_ocv_slope_V=0)]


def run(spec, frequencies, index):
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    native = battery_impedance(spec, frequencies, state=state)
    reference = analytic(spec, frequencies)
    result = {
        "native": native.to_dict(),
        "reference_real_ohm": reference.real.tolist(),
        "reference_imag_ohm": reference.imag.tolist(),
    }
    if not native.success:
        return {**result, "passed": False, "failure": "native_frequency_failure"}
    values = np.array(native.impedance_ohm)
    errors = np.abs(values - reference)
    bound = (
        (np.full(len(frequencies), 1e-8))
        if index == 6
        else 0.002 + 0.005 * np.abs(reference)
        if index >= 3
        else 0.01 + 0.02 * np.abs(reference)
    )
    unchanged = before == (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    return {
        **result,
        "passed": bool(np.all(errors <= bound) and unchanged),
        "error_ohm": errors.tolist(),
        "bound_ohm": bound.tolist(),
        "maximum_error_ohm": float(np.max(errors)),
        "state_unchanged": unchanged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frequencies = np.logspace(-5, -1, 13)
    data = {
        "mode": "development",
        "block": "six continuous-sphere impedance cases",
        "total": 6,
        "completed": 0,
        "frequency_count_per_case": 13,
        "frequencies_Hz": frequencies.tolist(),
        "units": [],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_conv45_complete": False,
    }
    start = time.monotonic()
    with args.output.open("x") as output:
        for index, spec in enumerate(cases(), 1):
            begin = time.monotonic()
            print(f"I{index:02}: starting ({index - 1}/6 complete)", flush=True)
            try:
                result = run(spec, frequencies, index)
            except Exception:
                result = {"passed": False, "exception": traceback.format_exc()}
            data["units"].append(
                {
                    "unit": f"I{index:02}",
                    "parameters": spec.to_dict(),
                    "elapsed_s": time.monotonic() - begin,
                    **result,
                }
            )
            data["completed"] = index
            data["elapsed_s"] = time.monotonic() - start
            output.seek(0)
            json.dump(data, output, indent=2, allow_nan=False)
            output.truncate()
            output.flush()
            print(
                f"I{index:02}: passed={result['passed']} ({index}/6 complete), "
                f"{data['elapsed_s']:.2f}s elapsed",
                flush=True,
            )
        errors = [row.get("maximum_error_ohm") for row in data["units"][:4]]
        converged = (
            all(e is not None for e in errors)
            and all(
                later <= earlier + 1e-8
                for earlier, later in pairwise(errors)
            )
            and errors[-1] < errors[0]
        )
        data["grid_converged"] = bool(converged)
        data["passed"] = sum(row["passed"] for row in data["units"])
        data["block_passed"] = bool(data["passed"] == 6 and converged)
        output.seek(0)
        json.dump(data, output, indent=2, allow_nan=False)
        output.truncate()
        print(f"Complete: {data['passed']}/6; grid_converged={converged}", flush=True)
    return 0 if data["block_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
