"""Read-only frequency response at an explicitly authored composite equilibrium.

The original composite time-domain example remains unequilibrated. This separate
example declares beta theta=.6 so both negative OCVs are .13 V; it does not relax
or alter a caller's operating state to manufacture stationarity.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from chemworld.examples.battery_composite import default_spec as time_domain_spec
from chemworld.physchem.battery_impedance import battery_impedance


def default_spec(*, cells=24):
    original = time_domain_spec(cells=cells)
    return replace(
        original,
        negative_phases=(
            original.negative_phases[0],
            replace(original.negative_phases[1], initial_fraction=.6),
        ),
        provenance="synthetic-composite-stationary-frequency-example-v1",
    )


def state_record(state):
    """Copy the original scalar/mapping fields without pickling read-only maps."""
    return {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state) for value in (getattr(state, field.name),)
    }


def run(output, *, spec=None, frequencies_Hz=(1e-4, 1e-3, 1e-2), state=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier analysis; choose a new output path")
    spec = default_spec() if spec is None else spec
    state = spec.initial_state() if state is None else state
    original = state_record(state)
    spectrum = battery_impedance(spec, frequencies_Hz, state=state)
    names = [v.name for v in spectrum.response.variables]
    indices = [names.index(k) for k in spec.algebraic_names]
    phases = []
    for point in spectrum.response.points:
        currents = None if point.response is None else (
            spec.reaction_map @ point.response[indices, 0]
        )
        phases.append({
            "frequency_Hz": point.frequency_Hz,
            "phase_current_response_A_per_A": {
                identity: None if currents is None else {
                    "real": float(currents[i].real), "imag": float(currents[i].imag),
                }
                for i, (*_, identity) in enumerate(spec.particle_bins)
            },
        })
    if state_record(state) != original:
        raise RuntimeError("frequency analysis mutated its input state")
    report = {
        "model": spec.to_dict(),
        "operating_state": original,
        "scope": "synthetic single-valued composite equilibrium; read-only physical API",
        "spectrum": spectrum.to_dict(),
        "full_response": spectrum.response.to_dict(),
        "phase_responses": phases,
        "state_unchanged": True,
    }
    # Full failures are useful evidence too; do not replace failed frequencies.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=24)
    args = parser.parse_args()
    report = run(args.output, spec=default_spec(cells=args.cells),
                 frequencies_Hz=np.logspace(-4, -2, 9))
    print(json.dumps({"success": report["spectrum"]["success"],
                      "frequencies": len(report["phase_responses"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
