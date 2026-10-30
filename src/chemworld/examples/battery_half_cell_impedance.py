"""Read-only impedance and finite-metal inventory response of a resting half cell.

The original example's uniform initial state is used directly. Fixed interface
geometry is explicit; the response is a derivative, not a finite oscillation
guaranteed to leave positive metal inventory at every frequency.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np

from chemworld.examples.battery_half_cell import default_spec
from chemworld.physchem.battery_impedance import battery_impedance


def state_record(state):
    return {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state) for value in (getattr(state, field.name),)
    }


def run(output, *, spec=None, state=None, frequencies_Hz=(1e-4, 1e-3, 1e-2)):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier analysis; choose a new output path")
    spec = default_spec(cells=24) if spec is None else spec
    state = spec.initial_state() if state is None else state
    original = state_record(state)
    spectrum = battery_impedance(spec, frequencies_Hz, state=state)
    names = [v.name for v in spectrum.response.variables]
    groups = {
        "metal_Li_mol_per_A": [names.index("metal:Li")],
        "working_Li_mol_per_A": [names.index(k) for k in spec.positive.keys("positive")],
        "signed_charge_C_per_A": [names.index("discharged_charge_C")],
        "signed_work_J_per_A": [names.index("delivered_work_J")],
    }
    inventory = []
    for point in spectrum.response.points:
        responses = {}
        for label, indices in groups.items():
            value = None if point.response is None else sum(point.response[indices, 0])
            responses[label] = None if value is None else {
                "real": float(value.real), "imag": float(value.imag),
            }
        inventory.append({"frequency_Hz": point.frequency_Hz, "responses": responses})
    if state_record(state) != original:
        raise RuntimeError("frequency analysis mutated its input state")
    report = {
        "model": spec.to_dict(), "operating_state": original,
        "scope": "synthetic fixed-geometry finite metal; read-only stationary derivative",
        "spectrum": spectrum.to_dict(), "full_response": spectrum.response.to_dict(),
        "inventory_responses": inventory, "state_unchanged": True,
    }
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
                      "frequencies": len(report["inventory_responses"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
