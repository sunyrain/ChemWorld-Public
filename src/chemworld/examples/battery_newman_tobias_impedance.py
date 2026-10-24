"""Read-only stationary frequency response of the Newman-Tobias example.

Two mean lithium stocks provide OCV inventory response. The existing thickness
mesh resolves porous electrical transport; there is no radial or salt diffusion.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np

from chemworld.examples.battery_newman_tobias import default_spec
from chemworld.physchem.battery_impedance import battery_impedance


def state_record(state):
    return {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state) for value in (getattr(state, field.name),)
    }


def run(output, *, spec=None, state=None, frequencies_Hz=(1e-5, 1e-2, 1.)):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier analysis; choose a new output path")
    spec = default_spec(cells=(24, 12, 24)) if spec is None else spec
    state = spec.initial_state() if state is None else state
    original = state_record(state)
    spectrum = battery_impedance(spec, frequencies_Hz, state=state)
    names = [v.name for v in spectrum.response.variables]
    groups = {
        "negative_Li_mol_per_A": [names.index("negative:Li")],
        "positive_Li_mol_per_A": [names.index("positive:Li")],
        "signed_charge_C_per_A": [names.index("discharged_charge_C")],
        "signed_work_J_per_A": [names.index("delivered_work_J")],
    }
    algebraic_indices = [names.index(key) for key in spec.algebraic_names]
    cell_labels = [
        f"{electrode}:cell:{i}"
        for electrode, region in (("negative", 0), ("positive", 2))
        for i in range(spec.geometry.cells[region])
    ]
    inventory = []
    spatial = []
    for point in spectrum.response.points:
        responses = {}
        for label, indices in groups.items():
            value = None if point.response is None else sum(point.response[indices, 0])
            responses[label] = None if value is None else {
                "real": float(value.real), "imag": float(value.imag),
            }
        inventory.append({"frequency_Hz": point.frequency_Hz, "responses": responses})
        # Each electrode stores N-1 independent reaction currents. The original
        # electrical map reconstructs its last cell from terminal conservation.
        currents = None if point.response is None else (
            spec.electrical_maps[0] @ point.response[algebraic_indices, 0]
        )
        spatial.append({
            "frequency_Hz": point.frequency_Hz,
            "cell_current_A_per_A": {
                label: None if currents is None else {
                    "real": float(currents[i].real), "imag": float(currents[i].imag),
                }
                for i, label in enumerate(cell_labels)
            },
        })
    if state_record(state) != original:
        raise RuntimeError("frequency analysis mutated its input state")
    report = {
        "model": spec.to_dict(), "operating_state": original,
        "scope": "synthetic isothermal Newman-Tobias; stationary derivative; two mean Li stocks",
        "reaction_current_variables": list(spec.reaction_keys),
        "cell_current_labels": cell_labels,
        "cell_current_responses": spatial,
        "thickness_cells": list(spec.geometry.cells),
        "spectrum": spectrum.to_dict(), "full_response": spectrum.response.to_dict(),
        "inventory_responses": inventory, "state_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", type=int, nargs=3, default=(24, 12, 24))
    args = parser.parse_args()
    report = run(args.output, spec=default_spec(cells=args.cells),
                 frequencies_Hz=np.logspace(-5, 1, 13))
    print(json.dumps({"success": report["spectrum"]["success"],
                      "frequencies": len(report["inventory_responses"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
