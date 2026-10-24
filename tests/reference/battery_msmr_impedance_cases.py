"""Fixed 10-unit/50-route stationary MSMR inputs and output identities."""

import json
from dataclasses import replace

import numpy as np
from battery_msmr_cases import case_data as time_case
from battery_msmr_cases import native_initial_state, native_spec

F, R = 96485.33212, 8.314462618
FREQUENCIES = np.geomspace(1e-5, 1e-2, 17)
GRIDS = {"coarse": 80, "fine": 160}
RTOL = 0.005
REFUSALS = (
    "missing-Li",
    "missing-QW",
    "Faraday-inconsistent",
    "zero-local-Li",
    "surface-below-margin",
    "voltage-face",
    "zero-frequency",
    "nan-frequency",
)


def case_data(index):
    if index not in range(1, 11):
        raise ValueError("MSMR EIS fixes M01-M10")
    data = time_case({1: 4, 2: 2, 3: 3, 4: 5, 5: 8, 9: 7}.get(index, 4))
    data.update(id=f"M{index:02d}", waveform=((1.0, 0.0, "A"),))
    if index == 6:
        for e, radius, volume in zip(
            data["electrodes"], (3e-6, 8e-6), (0.4e-6, 0.8e-6), strict=True
        ):
            e.update(radius_m=radius, active_volume_m3=volume)
    elif index == 7:
        data["resistance_ohm"] = 0.12
    elif index == 8:
        data.update(temperature_K=310.15, electrolyte_concentration_mol_m3=750.0)
    return data


def binding(data, kind, cells, reason=None):
    return {
        "unit": data["id"],
        "inputs": json.loads(json.dumps(data)),
        "kind": kind,
        "cells": cells,
        "reason": reason,
    }


def output_schema(data, cells=0):
    rows = [
        ("Z_ohm", "ohm", 0.002),
        ("current_A_A", "A/A", 1e-8),
        ("charge_C_A", "C/A", 1e-5),
        ("work_J_A", "J/A", 1e-5),
    ]
    for e in data["electrodes"]:
        name = e["name"]
        rows.extend(
            (f"{name}:{key}", unit, tol)
            for key, unit, tol in (
                ("U", "V/A", 2e-4),
                ("eta", "V/A", 2e-4),
                ("surface", "1/A", 5e-4),
                ("source_surface", "1/A", 5e-4),
                ("average", "1/A", 5e-4),
                ("amount", "mol/A", 1e-9),
            )
        )
        for j in range(len(e["rows"])):
            rows.extend(
                (f"{name}:{key}{j}", unit, tol)
                for key, unit, tol in (
                    ("kinetic_x", "1/A", 5e-4),
                    ("source_x", "1/A", 5e-4),
                    ("current", "A/A", 5e-4),
                )
            )
    for e in data["electrodes"]:
        for key, unit, tol in (("profile_U", "V/A", 2e-4), ("profile_x", "1/A", 5e-4)):
            rows.extend((f"{e['name']}:{key}:{i}", unit, tol) for i in range(cells))
    return rows


def response_order(data, cells=0):
    return [r[0] for r in output_schema(data, cells)]


def response_units(data, cells=0):
    return [r[1] for r in output_schema(data, cells)]


def refusal_input(data, name, cells):
    from chemworld.physchem.battery_msmr import msmr_readout

    spec = native_spec(data, cells)
    state = native_initial_state(spec, data, cells)
    frequencies = [0.01]
    amounts = dict(state.amounts_mol)
    keys = spec.negative.keys("negative")
    if name == "missing-Li":
        del amounts[keys[0]]
        state = replace(state, amounts_mol=amounts)
    elif name == "missing-QW":
        state = replace(state, coordinates={})
    elif name == "Faraday-inconsistent":
        amounts[keys[0]] *= 0.9
        state = replace(state, amounts_mol=amounts)
    elif name in ("zero-local-Li", "surface-below-margin"):
        capacity = spec.negative.mesh.cell_volumes_m3 * spec.negative.maximum_concentration_mol_m3
        selected = [0] if name == "zero-local-Li" else [cells - 2, cells - 1]
        previous = sum(amounts[keys[i]] for i in selected)
        for i in selected:
            amounts[keys[i]] = 0.0 if name == "zero-local-Li" else 0.005 * capacity[i]
        moved = previous - sum(amounts[keys[i]] for i in selected)
        others = [i for i in range(cells) if i not in selected]
        denominator = sum(capacity[i] for i in others)
        for i in others:
            amounts[keys[i]] += moved * capacity[i] / denominator
        state = replace(state, amounts_mol=amounts)
    elif name == "voltage-face":
        spec = replace(spec, minimum_voltage_V=msmr_readout(spec, state, 0.0).voltage_V)
    elif name == "zero-frequency":
        frequencies = [0.0]
    elif name == "nan-frequency":
        frequencies = [float("nan")]
    else:
        raise ValueError("unknown fixed refusal")
    return spec, state, frequencies


def route_plan(index):
    if index <= 8:
        return [(f"{kind}_{g}", kind, g) for kind in ("native", "reference") for g in GRIDS] + [
            ("analytic", "analytic", None)
        ]
    if index == 9:
        return [(k, k, "coarse") for k in ("native", "reference")]
    return [(name, "refusal", "coarse") for name in REFUSALS]
